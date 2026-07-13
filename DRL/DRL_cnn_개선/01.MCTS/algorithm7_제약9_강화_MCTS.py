from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict

import numpy as np
import onnxruntime as ort

from buffer_manager import BufferManager


# ---------------------------------------------------------------------------
# 입출력 스키마  (수정 금지)
# ---------------------------------------------------------------------------

class BoxInput(TypedDict):
    step: int
    id: int
    size: List[float]   # [length, width, height]
    mass: float


class PlacedBox(TypedDict):
    step: int
    id: int
    size: List[float]
    mass: float
    position: List[float]
    rotation: int       # 0 또는 90


class RunResult(TypedDict):
    buffer_size: int
    sequence: List[PlacedBox]
    terminated: bool
    terminated_step: Optional[int]
    finished_by_user: bool


# ---------------------------------------------------------------------------
# 설정 dataclass  (수정 금지)
# ---------------------------------------------------------------------------

@dataclass
class PalletConfig:
    length: float
    width: float
    height: float


# ---------------------------------------------------------------------------
# 참가자 개발 영역
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmConfig:
    allow_rotation: bool
    buffer_size: int


_DEFAULTS = dict(cell=0.01, max_mass=6.0, k_hol=1.8, com_alpha0=0.45, com_alpha1=0.20)


class Palletizer:

    def __init__(self, pallet_cfg: PalletConfig, algo_cfg: AlgorithmConfig) -> None:
        self.pallet = pallet_cfg
        self.algo = algo_cfg

        here = os.path.dirname(os.path.abspath(__file__))
        meta_path = os.path.join(here, "model_meta.json")
        meta = dict(_DEFAULTS)
        onnx_name = "model.onnx"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            meta.update(loaded)
            onnx_name = loaded.get("onnx_path", onnx_name)

        self.cell = float(meta["cell"])
        self.max_mass = float(meta["max_mass"])
        self.k_hol = float(meta["k_hol"])
        self.com_a0 = float(meta["com_alpha0"])
        self.com_a1 = float(meta["com_alpha1"])

        self.Lc = int(round(self.pallet.length / self.cell))
        self.Wc = int(round(self.pallet.width / self.cell))
        self.Hc = int(round(self.pallet.height / self.cell))
        self.N = int(meta.get("N", max(1, self.algo.buffer_size)))

        onnx_path = os.path.join(here, onnx_name)
        if not os.path.exists(onnx_path):
            found = [f for f in os.listdir(here) if f.lower().endswith(".onnx")]
            if not found:
                raise FileNotFoundError(f".onnx not found in {here}")
            onnx_path = os.path.join(here, sorted(found)[0])
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        # ---- MCTS 파라미터 ----
        # K = 각 스텝에서 시도할 상위 후보 수 (박스,회전,위치)
        # D = 각 후보에서 시뮬레이션할 rollout 깊이
        # 시간 예산: 스텝당 (K + K*D) × forward_ms. K=3,D=2 이면 9회 × 30ms ≈ 270ms
        # 250 스텝 × 270ms = 67초 (90초 제한 여유)
        self.mcts_K = int(meta.get("mcts_K", 3))
        self.mcts_D = int(meta.get("mcts_D", 2))
        self.mcts_enabled = bool(meta.get("mcts_enabled", True))

        self._reset_state()

    # -----------------------------------------------------------------------
    # 상태 관리
    # -----------------------------------------------------------------------
    def _reset_state(self) -> None:
        self.height = np.zeros((self.Lc, self.Wc), dtype=np.int32)
        self.topmass = np.zeros((self.Lc, self.Wc), dtype=np.float32)
        self.sequence: List[PlacedBox] = []
        self.finished = False
        self.terminated_step: Optional[int] = None
        self.finished_by_user = False
        self.cursor_x = 0.0
        self.row_depth = 0.0
        self.layer_height = 0.0

    # -----------------------------------------------------------------------
    # 참가자 수정 가능 함수
    # -----------------------------------------------------------------------
    def should_finish(self, current_buffer: List[BoxInput]) -> bool:
        return False

    # -----------------------------------------------------------------------
    # 헬퍼: 셀 단위 footprint
    # -----------------------------------------------------------------------
    def _cells(self, size, rot):
        l, w, h = float(size[0]), float(size[1]), float(size[2])
        if rot == 1:
            l, w = w, l
        return (max(1, math.ceil(l / self.cell)),
                max(1, math.ceil(w / self.cell)),
                max(1, math.ceil(h / self.cell)))

    # -----------------------------------------------------------------------
    # 헬퍼: feasibility mask (단일 박스, 단일 회전)
    # -----------------------------------------------------------------------
    def _mask_one(self, box, rot):
        lc, wc, hc = self._cells(box["size"], rot)
        Lc, Wc = self.Lc, self.Wc
        if lc > Lc or wc > Wc:
            return np.zeros((Lc, Wc), dtype=np.float32)

        H = self.height
        from numpy.lib.stride_tricks import sliding_window_view
        win_r = sliding_window_view(H, lc, axis=0).max(axis=2)
        base = sliding_window_view(win_r, wc, axis=1).max(axis=2)
        vr, vc = base.shape

        support_cnt = np.zeros((vr, vc), dtype=np.int32)
        corner_cnt = np.zeros((vr, vc), dtype=np.int32)              # 네 모서리 지지 카운트
        supp_mass_min = np.full((vr, vc), np.inf, dtype=np.float32)

        # 네 모서리 상대 좌표
        corners = {(0, 0), (lc - 1, 0), (0, wc - 1), (lc - 1, wc - 1)}

        for di in range(lc):
            for dj in range(wc):
                sub = H[di:di + vr, dj:dj + vc]
                eq = (sub == base)
                support_cnt += eq
                tm = self.topmass[di:di + vr, dj:dj + vc]
                supp_mass_min = np.where(eq, np.minimum(supp_mass_min, tm), supp_mass_min)
                if (di, dj) in corners:
                    corner_cnt += eq

        area = lc * wc

        # (2) 높이 초과
        overflow = (base + hc) > self.Hc

        # (3) 지지 기하 (강화 조건):
        #     면적 지지율 >= 75%  AND  네 모서리 중 3개 이상 지지
        #     바닥(base == 0)이면 무조건 통과.
        floor = (base == 0)
        geom = floor | ((support_cnt >= 0.75 * area) & (corner_cnt >= 3))

        # (4) CoM 마진 → 제거

        # (5) heavy-on-light (강화): 배치 박스 질량 <= 2 × 최소 지지 박스 질량
        safe_supp = np.where(np.isfinite(supp_mass_min), supp_mass_min, 0.0)
        hol_ok = floor | (float(box["mass"]) <= 2.0 * safe_supp)

        ok = (~overflow) & geom & hol_ok
        full = np.zeros((Lc, Wc), dtype=np.float32)
        full[:vr, :vc] = ok.astype(np.float32)
        return full

    def _build_state_single(self, box):
        Lc, Wc = self.Lc, self.Wc
        chans = [self.height.astype(np.float32) / self.Hc,
                 self.topmass / self.max_mass]
        for i in range(self.N):
            if i == 0:
                l, w, h = box["size"]
                chans += [np.full((Lc, Wc), l, np.float32),
                          np.full((Lc, Wc), w, np.float32),
                          np.full((Lc, Wc), h, np.float32),
                          np.full((Lc, Wc), float(box["mass"]) / self.max_mass, np.float32)]
            else:
                chans += [np.zeros((Lc, Wc), np.float32)] * 4
        return np.stack(chans, axis=0)[None].astype(np.float32)

    def _decode(self, action):
        per_rot = self.Lc * self.Wc
        ci = action // (2 * per_rot)
        rem = action % (2 * per_rot)
        rot = rem // per_rot
        pos = rem % per_rot
        x, y = pos // self.Wc, pos % self.Wc
        return ci, rot, int(x), int(y)

    # -----------------------------------------------------------------------
    # _find_position: run()이 호출. 박스 하나의 최적 (x,y,z,dims,rotation) 반환.
    #                 배치 불가면 None 반환.
    # -----------------------------------------------------------------------
    def _find_position(self, box):
        try:
            M = np.zeros((self.N, 2, self.Lc, self.Wc), dtype=np.float32)
            for rot in (0, 1):
                M[0, rot] = self._mask_one(box, rot)
            if M.sum() == 0:
                return None

            state = self._build_state_single(box)
            logits, _ = self.sess.run(["logits", "value"], {"state": state})
            logits = np.asarray(logits).reshape(-1)
            mask_flat = M.reshape(-1)
            if mask_flat.shape[0] != logits.shape[0]:
                return None

            masked = np.where(mask_flat > 0, logits, -1e30)
            per_cand = 2 * self.Lc * self.Wc
            sub = masked[:per_cand]
            action = int(sub.argmax())
            _, rot, x, y = self._decode(action)

            lc, wc, hc = self._cells(box["size"], rot)
            x = max(0, min(self.Lc - lc, x))
            y = max(0, min(self.Wc - wc, y))
            base = int(self.height[x:x + lc, y:y + wc].max())

            l, w, h = float(box["size"][0]), float(box["size"][1]), float(box["size"][2])
            if rot == 1:
                l, w = w, l

            xm = x * self.cell
            ym = y * self.cell
            zm = base * self.cell
            rotation = 90 if rot == 1 else 0
            return xm, ym, zm, (l, w, h), rotation
        except Exception:
            return None

    # =====================================================================
    #  MCTS 관련 헬퍼
    # =====================================================================

    def _find_topk_candidates(self, current_buffer, K):
        """
        현재 버퍼(current_buffer)의 모든 박스에 대해 CNN forward 를 한 번씩 돌리고,
        각 (박스, 회전, 위치) 조합의 logits 를 모아 상위 K 개를 반환.

        반환: List of dict with keys:
              {"box_idx": int, "rot": int, "x": int, "y": int, "logit": float}
              배치 불가능하면 빈 리스트.
        """
        all_candidates = []
        for bi, box in enumerate(current_buffer):
            if box is None:
                continue
            try:
                M = np.zeros((self.N, 2, self.Lc, self.Wc), dtype=np.float32)
                for rot in (0, 1):
                    M[0, rot] = self._mask_one(box, rot)
                if M.sum() == 0:
                    continue

                state = self._build_state_single(box)
                logits, _ = self.sess.run(["logits", "value"], {"state": state})
                logits = np.asarray(logits).reshape(-1)
                mask_flat = M.reshape(-1)
                if mask_flat.shape[0] != logits.shape[0]:
                    continue

                masked = np.where(mask_flat > 0, logits, -1e30)
                per_cand = 2 * self.Lc * self.Wc
                sub = masked[:per_cand]

                # 이 박스의 상위 K 개 위치 (전체 후보 풀에서 상위 K 를 뽑기 위해)
                topk_local = min(K, int((sub > -1e29).sum()))
                if topk_local <= 0:
                    continue
                top_idx = np.argpartition(sub, -topk_local)[-topk_local:]
                for a in top_idx:
                    _, rot, x, y = self._decode(int(a))
                    lc, wc, _ = self._cells(box["size"], rot)
                    x = max(0, min(self.Lc - lc, x))
                    y = max(0, min(self.Wc - wc, y))
                    all_candidates.append({
                        "box_idx": bi, "rot": rot, "x": x, "y": y,
                        "logit": float(sub[a])
                    })
            except Exception:
                continue

        # 전체 풀에서 상위 K 개
        all_candidates.sort(key=lambda c: -c["logit"])
        return all_candidates[:K]

    def _snapshot_state(self):
        """height, topmass 를 복사해서 반환 (rollout 후 복원용)."""
        return (self.height.copy(), self.topmass.copy(),
                len(self.sequence), self.placed_meta_len_snapshot())

    def placed_meta_len_snapshot(self):
        """rollout 중 sequence 에 추가된 것들을 뒤로 되돌리기 위한 길이."""
        return len(self.sequence)

    def _restore_state(self, snap):
        """스냅샷으로 복원."""
        H, T, seq_len, _ = snap
        self.height = H
        self.topmass = T
        # sequence 뒤로 rollout 중 추가된 것 제거
        del self.sequence[seq_len:]

    def _simulate_place(self, box, rot, x, y):
        """
        rollout 용 내부 배치. sequence 는 건드리지 않고 height/topmass 만 갱신.
        반환: 이번 배치가 채운 부피(팔레트 대비 비율).
        """
        lc, wc, hc = self._cells(box["size"], rot)
        base = int(self.height[x:x + lc, y:y + wc].max())
        self.height[x:x + lc, y:y + wc] = base + hc
        self.topmass[x:x + lc, y:y + wc] = float(box["mass"])
        l, w, h = float(box["size"][0]), float(box["size"][1]), float(box["size"][2])
        pallet_vol = self.pallet.length * self.pallet.width * self.pallet.height
        return (l * w * h) / pallet_vol

    def _rollout_score(self, first_choice, current_buffer, remaining_boxes, D):
        """
        first_choice 를 배치했다고 가정하고 D 스텝 앞까지 greedy rollout,
        추가로 채운 부피 비율의 합을 반환.

        first_choice: {"box_idx","rot","x","y",...}
        current_buffer: 지금 스텝의 버퍼 (list of BoxInput or None)
        remaining_boxes: 아직 컨베이어에서 안 온 박스들 (list)
        """
        # first_choice 배치
        chosen_box = current_buffer[first_choice["box_idx"]]
        gained = self._simulate_place(chosen_box, first_choice["rot"],
                                       first_choice["x"], first_choice["y"])

        # 버퍼 업데이트 (rollout 내에서만)
        sim_buffer = [b for i, b in enumerate(current_buffer) if i != first_choice["box_idx"]]
        sim_remaining = list(remaining_boxes)
        # 새 박스 하나 채움
        if sim_remaining:
            sim_buffer.append(sim_remaining.pop(0))

        # D 스텝 greedy rollout
        for _ in range(D):
            if not sim_buffer:
                break
            # 이 스텝의 최선 후보 하나
            best = None
            for bi, box in enumerate(sim_buffer):
                if box is None:
                    continue
                try:
                    M = np.zeros((self.N, 2, self.Lc, self.Wc), dtype=np.float32)
                    for rot in (0, 1):
                        M[0, rot] = self._mask_one(box, rot)
                    if M.sum() == 0:
                        continue
                    state = self._build_state_single(box)
                    logits, _ = self.sess.run(["logits", "value"], {"state": state})
                    logits = np.asarray(logits).reshape(-1)
                    mask_flat = M.reshape(-1)
                    if mask_flat.shape[0] != logits.shape[0]:
                        continue
                    masked = np.where(mask_flat > 0, logits, -1e30)
                    per_cand = 2 * self.Lc * self.Wc
                    sub = masked[:per_cand]
                    a = int(sub.argmax())
                    v = float(sub[a])
                    if best is None or v > best["logit"]:
                        _, rot, x, y = self._decode(a)
                        lc, wc, _ = self._cells(box["size"], rot)
                        x = max(0, min(self.Lc - lc, x))
                        y = max(0, min(self.Wc - wc, y))
                        best = {"box_idx": bi, "rot": rot, "x": x, "y": y, "logit": v}
                except Exception:
                    continue

            if best is None:
                break

            chosen = sim_buffer[best["box_idx"]]
            gained += self._simulate_place(chosen, best["rot"], best["x"], best["y"])
            sim_buffer = [b for i, b in enumerate(sim_buffer) if i != best["box_idx"]]
            if sim_remaining:
                sim_buffer.append(sim_remaining.pop(0))

        return gained

    def _mcts_choose(self, current_buffer, remaining_boxes):
        """
        MCTS 로 현재 스텝의 최선 (박스, 회전, 위치) 선택.
        반환: dict {"box_idx","rot","x","y"} 또는 None.
        """
        # 1) 상위 K 후보 뽑기
        candidates = self._find_topk_candidates(current_buffer, self.mcts_K)
        if not candidates:
            return None
        if len(candidates) == 1 or self.mcts_D <= 0:
            return candidates[0]

        # 2) 각 후보에 대해 D 깊이 rollout 점수 계산
        best_score = -1.0
        best_choice = None
        for cand in candidates:
            snap = self._snapshot_state()
            try:
                score = cand["logit"] * 0.0  # logit 은 무시, rollout gained 만 사용
                score = self._rollout_score(cand, current_buffer, remaining_boxes, self.mcts_D)
            except Exception:
                score = -1.0
            self._restore_state(snap)

            if score > best_score:
                best_score = score
                best_choice = cand

        return best_choice

    # -----------------------------------------------------------------------
    # _append_placed: run()이 호출. PlacedBox 기록 + height/topmass 갱신.
    # -----------------------------------------------------------------------
    def _append_placed(self, box, dims, rotation, x, y, z):
        dx, dy, dz = dims

        self.sequence.append({
            "step": int(box["step"]),
            "id": int(box["id"]),
            "size": [round(dx, 3), round(dy, 3), round(dz, 3)],
            "mass": float(box["mass"]),
            "position": [
                round(x + dx / 2.0, 3),
                round(y + dy / 2.0, 3),
                round(z + dz / 2.0, 3),
            ],
            "rotation": int(rotation),
        })

        xc = int(round(x / self.cell))
        yc = int(round(y / self.cell))
        lc = max(1, math.ceil(dx / self.cell))
        wc = max(1, math.ceil(dy / self.cell))
        hc = max(1, math.ceil(dz / self.cell))
        base = int(round(z / self.cell))
        xc = max(0, min(self.Lc - lc, xc))
        yc = max(0, min(self.Wc - wc, yc))
        self.height[xc:xc + lc, yc:yc + wc] = base + hc
        self.topmass[xc:xc + lc, yc:yc + wc] = float(box["mass"])

        self.cursor_x += dx
        self.row_depth = max(self.row_depth, dy)
        self.layer_height = max(self.layer_height, dz)

    # -----------------------------------------------------------------------
    # 실행 진입점  (수정 금지 시그니처)
    # -----------------------------------------------------------------------
    def run(self, boxes: List[BoxInput]) -> RunResult:
        self._reset_state()

        buf = BufferManager(self.algo.buffer_size)
        buf.reset(boxes)

        # remaining boxes (원 시퀀스에서 아직 안 나온 박스들) 추적용
        # BufferManager 가 내부적으로 컨베이어를 관리하니, MCTS rollout 에서만
        # 별도 복사본으로 시뮬. 매 스텝 boxes[처음부터 사용된 만큼:] 로 근사.
        # 여기선 간단히 "다 사용된 것 이후" 를 근사하기 위해 인덱스를 추적.
        step_counter = [0]  # closure 변수

        while buf.has_pending():
            if self.algo.buffer_size == 0:
                current = [buf.peek_next()]
            else:
                current = buf.get_buffer()

            if self.should_finish(current):
                self.finished_by_user = True
                break

            # ---------- MCTS 경로 ----------
            if self.mcts_enabled:
                # remaining: 아직 버퍼에도 안 온 박스들의 근사 (rollout 용)
                # 정확한 미래는 알 수 없지만 boxes 자체가 시퀀스라
                # 사용된 만큼 이후의 것을 넘겨줌
                used = step_counter[0] + len(current)
                remaining = boxes[used:used + self.mcts_D + 2] if used < len(boxes) else []

                choice = self._mcts_choose(current, remaining)
                if choice is None:
                    self.finished = True
                    if current:
                        self.terminated_step = int(current[0]["step"])
                    break

                selected_index = choice["box_idx"]
                box = current[selected_index]
                rot = choice["rot"]
                x, y = choice["x"], choice["y"]

                lc, wc, hc = self._cells(box["size"], rot)
                base = int(self.height[x:x + lc, y:y + wc].max())
                l, w, h = float(box["size"][0]), float(box["size"][1]), float(box["size"][2])
                if rot == 1:
                    l, w = w, l

                self._append_placed(
                    box=box,
                    dims=(l, w, h),
                    rotation=90 if rot == 1 else 0,
                    x=x * self.cell,
                    y=y * self.cell,
                    z=base * self.cell,
                )

                if self.algo.buffer_size == 0:
                    buf.pop_next()
                else:
                    buf.pop_selected(selected_index)

                step_counter[0] += 1
                continue

            # ---------- 기존 (MCTS 비활성) 경로 ----------
            placed = False
            for selected_index, box in enumerate(current):
                found = self._find_position(box)

                if found is None:
                    continue

                x, y, z, dims, rotation = found

                self._append_placed(
                    box=box,
                    dims=dims,
                    rotation=rotation,
                    x=x,
                    y=y,
                    z=z,
                )

                if self.algo.buffer_size == 0:
                    buf.pop_next()
                else:
                    buf.pop_selected(selected_index)

                placed = True
                break

            if placed:
                step_counter[0] += 1
                continue

            self.finished = True

            if current:
                self.terminated_step = int(current[0]["step"])

            break

        return {
            "buffer_size": self.algo.buffer_size,
            "sequence": self.sequence,
            "terminated": self.finished,
            "terminated_step": self.terminated_step,
            "finished_by_user": self.finished_by_user,
        }