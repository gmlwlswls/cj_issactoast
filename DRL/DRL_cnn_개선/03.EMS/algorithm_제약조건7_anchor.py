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
        corner_cnt = np.zeros((vr, vc), dtype=np.int32)
        supp_mass_min = np.full((vr, vc), np.inf, dtype=np.float32)
        sum_di = np.zeros((vr, vc), dtype=np.float32)
        sum_dj = np.zeros((vr, vc), dtype=np.float32)

        corners = {(0, 0), (lc - 1, 0), (0, wc - 1), (lc - 1, wc - 1)}
        for di in range(lc):
            for dj in range(wc):
                sub = H[di:di + vr, dj:dj + vc]
                eq = (sub == base)
                support_cnt += eq
                sum_di += eq * di
                sum_dj += eq * dj
                tm = self.topmass[di:di + vr, dj:dj + vc]
                supp_mass_min = np.where(eq, np.minimum(supp_mass_min, tm), supp_mass_min)
                if (di, dj) in corners:
                    corner_cnt += eq

        area = lc * wc
        ratio = support_cnt / area
        overflow = (base + hc) > self.Hc
        geom = (((ratio > 0.60) & (corner_cnt >= 4)) |
                ((ratio > 0.80) & (corner_cnt >= 3)) |
                (ratio > 0.95))

        with np.errstate(invalid="ignore"):
            cen_di = np.where(support_cnt > 0, sum_di / np.maximum(support_cnt, 1), lc / 2.0)
            cen_dj = np.where(support_cnt > 0, sum_dj / np.maximum(support_cnt, 1), wc / 2.0)
        off_i = np.abs(cen_di - (lc - 1) / 2.0) / max(lc / 2.0, 1e-6)
        off_j = np.abs(cen_dj - (wc - 1) / 2.0) / max(wc / 2.0, 1e-6)
        m = float(box["mass"]) / self.max_mass
        alpha = self.com_a0 + (self.com_a1 - self.com_a0) * m
        com_ok = (off_i <= alpha) & (off_j <= alpha)

        floor = (base == 0)
        safe_supp = np.where(np.isfinite(supp_mass_min), supp_mass_min, 0.0)
        hol_ok = floor | (float(box["mass"]) <= self.k_hol * safe_supp)

        ok = (~overflow) & geom & com_ok & hol_ok
        full = np.zeros((Lc, Wc), dtype=np.float32)
        full[:vr, :vc] = ok.astype(np.float32)
        return full

    # -----------------------------------------------------------------------
    # 헬퍼: 상태 텐서 (단일 박스를 슬롯0에 넣음)
    # -----------------------------------------------------------------------
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

    # =====================================================================
    #  Phase 1 헬퍼: CNN + Anchor 재순위
    # =====================================================================
    def _contact_score(self, rot, x, y, box):
        lc, wc, hc = self._cells(box["size"], rot)
        Lc, Wc, Hc = self.Lc, self.Wc, self.Hc
        H = self.height
        base = int(H[x:x+lc, y:y+wc].max())
        score = 0.0
        p = max(1, 2*(lc+wc))
        if x == 0:       score += lc/p
        if x+lc >= Lc:   score += lc/p
        if y == 0:        score += wc/p
        if y+wc >= Wc:   score += wc/p
        if x > 0:         score += ((H[x-1, y:y+wc] >= base) & (H[x-1, y:y+wc] > 0)).sum()/p
        if x+lc < Lc:    score += ((H[x+lc, y:y+wc] >= base) & (H[x+lc, y:y+wc] > 0)).sum()/p
        if y > 0:         score += ((H[x:x+lc, y-1] >= base) & (H[x:x+lc, y-1] > 0)).sum()/p
        if y+wc < Wc:    score += ((H[x:x+lc, y+wc] >= base) & (H[x:x+lc, y+wc] > 0)).sum()/p
        score += (1.0 - base/Hc) * 0.5
        return score

    # =====================================================================
    #  Phase 2: 바닥 갭 채우기
    #   height=0 인 빈 바닥에 박스를 Z=0 으로 배치. 물리 조건 없음.
    #   접촉 점수로 기존 박스에 밀착된 위치 선호.
    # =====================================================================
    def _phase2_floor_gap(self, box):
        from numpy.lib.stride_tricks import sliding_window_view
        H = self.height
        best_result = None
        best_contact = -1.0

        for rot in (0, 1):
            lc, wc, hc = self._cells(box["size"], rot)
            if lc > self.Lc or wc > self.Wc or hc > self.Hc:
                continue
            win_r = sliding_window_view(H, lc, axis=0).max(axis=2)
            base_map = sliding_window_view(win_r, wc, axis=1).max(axis=2)
            floor_pos = np.where(base_map == 0)
            if len(floor_pos[0]) == 0:
                continue
            for idx in range(min(20, len(floor_pos[0]))):
                x, y = int(floor_pos[0][idx]), int(floor_pos[1][idx])
                x = max(0, min(self.Lc - lc, x))
                y = max(0, min(self.Wc - wc, y))
                contact = self._contact_score(rot, x, y, box)
                if contact > best_contact:
                    best_contact = contact
                    l, w, h = float(box["size"][0]), float(box["size"][1]), float(box["size"][2])
                    if rot == 1: l, w = w, l
                    best_result = (x * self.cell, y * self.cell, 0.0,
                                   (l, w, h), 90 if rot == 1 else 0)
        return best_result

    # =====================================================================
    #  Phase 3: 기존 박스의 column 위에 쌓기
    #
    #  로직:
    #   1) self.sequence 에서 바닥 박스(z_bottom ≈ 0) 를 모두 찾음
    #   2) 각 바닥 박스의 (x,y) 풋프린트로 column 을 정의
    #   3) column 안에서 가장 높이 쌓인 박스(최상단)를 찾음
    #   4) 현재 박스(버퍼)가 최상단 박스보다:
    #      - x,y 가 같거나 작고 (넘어서지 않음)
    #      - 질량이 같거나 작고
    #      - 쌓은 후 컨테이너 높이를 안 넘으면
    #   5) x,y 크기 편차가 가장 작은(밀착) column 에 배치
    #   6) 회전(0°, 90°) 모두 시도
    # =====================================================================
    def _phase3_column_stack(self, box):
        if not self.sequence:
            return None

        box_mass = float(box["mass"])
        pallet_h = self.pallet.height
        best_result = None
        best_deviation = float("inf")

        # 1) 바닥 박스 찾기 (z_bottom ≈ 0)
        bottom_boxes = []
        for placed in self.sequence:
            pz = placed["position"][2]
            pdz = placed["size"][2]
            z_bottom = pz - pdz / 2.0
            if abs(z_bottom) < 0.005:   # 바닥에 놓인 박스
                bottom_boxes.append(placed)

        if not bottom_boxes:
            return None

        # 2) 각 바닥 박스의 column 에서 최상단 박스 찾기
        for bb in bottom_boxes:
            bb_x0 = bb["position"][0] - bb["size"][0] / 2.0
            bb_y0 = bb["position"][1] - bb["size"][1] / 2.0
            bb_x1 = bb_x0 + bb["size"][0]
            bb_y1 = bb_y0 + bb["size"][1]

            # column 안의 최상단 박스 찾기
            topmost = bb
            topmost_ztop = bb["position"][2] + bb["size"][2] / 2.0
            for placed in self.sequence:
                px = placed["position"][0]
                py = placed["position"][1]
                # centroid 가 column 안에 있는지
                if bb_x0 - 0.001 <= px <= bb_x1 + 0.001 and \
                   bb_y0 - 0.001 <= py <= bb_y1 + 0.001:
                    ztop = placed["position"][2] + placed["size"][2] / 2.0
                    if ztop > topmost_ztop:
                        topmost = placed
                        topmost_ztop = ztop

            # 최상단 박스 정보
            top_l = topmost["size"][0]   # 배치된 상태의 l (이미 회전 반영)
            top_w = topmost["size"][1]   # 배치된 상태의 w
            top_mass = topmost["mass"]
            top_z = topmost_ztop         # 최상단 표면 z 좌표

            # 3) 현재 박스가 최상단 위에 올라갈 수 있는지 확인
            for rot in (0, 1):
                bl, bw, bh = float(box["size"][0]), float(box["size"][1]), float(box["size"][2])
                if rot == 1:
                    bl, bw = bw, bl

                # 조건: 크기 ≤ 최상단 (넘어서지 않음)
                if bl > top_l + 0.001 or bw > top_w + 0.001:
                    continue
                # 조건: 질량 ≤ 최상단 질량
                if box_mass > top_mass + 0.001:
                    continue
                # 조건: 높이 제한
                if top_z + bh > pallet_h + 0.001:
                    continue

                # 크기 편차 (작을수록 밀착)
                deviation = abs(bl - top_l) + abs(bw - top_w)

                if deviation < best_deviation:
                    best_deviation = deviation
                    # 최상단 박스 위에 중앙 정렬로 배치
                    # anchor (front-left-bottom) 좌표 계산
                    top_cx = topmost["position"][0]
                    top_cy = topmost["position"][1]
                    anchor_x = top_cx - bl / 2.0
                    anchor_y = top_cy - bw / 2.0
                    anchor_z = top_z
                    best_result = (anchor_x, anchor_y, anchor_z,
                                   (bl, bw, bh),
                                   90 if rot == 1 else 0)

        return best_result

    # -----------------------------------------------------------------------
    # _find_position: 3단계 전략
    #   Phase 1: CNN + anchor (기존 모델 그대로)
    #   Phase 2: 바닥 갭 채우기 (CNN 포기 후, 물리조건 없이)
    #   Phase 3: column 위에 쌓기 (크기+질량+높이만 검사)
    # -----------------------------------------------------------------------
    def _find_position(self, box):
        # ---- Phase 1: CNN + anchor ----
        try:
            M = np.zeros((self.N, 2, self.Lc, self.Wc), dtype=np.float32)
            for rot in (0, 1):
                M[0, rot] = self._mask_one(box, rot)

            if M.sum() > 0:
                state = self._build_state_single(box)
                logits, _ = self.sess.run(["logits", "value"], {"state": state})
                logits = np.asarray(logits).reshape(-1)
                mask_flat = M.reshape(-1)
                if mask_flat.shape[0] == logits.shape[0]:
                    masked = np.where(mask_flat > 0, logits, -1e30)
                    per_cand = 2 * self.Lc * self.Wc
                    sub = masked[:per_cand]

                    # Anchor 재순위
                    K = 10
                    valid_count = int((sub > -1e29).sum())
                    K = min(K, max(1, valid_count))
                    if K <= 1:
                        action = int(sub.argmax())
                    else:
                        top_idx = np.argpartition(sub, -K)[-K:]
                        top_logits = sub[top_idx]
                        lo, hi = top_logits.min(), top_logits.max()
                        best_combined = -1e30
                        action = int(top_idx[np.argmax(top_logits)])
                        alpha = 0.5
                        for i, a in enumerate(top_idx):
                            a = int(a)
                            _, ri, xi, yi = self._decode(a)
                            lc, wc, _ = self._cells(box["size"], ri)
                            xi = max(0, min(self.Lc - lc, xi))
                            yi = max(0, min(self.Wc - wc, yi))
                            norm = (float(sub[a]) - lo) / (hi - lo) if hi - lo > 1e-8 else 1.0
                            contact = self._contact_score(ri, xi, yi, box)
                            combined = norm * (1 - alpha) + contact * alpha
                            if combined > best_combined:
                                best_combined = combined
                                action = a

                    _, rot, x, y = self._decode(action)
                    lc, wc, hc = self._cells(box["size"], rot)
                    x = max(0, min(self.Lc - lc, x))
                    y = max(0, min(self.Wc - wc, y))
                    base = int(self.height[x:x+lc, y:y+wc].max())
                    l, w, h = float(box["size"][0]), float(box["size"][1]), float(box["size"][2])
                    if rot == 1: l, w = w, l
                    return (x * self.cell, y * self.cell, base * self.cell,
                            (l, w, h), 90 if rot == 1 else 0)
        except Exception:
            pass

        # ---- Phase 2: 바닥 갭 채우기 ----
        try:
            result = self._phase2_floor_gap(box)
            if result is not None:
                return result
        except Exception:
            pass

        # ---- Phase 3: column 위에 쌓기 ----
        try:
            result = self._phase3_column_stack(box)
            if result is not None:
                return result
        except Exception:
            pass

        return None

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

        while buf.has_pending():
            if self.algo.buffer_size == 0:
                current = [buf.peek_next()]
            else:
                current = buf.get_buffer()

            if self.should_finish(current):
                self.finished_by_user = True
                break

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