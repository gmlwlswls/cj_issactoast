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

    # -----------------------------------------------------------------------
    # Anchor 기반 후보 재순위 (CNN top-K 중 배치 박스에 가장 밀착되는 위치 선택)
    # -----------------------------------------------------------------------
    def _contact_score(self, rot, x, y, box):
        """
        (rot, x, y) 위치에 box 를 놓았을 때 기존 구조물 및 팔레트 벽과의
        접촉 점수를 계산. 높을수록 빈틈 없이 밀착된 배치.

        접촉 요소:
          - 벽 접촉 (x=0, x+lc=Lc, y=0, y+wc=Wc) 각 +1
          - 옆면 접촉: 인접 셀에 같은 높이 범위의 박스가 있으면 +접촉길이/둘레
          - 낮은 base 선호: base 가 낮을수록 가산 (아래부터 채우기 유도)
        """
        lc, wc, hc = self._cells(box["size"], rot)
        Lc, Wc, Hc = self.Lc, self.Wc, self.Hc
        H = self.height

        base = int(H[x:x + lc, y:y + wc].max())
        score = 0.0
        perimeter = 2 * (lc + wc)

        # 벽 접촉 (팔레트 경계에 붙으면 가산)
        if x == 0:          score += lc / perimeter
        if x + lc >= Lc:    score += lc / perimeter
        if y == 0:          score += wc / perimeter
        if y + wc >= Wc:    score += wc / perimeter

        # 왼쪽 옆면 접촉 (x-1 열에 base~base+hc 범위에 박스가 있는 셀 수)
        if x > 0:
            col = H[x - 1, y:y + wc]
            touch = ((col >= base) & (col > 0)).sum()
            score += touch / perimeter

        # 오른쪽 옆면 접촉
        if x + lc < Lc:
            col = H[x + lc, y:y + wc]
            touch = ((col >= base) & (col > 0)).sum()
            score += touch / perimeter

        # 앞쪽 옆면 접촉
        if y > 0:
            row = H[x:x + lc, y - 1]
            touch = ((row >= base) & (row > 0)).sum()
            score += touch / perimeter

        # 뒤쪽 옆면 접촉
        if y + wc < Wc:
            row = H[x:x + lc, y + wc]
            touch = ((row >= base) & (row > 0)).sum()
            score += touch / perimeter

        # 낮은 base 가산 (아래부터 채우기 유도)
        score += (1.0 - base / Hc) * 0.5

        return score

    # -----------------------------------------------------------------------
    # _find_position: run()이 호출. 박스 하나의 최적 (x,y,z,dims,rotation) 반환.
    #                 배치 불가면 None 반환.
    #
    # [Anchor 확장] CNN top-K 후보를 뽑은 뒤, 각 후보의 anchor 접촉 점수로
    #               재순위하여 기존 박스에 밀착되는 위치를 선호.
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

            # ---- Anchor re-ranking ----
            # CNN 상위 K개 후보를 뽑고, 각각의 접촉 점수를 계산해서 재순위
            K = 10                          # top-K 후보 수 (시간 여유 충분: 10회 접촉 계산 < 1ms)
            valid_count = int((sub > -1e29).sum())
            K = min(K, max(1, valid_count))

            if K <= 1:
                # 유효 후보 1개면 재순위 불필요
                action = int(sub.argmax())
            else:
                top_idx = np.argpartition(sub, -K)[-K:]

                # logits 를 0~1 로 정규화 (재순위 결합용)
                top_logits = sub[top_idx]
                lo, hi = top_logits.min(), top_logits.max()
                if hi - lo > 1e-8:
                    norm_logits = (top_logits - lo) / (hi - lo)
                else:
                    norm_logits = np.ones_like(top_logits)

                best_combined = -1e30
                best_action = int(top_idx[np.argmax(top_logits)])  # fallback

                for i, a in enumerate(top_idx):
                    a = int(a)
                    _, rot_i, x_i, y_i = self._decode(a)
                    lc, wc, _ = self._cells(box["size"], rot_i)
                    x_i = max(0, min(self.Lc - lc, x_i))
                    y_i = max(0, min(self.Wc - wc, y_i))

                    contact = self._contact_score(rot_i, x_i, y_i, box)

                    # 결합 점수: CNN 정규화 logit × (1-α) + 접촉 점수 × α
                    # α = 0.4 → CNN 60%, 접촉 40% 비중
                    alpha = 0.5
                    combined = norm_logits[i] * (1 - alpha) + contact * alpha

                    if combined > best_combined:
                        best_combined = combined
                        best_action = a

                action = best_action

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

    # =======================================================================
    # [Phase 2] 바닥 갭 채우기
    #   위에서 내려다본 XY 평면에서 height==0 인 빈 바닥 영역을 찾아,
    #   버퍼의 박스가 그 빈 바닥에 통째로 들어가면 Z=0 에 배치.
    #   바닥이므로 지지/무게중심/holding 등 물리 조건은 검사하지 않는다.
    # =======================================================================
    def _find_floor_gap_position(self, box):
        from numpy.lib.stride_tricks import sliding_window_view

        rotations = (0, 1) if self.algo.allow_rotation else (0,)
        best = None
        best_score = -1e30

        for rot in rotations:
            lc, wc, hc = self._cells(box["size"], rot)
            if lc > self.Lc or wc > self.Wc:      # 평면상 안 들어감
                continue
            if hc > self.Hc:                       # 높이 초과
                continue

            # 각 (x,y) 시작점에서 footprint 내부의 최대 높이
            win_r = sliding_window_view(self.height, lc, axis=0).max(axis=2)
            base = sliding_window_view(win_r, wc, axis=1).max(axis=2)  # (vr, wc후)

            # base==0 → 그 footprint 전체가 빈 바닥
            ys, xs = np.where(base == 0)   # 주의: base 축0=x, 축1=y
            for x, y in zip(ys, xs):
                x, y = int(x), int(y)
                # 벽/기존 구조물에 밀착되는 곳을 우선(갭을 연속되게 유지)
                score = self._contact_score(rot, x, y, box)
                if score > best_score:
                    best_score = score
                    best = (rot, x, y)

        if best is None:
            return None

        rot, x, y = best
        l, w, h = float(box["size"][0]), float(box["size"][1]), float(box["size"][2])
        if rot == 1:
            l, w = w, l
        xm = x * self.cell
        ym = y * self.cell
        zm = 0.0                       # 바닥
        rotation = 90 if rot == 1 else 0
        return xm, ym, zm, (l, w, h), rotation

    # =======================================================================
    # [Phase 3] 기존 박스 위에 쌓기
    #   바닥에 놓인 박스(=기둥)의 x,y footprint 를 z축으로 확장한 공간을 만들고,
    #   그 공간에서 '가장 최상단' 박스를 찾은 뒤,
    #   버퍼 박스 중  (1) x,y 가 각각 최상단 박스 이하이고
    #                (2) 쌓아도 컨테이너 최대높이를 넘지 않으며
    #                (3) 부피(x*y*z)가 최상단 박스 이하인 것 중
    #   x,y 길이 편차가 가장 작은(size 가 가장 비슷한) 박스를 그 위에 올린다.
    # =======================================================================
    def _floor_boxes(self):
        """sequence 중 바닥(z_min≈0)에 놓인 박스 목록(=기둥)."""
        tol = self.cell
        res = []
        for pb in self.sequence:
            z_min = pb["position"][2] - pb["size"][2] / 2.0
            if z_min <= tol:
                res.append(pb)
        return res

    def _topmost_in_column(self, floor_box):
        """floor_box 의 x,y footprint 와 겹치는 박스 중 윗면 z 가 가장 높은 박스."""
        fcx, fcy = floor_box["position"][0], floor_box["position"][1]
        fx, fy = floor_box["size"][0], floor_box["size"][1]
        cx0, cx1 = fcx - fx / 2.0, fcx + fx / 2.0
        cy0, cy1 = fcy - fy / 2.0, fcy + fy / 2.0

        best, best_top = None, -1.0
        for pb in self.sequence:
            px, py = pb["position"][0], pb["position"][1]
            dx, dy = pb["size"][0], pb["size"][1]
            bx0, bx1 = px - dx / 2.0, px + dx / 2.0
            by0, by1 = py - dy / 2.0, py + dy / 2.0
            # 기둥과 XY 겹침 여부
            if min(cx1, bx1) - max(cx0, bx0) <= 1e-9:
                continue
            if min(cy1, by1) - max(cy0, by0) <= 1e-9:
                continue
            top = pb["position"][2] + pb["size"][2] / 2.0
            if top > best_top:
                best_top, best = top, pb
        return best

    def _find_stack_placement(self, current):
        """Phase 3 본체. (selected_index, (x,y,z,dims,rotation)) 또는 None."""
        floor_boxes = self._floor_boxes()
        if not floor_boxes:
            return None

        eps = 1e-9
        rotations = (0, 1) if self.algo.allow_rotation else (0,)

        best = None
        best_dev = float("inf")

        for fb in floor_boxes:
            top = self._topmost_in_column(fb)
            if top is None:
                continue
            tcx, tcy = top["position"][0], top["position"][1]
            tx, ty, tz = top["size"][0], top["size"][1], top["size"][2]
            top_vol = tx * ty * tz

            for idx, cand in enumerate(current):
                cl = float(cand["size"][0])
                cw = float(cand["size"][1])
                ch = float(cand["size"][2])
                for rot in rotations:
                    bx, by = (cw, cl) if rot == 1 else (cl, cw)
                    bz = ch

                    # (1) x,y 각각 최상단 박스 이하
                    if bx > tx + eps or by > ty + eps:
                        continue
                    # (3) 부피 최상단 박스 이하
                    if bx * by * bz > top_vol + eps:
                        continue

                    lc = max(1, math.ceil(bx / self.cell))
                    wc = max(1, math.ceil(by / self.cell))
                    hc = max(1, math.ceil(bz / self.cell))
                    if lc > self.Lc or wc > self.Wc:
                        continue

                    # 최상단 박스 중앙에 정렬 → footprint 가 최상단 박스 안에 들어감
                    xc = int(round((tcx - bx / 2.0) / self.cell))
                    yc = int(round((tcy - by / 2.0) / self.cell))
                    xc = max(0, min(self.Lc - lc, xc))
                    yc = max(0, min(self.Wc - wc, yc))

                    # 실제 height map 기준 착지면(= 최상단 박스 윗면)
                    base = int(self.height[xc:xc + lc, yc:yc + wc].max())
                    # (2) 컨테이너 최대 높이 초과 금지
                    if base + hc > self.Hc:
                        continue

                    # x,y 길이 편차(작을수록 size 유사)
                    dev = (tx - bx) + (ty - by)
                    if dev < best_dev - eps:
                        best_dev = dev
                        best = (idx, xc * self.cell, yc * self.cell,
                                base * self.cell, (bx, by, bz),
                                90 if rot == 1 else 0)

        if best is None:
            return None
        idx, xm, ym, zm, dims, rotation = best
        return idx, (xm, ym, zm, dims, rotation)

    # =======================================================================
    # 3-Phase 배치 선택기: Phase1 → Phase2 → Phase3 폴백
    #   (selected_index, x, y, z, dims, rotation, phase) 또는 None
    # =======================================================================
    def _select_placement(self, current):
        # Phase 1: CNN + anchor (기존)
        for idx, box in enumerate(current):
            found = self._find_position(box)
            if found is not None:
                x, y, z, dims, rot = found
                return idx, x, y, z, dims, rot, 1

        # Phase 2: 바닥 갭 채우기
        for idx, box in enumerate(current):
            found = self._find_floor_gap_position(box)
            if found is not None:
                x, y, z, dims, rot = found
                return idx, x, y, z, dims, rot, 2

        # Phase 3: 기존 박스 위에 쌓기
        res = self._find_stack_placement(current)
        if res is not None:
            idx, (x, y, z, dims, rot) = res
            return idx, x, y, z, dims, rot, 3

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

            # Phase 1 → Phase 2 → Phase 3 순으로 배치 위치를 찾는다.
            placement = self._select_placement(current)

            if placement is None:
                # 세 Phase 모두 실패 → 버퍼에 놓을 수 있는 박스가 없음 → 종료
                self.finished = True
                if current:
                    self.terminated_step = int(current[0]["step"])
                break

            selected_index, x, y, z, dims, rotation, phase = placement
            box = current[selected_index]

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

            # 하나 배치했으니 다음 스텝으로 (Phase 3 '반복'은 이 루프가 담당)
            continue

        return {
            "buffer_size": self.algo.buffer_size,
            "sequence": self.sequence,
            "terminated": self.finished,
            "terminated_step": self.terminated_step,
            "finished_by_user": self.finished_by_user,
        }