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


_DEFAULTS = dict(cell=0.01, max_mass=6.0)


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