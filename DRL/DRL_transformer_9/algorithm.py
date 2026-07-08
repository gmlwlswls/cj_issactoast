"""
algorithm.py — O4M-SP 방식 추론 (제출용)
==============================================================================
학습된 ONNX 모델을 onnxruntime 으로 불러 추론한다.
EMS 기반 공간 관리 + Stability Checker + Transformer(ONNX) 사용.

run() 은 템플릿 골격 유지, 우리 로직은 _find_position 과 _append_placed 에.
==============================================================================
"""
from __future__ import annotations

import os, json, math
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict

import numpy as np
import onnxruntime as ort

from buffer_manager import BufferManager
from ems import EMSManager
from stability_checker import StabilityChecker


# ---------------------------------------------------------------------------
# 입출력 스키마  (수정 금지)
# ---------------------------------------------------------------------------

class BoxInput(TypedDict):
    step: int
    id: int
    size: List[float]
    mass: float


class PlacedBox(TypedDict):
    step: int
    id: int
    size: List[float]
    mass: float
    position: List[float]
    rotation: int


class RunResult(TypedDict):
    buffer_size: int
    sequence: List[PlacedBox]
    terminated: bool
    terminated_step: Optional[int]
    finished_by_user: bool


@dataclass
class PalletConfig:
    length: float
    width: float
    height: float


@dataclass
class AlgorithmConfig:
    allow_rotation: bool
    buffer_size: int


class Palletizer:

    def __init__(self, pallet_cfg: PalletConfig, algo_cfg: AlgorithmConfig) -> None:
        self.pallet = pallet_cfg
        self.algo = algo_cfg

        here = os.path.dirname(os.path.abspath(__file__))

        # ---- 메타 로드 ----
        meta_path = os.path.join(here, "model_meta.json")
        meta = {"cell": 0.01, "r_s": 0.66, "r_w": 3.0, "buffer_B": algo_cfg.buffer_size}
        onnx_name = "model.onnx"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            meta.update(loaded)
            onnx_name = loaded.get("onnx_path", onnx_name)

        self.cell = float(meta["cell"])
        self.r_s = float(meta["r_s"])
        self.r_w = float(meta["r_w"])
        self.buffer_B = int(meta.get("buffer_B", algo_cfg.buffer_size))

        # 격자 크기
        self.Lc = int(round(self.pallet.length / self.cell))
        self.Wc = int(round(self.pallet.width / self.cell))
        self.Hc = int(round(self.pallet.height / self.cell))
        self.max_placed = 80

        # ---- ONNX 세션 ----
        onnx_path = os.path.join(here, onnx_name)
        if not os.path.exists(onnx_path):
            found = [f for f in os.listdir(here) if f.lower().endswith(".onnx")]
            if not found:
                raise FileNotFoundError(f".onnx not found in {here}")
            onnx_path = os.path.join(here, sorted(found)[0])
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        # EMS, Stability
        self.ems = EMSManager(self.pallet.length, self.pallet.width, self.pallet.height)
        self.stab = StabilityChecker(self.Lc, self.Wc, self.Hc, self.cell, self.r_s, self.r_w)

        self._reset_state()

    def _reset_state(self) -> None:
        self.sequence: List[PlacedBox] = []
        self.finished = False
        self.terminated_step: Optional[int] = None
        self.finished_by_user = False
        self.placed_meta: List[dict] = []
        self.ems.reset()
        self.stab.reset()
        # 템플릿 호환 필드
        self.cursor_x = 0.0; self.row_depth = 0.0; self.layer_height = 0.0

    def should_finish(self, current_buffer: List[BoxInput]) -> bool:
        return False

    @staticmethod
    def _valid_box(b) -> bool:
        try:
            s = b["size"]
            if not (isinstance(s, (list, tuple)) and len(s) == 3): return False
            float(s[0]); float(s[1]); float(s[2]); float(b["mass"])
            int(b["step"]); int(b["id"]); return True
        except Exception:
            return False

    def _rotated(self, size, rot):
        l, w, h = float(size[0]), float(size[1]), float(size[2])
        return (w, l, h) if rot == 1 else (l, w, h)

    # -----------------------------------------------------------------------
    # 상태 텐서 구성 (학습과 동일)
    # -----------------------------------------------------------------------
    def _find_current_space(self, current_buffer):
        """EMS 중 버퍼 박스가 들어가고 안정한 첫 공간."""
        for sp in self.ems.get_sorted_spaces():
            for box in current_buffer:
                if not self._valid_box(box): continue
                l, w, h = box["size"]; mass = box["mass"]
                for rot in (0, 1):
                    rl, rw, rh = self._rotated((l, w, h), rot)
                    if self.ems.can_fit(sp, (rl, rw, rh)):
                        if self.stab.check(sp[0], sp[1], (rl, rw, rh), mass):
                            return sp
        return None

    def _build_obs(self, current_buffer, cur_space):
        L, W, H = self.pallet.length, self.pallet.width, self.pallet.height

        # S_bin
        rows = [[L, W, H, 0, 0, 0, 0]]
        if cur_space:
            si = self.ems.get_space_info(cur_space)
            rows.append(list(si) + [0])
        else:
            rows.append([0]*7)
        for p in self.placed_meta[-self.max_placed:]:
            rows.append([p["x"], p["y"], p["z"], p["l"], p["w"], p["h"], p["mass"]])
        s_bin = np.array(rows, np.float32)

        # S_items
        items = []
        B = self.buffer_B
        for i in range(B):
            if i < len(current_buffer) and self._valid_box(current_buffer[i]):
                b = current_buffer[i]
                items.append([b["size"][0], b["size"][1], b["size"][2], b["mass"]])
            else:
                items.append([0, 0, 0, 0])
        s_items = np.array(items, np.float32)

        # rot_candidates + valid_mask
        rc, vm = [], []
        for i in range(B):
            box = current_buffer[i] if i < len(current_buffer) else None
            if not self._valid_box(box):
                rc += [[0]*5, [0]*5]; vm += [0, 0]; continue
            l, w, h = box["size"]; mass = box["mass"]
            for rot in (0, 1):
                rl, rw, rh = self._rotated((l, w, h), rot)
                rc.append([rl, rw, rh, mass, float(rot)])
                ok = cur_space is not None and self.ems.can_fit(cur_space, (rl, rw, rh))
                ok = ok and self.stab.check(cur_space[0], cur_space[1], (rl, rw, rh), mass)
                vm.append(1 if ok else 0)
        return s_bin, s_items, np.array(rc, np.float32), np.array(vm, np.float32)

    # -----------------------------------------------------------------------
    # 헬퍼: _find_position / _append_placed  (run() 이 호출)
    # -----------------------------------------------------------------------
    def _find_position(self, current_buffer):
        """
        current_buffer 를 받아 (selected_index, x, y, z, dims, rotation) 반환.
        배치 불가면 None.
        """
        try:
            cur_space = self._find_current_space(current_buffer)
            if cur_space is None:
                return None

            s_bin, s_items, rc, vm = self._build_obs(current_buffer, cur_space)
            if vm.sum() == 0:
                return None

            logits, _ = self.sess.run(
                ["logits", "value"],
                {"s_bin": s_bin[None], "s_items": s_items[None], "rot_cands": rc[None]}
            )
            logits = np.asarray(logits).reshape(-1)
            masked = np.where(vm > 0, logits, -1e30)
            action = int(masked.argmax())
            bi = action // 2; rot = action % 2

            box = current_buffer[bi]
            if not self._valid_box(box):
                return None
            l, w, h = box["size"]; mass = box["mass"]
            rl, rw, rh = self._rotated((l, w, h), rot)

            sx, sy, sz = cur_space[0], cur_space[1], cur_space[2]
            # 실제 배치 base_z (셀 격자 상의 최대 높이)
            xc = int(round(sx / self.cell))
            yc = int(round(sy / self.cell))
            lc = max(1, math.ceil(rl / self.cell))
            wc = max(1, math.ceil(rw / self.cell))
            xc = max(0, min(self.Lc - lc, xc))
            yc = max(0, min(self.Wc - wc, yc))
            base_cells = int(self.stab.height[xc:xc + lc, yc:yc + wc].max())
            base_z = base_cells * self.cell

            return bi, sx, sy, base_z, (rl, rw, rh), (90 if rot == 1 else 0)
        except Exception:
            return None

    def _append_placed(self, box, dims, rotation, x, y, z):
        dx, dy, dz = dims
        # anchor → centroid (템플릿 규약과 동일)
        cx = x + dx / 2.0
        cy = y + dy / 2.0
        cz = z + dz / 2.0
        self.sequence.append({
            "step": int(box["step"]),
            "id": int(box["id"]),
            "size": [round(dx, 3), round(dy, 3), round(dz, 3)],
            "mass": float(box["mass"]),
            "position": [round(cx, 3), round(cy, 3), round(cz, 3)],
            "rotation": int(rotation),
        })
        # height/topmass 갱신 + EMS 갱신
        self.stab.place(x, y, (dx, dy, dz), float(box["mass"]))
        self.ems.update((x, y, z), (dx, dy, dz))
        self.placed_meta.append({"x": x, "y": y, "z": z,
                                 "l": dx, "w": dy, "h": dz,
                                 "mass": float(box["mass"])})
        self.cursor_x += dx
        self.row_depth = max(self.row_depth, dy)
        self.layer_height = max(self.layer_height, dz)

    # -----------------------------------------------------------------------
    # 실행 진입점  (템플릿 골격 유지)
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

            # 한 번의 추론으로 (박스, 회전, 위치) 결정
            found = self._find_position(current)

            if found is None:
                self.finished = True
                if current:
                    self.terminated_step = int(current[0]["step"])
                break

            selected_index, x, y, z, dims, rotation = found
            box = current[selected_index]

            self._append_placed(box=box, dims=dims, rotation=rotation, x=x, y=y, z=z)

            if self.algo.buffer_size == 0:
                buf.pop_next()
            else:
                buf.pop_selected(selected_index)

        return {
            "buffer_size": self.algo.buffer_size,
            "sequence": self.sequence,
            "terminated": self.finished,
            "terminated_step": self.terminated_step,
            "finished_by_user": self.finished_by_user,
        }
