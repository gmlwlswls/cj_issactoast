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


# 학습(model_learn.py)의 Config 기본값. model_meta.json 이 있으면 그 값으로 덮어씀.
_DEFAULTS = dict(cell=0.01, max_mass=6.0, k_hol=1.8, com_alpha0=0.45, com_alpha1=0.20)


class Palletizer:
    """
    학습한 DRL(CNN Actor-Critic) 모델을 ONNX로 추론하여 박스 선택·위치·회전을 결정한다.

    핵심: 학습 환경(PalletizingEnv)과 '동일한' 상태/마스크/디코드 로직을 사용해야
          모델이 올바르게 동작한다. (height map + top-mass + 후보채널 / feasibility mask)
    """

    def __init__(self, pallet_cfg: PalletConfig, algo_cfg: AlgorithmConfig) -> None:
        self.pallet = pallet_cfg
        self.algo = algo_cfg

        here = os.path.dirname(os.path.abspath(__file__))
        # ---- 모델 메타 로드 (격자·mask 파라미터) ----
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

        # 격자 칸 수: 팔레트 크기 / cell (학습과 동일 규약)
        self.Lc = int(round(self.pallet.length / self.cell))   # X (length)
        self.Wc = int(round(self.pallet.width / self.cell))    # Y (width)
        self.Hc = int(round(self.pallet.height / self.cell))   # Z (height)
        self.N = int(meta.get("N", self.algo.buffer_size))     # 후보 수 = 버퍼 크기

        # ---- ONNX 세션 ----
        onnx_path = os.path.join(here, onnx_name)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"ONNX 모델을 찾을 수 없습니다: {onnx_path}\n"
                f"export_onnx.py 로 .pt → .onnx 변환 후, model_meta.json 과 함께 두세요.")
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        self._reset_state()

    def _reset_state(self) -> None:
        self.height = np.zeros((self.Lc, self.Wc), dtype=np.int32)
        self.topmass = np.zeros((self.Lc, self.Wc), dtype=np.float32)
        self.sequence: List[PlacedBox] = []
        self.finished = False
        self.terminated_step: Optional[int] = None
        self.finished_by_user = False

    # -----------------------------------------------------------------------
    # 참가자 수정 가능 함수
    # -----------------------------------------------------------------------
    def should_finish(self, current_buffer: List[BoxInput]) -> bool:
        """명시적 종료 판단. (현재는 mask 기반 자동 종료에 위임)"""
        return False

    # -----------------------------------------------------------------------
    # 셀 단위 footprint
    # -----------------------------------------------------------------------
    def _cells(self, size: List[float], rot: int) -> Tuple[int, int, int]:
        l, w, h = float(size[0]), float(size[1]), float(size[2])
        if rot == 1:                 # 90도: l,w swap
            l, w = w, l
        return (max(1, math.ceil(l / self.cell)),
                max(1, math.ceil(w / self.cell)),
                max(1, math.ceil(h / self.cell)))

    # -----------------------------------------------------------------------
    # feasibility mask  (학습 PalletizingEnv 와 동일 로직)
    # -----------------------------------------------------------------------
    def _mask_one(self, box: BoxInput, rot: int) -> np.ndarray:
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
        sum_di = np.zeros((vr, vc), dtype=np.float32)
        sum_dj = np.zeros((vr, vc), dtype=np.float32)
        supp_mass_min = np.full((vr, vc), np.inf, dtype=np.float32)

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

    def _feasibility_mask(self, candidates: List[Optional[BoxInput]]) -> np.ndarray:
        M = np.zeros((self.N, 2, self.Lc, self.Wc), dtype=np.float32)
        for ci, box in enumerate(candidates):
            if box is None:
                continue
            for rot in (0, 1):
                M[ci, rot] = self._mask_one(box, rot)
        return M

    # -----------------------------------------------------------------------
    # 상태 텐서  (학습과 동일 채널 구성: height, topmass, 후보별 l,w,h,mass)
    # -----------------------------------------------------------------------
    def _build_state(self, candidates: List[Optional[BoxInput]]) -> np.ndarray:
        Lc, Wc = self.Lc, self.Wc
        chans = [self.height.astype(np.float32) / self.Hc,
                 self.topmass / self.max_mass]
        for box in candidates:
            if box is None:
                chans += [np.zeros((Lc, Wc), np.float32)] * 4
            else:
                l, w, h = box["size"]
                chans += [np.full((Lc, Wc), l, np.float32),
                          np.full((Lc, Wc), w, np.float32),
                          np.full((Lc, Wc), h, np.float32),
                          np.full((Lc, Wc), float(box["mass"]) / self.max_mass, np.float32)]
        return np.stack(chans, axis=0)[None].astype(np.float32)   # (1, C, Lc, Wc)

    def _decode(self, action: int) -> Tuple[int, int, int, int]:
        per_rot = self.Lc * self.Wc
        ci = action // (2 * per_rot)
        rem = action % (2 * per_rot)
        rot = rem // per_rot
        pos = rem % per_rot
        x, y = pos // self.Wc, pos % self.Wc
        return ci, rot, int(x), int(y)

    # -----------------------------------------------------------------------
    # 박스 적재 (height/topmass 갱신 + 출력 PlacedBox 생성)
    # -----------------------------------------------------------------------
    def _place(self, box: BoxInput, rot: int, x: int, y: int) -> None:
        lc, wc, hc = self._cells(box["size"], rot)
        region = self.height[x:x + lc, y:y + wc]
        base = int(region.max())
        self.height[x:x + lc, y:y + wc] = base + hc
        self.topmass[x:x + lc, y:y + wc] = float(box["mass"])

        # 회전 반영한 실제 크기 (m)
        l, w, h = float(box["size"][0]), float(box["size"][1]), float(box["size"][2])
        if rot == 1:
            l, w = w, l
        # anchor(FLB) → centroid(무게중심)
        px = x * self.cell + l / 2.0
        py = y * self.cell + w / 2.0
        pz = base * self.cell + h / 2.0

        self.sequence.append({
            "step": int(box["step"]),
            "id": int(box["id"]),
            "size": [round(l, 3), round(w, 3), round(h, 3)],
            "mass": float(box["mass"]),
            "position": [round(px, 3), round(py, 3), round(pz, 3)],
            "rotation": 90 if rot == 1 else 0,
        })

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

            # 후보 N개 구성 (부족하면 None 패딩)
            candidates: List[Optional[BoxInput]] = list(current)
            while len(candidates) < self.N:
                candidates.append(None)
            candidates = candidates[:self.N]

            # feasibility mask → 전부 0이면 종료
            mask = self._feasibility_mask(candidates)
            if mask.sum() == 0:
                self.finished = True
                if current:
                    self.terminated_step = int(current[0]["step"])
                break

            # ONNX 추론 → mask 적용 → argmax → 디코드
            state = self._build_state(candidates)
            logits, _ = self.sess.run(["logits", "value"], {"state": state})
            logits = np.asarray(logits).reshape(-1)
            mask_flat = mask.reshape(-1)
            masked = np.where(mask_flat > 0, logits, -1e30)
            action = int(masked.argmax())
            ci, rot, x, y = self._decode(action)

            box = candidates[ci]
            self._place(box, rot, x, y)

            # 선택한 후보 소비 (버퍼에서 pop → 자동 보충)
            if self.algo.buffer_size == 0:
                buf.pop_next()
            else:
                buf.pop_selected(ci)

        return {
            "buffer_size": self.algo.buffer_size,
            "sequence": self.sequence,
            "terminated": self.finished,
            "terminated_step": self.terminated_step,
            "finished_by_user": self.finished_by_user,
        }
