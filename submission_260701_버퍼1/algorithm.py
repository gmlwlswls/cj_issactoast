from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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


# # 학습(model_learn.py)의 Config 기본값. model_meta.json 이 있으면 그 값으로 덮어씀.
# _DEFAULTS = dict(cell=0.01, max_mass=6.0, k_hol=1.8, com_alpha0=0.45, com_alpha1=0.20)


class Palletizer:
    """
    학습한 DRL(CNN Actor-Critic) 모델을 ONNX로 추론하여 박스 선택·위치·회전을 결정한다.

    핵심: 학습 환경(PalletizingEnv)과 '동일한' 상태/마스크/디코드 로직을 사용해야
          모델이 올바르게 동작한다. (height map + top-mass + 후보채널 / feasibility mask)

    run()은 원본 템플릿 골격을 그대로 유지한다. ONNX 추론과 모든 방어 로직은
    should_finish / _mask_one / _feasibility_mask / _build_state / _place 등
    참가자 수정 영역(헬퍼 메서드) 내부에서만 처리하며, 각 헬퍼는 어떤 입력에도
    예외를 밖으로 던지지 않도록 스스로 방어한다. (예: 유효하지 않은 박스는
    feasibility mask를 전부 0으로 만들어 자연스럽게 후보에서 제외되게 함 →
    run()에 이미 있는 "mask.sum() == 0 → 안전 종료" 분기가 그대로 처리)
    """

    def __init__(self, pallet_cfg: PalletConfig, algo_cfg: AlgorithmConfig) -> None:
            self.pallet = pallet_cfg
            self.algo = algo_cfg
    
            here = os.path.dirname(os.path.abspath(__file__))
    
            # 1. 시스템 기본값 설정 (YAML이나 meta에 아무것도 없을 때를 위한 최소한의 방어)
            base_meta = {
                "cell": 0.01,
                "max_mass": 6.0,
                "k_hol": 1.8,
                "com_alpha0": 0.45,
                "com_alpha1": 0.20,
                "onnx_path": "model.onnx"
            }
    
            # 2. config/algorithm_config.yaml 파일 읽기
            # 프로젝트 루트(한 단계 위) 기준과 현재 파일 기준 모두 탐색하여 안전하게 로드
            import yaml  # 코드 상단에 import yaml이 없다면 여기에 선언해도 안전합니다.
            
            yaml_path = os.path.join(os.path.dirname(here), "config", "algorithm_config.yaml")
            if not os.path.exists(yaml_path):
                yaml_path = os.path.join(here, "config", "algorithm_config.yaml")
    
            if os.path.exists(yaml_path):
                with open(yaml_path, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f)
                    if yaml_data:
                        # YAML에 적힌 하이퍼파라미터가 있다면 기본값에 동적으로 덮어씀
                        for k, v in yaml_data.items():
                            if k in base_meta:
                                base_meta[k] = v
    
            # 3. 기존 model_meta.json 파일 읽기 (기존 규약 및 메타 매핑 유지)
            meta_path = os.path.join(here, "model_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                base_meta.update(loaded)
    
            # 4. 최종 변수 할당
            self.cell = float(base_meta["cell"])
            self.max_mass = float(base_meta["max_mass"])
            self.k_hol = float(base_meta["k_hol"])
            self.com_a0 = float(base_meta["com_alpha0"])
            self.com_a1 = float(base_meta["com_alpha1"])
            
            onnx_name = base_meta.get("onnx_path", "model.onnx")
    
            # 격자 칸 수: 팔레트 크기 / cell (학습과 동일 규약)
            self.Lc = int(round(self.pallet.length / self.cell))   # X (length)
            self.Wc = int(round(self.pallet.width / self.cell))    # Y (width)
            self.Hc = int(round(self.pallet.height / self.cell))   # Z (height)
            self.N = int(base_meta.get("N", self.algo.buffer_size))     # 후보 수 = 버퍼 크기
    
            # ---- ONNX 세션 ----
            # 1) meta 또는 YAML에 지정된 이름이 실제로 있으면 그걸 사용
            # 2) 없으면 같은 폴더의 .onnx 파일을 자동 탐색
            onnx_path = os.path.join(here, onnx_name)
            if not os.path.exists(onnx_path):
                found = [f for f in os.listdir(here) if f.lower().endswith(".onnx")]
                if found:
                    onnx_path = os.path.join(here, sorted(found)[0])
                else:
                    raise FileNotFoundError(
                        f"ONNX 모델을 찾을 수 없습니다: {here} 폴더에 .onnx 파일이 없습니다.\n"
                        f"export_onnx.py 로 .pt → .onnx 변환 후, 이 폴더에 두세요.")
            
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
        """명시적 종료 판단. (현재는 mask 기반 자동 종료에 위임 → 항상 False)"""
        return False

    # -----------------------------------------------------------------------
    # 입력 박스 유효성 검사 (평가 데이터 엣지케이스 방어)
    # -----------------------------------------------------------------------
    @staticmethod
    def _valid_box(b) -> bool:
        try:
            s = b["size"]
            if not (isinstance(s, (list, tuple)) and len(s) == 3):
                return False
            l, w, h = float(s[0]), float(s[1]), float(s[2])
            if l <= 0 or w <= 0 or h <= 0:
                return False
            float(b["mass"])
            int(b["step"])
            int(b["id"])
            return True
        except Exception:
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
    #   방어: 어떤 이유로든 계산이 실패하면(비정상 데이터 포함) 해당 후보는
    #         전부 0인 mask를 반환한다 → run()이 자동으로 후보에서 제외한다.
    # -----------------------------------------------------------------------
    def _mask_one(self, box: Optional[BoxInput], rot: int) -> np.ndarray:
        Lc, Wc = self.Lc, self.Wc
        try:
            if box is None or not self._valid_box(box):
                return np.zeros((Lc, Wc), dtype=np.float32)

            lc, wc, hc = self._cells(box["size"], rot)
            if lc > Lc or wc > Wc:
                return np.zeros((Lc, Wc), dtype=np.float32)

            H = self.height
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
        except Exception:
            # 어떤 예외든 이 후보는 "놓을 곳 없음"으로 처리
            return np.zeros((Lc, Wc), dtype=np.float32)

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
    #   방어: 후보가 없거나 유효하지 않으면 해당 채널을 0으로 채운다
    #         (그 후보는 _mask_one에서 이미 mask 0으로 배제되므로 argmax에서
    #          선택될 일이 없어 값 자체는 영향이 없다).
    # -----------------------------------------------------------------------
    def _build_state(self, candidates: List[Optional[BoxInput]]) -> np.ndarray:
        Lc, Wc = self.Lc, self.Wc
        chans = [self.height.astype(np.float32) / self.Hc,
                 self.topmass / self.max_mass]
        for box in candidates:
            if box is None or not self._valid_box(box):
                chans += [np.zeros((Lc, Wc), np.float32)] * 4
                continue
            try:
                l, w, h = float(box["size"][0]), float(box["size"][1]), float(box["size"][2])
                m = float(box["mass"]) / self.max_mass
            except Exception:
                chans += [np.zeros((Lc, Wc), np.float32)] * 4
                continue
            chans += [np.full((Lc, Wc), l, np.float32),
                      np.full((Lc, Wc), w, np.float32),
                      np.full((Lc, Wc), h, np.float32),
                      np.full((Lc, Wc), m, np.float32)]
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
    #   방어: 좌표는 격자 범위로 clamp, 타입 변환 실패 시 안전한 기본값 사용,
    #         좌표 반올림으로 팔레트 경계를 벗어나지 않도록 clamp.
    # -----------------------------------------------------------------------
    def _place(self, box: BoxInput, rot: int, x: int, y: int) -> None:
        try:
            lc, wc, hc = self._cells(box["size"], rot)
        except Exception:
            lc, wc, hc = 1, 1, 1

        # 격자 범위 밖 인덱스 방지
        x = max(0, min(self.Lc - lc, int(x)))
        y = max(0, min(self.Wc - wc, int(y)))

        region = self.height[x:x + lc, y:y + wc]
        base = int(region.max()) if region.size else 0
        self.height[x:x + lc, y:y + wc] = base + hc

        try:
            mass_val = float(box["mass"])
        except Exception:
            mass_val = 0.0
        self.topmass[x:x + lc, y:y + wc] = mass_val

        # 회전 반영한 실제 크기 (m)
        try:
            l, w, h = float(box["size"][0]), float(box["size"][1]), float(box["size"][2])
        except Exception:
            l, w, h = lc * self.cell, wc * self.cell, hc * self.cell
        if rot == 1:
            l, w = w, l

        # anchor(FLB) → centroid(무게중심)
        px = x * self.cell + l / 2.0
        py = y * self.cell + w / 2.0
        pz = base * self.cell + h / 2.0

        # 출력 좌표: 박스 끝이 [0, 팔레트] 안에 정확히 들어오도록 clamp 후 반올림
        # (소수 반올림으로 원점을 0.5mm 넘는 아티팩트 방지)
        def _clamp_center(c, half, limit):
            lo, hi = half, limit - half
            if lo > hi:            # 박스가 팔레트보다 크면 중앙
                return limit / 2.0
            return min(max(c, lo), hi)

        px = _clamp_center(px, l / 2.0, self.pallet.length)
        py = _clamp_center(py, w / 2.0, self.pallet.width)
        pz = _clamp_center(pz, h / 2.0, self.pallet.height)

        try:
            step_val = int(box["step"])
        except Exception:
            step_val = 0
        try:
            id_val = int(box["id"])
        except Exception:
            id_val = 0

        self.sequence.append({
            "step": step_val,
            "id": id_val,
            "size": [round(l, 4), round(w, 4), round(h, 4)],
            "mass": mass_val,
            "position": [round(px, 4), round(py, 4), round(pz, 4)],
            "rotation": 90 if rot == 1 else 0,
        })

    # -----------------------------------------------------------------------
    # 실행 진입점  (원본 템플릿 골격 그대로 유지 — 이 함수는 수정하지 않음)
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