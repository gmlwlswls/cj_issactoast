"""
stability_checker.py — 물리 안정성 검사
==============================================================================
논문 9번(O4M-SP)의 Stability Checker 구현.
  - 지지 제약 (r_s): 밑면의 r_s 이상이 받쳐져야 함 (기본 0.66)
  - 무게 제약 (r_w): 위 박스 무게 ≤ r_w × 아래 박스 무게 (기본 3.0)

height map (Lc×Wc 정수 격자) 기반으로 검사한다.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass
class PlaceResult:
    xc: int
    yc: int
    base: int


class StabilityChecker:
    """
    물리적 제약(간이) 체크 + height/topmass map 유지.

    - 지지 제약: 지지 면적 / 바닥 면적 >= r_s
      여기서 "지지"는 footprint 내에서 base 높이에 닿아있는 셀(=최대 높이 평탄 지지)로 근사.
    - 무게 제약: 배치 박스 mass <= r_w * (지지 셀들 아래의 topmass 중 최소값)
      (바닥(base==0)이면 무게 제약은 항상 통과로 처리)
    """

    def __init__(self, Lc: int, Wc: int, Hc: int, cell: float, r_s: float = 0.66, r_w: float = 3.0):
        self.Lc = int(Lc)
        self.Wc = int(Wc)
        self.Hc = int(Hc)
        self.cell = float(cell)
        self.r_s = float(r_s)
        self.r_w = float(r_w)

        self.height = np.zeros((self.Lc, self.Wc), dtype=np.int32)
        self.topmass = np.zeros((self.Lc, self.Wc), dtype=np.float32)

    def reset(self) -> None:
        self.height.fill(0)
        self.topmass.fill(0.0)

    def _cells(self, dims: Tuple[float, float, float]) -> Tuple[int, int, int]:
        l, w, h = float(dims[0]), float(dims[1]), float(dims[2])
        lc = max(1, int(math.ceil(l / self.cell)))
        wc = max(1, int(math.ceil(w / self.cell)))
        hc = max(1, int(math.ceil(h / self.cell)))
        return lc, wc, hc

    def _anchor_to_cell(self, x: float, y: float, lc: int, wc: int) -> Tuple[int, int]:
        xc = int(round(x / self.cell))
        yc = int(round(y / self.cell))
        xc = max(0, min(self.Lc - lc, xc))
        yc = max(0, min(self.Wc - wc, yc))
        return xc, yc

    def check(self, x: float, y: float, dims: Tuple[float, float, float], mass: float) -> bool:
        lc, wc, hc = self._cells(dims)
        xc, yc = self._anchor_to_cell(x, y, lc, wc)

        region = self.height[xc:xc + lc, yc:yc + wc]
        base = int(region.max())
        if base + hc > self.Hc:
            return False

        # 지지율: footprint 중 base 높이와 같은 셀의 비율(평탄 지지 근사)
        support = (region == base)
        r_support = float(support.mean()) if support.size > 0 else 0.0
        if r_support < self.r_s:
            return False

        # 무게비: 바닥이면 OK, 아니면 지지 셀 아래 topmass 최소와 비교
        if base == 0:
            return True

        tm = self.topmass[xc:xc + lc, yc:yc + wc]
        if support.any():
            supp_min = float(np.min(tm[support]))
        else:
            return False

        # 지지체가 "빈 topmass(0)"이면 무게비가 무조건 깨지도록 처리(떠있는 상황 방지)
        if supp_min <= 0.0:
            return False

        return float(mass) <= self.r_w * supp_min

    def place(self, x: float, y: float, dims: Tuple[float, float, float], mass: float) -> PlaceResult:
        """
        check를 통과한다고 가정하고 height/topmass를 갱신한다.
        반환 base는 cell 단위 높이(정수).
        """
        lc, wc, hc = self._cells(dims)
        xc, yc = self._anchor_to_cell(x, y, lc, wc)

        region = self.height[xc:xc + lc, yc:yc + wc]
        base = int(region.max())
        self.height[xc:xc + lc, yc:yc + wc] = base + hc
        self.topmass[xc:xc + lc, yc:yc + wc] = float(mass)
        return PlaceResult(xc=xc, yc=yc, base=base)

    def get_height_std(self) -> float:
        return float(self.height.std())
