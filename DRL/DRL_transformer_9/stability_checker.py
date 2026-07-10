"""
stability_checker.py — 물리 안정성 검사 (강화 조건)
==============================================================================
논문 9번(O4M-SP) 기반 + 강화:
  - 지지 제약 (r_s): 밑면의 r_s 이상이 받쳐져야 함 (강화: 0.75)
  - 무게 제약 (r_w): 위 박스 무게 ≤ r_w × 아래 박스 무게 (강화: 2.0)
  - 모서리 제약: 네 모서리 중 min_corners 개 이상 지지되어야 함 (강화: 3)

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
    물리적 제약(강화) 체크 + height/topmass map 유지.

    - 지지 제약: 지지 면적 / 바닥 면적 >= r_s  (기본 0.75)
    - 모서리 제약: 네 모서리(0,0)(lc-1,0)(0,wc-1)(lc-1,wc-1) 중 min_corners 개 이상 지지
    - 무게 제약: 배치 박스 mass <= r_w * (지지 셀들 아래 topmass 중 최소)  (기본 2.0)
      (바닥(base==0)이면 지지·무게 제약 모두 통과)
    """

    def __init__(self, Lc: int, Wc: int, Hc: int, cell: float,
                 r_s: float = 0.75, r_w: float = 2.0, min_corners: int = 3):
        self.Lc = int(Lc)
        self.Wc = int(Wc)
        self.Hc = int(Hc)
        self.cell = float(cell)
        self.r_s = float(r_s)
        self.r_w = float(r_w)
        self.min_corners = int(min_corners)

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
        if region.size == 0:
            return False
        base = int(region.max())
        if base + hc > self.Hc:
            return False

        # 바닥이면 모든 제약 통과
        if base == 0:
            return True

        # 지지 셀(같은 base 높이) 판정
        support = (region == base)
        r_support = float(support.mean()) if support.size > 0 else 0.0
        if r_support < self.r_s:
            return False

        # 모서리 조건: 네 모서리 중 min_corners 개 이상 지지
        corner_ok = 0
        for (ci, cj) in [(0, 0), (lc - 1, 0), (0, wc - 1), (lc - 1, wc - 1)]:
            if support[ci, cj]:
                corner_ok += 1
        if corner_ok < self.min_corners:
            return False

        # 무게 제약: 지지 셀 아래 topmass 최소와 비교
        tm = self.topmass[xc:xc + lc, yc:yc + wc]
        if support.any():
            supp_min = float(np.min(tm[support]))
        else:
            return False
        if supp_min <= 0.0:      # 지지체가 "빈 topmass(0)"이면 떠있는 상황 방지
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
