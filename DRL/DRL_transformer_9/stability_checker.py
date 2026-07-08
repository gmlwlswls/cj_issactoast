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


class StabilityChecker:
    def __init__(self, Lc: int, Wc: int, Hc: int, cell: float,
                 r_s: float = 0.66, r_w: float = 3.0):
        self.Lc, self.Wc, self.Hc = Lc, Wc, Hc
        self.cell = cell
        self.r_s = r_s
        self.r_w = r_w
        self.height = np.zeros((Lc, Wc), dtype=np.int32)
        self.topmass = np.zeros((Lc, Wc), dtype=np.float32)

    def reset(self):
        self.height[:] = 0
        self.topmass[:] = 0.0

    def _to_cells(self, l: float, w: float, h: float) -> Tuple[int, int, int]:
        return (max(1, math.ceil(l / self.cell)),
                max(1, math.ceil(w / self.cell)),
                max(1, math.ceil(h / self.cell)))

    def check(self, x: float, y: float, dims: Tuple[float, float, float],
              mass: float) -> bool:
        """
        (x, y) = FLB anchor 미터 좌표, dims = (l, w, h) 미터.
        지지 제약 + 무게 제약 + 높이 제약 모두 통과하면 True.
        """
        l, w, h = dims
        lc, wc, hc = self._to_cells(l, w, h)
        xc = int(round(x / self.cell))
        yc = int(round(y / self.cell))
        xc = max(0, min(self.Lc - lc, xc))
        yc = max(0, min(self.Wc - wc, yc))

        region = self.height[xc:xc + lc, yc:yc + wc]
        if region.size == 0:
            return False
        base = int(region.max())

        # 높이 제약: base + hc <= Hc
        if base + hc > self.Hc:
            return False

        # 바닥(z=0)이면 지지/무게 제약 없이 통과
        if base == 0:
            return True

        # 지지 제약: 밑면 중 base 높이에 닿는 셀 비율 >= r_s
        support = (region == base).sum()
        area = lc * wc
        if support / area < self.r_s:
            return False

        # 무게 제약: 지지하는 셀의 최소 질량 기준, mass <= r_w * min_mass
        support_mask = (region == base)
        supp_mass = self.topmass[xc:xc + lc, yc:yc + wc][support_mask]
        if supp_mass.size > 0:
            min_mass = supp_mass.min()
            if min_mass > 0 and mass > self.r_w * min_mass:
                return False

        return True

    def place(self, x: float, y: float, dims: Tuple[float, float, float],
              mass: float) -> Tuple[int, int, int]:
        """배치 후 height/topmass 갱신. base_z(셀 단위)를 반환."""
        l, w, h = dims
        lc, wc, hc = self._to_cells(l, w, h)
        xc = int(round(x / self.cell))
        yc = int(round(y / self.cell))
        xc = max(0, min(self.Lc - lc, xc))
        yc = max(0, min(self.Wc - wc, yc))

        base = int(self.height[xc:xc + lc, yc:yc + wc].max())
        self.height[xc:xc + lc, yc:yc + wc] = base + hc
        self.topmass[xc:xc + lc, yc:yc + wc] = mass
        return xc, yc, base

    def get_max_height(self) -> int:
        return int(self.height.max())

    def get_height_std(self) -> float:
        """현재 높이맵의 표준편차 (평탄도 측정용)."""
        return float(self.height.astype(np.float32).std())
