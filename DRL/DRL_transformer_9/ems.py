"""
ems.py — Empty Maximal Space 관리
==============================================================================
EMS 는 팔레트 안의 "비어 있는 직육면체 공간"을 목록으로 관리한다.
박스를 놓을 때마다 겹치는 EMS 를 쪼개고, 포함된 공간을 제거(maximal filtering)한다.
XYZ 우선순위로 정렬해 stack 의 맨 위가 "가장 먼저 써야 할 공간"이 되도록 한다.

각 EMS 는 (x1,y1,z1, x2,y2,z2) — 좌하단바닥 ~ 우상단천장 좌표.
"""
from __future__ import annotations
from typing import List, Tuple

Space = Tuple[float, float, float, float, float, float]  # x1,y1,z1, x2,y2,z2


class EMSManager:
    def __init__(self, L: float, W: float, H: float):
        """팔레트 전체를 하나의 초기 EMS 로."""
        self.L, self.W, self.H = L, W, H
        self.spaces: List[Space] = [(0.0, 0.0, 0.0, L, W, H)]

    def reset(self):
        self.spaces = [(0.0, 0.0, 0.0, self.L, self.W, self.H)]

    @staticmethod
    def _overlaps(a: Space, b: Space) -> bool:
        return (a[0] < b[3] and a[3] > b[0] and
                a[1] < b[4] and a[4] > b[1] and
                a[2] < b[5] and a[5] > b[2])

    @staticmethod
    def _contains(outer: Space, inner: Space) -> bool:
        return (outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] <= inner[2] and
                outer[3] >= inner[3] and outer[4] >= inner[4] and outer[5] >= inner[5])

    def _split(self, ems: Space, box: Space) -> List[Space]:
        """ems 와 box 가 겹치면, ems 에서 box 를 빼고 남는 최대 6개 조각을 반환."""
        if not self._overlaps(ems, box):
            return [ems]
        parts: List[Space] = []
        x1, y1, z1, x2, y2, z2 = ems
        bx1, by1, bz1, bx2, by2, bz2 = box
        # X 방향 (왼쪽 / 오른쪽)
        if bx1 > x1:
            parts.append((x1, y1, z1, bx1, y2, z2))
        if bx2 < x2:
            parts.append((bx2, y1, z1, x2, y2, z2))
        # Y 방향 (앞쪽 / 뒤쪽)
        if by1 > y1:
            parts.append((x1, y1, z1, x2, by1, z2))
        if by2 < y2:
            parts.append((x1, by2, z1, x2, y2, z2))
        # Z 방향 (위쪽 — 아래쪽은 보통 안 생김)
        if bz2 < z2:
            parts.append((x1, y1, bz2, x2, y2, z2))
        if bz1 > z1:
            parts.append((x1, y1, z1, x2, y2, bz1))
        # 유효 크기 필터 (너무 작은 공간 제거 — 최소 박스보다 작으면 쓸모없음)
        MIN = 0.05  # 5cm 미만 공간 제거
        return [(a, b, c, d, e, f) for a, b, c, d, e, f in parts
                if d - a > MIN and e - b > MIN and f - c > MIN]

    def _maximal_filter(self, spaces: List[Space]) -> List[Space]:
        """다른 공간에 완전히 포함되는 공간을 제거."""
        keep = []
        for i, s in enumerate(spaces):
            contained = False
            for j, t in enumerate(spaces):
                if i != j and self._contains(t, s):
                    contained = True
                    break
            if not contained:
                keep.append(s)
        return keep

    def update(self, box_pos: Tuple[float, float, float],
               box_dims: Tuple[float, float, float]) -> None:
        """박스 배치 후 EMS 갱신. box_pos=(x,y,z) 는 FLB(앞-왼-아래) 좌표."""
        x, y, z = box_pos
        dx, dy, dz = box_dims
        placed = (x, y, z, x + dx, y + dy, z + dz)
        new_spaces: List[Space] = []
        for ems in self.spaces:
            new_spaces.extend(self._split(ems, placed))
        self.spaces = self._maximal_filter(new_spaces)

    def get_sorted_spaces(self) -> List[Space]:
        """XYZ 우선순위로 정렬 (X 작은 것 → Y 작은 것 → Z 낮은 것 우선)."""
        return sorted(self.spaces, key=lambda s: (s[0], s[1], s[2]))

    def can_fit(self, space: Space, dims: Tuple[float, float, float]) -> bool:
        """dims=(l,w,h) 가 space 안에 들어가는지."""
        sx = space[3] - space[0]
        sy = space[4] - space[1]
        sz = space[5] - space[2]
        return dims[0] <= sx + 1e-6 and dims[1] <= sy + 1e-6 and dims[2] <= sz + 1e-6

    def get_space_info(self, space: Space) -> Tuple[float, float, float, float, float, float]:
        """공간의 (x, y, z, size_x, size_y, size_z) 반환."""
        return (space[0], space[1], space[2],
                space[3] - space[0], space[4] - space[1], space[5] - space[2])
