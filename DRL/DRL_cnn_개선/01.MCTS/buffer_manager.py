"""
================================================================================
[버퍼 관리] 수정 금지
================================================================================
컨베이어 시퀀스를 sliding-window 방식으로 관리하는 인프라 코드입니다.
버퍼 크기(capacity)는 config/algorithm_config.yaml 의 buffer.size 로만 조정하세요.

알고리즘(SimplePalletizer)은 이 클래스를 통해 최대 capacity 개의 박스를
미리 보고 적재 순서를 자유롭게 결정할 수 있습니다.

흐름:
  1. reset(boxes)    — 전체 박스 리스트를 받아 버퍼 초기 적재
  2. get_buffer()    — 현재 버퍼 스냅샷 조회 (복사본 반환)
  3. pop_selected(i) — i번째 박스를 꺼내고 다음 박스를 자동 보충
  4. has_pending()   — 버퍼 또는 대기열에 남은 박스가 있으면 True
================================================================================
"""
from __future__ import annotations

from typing import Dict, List


class BufferManager:
    """Sliding-window 버퍼: capacity 개의 박스를 유지한다. capacity=0이면 버퍼 미사용."""

    def __init__(self, capacity: int) -> None:
        """capacity: 동시에 유지할 최대 박스 수 (≥ 0)."""
        if capacity < 0:
            raise ValueError("buffer_size must be >= 0")

        self.capacity = int(capacity)
        self._source: List[Dict] = []
        self._next: int = 0
        self._buffer: List[Dict] = []

    def reset(self, boxes: List[Dict]) -> None:
        self._source = list(boxes)
        self._next = 0
        self._buffer = []
        self._fill()

    def _fill(self) -> None:
        while len(self._buffer) < self.capacity and self._next < len(self._source):
            self._buffer.append(self._source[self._next])
            self._next += 1

    def has_pending(self) -> bool:
        return bool(self._buffer) or self._next < len(self._source)

    def get_buffer(self) -> List[Dict]:
        if self.capacity == 0:
            return []
        return list(self._buffer)

    def pop_selected(self, index: int) -> Dict:
        if self.capacity == 0:
            raise RuntimeError("buffer_size is 0, pop_selected() cannot be used")

        box = self._buffer.pop(index)
        self._fill()
        return box

    def pop_next(self) -> Dict:
        """버퍼를 사용하지 않을 때 다음 박스를 순서대로 꺼낸다."""
        if self._next >= len(self._source):
            raise IndexError("no boxes remaining")

        box = self._source[self._next]
        self._next += 1
        return box

    def drain_remaining(self) -> List[Dict]:
        remaining = list(self._buffer)
        remaining.extend(self._source[self._next:])
        self._buffer.clear()
        self._next = len(self._source)
        return remaining
    def peek_next(self) -> Dict:
        """buffer_size=0 일 때 다음 박스를 제거하지 않고 조회."""
        if self._next >= len(self._source):
            raise IndexError("no boxes remaining")

        return self._source[self._next]
