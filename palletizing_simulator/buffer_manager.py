from __future__ import annotations

import json
import os
from typing import Dict, List, Optional


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_boxes(path: str) -> List[Dict]:
    boxes: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no} -> {e}") from e

            if not isinstance(item, dict):
                raise TypeError(f"Each line must be a JSON object: {path}:{line_no}")

            for key in ("step", "id", "size", "mass"):
                if key not in item:
                    raise KeyError(f"Missing key '{key}' in {path}:{line_no}")

            boxes.append(item)
    return boxes


class BufferManager:
    """
    역할
    1) box_sequence 원본에서 초기 buffer_size개를 버퍼에 채운다.
    2) algorithm_results.sequence 순서대로 박스를 꺼낼 때,
       다음 원본 박스를 자동으로 버퍼에 보충한다.
    """

    def __init__(
        self,
        source_boxes: List[Dict],
        plan_sequence: List[Dict],
        buffer_size: int,
    ) -> None:
        if buffer_size < 0:
            raise ValueError("buffer_size must be > 0")

        self.source_boxes: List[Dict] = [dict(x) for x in source_boxes]
        self.plan_sequence: List[Dict] = [dict(x) for x in plan_sequence]
        self.buffer_size = int(buffer_size)

        self.source_by_id: Dict[int, Dict] = {}
        for box in self.source_boxes:
            bid = int(box["id"])
            if bid in self.source_by_id:
                raise ValueError(f"Duplicated id in box_sequence: {bid}")
            self.source_by_id[bid] = box

        self.plan_by_id: Dict[int, Dict] = {}
        for item in self.plan_sequence:
            bid = int(item["id"])
            if bid in self.plan_by_id:
                raise ValueError(f"Duplicated id in algorithm_results.sequence: {bid}")
            self.plan_by_id[bid] = item

        self._next_source_index = 0
        self.current_buffer: List[Dict] = []
        self.current_buffer_ids: set[int] = set()
        self.placed_ids: set[int] = set()

        self._fill_initial_buffer()
        self._validate_plan_ids()

    # ------------------------------------------------------------
    # init / validate
    # ------------------------------------------------------------
    def _fill_initial_buffer(self) -> None:
        while len(self.current_buffer) < self.buffer_size and self._next_source_index < len(self.source_boxes):
            self._push_next_source_box()

    def _push_next_source_box(self) -> Optional[Dict]:
        if self._next_source_index >= len(self.source_boxes):
            return None

        box = dict(self.source_boxes[self._next_source_index])
        self._next_source_index += 1

        bid = int(box["id"])
        if bid in self.current_buffer_ids:
            raise ValueError(f"Box already exists in buffer: id={bid}")
        if bid in self.placed_ids:
            raise ValueError(f"Box already placed before refill: id={bid}")

        self.current_buffer.append(box)
        self.current_buffer_ids.add(bid)
        return box

    def _validate_plan_ids(self) -> None:
        for item in self.plan_sequence:
            bid = int(item["id"])
            if bid not in self.source_by_id:
                raise KeyError(f"id={bid} exists in algorithm_results but not in box_sequence")

    # ------------------------------------------------------------
    # public api
    # ------------------------------------------------------------
    def get_initial_buffer_boxes(self) -> List[Dict]:
        return [dict(x) for x in self.current_buffer]

    def get_box_info(self, box_id: int) -> Dict:
        if box_id not in self.source_by_id:
            raise KeyError(f"Unknown box id: {box_id}")
        return dict(self.source_by_id[box_id])

    def has_in_buffer(self, box_id: int) -> bool:
        return int(box_id) in self.current_buffer_ids

    def get_current_buffer_ids(self) -> List[int]:
        return [int(x["id"]) for x in self.current_buffer]

    def consume(self, box_id: int) -> Dict:
        """
        box_id를 현재 buffer에서 꺼내고, 다음 source box 1개를 자동 보충한다.

        return:
        {
            "placed_box": {...},         # source box metadata
            "refilled_box": {...}|None,  # 자동 보충된 박스
            "buffer_ids": [...]
        }
        """
        bid = int(box_id)

        if bid not in self.current_buffer_ids:
            raise ValueError(
                f"Box id={bid} is not in current buffer. "
                f"current_buffer={self.get_current_buffer_ids()}"
            )

        found_idx = -1
        placed_box: Optional[Dict] = None
        for i, box in enumerate(self.current_buffer):
            if int(box["id"]) == bid:
                found_idx = i
                placed_box = box
                break

        if found_idx < 0 or placed_box is None:
            raise RuntimeError(f"Internal buffer mismatch for id={bid}")

        self.current_buffer.pop(found_idx)
        self.current_buffer_ids.remove(bid)
        self.placed_ids.add(bid)

        refilled_box = self._push_next_source_box()

        return {
            "placed_box": dict(placed_box),
            "refilled_box": None if refilled_box is None else dict(refilled_box),
            "buffer_ids": self.get_current_buffer_ids(),
        }

    def total_source_count(self) -> int:
        return len(self.source_boxes)

    def total_plan_count(self) -> int:
        return len(self.plan_sequence)

    def remaining_source_count(self) -> int:
        return len(self.source_boxes) - len(self.placed_ids)

    @classmethod
    def from_files(cls, algorithm_result_path: str, box_sequence_path: str) -> "BufferManager":
        plan_data = load_json(algorithm_result_path)

        if "buffer_size" not in plan_data:
            raise KeyError(f"'buffer_size' not found in {algorithm_result_path}")
        if "sequence" not in plan_data:
            raise KeyError(f"'sequence' not found in {algorithm_result_path}")
        if not isinstance(plan_data["sequence"], list):
            raise TypeError(f"'sequence' must be a list: {algorithm_result_path}")

        source_boxes = load_jsonl_boxes(box_sequence_path)
        plan_sequence = plan_data["sequence"]
        buffer_size = int(plan_data["buffer_size"])

        return cls(
            source_boxes=source_boxes,
            plan_sequence=plan_sequence,
            buffer_size=buffer_size,
        )


def build_matching_box_sequence_path(algorithm_result_path: str, box_sequence_dir: str) -> str:
    file_name = os.path.basename(algorithm_result_path)
    path = os.path.join(box_sequence_dir, file_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Matching box_sequence file not found: {path} "
            f"(algorithm_results file: {algorithm_result_path})"
        )
    return path