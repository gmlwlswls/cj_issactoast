from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import yaml

from algorithm import (
    AlgorithmConfig,
    PalletConfig,
    RunResult,
    Palletizer,
)
from visualize import visualize


def load_config(config_path: str) -> Dict:
    """[수정 금지] YAML 설정 파일을 읽어 딕셔너리로 반환한다."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Empty config file: {config_path}")

    return cfg


def validate_config(cfg: Dict) -> None:
    """
    [수정 금지]
    필수 설정 키가 모두 존재하는지 확인한다.
    누락 시 KeyError를 발생시킨다.
    """
    for key in ("input_path", "output_dir", "pallet", "algorithm", "buffer"):
        if key not in cfg:
            raise KeyError(f"Missing config key: {key}")

    for key in ("length", "width", "height"):
        if key not in cfg["pallet"]:
            raise KeyError(f"Missing pallet config: {key}")

    if "size" not in cfg["buffer"]:
        raise KeyError("Missing buffer.size in config")


def list_input_files(input_path: Path) -> List[Path]:
    """
    [수정 금지]
    경로가 파일이면 단일 리스트,
    디렉토리면 내부 .json 파일 목록을 반환한다.
    """
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    files = sorted(p for p in input_path.iterdir() if p.suffix.lower() == ".json")

    if not files:
        raise FileNotFoundError(f"No .json files found in directory: {input_path}")

    return files


def load_boxes(file_path: Path) -> List[Dict]:
    """
    [수정 금지]
    JSON 배열 또는 JSONL(줄별 JSON) 형식의 파일을 읽어
    박스 리스트로 반환한다.
    """
    with file_path.open("r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    if text[0] == "[":
        data = json.loads(text)

        if not isinstance(data, list):
            raise ValueError(f"Input JSON must be a list: {file_path}")

        return data

    boxes = []

    for line in text.splitlines():
        line = line.strip()

        if line:
            boxes.append(json.loads(line))

    return boxes


def save_result(
    result: RunResult,
    output_path: Path,
    ensure_ascii: bool = False,
    indent: int = 2,
) -> None:
    """
    [수정 금지]
    RunResult를 JSON 형식으로 파일에 저장한다.

    저장 항목:
      - buffer_size
      - sequence
      - terminated
      - terminated_step
      - finished_by_user
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "buffer_size": result["buffer_size"],
        "sequence": result["sequence"],
        "terminated": result["terminated"],
        "terminated_step": result["terminated_step"],
        "finished_by_user": result["finished_by_user"],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            output_data,
            f,
            indent=indent,
            ensure_ascii=ensure_ascii,
        )


def build_output_path(output_dir: Path, input_file_path: Path) -> Path:
    """
    [수정 금지]
    출력 디렉토리를 생성하고 결과 JSON 파일 경로를 반환한다.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / input_file_path.name


def build_vis_path(output_dir: Path, input_file_path: Path) -> Path:
    """
    [수정 금지]
    시각화 이미지(PNG)가 저장될 경로를 반환한다.
    output_dir/vis/ 하위에 저장된다.
    """
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    return vis_dir / f"{input_file_path.stem}.png"


def compute_utilization(result: RunResult, pallet_cfg: PalletConfig) -> Dict[str, float]:
    """
    적재 결과로부터 공간 활용률(utilization)을 계산한다.

    utilization =
        박스 총 부피 / (팔레트 길이 × 팔레트 폭 × 팔레트 최대 높이)
    """
    seq = result.get("sequence", [])

    if not seq:
        return {
            "placed_count": 0,
            "box_volume_sum": 0.0,
            "max_top_height": 0.0,
            "pallet_volume": pallet_cfg.length * pallet_cfg.width * pallet_cfg.height,
            "utilization": 0.0,
        }

    total_box_volume = 0.0
    max_top_z = 0.0
    placed_count = 0

    for item in seq:
        size = item.get("size")
        pos = item.get("position")

        if not size or len(size) != 3:
            continue

        if not pos or len(pos) != 3:
            continue

        dx = float(size[0])
        dy = float(size[1])
        dz = float(size[2])

        cz = float(pos[2])
        top_z = cz + dz / 2.0

        total_box_volume += dx * dy * dz
        max_top_z = max(max_top_z, top_z)
        placed_count += 1

    pallet_volume = pallet_cfg.length * pallet_cfg.width * pallet_cfg.height

    if pallet_volume <= 0.0:
        utilization = 0.0
    else:
        utilization = total_box_volume / pallet_volume

    return {
        "placed_count": placed_count,
        "box_volume_sum": total_box_volume,
        "max_top_height": max_top_z,
        "pallet_volume": pallet_volume,
        "utilization": utilization,
    }


def resolve_path(base_dir: Path, path_value: str) -> Path:
    """
    상대 경로를 main.py 기준 절대 경로로 변환한다.
    절대 경로가 입력되면 그대로 사용한다.
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    return base_dir / path


def main() -> None:
    """
    설정 로드부터 결과 저장·통계 출력까지 전체 실행 흐름을 조율한다.

    실행 순서:
      1. config/algorithm_config.yaml 로드
      2. 입력 JSON 파일 목록 조회
      3. 각 입력 파일에 대해 Palletizer 실행
      4. 결과 JSON 저장
      5. 시각화 PNG 저장
      6. 통계 출력
    """
    total_start_time = time.time()

    here = Path(__file__).resolve().parent
    config_path = here / "config" / "algorithm_config.yaml"

    cfg = load_config(str(config_path))
    validate_config(cfg)

    pallet_cfg = PalletConfig(
        length=float(cfg["pallet"]["length"]),
        width=float(cfg["pallet"]["width"]),
        height=float(cfg["pallet"]["height"]),
    )

    algo_cfg = AlgorithmConfig(
        allow_rotation=bool(cfg["algorithm"]["allow_rotation"]),
        buffer_size=int(cfg["buffer"]["size"]),
    )

    input_path = resolve_path(here, cfg["input_path"])
    output_dir = resolve_path(here, cfg["output_dir"])

    json_cfg = cfg.get("json", {})
    json_indent = int(json_cfg.get("indent", 2))
    ensure_ascii = bool(json_cfg.get("ensure_ascii", False))

    input_files = list_input_files(input_path)

    print(f"[INFO] input_path            : {input_path}")
    print(f"[INFO] output_dir            : {output_dir}")
    print(f"[INFO] buffer_size           : {algo_cfg.buffer_size}")
    print(f"[INFO] pallet_length         : {pallet_cfg.length}")
    print(f"[INFO] pallet_width          : {pallet_cfg.width}")
    print(f"[INFO] pallet_height         : {pallet_cfg.height}")
    print(f"[INFO] allow_rotation        : {algo_cfg.allow_rotation}")
    print(f"[INFO] total_input_files     : {len(input_files)}")

    total_input_boxes_all = 0
    total_output_boxes_all = 0
    total_placed_volume_all = 0.0
    total_processing_time_files = 0.0

    for idx, input_file in enumerate(input_files):
        file_start_time = time.time()

        boxes = load_boxes(input_file)

        palletizer = Palletizer(
            pallet_cfg=pallet_cfg,
            algo_cfg=algo_cfg,
        )

        result = palletizer.run(boxes)

        output_path = build_output_path(output_dir, input_file)

        save_result(
            result=result,
            output_path=output_path,
            ensure_ascii=ensure_ascii,
            indent=json_indent,
        )

        vis_path = build_vis_path(output_dir, input_file)

        visualize(
            result=result,
            pallet_size=[
                pallet_cfg.length,
                pallet_cfg.width,
                pallet_cfg.height,
            ],
            save_path=vis_path,
        )

        util_info = compute_utilization(result, pallet_cfg)

        file_end_time = time.time()
        file_elapsed = file_end_time - file_start_time

        total_input_boxes_all += len(boxes)
        total_output_boxes_all += len(result["sequence"])
        total_placed_volume_all += util_info["box_volume_sum"]
        total_processing_time_files += file_elapsed

        print()
        print(f"[INFO] file_index            : {idx + 1}/{len(input_files)}")
        print(f"[INFO] input_file            : {input_file}")
        print(f"[INFO] output_file           : {output_path}")
        print(f"[INFO] vis_file              : {vis_path}")
        print(f"[INFO] total_input_boxes     : {len(boxes)}")
        print(f"[INFO] output_boxes          : {len(result['sequence'])}")
        print(f"[INFO] placed_count          : {util_info['placed_count']}")
        print(f"[INFO] placed_volume         : {util_info['box_volume_sum']:.6f}")
        print(f"[INFO] pallet_volume         : {util_info['pallet_volume']:.6f}")
        print(f"[INFO] max_top_height        : {util_info['max_top_height']:.6f}")
        print(f"[INFO] utilization           : {util_info['utilization']:.4f}")
        print(f"[INFO] utilization_percent   : {util_info['utilization'] * 100.0:.2f}%")
        print(f"[INFO] terminated            : {result['terminated']}")
        print(f"[INFO] terminated_step       : {result['terminated_step']}")
        print(f"[INFO] finished_by_user      : {result['finished_by_user']}")
        print(f"[INFO] file_processing_time  : {file_elapsed:.3f} sec")

    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time

    print()
    print("===================================================")
    print("[INFO] ALL FILES SUMMARY")
    print("===================================================")
    print(f"[INFO] total_input_files      : {len(input_files)}")
    print(f"[INFO] total_input_boxes      : {total_input_boxes_all}")
    print(f"[INFO] total_output_boxes     : {total_output_boxes_all}")
    print(f"[INFO] total_placed_volume    : {total_placed_volume_all:.6f}")
    print(f"[INFO] files_processing_time  : {total_processing_time_files:.3f} sec")
    print(f"[INFO] total_processing_time  : {total_elapsed:.3f} sec")


if __name__ == "__main__":
    main()