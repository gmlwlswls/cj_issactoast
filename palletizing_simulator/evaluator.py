# evaluator.py

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

_cfg: dict = {}


def init(cfg: dict) -> None:
    global _cfg
    _cfg = cfg


def _drift(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _evaluate_box(
    box_id: int,
    box_size: list[float],
    intended_pos_world: list[float],
    final_pos: list[float],
    pallet_size: list[float],
    pallet_thickness: float,
    rotation_deg: float = 0.0,
) -> dict[str, Any]:
    ev = _cfg["evaluation"]
    drift_thresh = ev["drift_threshold_m"]
    tol = ev["bounds_tolerance_m"]

    pallet_lx, pallet_ly, pallet_lz = pallet_size
    rot_is_90 = abs(rotation_deg % 180 - 90) < 1.0
    half_x = box_size[1] / 2.0 if rot_is_90 else box_size[0] / 2.0
    half_y = box_size[0] / 2.0 if rot_is_90 else box_size[1] / 2.0

    drift_m = _drift(intended_pos_world, final_pos)

    out_of_bounds = (
        (final_pos[0] - half_x < -tol) or (final_pos[0] + half_x > pallet_lx + tol) or
        (final_pos[1] - half_y < -tol) or (final_pos[1] + half_y > pallet_ly + tol) or
        (final_pos[2] < pallet_thickness - tol) or
        (final_pos[2] > pallet_thickness + pallet_lz + tol)
    )

    # 박스 상단(center + half_z)이 팔레트 최대 적재 높이를 초과하는지 체크
    box_top_z = final_pos[2] + box_size[2] / 2.0
    height_overflow = box_top_z > pallet_thickness + pallet_lz + tol

    unstable = drift_m > drift_thresh

    if out_of_bounds and unstable:
        failure_type = "drop"
    elif out_of_bounds:
        failure_type = "out_of_bounds"
    elif unstable:
        failure_type = "collapse"
    else:
        failure_type = None

    return {
        "id": box_id,
        "drift_m": round(drift_m, 4),
        "in_bounds": not out_of_bounds,
        "height_overflow": height_overflow,
        "success": failure_type is None,
        "failure_type": failure_type,
    }


_BUFFER_MAX = 20


def evaluate_file(
    source: str,
    sim_results: list[dict],
    final_stack_by_id: dict[int, dict],
    pallet_size: list[float],
    pallet_thickness: float,
    buffer_size: int = 0,
) -> dict[str, Any]:
    ev = _cfg["evaluation"]
    episode_min_rate = ev["episode_success_min_rate"]
    pallet_volume = pallet_size[0] * pallet_size[1] * pallet_size[2]

    sim_by_id = {r["id"]: r for r in sim_results}

    success_count = 0
    collapse_count = 0
    out_of_bounds_count = 0
    drop_count = 0
    height_overflow_count = 0
    stacked_volume = 0.0
    max_height = 0.0

    for bid, intended in final_stack_by_id.items():
        if bid not in sim_by_id:
            continue

        box_size = intended["size"]
        intended_pos = intended["position"]

        intended_world = [
            intended_pos[0],
            intended_pos[1],
            intended_pos[2] + pallet_thickness,
        ]

        final_pos = sim_by_id[bid]["position"]
        result = _evaluate_box(
            box_id=bid,
            box_size=box_size,
            intended_pos_world=intended_world,
            final_pos=final_pos,
            pallet_size=pallet_size,
            pallet_thickness=pallet_thickness,
            rotation_deg=intended.get("rotation", 0.0),
        )

        box_top = final_pos[2] + box_size[2] / 2.0 - pallet_thickness
        max_height = max(max_height, box_top)

        if result["height_overflow"]:
            height_overflow_count += 1

        ft = result["failure_type"]
        if ft is None:
            success_count += 1
            stacked_volume += box_size[0] * box_size[1] * box_size[2]
        elif ft == "collapse":
            collapse_count += 1
        elif ft == "out_of_bounds":
            out_of_bounds_count += 1
        else:
            drop_count += 1

    total = len(final_stack_by_id)
    failure_count = collapse_count + out_of_bounds_count + drop_count
    success_rate = success_count / total if total > 0 else 0.0

    stacking_rate_raw = stacked_volume / pallet_volume if pallet_volume > 0 else 0.0
    episode_ok = success_rate >= episode_min_rate
    stacking_rate_official = stacking_rate_raw if episode_ok else 0.0

    episode_height_overflow = height_overflow_count > 0

    # 점수 계산: 물리 실패 또는 높이 초과 시 0점
    if not episode_ok or episode_height_overflow:
        episode_score = 0.0
        buffer_bonus = 0
    else:
        # 적재율 점수: 100% → 100점 (선형)
        episode_score = round(stacking_rate_official * 100.0, 1)
        # 버퍼 보너스: 20개 사용 → 0점, 0개 사용 → 20점
        buffer_bonus = max(0, _BUFFER_MAX - buffer_size)

    total_score = round(episode_score + buffer_bonus, 1)

    return {
        "source": source,
        "episode": "success" if episode_ok else "failure",
        "height_overflow": episode_height_overflow,
        "height_overflow_count": height_overflow_count,
        "pallet_volume_m3": round(pallet_volume, 4),
        "stacked_volume_m3": round(stacked_volume, 4),

        # raw / official 둘 다 저장
        "stacking_rate_pct_raw": round(stacking_rate_raw * 100.0, 1),
        "stacking_rate_pct_official": round(stacking_rate_official * 100.0, 1),

        # 하위 호환용: 앞으로는 official을 기본 표시값으로 사용
        "stacking_rate_pct": round(stacking_rate_official * 100.0, 1),

        "success_count": success_count,
        "failure_count": failure_count,
        "collapse_count": collapse_count,
        "out_of_bounds_count": out_of_bounds_count,
        "drop_count": drop_count,
        "total_boxes": total,
        "success_rate_pct": round(success_rate * 100.0, 1),

        "max_height_m": round(max_height, 3),
        "buffer_size": buffer_size,
        "episode_score": episode_score,
        "buffer_bonus": buffer_bonus,
        "total_score": total_score,
    }


def build_result(file_results: list[dict]) -> dict[str, Any]:
    total_boxes = sum(f["total_boxes"] for f in file_results)
    total_success = sum(f["success_count"] for f in file_results)
    total_collapse = sum(f["collapse_count"] for f in file_results)
    total_oob = sum(f["out_of_bounds_count"] for f in file_results)
    total_drop = sum(f["drop_count"] for f in file_results)

    ep_success = sum(1 for f in file_results if f["episode"] == "success")
    ep_failure = len(file_results) - ep_success
    ep_height_overflow = sum(1 for f in file_results if f.get("height_overflow", False))

    # 공식 평균: failure는 0으로 처리된 official 값 평균
    avg_stacking_official = (
        sum(f["stacking_rate_pct_official"] for f in file_results) / len(file_results)
        if file_results else 0.0
    )

    # raw 평균: failure여도 실제 적재율 반영
    avg_stacking_raw = (
        sum(f["stacking_rate_pct_raw"] for f in file_results) / len(file_results)
        if file_results else 0.0
    )

    avg_success = (total_success / total_boxes * 100.0) if total_boxes > 0 else 0.0

    avg_score = (
        sum(f.get("total_score", 0.0) for f in file_results) / len(file_results)
        if file_results else 0.0
    )

    file_logs = []
    for f in file_results:
        file_logs.append({
            "source": f["source"],
            "episode": f["episode"],
            "height_overflow": f.get("height_overflow", False),
            "stacking_rate_pct": f["stacking_rate_pct_official"],
            "stacking_rate_pct_raw": f["stacking_rate_pct_raw"],
            "success_count": f["success_count"],
            "failure_count": f["failure_count"],
            "collapse_count": f["collapse_count"],
            "out_of_bounds_count": f["out_of_bounds_count"],
            "drop_count": f["drop_count"],
            "total_boxes": f["total_boxes"],
            "max_height_m": f.get("max_height_m", 0.0),
            "buffer_size": f.get("buffer_size", 0),
            "episode_score": f.get("episode_score", 0.0),
            "buffer_bonus": f.get("buffer_bonus", 0),
            "total_score": f.get("total_score", 0.0),
        })

    return {
        "results": {
            "avg_score": round(avg_score, 2),
            "episodes": [
                {
                    "source": f["source"],
                    "status": "success" if (f["episode"] == "success" and not f.get("height_overflow")) else "failure",
                    "reason": (
                        None if (f["episode"] == "success" and not f.get("height_overflow")) else
                        "height_overflow" if f.get("height_overflow") else
                        ", ".join(
                            f"{k}:{v}" for k, v in [
                                ("drop", f["drop_count"]),
                                ("collapse", f["collapse_count"]),
                                ("out_of_bounds", f["out_of_bounds_count"]),
                            ] if v > 0
                        ) or "physics_failure"
                    ),
                    "max_height_m": f["max_height_m"],
                    "buffer_size": f["buffer_size"],
                    "score": f["total_score"],
                }
                for f in file_logs
            ],
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": file_logs,
        "summary": {
            "total_episodes": len(file_results),
            "success_episodes": ep_success,
            "failure_episodes": ep_failure,
            "height_overflow_episodes": ep_height_overflow,
            "total_boxes": total_boxes,
            "total_success_boxes": total_success,
            "total_failure_boxes": total_collapse + total_oob + total_drop,
            "total_collapse": total_collapse,
            "total_out_of_bounds": total_oob,
            "total_drop": total_drop,
            "avg_success_rate_pct": round(avg_success, 1),

            # 표시용
            "avg_stacking_rate_pct": round(avg_stacking_official, 1),
            "avg_stacking_rate_pct_raw": round(avg_stacking_raw, 1),
        },
    }