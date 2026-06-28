"""
simulator.py – 팔레타이징 물리 시뮬레이터 진입점

사용법:
    ./python.sh palletizing_2026/simulator.py [옵션]

옵션:
    --config        설정 파일 경로 (기본: config.yaml)
    --input-dir     입력 JSON 디렉토리 (config 값 override)
    --output / -o   출력 디렉토리      (config 값 override)
    --settle-steps  박스당 최대 안착 스텝 (config 값 override)
    --settle-vel    안착 판정 속도 m/s   (config 값 override)
    --final-steps   전체 적재 후 추가 안정화 스텝 (config 값 override)
    --drop-offset   스폰 높이 오프셋 m   (config 값 override)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import yaml
import omni.kit.app
import tempfile
import fcntl
from contextlib import contextmanager
# ──────────────────────────────────────────────
# Config 로드  (SimulationApp 보다 먼저)
# ──────────────────────────────────────────────
_HERE = os.path.dirname(__file__)
_DEFAULT_CONFIG = os.path.join(_HERE, "config/sim_config.yaml")

def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_HERE, path)


def ensure_dirs() -> None:
    input_dir = resolve_path(cfg["paths"]["input_dir"])
    box_sequence_dir = resolve_path(cfg["paths"]["box_sequence_dir"])
    output_dir = resolve_path(cfg["paths"]["output_dir"])

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(box_sequence_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 1단계: --config 경로만 먼저 파싱 ──────────
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", default=_DEFAULT_CONFIG)
_pre_args, _ = _pre.parse_known_args()

cfg = load_config(_pre_args.config)

# ── 2단계: 전체 CLI 파싱 (config 기본값 + override) ──
parser = argparse.ArgumentParser(description="Palletizing physics simulator")
parser.add_argument("--config",       default=_DEFAULT_CONFIG)
parser.add_argument("--input-dir", default=resolve_path(cfg["paths"]["input_dir"]))
parser.add_argument("--output", "-o", default=None)
parser.add_argument("--settle-steps", type=int,   default=cfg["settling"]["max_steps"])
parser.add_argument("--settle-vel",   type=float, default=cfg["settling"]["velocity_threshold"])
parser.add_argument("--final-steps",  type=int,   default=cfg["settling"]["final_steps"])
parser.add_argument("--drop-offset",  type=float, default=cfg["settling"]["drop_offset"])
args, _ = parser.parse_known_args()

# ──────────────────────────────────────────────
# SimulationApp 시작
# ──────────────────────────────────────────────
from omni.isaac.kit import SimulationApp  # noqa: E402
from buffer_manager import BufferManager, build_matching_box_sequence_path

_app_cfg = cfg["app"]
SIM_CONFIG = {
    "experience":      _app_cfg["experience"],
    "width":           _app_cfg["width"],
    "height":          _app_cfg["height"],
    "window_width":    _app_cfg["width"],
    "window_height":   _app_cfg["height"],
    "headless":        _app_cfg["headless"],
    "hide_ui":         _app_cfg["hide_ui"],
    "renderer":        _app_cfg["renderer"],
    "display_options": _app_cfg["display_options"],
}

simulation_app = SimulationApp(SIM_CONFIG)

# from omni.isaac.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension


def _try_enable(ext: str) -> None:
    try:
        enable_extension(ext)
    except Exception as e:
        print(f"[simulator] Warning: could not enable {ext}: {e}", flush=True)


for _ext in _app_cfg.get("extensions", []):
    _try_enable(_ext)

simulation_app.update()

# SimulationApp 이후에 Isaac Sim 모듈 import
import omni.replicator.core as rep  # noqa: E402
import omni.usd                     # noqa: E402
from pxr import Sdf                 # noqa: E402
from isaacsim.core.api import World # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid  # noqa: E402

import scene      # noqa: E402  (SimulationApp 이후에 import)
import evaluator  # noqa: E402

scene.init(cfg)
evaluator.init(cfg)

import monitor    # noqa: E402
import carb

monitor_window = monitor.MonitorWindow()
# settings = carb.settings.get_settings()
# settings.set("/rtx/post/autoExposure/enabled", False)
# settings.set("/rtx/post/exposure", -1.0)
# ──────────────────────────────────────────────
# Input I/O
# ──────────────────────────────────────────────
@contextmanager
def file_lock(lock_path: str, exclusive: bool):
    """
    Linux/Isaac Sim 환경용 파일락.
    exclusive=True  : 저장/쓰기 잠금
    exclusive=False : 읽기 잠금
    """
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(lock_file.fileno(), lock_type)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def atomic_write_json(path: str, data: dict) -> None:
    """
    result.json 저장 중 다른 프로세스가 깨진 JSON을 읽지 않도록
    tmp 파일에 먼저 쓰고 os.replace로 원자적 교체.
    """
    out_dir = os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)

    lock_path = path + ".lock"

    with file_lock(lock_path, exclusive=True):
        fd, tmp_path = tempfile.mkstemp(
            prefix=".tmp_",
            suffix=".json",
            dir=out_dir,
            text=True,
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, path)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
def load_input(path: str) -> dict:
    """# 및 // 주석이 포함된 JSON 파일을 안전하게 파싱한다."""
    lock_path = path + ".lock"

    with file_lock(lock_path, exclusive=False):
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

    raw = re.sub(r"//[^\n]*", "", raw)
    raw = re.sub(r"#[^\n]*", "", raw)
    data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError(f"Input JSON must be an object: {path}")

    if "sequence" not in data:
        raise KeyError(f"Input JSON must contain 'sequence': {path}")

    if not isinstance(data["sequence"], list):
        raise TypeError(f"'sequence' must be a list: {path}")

    return data

def get_pallet_size_from_config(cfg: dict) -> list[float]:
    pallet_cfg = cfg.get("pallet", {})
    pallet_size = pallet_cfg.get("size", None)

    if pallet_size is None:
        raise KeyError("sim_config.yaml must contain pallet.size")

    if not isinstance(pallet_size, list) or len(pallet_size) != 3:
        raise ValueError("pallet.size must be a list of 3 values: [length, width, height]")

    return [float(pallet_size[0]), float(pallet_size[1]), float(pallet_size[2])]

def collect_input_files(directory: str) -> list[str]:
    os.makedirs(directory, exist_ok=True)

    files = sorted(glob.glob(os.path.join(directory, "*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON files found in: {directory}")
    return files

def wait_for_timeline_play(auto_start: bool) -> None:
    if auto_start:
        print("[simulator] auto_start=True -> start immediately", flush=True)
        return

    usd_ctx = omni.usd.get_context()
    timeline = usd_ctx.get_timeline()

    print("[simulator] Waiting for Isaac Sim PLAY button...", flush=True)

    while simulation_app.is_running():
        simulation_app.update()
        if timeline.is_playing():
            print("[simulator] PLAY detected. Starting simulation.", flush=True)
            break
        

    
# ──────────────────────────────────────────────
# Headless screenshot
# ──────────────────────────────────────────────

def save_screenshot(img_path: str, pallet_lx: float, pallet_ly: float) -> None:
    """오프스크린 카메라로 PNG를 저장한다."""
    try:
        from PIL import Image
    except ImportError:
        print("[simulator] Warning: Pillow not found, skipping screenshot.", flush=True)
        return

    ss = cfg["screenshot"]
    cx, cy = pallet_lx / 2.0, pallet_ly / 2.0
    ox, oy, oz = ss["camera_offset"]

    # annotator는 싱글턴 – 새 render_product attach 전에 이전 연결을 끊는다.
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    try:
        annotator.detach()
    except Exception:
        pass

    camera = rep.create.camera(
        position=(cx + ox, cy + oy, oz),
        look_at=(cx, cy, ss["look_at_z"]),
    )
    render_product = rep.create.render_product(camera, (ss["width"], ss["height"]))
    annotator.attach([render_product])

    rep.orchestrator.step(rt_subframes=ss["rt_subframes"], pause_timeline=False)
    for _ in range(ss["flush_frames"]):
        simulation_app.update()

    try:
        rgba = annotator.get_data()
        if rgba is not None and rgba.size > 0:
            Image.fromarray(rgba[:, :, :3]).save(img_path)
            print(f"[simulator] Screenshot → {img_path}", flush=True)
        else:
            print("[simulator] Warning: screenshot empty, retrying …", flush=True)
            for _ in range(8):
                simulation_app.update()
            rgba = annotator.get_data()
            if rgba is not None and rgba.size > 0:
                Image.fromarray(rgba[:, :, :3]).save(img_path)
                print(f"[simulator] Screenshot (retry) → {img_path}", flush=True)
            else:
                print("[simulator] Warning: screenshot still empty, skipping.", flush=True)
    except Exception as e:
        print(f"[simulator] Screenshot error: {e}", flush=True)
    # render_product는 destroy하지 않음 – stage.RemovePrim("/World") 시 정리됨.


# ──────────────────────────────────────────────
# Single-file simulation
# ──────────────────────────────────────────────

def run_one_file(input_path: str, box_sequence_dir: str, out_dir: str) -> dict:
    stem = os.path.splitext(os.path.basename(input_path))[0]
    print(f"\n[simulator] ══ Processing: {os.path.basename(input_path)} ══", flush=True)

    t0 = time.time()

    box_sequence_path = build_matching_box_sequence_path(input_path, box_sequence_dir)
    manager = BufferManager.from_files(input_path, box_sequence_path)

    data = load_input(input_path)
    sequence = data["sequence"]
    pallet_size = get_pallet_size_from_config(cfg)
    pallet_thickness = cfg["physics"]["pallet"]["thickness"]

    print(
        f"[simulator] plan={len(sequence)}  "
        f"buffer_size={manager.buffer_size}  "
        f"source_boxes={manager.total_source_count()}",
        flush=True,
    )

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    pallet_thickness, buffer_thickness = scene.build_base_scene(
        world, pallet_size, manager.buffer_size
    )
    world.reset()

    placed_pairs: list[tuple] = []
    buffer_boxes: dict[int, DynamicCuboid] = {}
    buffer_slot_map: dict[int, int] = {}

    settle_kw = dict(
        simulation_app=simulation_app,
        settle_steps=args.settle_steps,
        settle_vel=args.settle_vel,
        drop_offset=args.drop_offset,
    )

    # --------------------------------------------------
    # 1) 초기 버퍼 채우기
    # --------------------------------------------------
    initial_boxes = manager.get_initial_buffer_boxes()
    for slot_idx, box in enumerate(initial_boxes):
        bid = int(box["id"])
        slot = scene.buffer_slot_world_pos(
            slot_idx, pallet_size[0], buffer_thickness
        )

        cube = scene.spawn_in_buffer(
            world=world,
            bid=bid,
            size=box["size"],
            mass=box["mass"],
            rotation_deg=0.0,
            slot_pos=slot,
            label_z=0.0,
            **settle_kw,
        )
        buffer_boxes[bid] = cube
        buffer_slot_map[bid] = slot_idx

    print(
        f"[simulator] initial buffer ids = {manager.get_current_buffer_ids()}",
        flush=True,
    )

    # --------------------------------------------------
    # 2) algorithm_results.sequence 순서대로 적재
    # --------------------------------------------------
    for i, step in enumerate(sequence):
        if not simulation_app.is_running():
            break

        if not isinstance(step, dict):
            raise TypeError(f"sequence[{i}] must be an object")

        for key in ("id", "position"):
            if key not in step:
                raise KeyError(f"sequence[{i}] missing required key: '{key}'")

        sid = int(step["id"])
        pos = step["position"]
        rot = float(step.get("rotation", 0.0))

        if not isinstance(pos, list) or len(pos) != 3:
            raise ValueError(f"sequence[{i}]['position'] must be [x, y, z]")

        # =========================================================
        # buffer_size == 0
        # 버퍼 없이 즉시 팔레트 적재
        # =========================================================
        if manager.buffer_size == 0:

            box_info = manager.get_box_info(sid)

            world_xyz = [
                pos[0],
                pos[1],
                pos[2] + pallet_thickness,
            ]

            cube = scene.spawn_on_pallet(
                world=world,
                bid=sid,
                size=box_info["size"],
                mass=box_info["mass"],
                rotation_deg=rot,
                target_xyz_world=world_xyz,
                placed_pairs=placed_pairs,
                floor_z=pallet_thickness,
                **settle_kw,
            )

            placed_pairs.append((cube, rot))
            continue

        # =========================================================
        # 기존 buffer 기반 로직
        # =========================================================
        if sid not in buffer_boxes:
            raise ValueError(
                f"Step id={sid} is not in spawned buffer boxes. "
                f"current_buffer={manager.get_current_buffer_ids()}"
            )

        consume_info = manager.consume(sid)
        refilled_box = consume_info["refilled_box"]

        cube = buffer_boxes.pop(sid)
        freed_slot_idx = buffer_slot_map.pop(sid)

        world_xyz = [pos[0], pos[1], pos[2] + pallet_thickness]

        scene.teleport_and_settle(
            world=world,
            cube=cube,
            target_xyz_world=world_xyz,
            rotation_deg=rot,
            placed_pairs=placed_pairs,
            floor_z=pallet_thickness,
            **settle_kw,
        )

        placed_pairs.append((cube, rot))

        # 1개 적재 후 자동 보충
        if refilled_box is not None:

            refill_id = int(refilled_box["id"])

            refill_slot = scene.buffer_slot_world_pos(
                freed_slot_idx,
                pallet_size[0],
                buffer_thickness,
            )

            refill_cube = scene.spawn_in_buffer(
                world=world,
                bid=refill_id,
                size=refilled_box["size"],
                mass=refilled_box["mass"],
                rotation_deg=0.0,
                slot_pos=refill_slot,
                label_z=0.0,
                **settle_kw,
            )

            buffer_boxes[refill_id] = refill_cube
            buffer_slot_map[refill_id] = freed_slot_idx

            refill_msg = (
                f" -> refill id={refill_id} "
                f"slot={freed_slot_idx}"
            )

        else:
            refill_msg = " -> refill none"

        # print(
        #     f"[simulator] step {i+1:>3d}/{len(sequence)}  "
        #     f"place id={sid:>3d}  slot={freed_slot_idx}{refill_msg}  "
        #     f"buffer={manager.get_current_buffer_ids()}  "
        #     f"({time.time()-t0:.1f}s)",
        #     flush=True,
        # )

    # ── 전체 적재 후 추가 안정화 ─────────────────
    print(f"[simulator] Final settle {args.final_steps} steps …", flush=True)
    for _ in range(args.final_steps):
        if not simulation_app.is_running():
            break
        world.step(render=True)

    # ── 결과 수집 (팔레트 위 박스만) ─────────────
    sim_results = []
    for cube, _ in placed_pairs:
        pos, orient = cube.get_world_pose()
        sim_results.append({
            "id": int(cube.name.split("_")[-1]),
            "position": pos.tolist(),
            "orientation": orient.tolist(),
        })

    elapsed = time.time() - t0
    print(f"[simulator] All steps done  (total {elapsed:.1f}s)", flush=True)

    # --------------------------------------------------
    # evaluator용 final_stack_by_id 구성
    # algorithm_results.sequence가 최종 적재 계획이므로 여기서 생성
    # --------------------------------------------------
    final_stack_by_id: dict[int, dict] = {}
    for step in sequence:
        sid = int(step["id"])
        final_stack_by_id[sid] = {
            "size": manager.get_box_info(sid)["size"],
            "position": step["position"],
            "rotation": float(step.get("rotation", 0.0)),
        }

    file_eval = evaluator.evaluate_file(
        source=os.path.basename(input_path),
        sim_results=sim_results,
        final_stack_by_id=final_stack_by_id,
        pallet_size=pallet_size,
        pallet_thickness=pallet_thickness,
        buffer_size=manager.buffer_size,
    )

    monitor_window.update_episode(file_eval)
    simulation_app.update()

    fe = file_eval
    height_tag = "  HEIGHT_OVERFLOW" if fe.get("height_overflow") else ""
    print(
        f"[evaluator] [{fe['episode'].upper()}]{height_tag}  "
        f"ok={fe['success_count']}/{fe['total_boxes']}  "
        f"stacking={fe['stacking_rate_pct_official']:.1f}% "
        f"[{fe['stacking_rate_pct_raw']:.1f}%]  "
        f"collapse={fe['collapse_count']}  "
        f"oob={fe['out_of_bounds_count']}  "
        f"drop={fe['drop_count']}  "
        f"buffer={fe['buffer_size']}  "
        f"score={fe['total_score']:.1f} "
        f"(stacking={fe['episode_score']:.1f} + buffer_bonus={fe['buffer_bonus']})",
        flush=True,
    )

    img_path = os.path.join(out_dir, f"{stem}_result.png")
    save_screenshot(img_path, pallet_size[0], pallet_size[1])

    world.clear()
    scene.reset_shared_box_mat()

    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath("/World").IsValid():
        stage.RemovePrim(Sdf.Path("/World"))

    simulation_app.update()
    simulation_app.update()
    file_eval["processing_time_sec"] = round(elapsed, 3)

    return file_eval



# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    total_t0 = time.time()

    ensure_dirs()

    input_dir = resolve_path(args.input_dir)
    out_dir = args.output or resolve_path(cfg["paths"]["output_dir"])

    input_files = collect_input_files(input_dir)

    os.makedirs(out_dir, exist_ok=True)

    print(f"[simulator] Config     : {_pre_args.config}", flush=True)
    print(f"[simulator] Input dir  : {args.input_dir}", flush=True)
    print(f"[simulator] Output dir : {out_dir}", flush=True)
    print(f"[simulator] Found {len(input_files)} input file(s)", flush=True)
    for p in input_files:
        print(f"  - {os.path.basename(p)}", flush=True)

    print(
        f"[simulator] drop_offset={args.drop_offset}m  "
        f"settle_steps(max)={args.settle_steps}  "
        f"settle_vel={args.settle_vel}m/s  "
        f"final_steps={args.final_steps}",
        flush=True,
    )

    import omni.usd

    usd_ctx = omni.usd.get_context()
    usd_ctx.new_stage()

    simulation_app.update()
    simulation_app.update()

    auto_start = bool(cfg.get("runtime", {}).get("auto_start", True))
    print(f"[simulator] auto_start : {auto_start}", flush=True)
    wait_for_timeline_play(auto_start)

    if not simulation_app.is_running():
        simulation_app.close()
        return

    file_results = []
    box_sequence_dir = resolve_path(cfg["paths"]["box_sequence_dir"])

    for path in input_files:
        if not simulation_app.is_running():
            break

        file_eval = run_one_file(path, box_sequence_dir, out_dir)
        file_results.append(file_eval)

    if file_results:
        result = evaluator.build_result(file_results)
        total_elapsed = time.time() - total_t0
        result["summary"]["total_processing_time_sec"] = round(total_elapsed, 3)
        monitor_window.update_summary(result)
        simulation_app.update()
        result_path = os.path.join(out_dir, "result.json")
        atomic_write_json(result_path, result)

        print(f"\n[evaluator] ══ Result ══", flush=True)
        for fl in result["files"]:
            tag = "✓" if fl["episode"] == "success" else "✗"
            hoflg = "  [H]" if fl.get("height_overflow") else ""
            print(
                f"  {tag}{hoflg} {fl['source']:<12}  "
                f"stacking={fl['stacking_rate_pct']:.1f}% "
                f"[{fl['stacking_rate_pct_raw']:.1f}%]  "
                f"ok={fl['success_count']}/{fl['total_boxes']}  "
                f"time={fl.get('processing_time_sec', 0.0):.1f}s  "
                f"buffer={fl.get('buffer_size', 0)}  "
                f"score={fl.get('total_score', 0.0):.1f} "
                f"(stacking={fl.get('episode_score', 0.0):.1f} + bonus={fl.get('buffer_bonus', 0)})",
                flush=True,
            )

        sm = result["summary"]
        rs = result["results"]
        print(f"  {'─'*55}", flush=True)
        print(
            f"  summary : episodes {sm['success_episodes']} success / {sm['failure_episodes']} failure"
            f" / {sm['height_overflow_episodes']} height_overflow",
            flush=True,
        )
        print(
            f"            avg stacking rate  {sm['avg_stacking_rate_pct']:.1f}% "
            f"[{sm['avg_stacking_rate_pct_raw']:.1f}%]",
            flush=True,
        )
        print(
            f"            avg success rate   {sm['avg_success_rate_pct']:.1f}%",
            flush=True,
        )
        print(
            f"            avg score          {rs['avg_score']:.2f}",
            flush=True,
        )
        print(
            f"            total processing time {sm['total_processing_time_sec']:.1f}s",
            flush=True,
        )
        print(f"[evaluator] Saved → {result_path}", flush=True)

    print("\n[simulator] All files processed.", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
