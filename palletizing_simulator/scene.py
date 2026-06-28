"""
scene.py – 물리 씬 구성 모듈

SimulationApp이 초기화된 후 import해야 합니다.
simulator.py에서 scene.init(cfg) 호출로 설정을 주입합니다.
"""
from __future__ import annotations

import math

import numpy as np
import omni.usd
from pxr import Gf, PhysxSchema, UsdLux, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, GroundPlane

_cfg: dict = {}
_shared_box_mat: PhysicsMaterial | None = None


def init(cfg: dict) -> None:
    global _cfg
    _cfg = cfg


# ──────────────────────────────────────────────
# 재질 / 색상
# ──────────────────────────────────────────────

# def box_color(label_z: float, height: float) -> np.ndarray:
#     """layer(팔레트 상면 기준 z)로 색상 결정."""
#     layer_h = _cfg["colors"]["layer_height"]
#     layers  = _cfg["colors"]["layers"]
#     layer   = int((label_z - height / 2.0) / layer_h)
#     return np.array(layers[min(max(layer, 0), len(layers) - 1)])
def box_color(label_z: float, height: float) -> np.ndarray:
    return np.array([0.20, 0.80, 0.20])


def _make_physics_material(path: str, section: str) -> PhysicsMaterial:
    mat = _cfg["physics"][section]
    return PhysicsMaterial(
        prim_path=path,
        static_friction=mat["static_friction"],
        dynamic_friction=mat["dynamic_friction"],
        restitution=mat["restitution"],
    )


def get_shared_box_mat() -> PhysicsMaterial:
    global _shared_box_mat
    if _shared_box_mat is None:
        _shared_box_mat = _make_physics_material("/World/Physics/box_mat", "box")
    return _shared_box_mat


def reset_shared_box_mat() -> None:
    global _shared_box_mat
    _shared_box_mat = None


# ──────────────────────────────────────────────
# Rotation 유틸
# ──────────────────────────────────────────────

def rotation_quat(degrees: float) -> np.ndarray:
    """Z축 회전 quaternion (w, x, y, z)."""
    rad = math.radians(degrees)
    return np.array([math.cos(rad / 2), 0.0, 0.0, math.sin(rad / 2)])


def rotated_half(size: list[float], rot_deg: float) -> list[float]:
    """rotation 적용 후 (hx, hy, hz). 90° 배수면 x↔y 교환."""
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    return [hy, hx, hz] if abs(rot_deg % 180 - 90) < 1.0 else [hx, hy, hz]


# ──────────────────────────────────────────────
# 씬 구성
# ──────────────────────────────────────────────

def setup_lighting() -> None:
    stage = omni.usd.get_context().get_stage()
    #  기존 조명 제거
    stage.RemovePrim("/World/Lights")
    

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(500.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    distant = UsdLux.DistantLight.Define(stage, "/World/Lights/DistantLight")
    distant.CreateIntensityAttr(500.0)
    distant.CreateAngleAttr(0.53)
    distant.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 5.0))
    distant.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 45.0))


def build_base_scene(
    world: World,
    pallet_size: list[float],
    n_buffer_slots: int,
) -> tuple[float, float]:
    """
    ground plane + 팔레트 + 버퍼 플랫폼을 생성한다.
    Returns: (pallet_thickness, buffer_thickness)
    """
    setup_lighting()
    scene = world.scene

    ground_mat = _make_physics_material("/World/Physics/ground_mat", "ground")
    scene.add(GroundPlane(
        prim_path="/World/GroundPlane",
        name="ground_plane",
        z_position=0.0,
        physics_material=ground_mat,
    ))

    pallet_lx, pallet_ly = pallet_size[0], pallet_size[1]
    pallet_cfg       = _cfg["physics"]["pallet"]
    pallet_thickness = pallet_cfg["thickness"]

    pallet_mat = _make_physics_material("/World/Physics/pallet_mat", "pallet")
    pallet = FixedCuboid(
        prim_path="/World/pallet",
        name="pallet",
        position=np.array([pallet_lx / 2.0, pallet_ly / 2.0, pallet_thickness / 2.0]),
        scale=np.array([pallet_lx, pallet_ly, pallet_thickness]),
        size=1.0,
        color=np.array(pallet_cfg["color"]),
        physics_material=pallet_mat,
    )
    pallet.set_contact_offset(pallet_cfg["contact_offset"])
    pallet.set_rest_offset(pallet_cfg["rest_offset"])
    scene.add(pallet)

    # ── 버퍼 플랫폼 ──────────────────────────────
    buf_cfg          = _cfg["buffer"]
    buffer_thickness = buf_cfg["platform_thickness"]
    gap              = buf_cfg["gap_from_pallet"]
    sw, sd           = buf_cfg["slot_size"]
    spr              = buf_cfg["slots_per_row"]
    n_cols           = spr
    n_rows           = max(1, math.ceil(n_buffer_slots / spr))
    buf_lx           = n_cols * sw
    buf_ly           = n_rows * sd

    buf_mat = _make_physics_material("/World/Physics/buffer_mat", "pallet")
    buf_platform = FixedCuboid(
        prim_path="/World/buffer_platform",
        name="buffer_platform",
        position=np.array([
            pallet_lx + gap + buf_lx / 2.0,
            buf_ly / 2.0,
            buffer_thickness / 2.0,
        ]),
        scale=np.array([buf_lx, buf_ly, buffer_thickness]),
        size=1.0,
        color=np.array(buf_cfg["color"]),
        physics_material=buf_mat,
    )
    buf_platform.set_contact_offset(pallet_cfg["contact_offset"])
    buf_platform.set_rest_offset(pallet_cfg["rest_offset"])
    scene.add(buf_platform)

    return pallet_thickness, buffer_thickness


def buffer_slot_world_pos(
    slot_idx: int,
    pallet_lx: float,
    buffer_thickness: float,
) -> list[float]:
    """버퍼 슬롯 중심의 월드 XY 좌표와 플랫폼 상면 z를 반환."""
    buf_cfg = _cfg["buffer"]
    gap     = buf_cfg["gap_from_pallet"]
    sw, sd  = buf_cfg["slot_size"]
    spr     = buf_cfg["slots_per_row"]
    col     = slot_idx % spr
    row     = slot_idx // spr
    return [
        pallet_lx + gap + sw / 2.0 + col * sw,
        sd / 2.0 + row * sd,
        buffer_thickness,           # 플랫폼 상면 z
    ]


# ──────────────────────────────────────────────
# 안착 대기
# ──────────────────────────────────────────────

def _settle(
    world: World,
    cube: DynamicCuboid,
    simulation_app,
    settle_steps: int,
    settle_vel: float,
) -> None:
    min_frames = _cfg["settling"]["min_frames"]
    for step in range(settle_steps):
        if not simulation_app.is_running():
            break
        world.step(render=True)
        lin_vel = np.linalg.norm(cube.get_linear_velocity())
        ang_vel = np.linalg.norm(cube.get_angular_velocity())
        if step >= min_frames and lin_vel < settle_vel and ang_vel < settle_vel:
            break


def _kinematic_place_and_settle(
    world: World,
    cube: DynamicCuboid,
    target_xyz: list[float],
    rotation_deg: float,
    simulation_app,
    settle_steps: int,
    settle_vel: float,
) -> None:
    """키네마틱 모드로 목표 위치에 정확히 고정 후 dynamic으로 전환해 안착시킨다."""
    rb_api = UsdPhysics.RigidBodyAPI.Apply(cube.prim)
    rb_api.CreateKinematicEnabledAttr().Set(True)
    cube.set_world_pose(
        position=np.array(target_xyz, dtype=float),
        orientation=rotation_quat(rotation_deg),
    )
    # kinematic 상태에서는 velocity API 호출 불가 – 프레임만 진행
    for _ in range(3):
        if simulation_app.is_running():
            world.step(render=True)
    rb_api.GetKinematicEnabledAttr().Set(False)
    # PhysX 가 dynamic 모드를 인식하도록 한 스텝 후 속도 초기화
    if simulation_app.is_running():
        world.step(render=True)
    cube.set_linear_velocity(np.zeros(3))
    cube.set_angular_velocity(np.zeros(3))
    _settle(world, cube, simulation_app, settle_steps, settle_vel)


# ──────────────────────────────────────────────
# 스폰 높이 계산 (XY 겹침 기반)
# ──────────────────────────────────────────────

def _safe_spawn_z(
    target_xy: list[float],
    size: list[float],
    rot_deg: float,
    placed_pairs: list[tuple],  # [(DynamicCuboid, rot_deg), ...]
    floor_z: float,
    drop_offset: float,
) -> float:
    new_half = rotated_half(size, rot_deg)
    max_top_z = floor_z

    for cube, c_rot in placed_pairs:
        c_pos, _ = cube.get_world_pose()
        c_scale   = cube.get_local_scale()
        c_half    = rotated_half(list(c_scale), c_rot)

        if (abs(c_pos[0] - target_xy[0]) < new_half[0] + c_half[0] and
                abs(c_pos[1] - target_xy[1]) < new_half[1] + c_half[1]):
            max_top_z = max(max_top_z, float(c_pos[2]) + c_half[2])

    return max_top_z + new_half[2] + drop_offset


# ──────────────────────────────────────────────
# 박스 생성 (공통)
# ──────────────────────────────────────────────

def _create_cube(
    bid: int,
    size: list[float],
    mass: float,
    rotation_deg: float,
    spawn_xyz: list[float],
    label_z: float,           # 색상 결정용 팔레트 상면 기준 z
    world: World,
) -> DynamicCuboid:
    box_cfg = _cfg["physics"]["box"]
    cube = DynamicCuboid(
        prim_path=f"/World/box_{bid:03d}",
        name=f"box_{bid:03d}",
        position=np.array(spawn_xyz),
        orientation=rotation_quat(rotation_deg),
        scale=np.array(size, dtype=float),
        size=1.0,
        mass=mass,
        color=box_color(label_z, size[2]),
        physics_material=get_shared_box_mat(),
    )
    cube.set_contact_offset(box_cfg["contact_offset"])
    cube.set_rest_offset(box_cfg["rest_offset"])
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(cube.prim)
    physx_rb.CreateLinearDampingAttr().Set(box_cfg["linear_damping"])
    physx_rb.CreateAngularDampingAttr().Set(box_cfg["angular_damping"])
    world.scene.add(cube)
    return cube


# ──────────────────────────────────────────────
# 팔레트에 직접 배치
# ──────────────────────────────────────────────

def spawn_on_pallet(
    world: World,
    bid: int,
    size: list[float],
    mass: float,
    rotation_deg: float,
    target_xyz_world: list[float],
    placed_pairs: list[tuple],
    floor_z: float,
    simulation_app,
    settle_steps: int,
    settle_vel: float,
    drop_offset: float,
) -> DynamicCuboid:
    label_z = target_xyz_world[2] - floor_z

    # 명백한 공중부양만 방지
    if not _has_min_support(
        target_xyz_world,
        size,
        rotation_deg,
        placed_pairs,
        floor_z,
        z_tol=0.01,
        min_support_ratio=0.30,
    ):
        safe_z = _safe_spawn_z(
            target_xy=target_xyz_world[:2],
            size=size,
            rot_deg=rotation_deg,
            placed_pairs=placed_pairs,
            floor_z=floor_z,
            drop_offset=0.0,
        )
        target_xyz_world = [target_xyz_world[0], target_xyz_world[1], safe_z]

    cube = _create_cube(bid, size, mass, rotation_deg, target_xyz_world, label_z, world)
    _kinematic_place_and_settle(world, cube, target_xyz_world, rotation_deg,
                                simulation_app, settle_steps, settle_vel)
    return cube

# ──────────────────────────────────────────────
# 버퍼에 임시 배치
# ──────────────────────────────────────────────

def spawn_in_buffer(
    world: World,
    bid: int,
    size: list[float],
    mass: float,
    rotation_deg: float,
    slot_pos: list[float],        # buffer_slot_world_pos() 반환값 [x, y, z_platform_top]
    label_z: float,               # 최종 팔레트 적재 z (색상용)
    simulation_app,
    settle_steps: int,
    settle_vel: float,
    drop_offset: float,
) -> DynamicCuboid:
    half_z  = size[2] / 2.0
    spawn_z = slot_pos[2] + half_z + drop_offset
    cube    = _create_cube(bid, size, mass, rotation_deg,
                            [slot_pos[0], slot_pos[1], spawn_z],
                            label_z, world)
    _settle(world, cube, simulation_app, settle_steps, settle_vel)
    return cube


# ──────────────────────────────────────────────
# 버퍼 → 팔레트 텔레포트
# ──────────────────────────────────────────────

def teleport_and_settle(
    world: World,
    cube: DynamicCuboid,
    target_xyz_world: list[float],
    rotation_deg: float,
    placed_pairs: list[tuple],
    floor_z: float,
    simulation_app,
    settle_steps: int,
    settle_vel: float,
    drop_offset: float,
) -> None:
    c_scale = cube.get_local_scale()

    if not _has_min_support(
        target_xyz_world,
        list(c_scale),
        rotation_deg,
        placed_pairs,
        floor_z,
        z_tol=0.01,
        min_support_ratio=0.30,
    ):
        safe_z = _safe_spawn_z(
            target_xy=target_xyz_world[:2],
            size=list(c_scale),
            rot_deg=rotation_deg,
            placed_pairs=placed_pairs,
            floor_z=floor_z,
            drop_offset=0.0,
        )
        target_xyz_world = [target_xyz_world[0], target_xyz_world[1], safe_z]

    _kinematic_place_and_settle(world, cube, target_xyz_world, rotation_deg,
                                simulation_app, settle_steps, settle_vel)
def _xy_overlap_area(center_a, half_a, center_b, half_b) -> float:
    dx = min(center_a[0] + half_a[0], center_b[0] + half_b[0]) - max(center_a[0] - half_a[0], center_b[0] - half_b[0])
    dy = min(center_a[1] + half_a[1], center_b[1] + half_b[1]) - max(center_a[1] - half_a[1], center_b[1] - half_b[1])
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return float(dx * dy)


def _has_min_support(
    target_xyz: list[float],
    size: list[float],
    rot_deg: float,
    placed_pairs: list[tuple],   # [(cube, rot_deg), ...]
    floor_z: float,
    *,
    z_tol: float = 0.01,
    min_support_ratio: float = 0.30,
) -> bool:
    new_half = rotated_half(size, rot_deg)
    bottom_z = float(target_xyz[2] - new_half[2])
    target_xy = [float(target_xyz[0]), float(target_xyz[1])]

    base_area = float((new_half[0] * 2.0) * (new_half[1] * 2.0))
    support_area = 0.0

    # 1) 팔레트 상면 지지
    if abs(bottom_z - floor_z) <= z_tol:
        return True

    # 2) 아래 박스 지지
    for cube, c_rot in placed_pairs:
        c_pos, _ = cube.get_world_pose()
        c_scale = cube.get_local_scale()
        c_half = rotated_half(list(c_scale), c_rot)
        top_z = float(c_pos[2] + c_half[2])

        if abs(bottom_z - top_z) <= z_tol:
            support_area += _xy_overlap_area(
                target_xy, new_half,
                [float(c_pos[0]), float(c_pos[1])], c_half,
            )

    support_ratio = support_area / max(base_area, 1e-6)
    return support_ratio >= min_support_ratio