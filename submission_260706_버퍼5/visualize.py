"""
시각화 유틸리티 (수정 불필요)
--------------------------------
적재 결과를 3D 이미지(PNG)로 저장합니다.
알고리즘 개발에 직접 관련이 없으므로 수정하지 않아도 됩니다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_result(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def create_box_vertices(cx, cy, cz, dx, dy, dz):
    x0 = cx - dx / 2
    x1 = cx + dx / 2
    y0 = cy - dy / 2
    y1 = cy + dy / 2
    z0 = cz - dz / 2
    z1 = cz + dz / 2

    return [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]


def create_faces(v):
    return [
        [v[0], v[1], v[2], v[3]],
        [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]],
        [v[1], v[2], v[6], v[5]],
        [v[2], v[3], v[7], v[6]],
        [v[3], v[0], v[4], v[7]],
    ]


def visualize(result: Dict, pallet_size: List[float], save_path: Optional[Path] = None) -> None:
    print(f"[VIS] matplotlib backend: {matplotlib.get_backend()}", flush=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    px, py, pz = pallet_size

    # pallet
    pallet_vertices = create_box_vertices(px / 2, py / 2, -0.025, px, py, 0.05)
    pallet_faces = create_faces(pallet_vertices)
    ax.add_collection3d(
        Poly3DCollection(
            pallet_faces,
            alpha=0.2,
            facecolor="brown",
            edgecolor="k",
            linewidths=0.5,
        )
    )

    seq = result["sequence"]

    for box in seq:
        if box["position"] == "buffer":
            continue

        cx, cy, cz = box["position"]
        dx, dy, dz = box["size"]
        rot = int(box.get("rotation", 0))

        # 여기서는 size가 이미 회전 반영된 값으로 저장되므로
        # 추가 스왑은 하지 않는다.
        vertices = create_box_vertices(cx, cy, cz, dx, dy, dz)
        faces = create_faces(vertices)

        ax.add_collection3d(
            Poly3DCollection(
                faces,
                facecolors=(0.2, 0.6, 0.8),
                linewidths=0.5,
                edgecolors="k",
                alpha=0.6,
            )
        )

    ax.set_xlim([0, px])
    ax.set_ylim([0, py])
    ax.set_zlim([0, pz])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Palletizing Result")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[VIS] saved figure: {save_path}", flush=True)

    # GUI가 있으면 창 표시
    plt.close(fig)