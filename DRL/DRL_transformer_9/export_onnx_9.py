"""
export_onnx.py (for model_learn_v3.py)
=====================================================
학습으로 저장된 v3 체크포인트(.pt)를 ONNX(.onnx)로 변환한다.

사용법:
    python export_onnx.py --pt O4MSP_B1.pt --onnx O4MSP_B1.onnx

.pt 체크포인트에는 최소한 state_dict가 있어야 하며,
가능하면 buffer_B, d_model, n_layers, n_heads도 포함되어야 한다.
=====================================================
"""
import argparse
import os
import torch

from model_learn_v3 import O4MSPNet, export_onnx  # ✅ v3 기준으로 변경


def export(pt_path: str, onnx_path: str, device: str = "cpu"):
    # torch.load 호환성 (PyTorch 버전에 따라 weights_only 인자 지원 여부가 다를 수 있음)
    try:
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(pt_path, map_location="cpu")

    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("Expected v3 checkpoint dict with key 'state_dict'.")

    buffer_B = int(ckpt.get("buffer_B", 1))
    d_model = int(ckpt.get("d_model", 128))
    n_layers = int(ckpt.get("n_layers", 3))
    n_heads = int(ckpt.get("n_heads", 4))

    net = O4MSPNet(d=d_model, layers=n_layers, heads=n_heads)
    net.load_state_dict(ckpt["state_dict"], strict=True)
    net.to(device).eval()

    # v3 export_onnx는 meta_path "파일 경로"를 받지만,
    # 내부에서 dirname(meta_path)에 model_meta.json을 저장함
    meta_path = os.path.join(os.path.dirname(onnx_path) or ".", "model_meta.json")

    export_onnx(
        net=net,
        buffer_B=buffer_B,
        onnx_path=onnx_path,
        meta_path=meta_path,
        device=device,
    )

    print(f"[OK] ONNX saved: {onnx_path}")
    print(f"     buffer_B={buffer_B}  d_model={d_model}  n_layers={n_layers}  n_heads={n_heads}")
    print(f"[OK] Meta saved in: {os.path.dirname(meta_path) or '.'}/model_meta.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pt", required=True, help="입력 .pt 경로 (v3 체크포인트)")
    p.add_argument("--onnx", required=True, help="출력 .onnx 경로")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    export(args.pt, args.onnx, device=args.device)