"""
export_onnx.py
==============================================================================
학습으로 저장된 best_model(.pt)을 ONNX로 변환한다.
제출 환경은 ONNX 전용이므로, 추론(algorithm.py)은 이 .onnx 를 사용한다.

사용법:
    python export_onnx.py --pt DRL_cnn_model_B1.pt --onnx DRL_cnn_model_B1.onnx

.pt 체크포인트에는 state_dict, cfg(vars), buffer_B, in_ch, N 이 들어 있다.
이 정보로 동일한 네트워크를 만들고 가중치를 올린 뒤 ONNX로 export 한다.
==============================================================================
"""
import argparse
import torch
from model_learn import ActorCritic   # 학습 때와 동일한 네트워크 정의


def export(pt_path, onnx_path):
    from model_learn import Config
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    # 체크포인트 cfg(vars)는 클래스 속성(Lc/Wc/cell 등)을 누락할 수 있으므로
    # Config() 기본값 위에 덮어써 완전한 cfg 를 복원한다.
    base = {k: getattr(Config, k) for k in dir(Config) if not k.startswith("_")}
    base.update(ckpt.get("cfg", {}))
    cfg = base

    in_ch = ckpt["in_ch"]
    N = ckpt["N"]
    Lc, Wc = cfg["Lc"], cfg["Wc"]

    net = ActorCritic(in_ch, N, Lc, Wc)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    dummy = torch.zeros(1, in_ch, Lc, Wc)
    torch.onnx.export(
        net, dummy, onnx_path,
        input_names=["state"], output_names=["logits", "value"],
        dynamic_axes={"state": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print(f"[OK] ONNX 저장: {onnx_path}")
    print(f"     buffer_B={ckpt.get('buffer_B')}  in_ch={in_ch}  N={N}  grid={Lc}x{Wc}")
    print(f"     util={ckpt.get('util')}  total_score={ckpt.get('total_score')}")

    # 추론(algorithm.py)이 학습과 동일하게 상태·mask 를 구성하도록 메타 저장
    import json, os
    meta = {
        "onnx_path": os.path.basename(onnx_path),
        "buffer_B": ckpt.get("buffer_B"), "in_ch": in_ch, "N": N,
        "Lc": Lc, "Wc": Wc, "Hc": cfg["Hc"], "cell": cfg["cell"],
        "max_mass": cfg["max_mass"], "k_hol": cfg["k_hol"],
        "com_alpha0": cfg["com_alpha0"], "com_alpha1": cfg["com_alpha1"],
    }
    meta_path = os.path.join(os.path.dirname(onnx_path) or ".", "model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[OK] 메타 저장: {meta_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pt", required=True, help="입력 .pt 경로")
    p.add_argument("--onnx", required=True, help="출력 .onnx 경로")
    args = p.parse_args()
    export(args.pt, args.onnx)
