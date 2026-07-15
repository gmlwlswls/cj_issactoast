"""
팔레타이징 DRL 학습용 박스 시퀀스 생성기
================================================
- 5개 고정 SKU(주 분포) + 부피-질량 거듭제곱 관계식을 따르는 합성 박스(증강)를 섞어
  기존 box_sequence_*.json 과 동일한 JSONL 포맷으로 시퀀스를 생성한다.
- 단위: 길이 m, 질량 kg (기존 데이터와 동일).
- 각 줄: {"step", "id", "type", "type_name", "size":[l,w,h], "mass"}
  * 합성 박스는 도메인 랜덤화(증강)용 → mass 는 mass=2544*vol^1.55 로 부여(밀도 증가 추세 반영).
  * 합성 박스는 실제 5개 type 으로 위장하지 않고 별도 type="synthetic"로 태깅.
  * 시퀀스 길이 250 → 총 부피가 팔레트(1.5 m^3)를 크게 초과하므로 항상 packing-limited.
"""
import os
import json
import argparse
import numpy as np

# ── 5개 고정 SKU 카탈로그 (주어진 데이터에서 추출) ─────────────────
#   size = [length, width, height] (m),  mass (kg)
SKU_CATALOG = [
    {"type": 0, "type_name": "1", "size": [0.195, 0.178, 0.134], "mass": 0.5},
    {"type": 1, "type_name": "2", "size": [0.245, 0.178, 0.140], "mass": 1.0},
    {"type": 2, "type_name": "3", "size": [0.245, 0.220, 0.158], "mass": 2.0},
    {"type": 3, "type_name": "4", "size": [0.310, 0.233, 0.210], "mass": 4.0},
    {"type": 4, "type_name": "5", "size": [0.315, 0.272, 0.257], "mass": 6.0},
]

# 합성 박스 치수 범위 (과제 스펙: W·L 170~320mm, H 130~260mm)
DIM_RANGE = {"L": (0.170, 0.320), "W": (0.170, 0.320), "H": (0.130, 0.260)}
MASS_CLAMP = (0.3, 7.5)  # 합성 질량 안전 범위 (kg)

SYNTH_TYPE = 5
SYNTH_TYPE_NAME = "synthetic"


def curriculum_fixed_ratio(progress, start=0.5, end=0.9, warmup=0.1):
    """
    학습 진행도(progress: 0.0~1.0)에 따라 fixed_ratio 를 매끄럽게 증가시키는 스케줄러.
      - 초반(다양성 우선): start (기본 0.5) → 합성 박스를 많이 섞어 일반화 워밍업
      - 후반(실전 분포 정밀화): end (기본 0.9) → 5 SKU 비중을 높여 평가 분포에 튜닝
      - warmup: 이 구간(progress < warmup)에서는 start 를 유지하다가 이후 선형 증가
    학습 루프에서 매 에피소드/스텝마다 호출해 그때의 fixed_ratio 로 시퀀스를 뽑으면 됨.

    예) for ep in range(N):
            fr = curriculum_fixed_ratio(ep / N)
            seq = generate_sequence(rng, a, b, 250, fr, 0.10, "uniform", 2.0)
    """
    progress = float(np.clip(progress, 0.0, 1.0))
    if progress <= warmup:
        return start
    # warmup 이후 [warmup,1] 구간을 [0,1] 로 정규화해 선형 보간
    frac = (progress - warmup) / (1.0 - warmup)
    return float(start + (end - start) * frac)


def fit_mass_law(catalog):
    """5개 SKU 로부터 mass = a * vol^b (로그선형 회귀) 계수를 적합."""
    vol = np.array([np.prod(s["size"]) for s in catalog])
    mass = np.array([s["mass"] for s in catalog])
    b, log_a = np.polyfit(np.log(vol), np.log(mass), 1)
    a = float(np.exp(log_a))
    return a, float(b)


def make_synthetic_box(rng, a, b, noise_pct):
    """치수를 균등 샘플링하고 거듭제곱식+곱셈노이즈로 질량을 부여한 합성 박스."""
    L = rng.uniform(*DIM_RANGE["L"])
    W = rng.uniform(*DIM_RANGE["W"])
    H = rng.uniform(*DIM_RANGE["H"])
    L, W = max(L, W), min(L, W)          # 기존 데이터 규약: length >= width
    vol = L * W * H
    mass = a * (vol ** b) * (1.0 + rng.uniform(-noise_pct, noise_pct))
    mass = float(np.clip(mass, *MASS_CLAMP))
    return {
        "type": SYNTH_TYPE,
        "type_name": SYNTH_TYPE_NAME,
        "size": [round(L, 3), round(W, 3), round(H, 3)],
        "mass": round(mass, 2),
    }


def generate_sequence(rng, a, b, n_boxes, fixed_ratio, noise_pct,
                      type_sampling, dirichlet_alpha):
    """박스 n_boxes 개로 이루어진 한 시퀀스(리스트[dict])를 생성."""
    # 이 시퀀스에서 고정 SKU 의 type 분포 결정
    if type_sampling == "dirichlet":
        type_p = rng.dirichlet([dirichlet_alpha] * len(SKU_CATALOG))
    else:  # uniform
        type_p = np.full(len(SKU_CATALOG), 1.0 / len(SKU_CATALOG))

    seq = []
    for step in range(n_boxes):
        if rng.random() < fixed_ratio:
            t = int(rng.choice(len(SKU_CATALOG), p=type_p))
            sku = SKU_CATALOG[t]
            box = {"type": sku["type"], "type_name": sku["type_name"],
                   "size": list(sku["size"]), "mass": sku["mass"]}
        else:
            box = make_synthetic_box(rng, a, b, noise_pct)
        box = {"step": step, "id": step, **box}
        seq.append(box)
    return seq


def write_jsonl(seq, path):
    with open(path, "w", encoding="utf-8") as f:
        for box in seq:
            f.write(json.dumps(box, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="/mnt/user-data/outputs/generated_sequences")
    p.add_argument("--n_sequences", type=int, default=8, help="생성할 시퀀스(파일) 수")
    p.add_argument("--n_boxes", type=int, default=250, help="시퀀스당 박스 수")
    p.add_argument("--fixed_ratio", type=float, default=0.80, help="고정 SKU 비율(나머지는 합성)")
    p.add_argument("--noise_pct", type=float, default=0.10, help="합성 박스 질량 곱셈노이즈 폭(±)")
    p.add_argument("--type_sampling", choices=["uniform", "dirichlet"], default="uniform")
    p.add_argument("--dirichlet_alpha", type=float, default=2.0,
                   help="dirichlet 일 때 작을수록 시퀀스별 type 편중이 커짐")
    p.add_argument("--seed", type=int, default=42)
    # ── 커리큘럼 모드 ────────────────────────────────────────────
    p.add_argument("--curriculum", action="store_true",
                   help="단계별 fixed_ratio 로 데이터셋을 미리 생성(스테이지별 하위폴더)")
    p.add_argument("--stages", type=int, default=4, help="커리큘럼 스테이지 수")
    p.add_argument("--cur_start", type=float, default=0.5, help="커리큘럼 시작 fixed_ratio")
    p.add_argument("--cur_end", type=float, default=0.9, help="커리큘럼 종료 fixed_ratio")
    p.add_argument("--cur_warmup", type=float, default=0.1, help="시작 비율 유지 구간(progress)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    a, b = fit_mass_law(SKU_CATALOG)
    pallet_vol = 1.2 * 1.0 * 1.25  # 1.5 m^3

    print(f"질량 관계식: mass = {a:.1f} * vol^{b:.3f}  (5 SKU 적합)")
    print(f"출력 폴더 : {args.out_dir}")
    print(f"설정      : n_seq={args.n_sequences}, n_boxes={args.n_boxes}, "
          f"fixed_ratio={args.fixed_ratio}, noise=±{args.noise_pct}, "
          f"type_sampling={args.type_sampling}\n")

    master = np.random.default_rng(args.seed)

    if args.curriculum:
        # 스테이지마다 progress 의 대표값으로 fixed_ratio 를 구해 별도 폴더에 생성
        print(f"[커리큘럼] stages={args.stages}, "
              f"fixed_ratio {args.cur_start}→{args.cur_end} (warmup={args.cur_warmup})\n")
        for s in range(args.stages):
            # 스테이지 중앙값을 progress 로 사용
            progress = (s + 0.5) / args.stages
            fr = curriculum_fixed_ratio(progress, args.cur_start, args.cur_end, args.cur_warmup)
            stage_dir = os.path.join(args.out_dir, f"stage_{s}_fr{fr:.2f}")
            os.makedirs(stage_dir, exist_ok=True)
            print(f"── stage {s}: progress≈{progress:.2f}, fixed_ratio={fr:.2f} → {stage_dir}")
            for i in range(args.n_sequences):
                rng = np.random.default_rng(master.integers(0, 2**31 - 1))
                seq = generate_sequence(rng, a, b, args.n_boxes, fr,
                                        args.noise_pct, args.type_sampling, args.dirichlet_alpha)
                write_jsonl(seq, os.path.join(stage_dir, f"box_sequence_gen_{i}.json"))
                n_synth = sum(1 for box in seq if box["type"] == SYNTH_TYPE)
                if i == 0:
                    print(f"     예시 gen_0: 합성 {n_synth}개({n_synth/len(seq):.0%})")
        print("\n[참고] 온라인 학습에서는 파일을 미리 만들지 말고 매 에피소드마다")
        print("       curriculum_fixed_ratio(ep/N) 로 fixed_ratio 를 구해 즉석 생성하는 것을 권장.")
        return

    for i in range(args.n_sequences):
        rng = np.random.default_rng(master.integers(0, 2**31 - 1))
        seq = generate_sequence(rng, a, b, args.n_boxes, args.fixed_ratio,
                                args.noise_pct, args.type_sampling, args.dirichlet_alpha)
        path = os.path.join(args.out_dir, f"box_sequence_gen_{i}.json")
        write_jsonl(seq, path)

        # 요약 통계
        vols = np.array([np.prod(box["size"]) for box in seq])
        n_synth = sum(1 for box in seq if box["type"] == SYNTH_TYPE)
        total = vols.sum()
        print(f"[gen_{i}] 박스 {len(seq)}개 | 합성 {n_synth}개({n_synth/len(seq):.0%}) | "
              f"총부피 {total:.3f} m^3 (팔레트 {total/pallet_vol:.2f}배)")


if __name__ == "__main__":
    main()