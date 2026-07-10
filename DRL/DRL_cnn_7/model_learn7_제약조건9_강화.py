"""
model_learn.py
python model_learn.py --buffers 1,10,20   
==============================================================================
팔레타이징 온라인 3D-BPP 용 DRL(Actor-Critic CNN) 학습 스크립트.

기반: Zhang & Shuai(2024) "Online 3D-BPP: A DRL Algorithm with the Buffer Zone"(7번 논문)
      + Zhao et al.(2021) 의 height-map / feasibility-mask / 0·90도 회전(3번 논문).

7번 논문 대비 우리가 바꾼 점(명시):
  (A) 무게(mass) 고려:
        - feasibility mask 에 heavy-on-light 규칙 추가(9번 논문 기준: mass <= 3 × 지지질량).
        - 무거운 박스를 낮게 두도록 하는 작은 CoG 소프트 보상항 추가(논문엔 없음).
  (B) 후보 선택 방식:
        - 논문은 "버퍼 최선참 1개 + 컨베이어 1개 중 랜덤 선택"이지만,
          여기서는 "버퍼 내 모든 박스 + 컨베이어 1개"를 전부 상태로 입력하고,
          네트워크가 (어느 박스 / 어느 회전 / 어디에) 를 한 번에 출력해 최선을 고른다.
  (C) feasibility mask 조건 변경 (9번 논문 반영):
        - 조건3 지지 기하: 60/80/95% + 모서리 규칙 →
          접촉 셀 수 >= 0.66 * footprint 면적 (바닥이면 무조건 통과).
        - 조건4 CoM 마진: 제거.
        - 조건5 heavy-on-light: mass <= k_hol*지지질량 → mass <= 3.0 * 지지질량.

상태(state) 채널 구성 (Lc x Wc 격자, 1cm 단위):
    ch 0            : height map (현재 적재 높이, 셀 단위)
    ch 1            : top-mass map (각 셀 최상단 박스 질량) → heavy-on-light 학습용
    ch 2 .. 2+4N-1  : 후보 N(=B+1)개 각각의 (l, w, h, mass) 를 격자에 펼친 4채널씩
  → 총 채널 수 = 2 + 4 * (B + 1)
  → 후보는 슬롯 순서 고정(슬롯0=가장 오래된 버퍼 … 슬롯B=신착), 빈 슬롯은 0 패딩.

행동(action): (후보 N) x (회전 2) x (위치 Lc*Wc) 중 하나 → 박스·회전·위치 동시 결정.

저장:
  - 최고 검증 성능 모델을 ./DRL_cnn_model 폴더에 저장(state_dict + ONNX).
  - 성능 수렴/early stopping 기준으로 학습 종료.
  - 재현성을 위해 random_state=42 고정.
==============================================================================
"""
import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 같은 폴더의 생성기 재사용 (5 SKU + 합성 박스, mass=a*vol^b)
from sequence_generator import (
    SKU_CATALOG, fit_mass_law, generate_sequence, curriculum_fixed_ratio, SYNTH_TYPE
)

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────
#  설정
# ──────────────────────────────────────────────────────────────────────────
class Config:
    # 팔레트 (m).  length(X)=1.2, width(Y)=1.0, height(Z)=1.25
    pallet_l, pallet_w, pallet_h = 1.2, 1.0, 1.25
    cell = 0.01                       # 1cm 격자
    # 격자 칸 수
    Lc = int(round(pallet_l / cell))  # 120
    Wc = int(round(pallet_w / cell))  # 100
    Hc = int(round(pallet_h / cell))  # 125

    buffer_B = 4                      # 버퍼 용량 (튜닝 대상; 후보 수 = B+1)
    n_boxes = 250                     # 시퀀스 길이
    fixed_ratio = 0.5                 # 50% 고정 SKU / 50% 합성 (사용자 지정)
    noise_pct = 0.10                  # 합성 박스 질량 노이즈 ±10%

    # feasibility mask 파라미터 (9번 논문 기준으로 변경)
    # k_hol, com_alpha0, com_alpha1 제거 → 조건3/4 단순화
    max_mass = 6.0                    # 정규화용 최대 질량

    # 보상
    phi = 10.0                        # 논문 스텝보상 스케일
    w_cog = 0.3                       # CoG 소프트 보상 가중치(우리 확장)

    # 학습
    gamma = 0.99
    lr = 2.5e-4
    entropy_coef = 0.01
    value_coef = 0.5
    max_episodes = 20000
    val_every = 200                   # 검증 주기(에피소드)
    val_size = 32                     # 검증 시퀀스 수(고정)
    patience = 10                     # early stopping: 검증 개선 없는 횟수 한도
    save_root = "./best_model_by_buffer_7_제약조건9"   # 버퍼별 모델 파일을 모아둘 폴더
    model_path = "./best_model_by_buffer_7_제약조건9/DRL_cnn_model_B4.pt"   # B에 맞춰 설정됨
    onnx_path = "./best_model_by_buffer_7_제약조건9/DRL_cnn_model_B4.onnx"


# ──────────────────────────────────────────────────────────────────────────
#  환경
# ──────────────────────────────────────────────────────────────────────────
class PalletizingEnv:
    def __init__(self, cfg, mass_a, mass_b):
        self.cfg = cfg
        self.a, self.b = mass_a, mass_b
        self.N = cfg.buffer_B          # 후보 수 = 버퍼 크기 (공식 get_buffer() 와 동일)
        self.n_actions = self.N * 2 * cfg.Lc * cfg.Wc

    # ---- 시퀀스/상태 초기화 ------------------------------------------------
    def reset(self, rng, fixed_ratio):
        cfg = self.cfg
        self.seq = generate_sequence(rng, self.a, self.b, cfg.n_boxes,
                                     fixed_ratio, cfg.noise_pct, "uniform", 2.0)
        self.ptr = 0                              # 다음에 버퍼로 채워질 박스 index
        self.height = np.zeros((cfg.Lc, cfg.Wc), dtype=np.int32)   # 셀 높이
        self.topmass = np.zeros((cfg.Lc, cfg.Wc), dtype=np.float32) # 셀 최상단 질량
        self.placed_vol = 0.0
        # 버퍼를 B개로 채움 (공식 BufferManager._fill 과 동일 동작)
        self.buffer = []
        self._fill()
        self._refresh_candidates()
        return self._state()

    def _fill(self):
        """버퍼가 B개가 될 때까지 시퀀스에서 보충."""
        while len(self.buffer) < self.cfg.buffer_B and self.ptr < len(self.seq):
            self.buffer.append(self.seq[self.ptr]); self.ptr += 1

    def _refresh_candidates(self):
        """후보 = 현재 버퍼 스냅샷(B개). 부족하면 None 패딩."""
        cands = list(self.buffer)
        while len(cands) < self.N:
            cands.append(None)
        self.candidates = cands[:self.N]

    # ---- 셀 단위 footprint -------------------------------------------------
    def _cells(self, size, rot):
        l, w, h = size
        if rot == 1:           # 90도: l,w swap
            l, w = w, l
        import math
        return (max(1, math.ceil(l / self.cfg.cell)),
                max(1, math.ceil(w / self.cfg.cell)),
                max(1, math.ceil(h / self.cfg.cell)))

    # ---- feasibility mask: (N, 2, Lc, Wc) ---------------------------------
    def feasibility_mask(self):
        cfg = self.cfg
        M = np.zeros((self.N, 2, cfg.Lc, cfg.Wc), dtype=np.float32)
        for ci, box in enumerate(self.candidates):
            if box is None:
                continue
            for rot in (0, 1):
                M[ci, rot] = self._mask_one(box, rot)
        return M

    def _mask_one(self, box, rot):
        cfg = self.cfg
        lc, wc, hc = self._cells(box["size"], rot)
        Lc, Wc = cfg.Lc, cfg.Wc
        if lc > Lc or wc > Wc:
            return np.zeros((Lc, Wc), dtype=np.float32)

        H = self.height
        # (1) 윈도우 최대 높이 base_z : 분리형(1D) 슬라이딩 맥스
        from numpy.lib.stride_tricks import sliding_window_view
        win_r = sliding_window_view(H, lc, axis=0).max(axis=2)       # (Lc-lc+1, Wc)
        base = sliding_window_view(win_r, wc, axis=1).max(axis=2)    # (Lc-lc+1, Wc-wc+1)
        vr, vc = base.shape

        # 누적용 버퍼
        support_cnt = np.zeros((vr, vc), dtype=np.int32)
        corner_cnt = np.zeros((vr, vc), dtype=np.int32)              # 네 모서리 지지 카운트
        supp_mass_min = np.full((vr, vc), np.inf, dtype=np.float32)

        # 네 모서리 상대 좌표 (di, dj)
        corners = {(0, 0), (lc - 1, 0), (0, wc - 1), (lc - 1, wc - 1)}

        for di in range(lc):
            for dj in range(wc):
                sub = H[di:di + vr, dj:dj + vc]
                eq = (sub == base)                      # 이 셀이 base 높이에 닿아 지지하는가
                support_cnt += eq
                tm = self.topmass[di:di + vr, dj:dj + vc]
                supp_mass_min = np.where(eq, np.minimum(supp_mass_min, tm), supp_mass_min)
                if (di, dj) in corners:
                    corner_cnt += eq

        area = lc * wc

        # (2) 높이 초과: base + 박스높이 <= Hc
        overflow = (base + hc) > cfg.Hc

        # (3) 지지 기하 (강화 조건):
        #     면적 지지율 >= 75%  AND  네 모서리 중 3개 이상 지지
        #     바닥(base == 0)이면 완전 지지로 간주 → 무조건 통과.
        floor = (base == 0)
        geom = floor | ((support_cnt >= 0.75 * area) & (corner_cnt >= 3))

        # (4) CoM 마진 → 제거

        # (5) heavy-on-light (강화): 배치 박스 질량 <= 2 × 최소 지지 박스 질량
        safe_supp = np.where(np.isfinite(supp_mass_min), supp_mass_min, 0.0)
        hol_ok = floor | (box["mass"] <= 2.0 * safe_supp)

        ok = (~overflow) & geom & hol_ok
        full = np.zeros((Lc, Wc), dtype=np.float32)
        full[:vr, :vc] = ok.astype(np.float32)   # anchor 가 유효영역 내일 때만 1
        return full

    # ---- 상태 텐서 ---------------------------------------------------------
    def _state(self):
        cfg = self.cfg
        Lc, Wc = cfg.Lc, cfg.Wc
        chans = [self.height.astype(np.float32) / cfg.Hc,
                 self.topmass / cfg.max_mass]
        for box in self.candidates:
            if box is None:
                chans += [np.zeros((Lc, Wc), np.float32)] * 4
            else:
                l, w, h = box["size"]
                chans += [np.full((Lc, Wc), l, np.float32),
                          np.full((Lc, Wc), w, np.float32),
                          np.full((Lc, Wc), h, np.float32),
                          np.full((Lc, Wc), box["mass"] / cfg.max_mass, np.float32)]
        return np.stack(chans, axis=0)            # (C, Lc, Wc)

    # ---- 행동 디코드 & 적용 ------------------------------------------------
    def decode(self, action):
        cfg = self.cfg
        per_rot = cfg.Lc * cfg.Wc
        ci = action // (2 * per_rot)
        rem = action % (2 * per_rot)
        rot = rem // per_rot
        pos = rem % per_rot
        x, y = pos // cfg.Wc, pos % cfg.Wc
        return ci, rot, x, y

    def step(self, action):
        cfg = self.cfg
        ci, rot, x, y = self.decode(action)
        box = self.candidates[ci]
        lc, wc, hc = self._cells(box["size"], rot)

        region = self.height[x:x + lc, y:y + wc]
        base = int(region.max())
        old = region.copy()
        # 박스 적재: footprint 셀을 base+hc 로, 최상단 질량 갱신
        self.height[x:x + lc, y:y + wc] = base + hc
        self.topmass[x:x + lc, y:y + wc] = box["mass"]

        # 부피/낭비공간
        vol = box["size"][0] * box["size"][1] * box["size"][2]
        self.placed_vol += vol
        pallet_vol = cfg.pallet_l * cfg.pallet_w * cfg.pallet_h
        v_box = vol / pallet_vol
        # 낭비공간: 박스 밑면 아래 갇힌 공극 (base 보다 낮았던 셀들)
        v_waste = ((base - old).clip(min=0).sum() * (cfg.cell ** 3)) / pallet_vol
        # CoG 소프트 보상(우리 확장): 무거운 박스를 낮게 둘수록 +
        cog = cfg.w_cog * (box["mass"] / cfg.max_mass) * (1.0 - base / cfg.Hc)
        reward = cfg.phi * (v_box - v_waste) + cog

        # 후보 소비 & 보충 (모든 후보는 버퍼 박스 → pop 후 _fill 로 보충)
        if ci < len(self.buffer):
            self.buffer.pop(ci)
        self._fill()
        self._refresh_candidates()

        # 종료: 후보 전체가 어디에도 못 놓이면 끝
        mask = self.feasibility_mask()
        done = (mask.sum() == 0) or all(c is None for c in self.candidates)
        return self._state(), reward, done, mask

    def stacking_rate(self):
        pallet_vol = self.cfg.pallet_l * self.cfg.pallet_w * self.cfg.pallet_h
        return self.placed_vol / pallet_vol


# ──────────────────────────────────────────────────────────────────────────
#  네트워크 (State CNN + Actor + Critic)
# ──────────────────────────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, in_ch, n_cand, Lc, Wc):
        super().__init__()
        self.n_cand, self.Lc, self.Wc = n_cand, Lc, Wc
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        # Actor: 후보*회전 = n_cand*2 채널의 Lc x Wc 맵 → 위치별 logit
        self.actor = nn.Conv2d(64, n_cand * 2, 3, padding=1)
        # Critic: 전역 풀링 → 스칼라
        self.critic = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        f = self.backbone(x)
        logits = self.actor(f)                          # (B, n_cand*2, Lc, Wc)
        logits = logits.reshape(x.size(0), -1)          # (B, n_cand*2*Lc*Wc)
        value = self.critic(f).squeeze(-1)              # (B,)
        return logits, value


def masked_policy(logits, mask_flat):
    """feasibility mask(0/1)를 곱해 불가능한 행동을 차단한 확률분포."""
    neg = torch.finfo(logits.dtype).min
    masked = torch.where(mask_flat > 0, logits, torch.full_like(logits, neg))
    return F.softmax(masked, dim=-1)


# ──────────────────────────────────────────────────────────────────────────
#  검증 (고정 시퀀스에서 greedy 적재율)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(net, cfg, mass_a, mass_b, device):
    """검증: 실제 5 SKU 분포에서 greedy 적재율 + 과제 총점(적재율*100 + 버퍼 가산점)."""
    env = PalletizingEnv(cfg, mass_a, mass_b)
    rng = np.random.default_rng(SEED)        # 검증셋 고정
    rates = []
    for _ in range(cfg.val_size):
        s = env.reset(rng, fixed_ratio=1.0)  # 검증은 실제 분포(5 SKU)로
        mask = env.feasibility_mask()
        done = False
        while not done:
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            mf = torch.tensor(mask.reshape(-1), dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = net(st)
            probs = masked_policy(logits, mf)
            if probs.sum() == 0:
                break
            a = int(torch.argmax(probs, dim=-1).item())
            s, r, done, mask = env.step(a)
        rates.append(env.stacking_rate())
    util = float(np.mean(rates))
    buffer_bonus = max(0, 20 - cfg.buffer_B)
    total_score = util * 100.0 + buffer_bonus   # 과제 총점 기준(버퍼 B 반영)
    return util, total_score


# ──────────────────────────────────────────────────────────────────────────
#  학습 루프 (A2C 스타일 on-policy actor-critic)
# ──────────────────────────────────────────────────────────────────────────
def train(cfg):
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mass_a, mass_b = fit_mass_law(SKU_CATALOG)
    env = PalletizingEnv(cfg, mass_a, mass_b)
    in_ch = 2 + 4 * cfg.buffer_B      # height + topmass + 후보 B개 × (l,w,h,mass)
    net = ActorCritic(in_ch, env.N, cfg.Lc, cfg.Wc).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    os.makedirs(cfg.save_root, exist_ok=True)
    rng = np.random.default_rng(SEED)
    best_val, no_improve = -1.0, 0

    for ep in range(cfg.max_episodes):
        fr = curriculum_fixed_ratio(ep / cfg.max_episodes,
                                    start=cfg.fixed_ratio, end=0.9, warmup=0.1)
        s = env.reset(rng, fixed_ratio=fr)
        mask = env.feasibility_mask()
        logps, values, rewards, entropies = [], [], [], []
        done = False
        while not done:
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            mf = torch.tensor(mask.reshape(-1), dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = net(st)
            probs = masked_policy(logits, mf)
            if probs.sum() == 0:
                break
            dist = torch.distributions.Categorical(probs=probs)
            a = dist.sample()
            logps.append(dist.log_prob(a))
            entropies.append(dist.entropy())
            values.append(value)
            s, r, done, mask = env.step(int(a.item()))
            rewards.append(r)

        if not rewards:
            continue
        # 할인 누적보상 → advantage
        returns, R = [], 0.0
        for rr in reversed(rewards):
            R = rr + cfg.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        values = torch.cat(values)
        logps = torch.cat(logps)
        entropies = torch.cat(entropies)
        adv = returns - values.detach()

        actor_loss = -(logps * adv).mean()
        critic_loss = F.mse_loss(values, returns)
        ent = entropies.mean()
        loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * ent

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()

        # 검증 + best 저장 + early stopping (총점 기준)
        if (ep + 1) % cfg.val_every == 0:
            util, total_score = validate(net, cfg, mass_a, mass_b, device)
            print(f"[ep {ep+1}] B={cfg.buffer_B} fr={fr:.2f} loss={loss.item():.3f} "
                  f"util={util:.4f} total={total_score:.2f} best={max(best_val,0):.2f}")
            if total_score > best_val:
                best_val, no_improve = total_score, 0
                full_cfg = {k: getattr(Config, k) for k in dir(Config) if not k.startswith("_")}
                full_cfg.update(vars(cfg))
                torch.save({"state_dict": net.state_dict(), "cfg": full_cfg,
                            "util": util, "total_score": total_score,
                            "buffer_B": cfg.buffer_B, "in_ch": in_ch, "N": env.N},
                           cfg.model_path)
                _export_onnx(net, in_ch, cfg, device)
                print(f"   ↳ best 갱신 → {cfg.model_path} 저장 (util={util:.4f}, total={total_score:.2f})")
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    print(f"[early stopping] {cfg.patience}회 개선 없음 → 종료 (best total={best_val:.2f})")
                    break
    print(f"학습 종료. best total_score={best_val:.2f} (B={cfg.buffer_B}), 모델 파일={cfg.model_path}")
    return best_val


def _export_onnx(net, in_ch, cfg, device):
    """best 갱신 시 ONNX 로도 저장. 환경에 따라 실패할 수 있으므로 학습은 막지 않음."""
    try:
        net.eval()
        dummy = torch.zeros(1, in_ch, cfg.Lc, cfg.Wc, device=device)
        torch.onnx.export(net, dummy, cfg.onnx_path,
                          input_names=["state"], output_names=["logits", "value"],
                          dynamic_axes={"state": {0: "batch"}}, opset_version=17,
                          dynamo=False)
    except Exception as e:
        print(f"   [경고] ONNX export 건너뜀: {e}")
    finally:
        net.train()


def _make_cfg(args, buffer_B):
    cfg = Config()
    cfg.buffer_B = buffer_B
    cfg.fixed_ratio = args.fixed_ratio
    cfg.max_episodes = args.max_episodes
    cfg.val_every = args.val_every
    cfg.patience = args.patience
    cfg.save_root = args.save_root
    cfg.model_path = os.path.join(args.save_root, f"DRL_cnn_model_B{buffer_B}.pt")
    cfg.onnx_path = os.path.join(args.save_root, f"DRL_cnn_model_B{buffer_B}.onnx")
    return cfg


def _parse_buffers(spec):
    """'1,5,10,15,20' 또는 '1-20' 형태를 정수 리스트로 변환."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # 단일 B 학습
    p.add_argument("--buffer_B", type=int, default=Config.buffer_B,
                   help="단일 버퍼 크기로 학습 (sweep 미사용 시)")
    # 여러 B 순회 학습
    p.add_argument("--buffers", type=str, default=None,
                   help="여러 버퍼 크기를 순회 학습. 예: '1,5,10,15,20' 또는 '1-20'")
    p.add_argument("--fixed_ratio", type=float, default=Config.fixed_ratio)
    p.add_argument("--max_episodes", type=int, default=Config.max_episodes)
    p.add_argument("--save_root", default=Config.save_root,
                   help="버퍼별 모델 상위 폴더 (하위에 DRL_cnn_model_B{B} 로 저장)")
    p.add_argument("--val_every", type= int, default=Config.val_every,
                   help='검증 주기(episode)')
    p.add_argument('--patience', type= int, default= Config.patience,
                   help= 'early stopping patience(개선 없는 횟수)')
    args = p.parse_args()

    if args.buffers:
        # ── sweep: 여러 B 를 차례로 학습하고 총점 비교 ──────────────
        buffers = _parse_buffers(args.buffers)
        print(f"[SWEEP] 버퍼 크기 순회 학습: {buffers}\n")
        results = {}
        for B in buffers:
            print(f"\n{'='*60}\n[SWEEP] buffer_B = {B} 학습 시작\n{'='*60}")
            cfg = _make_cfg(args, B)
            best = train(cfg)
            results[B] = best
        # 요약
        print(f"\n{'='*60}\n[SWEEP 결과 요약] (총점 = 적재율*100 + (20-B))\n{'='*60}")
        for B in buffers:
            print(f"  B={B:>2} → best total_score = {results[B]:.2f}")
        best_B = max(results, key=results.get)
        print(f"\n>>> 최적 버퍼 크기 B = {best_B} (총점 {results[best_B]:.2f})")
        print(f">>> 해당 모델: {os.path.join(args.save_root, f'DRL_cnn_model_B{best_B}.pt')}")
    else:
        # ── 단일 B 학습 ────────────────────────────────────────────
        cfg = _make_cfg(args, args.buffer_B)
        train(cfg)
