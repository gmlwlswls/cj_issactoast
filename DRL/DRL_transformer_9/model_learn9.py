"""
model_learn_v2.py — O4M-SP 스타일(9번 논문 요약 기반) PPO 학습 스크립트 (EMS + Screening Heuristic)
==============================================================================
구성요소:
  1) Network: Transformer Feature Extractor (3층, hidden_dim=128)
             + Actor (아이템 feature × rotation feature → logits)
             + Critic (→ V(s))
  2) Environment: EMS 기반 공간 관리 + Screening heuristic + StabilityChecker
  3) PPO 학습: AdamW, cosine annealing(3e-5→1e-6), warmup 100,
              max_iter=10000, val_every=20, val_size=64, patience=100

핵심 아이디어(스크리닝 휴리스틱):
  - EMS 전체 공간을 X > Y > Z 기준으로 정렬해 stack 구성
  - stack top 공간을 꺼내고, 그 공간에서 버퍼 박스들(0/90도 회전 포함) 중
    "들어가고 + 안정성 통과"하는 후보가 있으면:
        -> 그 공간이 current space
        -> 통과한 후보들만 valid action (2B mask)
  - 후보가 없으면 그 공간은 pop(버림)하고 다음 공간 평가
  - stack이 빌 때까지 current space 못 찾으면 종료

주의:
  - 이 파일은 학습 스크립트(훈련용)이며, 온라인 제출용 algorithm.py와는 별개입니다.

변경 이력 (이번 수정):
  1) Cross-attention 방향을 논문대로 수정: Query=S_new(빈), Key/Value=S_valid(아이템).
     출력은 Query shape(n+2,d)를 유지하며, 이를 풀링해 상태 벡터 1개(d,)를 만든다.
  2) Actor를 논문대로 수정: 상태 벡터 1개(1,d) x 회전 임베딩(2k,d) 곱 -> MLP -> logits(2k).
     (기존처럼 아이템별 feature(k,d)를 그대로 쓰지 않음)
  3) NaN 방어 5종 추가/명시: adv·return clamp(-10,10), log_ratio clamp(-10,10),
     std clamp(min=1e-6), loss NaN/Inf 시 epoch 스킵, logits/value NaN/Inf 감지 시 스킵.
  4) torch.export/ONNX export 오류 수정: forward() 내 `if tensor.any(): ...` 형태의
     데이터-종속(data-dependent) 제어흐름이 GuardOnDataDependentSymNode 오류를 유발해
     torch.export(및 이를 사용하는 torch.onnx.export)가 실패했음. 조건문을 제거하고
     logits/value에 nan_to_num을 항상(unconditional) 적용하도록 수정 (원식과 동일한
     방어 효과, export 호환).
  5) 학습 하이퍼파라미터 조정: lr 시작값 3e-5(기존 1e-4), warmup 100iter(기존 20),
     val_size 64(기존 10000).
  - EMS 정렬은 X>Y>Z를 그대로 유지 (합의: 우선 이대로 학습해보고 결과를 본 뒤 재검토).
==============================================================================
"""
from __future__ import annotations

import os
import sys
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ems import EMSManager
from stability_checker import StabilityChecker


# ─────────────────────────────────────────────────────────────────────────────
# sequence_generator 사용 (동일 폴더에 존재한다고 가정)
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sequence_generator import (
    generate_sequence,
    fit_mass_law,
    SKU_CATALOG,
    curriculum_fixed_ratio,
)


SEED = 42


def set_seed(s: int = SEED) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ─────────────────────────────────────────────────────────────────────────────
# 정규화 상수 (기존 v2 컨셉 유지)
# ─────────────────────────────────────────────────────────────────────────────
POS_SCALE = 1.3
MASS_SCALE = 8.0


# ─────────────────────────────────────────────────────────────────────────────
# 1) NETWORK (Transformer 3-layer, hidden_dim=128)
# ─────────────────────────────────────────────────────────────────────────────
class ManualMHA(nn.Module):
    """nn.MultiheadAttention(batch_first=True) 래퍼 + key_padding_mask 지원"""

    def __init__(self, d: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d, num_heads=heads, dropout=dropout, batch_first=True)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        out, _ = self.mha(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = ManualMHA(d, heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_ratio * d),
            nn.GELU(),
            nn.Linear(mlp_ratio * d, d),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        y = self.ln1(x)
        x = x + self.attn(y, y, y, key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """
    bin(query) -> item(key,value)   (논문 방향: Query=S_new(빈), Key/Value=S_valid(아이템))
    Attention의 출력 shape는 Query를 따라가므로, 이 블록의 출력은 (B, n+2, d) 로
    bin(S_new)의 shape를 유지한다. item 쪽 k는 attention 내부에서 흡수되어 사라진다.
    """

    def __init__(self, d: int, heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.attn = ManualMHA(d, heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_ratio * d),
            nn.GELU(),
            nn.Linear(mlp_ratio * d, d),
        )

    def forward(self, bin_tokens: torch.Tensor, item_tokens: torch.Tensor, item_mask: Optional[torch.Tensor] = None):
        q = self.ln_q(bin_tokens)
        kv = self.ln_kv(item_tokens)
        bin_tokens = bin_tokens + self.attn(q, kv, kv, key_padding_mask=item_mask)
        bin_tokens = bin_tokens + self.mlp(self.ln2(bin_tokens))
        return bin_tokens


class O4MSPNet(nn.Module):
    """
    입력:
      - s_bin: (B, n, 7)
      - s_items: (B, k, 4)        k = buffer_B
      - rot_cands: (B, 2k, 3)     (l,w,h) for each (item,rot)
      - bin_mask: (B, n) bool True=padding
      - item_mask: (B, k) float, 1=valid/0=padding (cross-attn의 item key padding mask로 사용)

    구조 (논문 방향):
      S_bin (n,7) --embed--> (n,d) --self-attn--> S_new (n,d)
      S_valid (k,4) --embed--> (k,d) --self-attn--> (k,d)
      cross-attn: Query=S_new(n,d), Key/Value=S_valid(k,d) -> 출력 (n,d)  (Query shape 유지)
      pooled = mean-pool(n,d) -> (d,)                                    상태 벡터 1개
      Actor:  pooled(1,d) x rot_emb(2k,d) -> elementwise -> MLP -> logits(2k)
      Critic: pooled(1,d) -> MLP -> value(1)

    출력:
      - logits: (B, 2k)
      - value:  (B,)
    """

    def __init__(self, bin_dim: int = 7, item_dim: int = 4, rot_dim: int = 3,
                 d: int = 128, layers: int = 3, heads: int = 4):
        super().__init__()
        self.d = d
        self.layers = layers
        self.heads = heads

        self.emb_bin = nn.Linear(bin_dim, d)
        self.emb_item = nn.Linear(item_dim, d)
        self.emb_rot = nn.Linear(rot_dim, d)

        self.bin_blocks = nn.ModuleList([TransformerBlock(d, heads) for _ in range(layers)])
        self.item_self_blocks = nn.ModuleList([TransformerBlock(d, heads) for _ in range(layers)])
        self.cross_blocks = nn.ModuleList([CrossAttentionBlock(d, heads) for _ in range(layers)])

        # Actor: pooled_state(1,d) * rot_emb(2k,d) 융합 결과(2k,d) -> scalar logits(2k)
        self.actor_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

        # Critic: pooled bin state(1,d) -> scalar V(s)
        self.critic_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(self, s_bin, s_items, rot_cands, bin_mask=None, item_mask=None):
        hb = self.emb_bin(s_bin)          # (B, n, d)  -- S_bin
        hi = self.emb_item(s_items)       # (B, k, d)  -- S_valid

        for blk in self.bin_blocks:
            hb = blk(hb, key_padding_mask=bin_mask)              # S_new = self-attn(S_bin)

        for blk in self.item_self_blocks:
            hi = blk(hi, key_padding_mask=None)                  # self-attn(S_valid)

        # item padding mask (bool, True=padding) : item_mask는 1=valid/0=padding 이므로 반전
        item_pad_mask = (item_mask < 0.5) if item_mask is not None else None

        # Cross-attention (논문 방향): Query = S_new(빈), Key/Value = S_valid(아이템)
        # attention은 Query shape를 유지하므로 출력은 계속 (B, n, d) 이다.
        for blk in self.cross_blocks:
            hb = blk(hb, hi, item_mask=item_pad_mask)

        # 풀링: (B, n, d) -> 상태 벡터 1개 (B, d).  bin_mask=True는 padding이므로 제외.
        if bin_mask is not None:
            valid = (~bin_mask).unsqueeze(-1).float()            # (B, n, 1)
            summed = (hb * valid).sum(dim=1)
            denom = valid.sum(dim=1).clamp(min=1.0)
            pooled = summed / denom                              # (B, d)
        else:
            pooled = hb.mean(dim=1)

        # Actor: 상태벡터(1,d) x 회전 임베딩(2k,d) -> element-wise(브로드캐스트) -> MLP -> 스칼라(2k)
        hrot = self.emb_rot(rot_cands)                           # (B, 2k, d)
        fused = pooled.unsqueeze(1) * hrot                       # (B, 2k, d)
        logits = self.actor_mlp(fused).squeeze(-1)               # (B, 2k)

        # Critic: 상태벡터(1,d) -> V(s)
        value = self.critic_mlp(pooled).squeeze(-1)              # (B,)

        # NaN 방어 #5: 네트워크 출력(logits/value) 자체가 오염된 경우 즉시 정리
        # (torch.export/onnx export 호환) `if tensor.any(): ...` 형태의 데이터-종속
        # 제어흐름은 torch.export에서 GuardOnDataDependentSymNode 오류를 유발하므로,
        # 조건문 없이 항상 nan_to_num을 적용한다. NaN/Inf가 없으면 값이 그대로
        # 보존되므로 원래의 방어 로직(원식)과 동일하게 동작한다.
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        return logits, value


def safe_categorical(logits: torch.Tensor, vmask: torch.Tensor):
    """valid mask를 반영한 Categorical 분포 (NaN 방어 포함)"""
    neg = torch.finfo(logits.dtype).min
    masked = torch.where(vmask > 0, logits, torch.full_like(logits, neg))

    zero_rows = (vmask.sum(dim=-1) == 0)
    if zero_rows.any():
        masked[zero_rows] = 0.0

    if torch.isnan(masked).any():
        masked = torch.nan_to_num(masked, nan=0.0)

    probs = F.softmax(masked, dim=-1).clamp(min=1e-8)
    return torch.distributions.Categorical(probs=probs, validate_args=False)


# ─────────────────────────────────────────────────────────────────────────────
# 2) ENVIRONMENT (EMS + Screening heuristic + StabilityChecker)
# ─────────────────────────────────────────────────────────────────────────────
Space = Tuple[float, float, float, float, float, float]


@dataclass
class CurrentDecision:
    space: Space
    valid_pairs: List[Tuple[int, int]]  # (buffer_index, rot)


class PackingEnv:
    def __init__(
        self,
        L: float = 1.2,
        W: float = 1.0,
        H: float = 1.25,
        cell: float = 0.01,
        buffer_B: int = 1,
        n_boxes: int = 250,
        r_s: float = 0.75,           # 강화: 0.66 → 0.75
        r_w: float = 2.0,            # 강화: 3.0 → 2.0
        min_corners: int = 3,        # 강화: 네 모서리 중 3개 이상 지지
        alpha_lr: float = 1.0,
        alpha_hd: float = 0.5,
        max_placed: int = 80,
    ):
        self.L, self.W, self.H, self.cell = float(L), float(W), float(H), float(cell)
        self.Lc = int(round(self.L / self.cell))
        self.Wc = int(round(self.W / self.cell))
        self.Hc = int(round(self.H / self.cell))

        self.buffer_B = int(buffer_B)
        self.n_boxes = int(n_boxes)

        self.alpha_lr = float(alpha_lr)
        self.alpha_hd = float(alpha_hd)
        self.max_placed = int(max_placed)

        self.ems = EMSManager(self.L, self.W, self.H)
        self.stab = StabilityChecker(self.Lc, self.Wc, self.Hc, self.cell,
                                     r_s=r_s, r_w=r_w, min_corners=min_corners)

        self.mass_a, self.mass_b = fit_mass_law(SKU_CATALOG)

        self.seq = []
        self.ptr = 0
        self.buffer = []
        self.placed = []
        self.placed_vol = 0.0
        self.prev_util = 0.0
        self.prev_improvement = 0.0

        self._cached_decision: Optional[CurrentDecision] = None

    def reset(self, rng: np.random.Generator, fixed_ratio: float = 0.5, noise_pct: float = 0.10):
        self.seq = generate_sequence(
            rng, self.mass_a, self.mass_b, self.n_boxes,
            fixed_ratio, noise_pct, "uniform", 2.0
        )
        self.ptr = 0
        self.buffer = []
        self._fill()

        self.ems.reset()
        self.stab.reset()

        self.placed = []
        self.placed_vol = 0.0
        self.prev_util = 0.0
        self.prev_improvement = 0.0
        self._cached_decision = None

        return self._obs()

    def _fill(self):
        while len(self.buffer) < self.buffer_B and self.ptr < len(self.seq):
            self.buffer.append(self.seq[self.ptr])
            self.ptr += 1

    @staticmethod
    def _sort_xyz(sp: Space):
        # Z-first 정렬: 낮은 층 → 왼쪽(x) → 앞쪽(y)
        # 큰 팔레트에서 X-first 는 (0,0) 위로만 탑을 쌓는 편향이 있어
        # 바닥층을 먼저 채우도록 Z 우선으로 변경.
        return (sp[2], sp[0], sp[1])

    def _screen_current_space(self) -> Optional[CurrentDecision]:
        spaces = list(self.ems.spaces)
        if not spaces:
            return None

        # 1) Z > X > Y 로 정렬 (낮은 층 먼저)
        spaces_sorted = sorted(spaces, key=self._sort_xyz)

        # 2) stack push
        stack: List[Space] = list(spaces_sorted)

        # 3) top 공간부터 pop 하며 valid 후보가 생기면 채택
        while stack:
            sp = stack.pop(0)  # 가장 작은 x 를 우선 (스택을 리스트로 처리)
            valid: List[Tuple[int, int]] = []

            for bi, box in enumerate(self.buffer):
                l, w, h = box["size"]
                mass = float(box["mass"])
                for rot in (0, 1):
                    rl, rw = (l, w) if rot == 0 else (w, l)
                    if not self.ems.can_fit(sp, (rl, rw, h)):
                        continue
                    if self.stab.check(sp[0], sp[1], (rl, rw, h), mass):
                        valid.append((bi, rot))

            if valid:
                return CurrentDecision(space=sp, valid_pairs=valid)

        return None

    def _obs(self):
        decision = self._screen_current_space()
        self._cached_decision = decision

        # S_bin: (n+2, 7)
        rows: List[List[float]] = []

        # 1행: bin 속성 (L,W,H,0,0,0,0)
        rows.append([self.L, self.W, self.H, 0.0, 0.0, 0.0, 0.0])

        # 2행: current space 속성
        if decision is not None:
            x1, y1, z1, x2, y2, z2 = decision.space
            rows.append([x1, y1, z1, x2 - x1, y2 - y1, z2 - z1, 0.0])
        else:
            rows.append([0.0] * 7)

        # 3행~: placed item
        for p in self.placed[-self.max_placed:]:
            rows.append([p["x"], p["y"], p["z"], p["l"], p["w"], p["h"], p["mass"]])

        s_bin = np.array(rows, np.float32)
        s_bin[:, 0:3] /= POS_SCALE
        s_bin[:, 3:6] /= POS_SCALE
        s_bin[:, 6] /= MASS_SCALE

        # S_items: (B,4)
        items = []
        for b in self.buffer[:self.buffer_B]:
            items.append([b["size"][0], b["size"][1], b["size"][2], float(b["mass"])])
        n_real = len(items)
        while len(items) < self.buffer_B:
            items.append([0.0, 0.0, 0.0, 0.0])
        s_items = np.array(items[:self.buffer_B], np.float32)
        s_items[:, 0:3] /= POS_SCALE
        s_items[:, 3] /= MASS_SCALE

        # rotation candidates: (2B,3) (회전된 l,w,h만)
        rc = []
        for i in range(self.buffer_B):
            l, w, h, _m = s_items[i]
            rc.append([l, w, h])  # rot=0
            rc.append([w, l, h])  # rot=90
        rc = np.array(rc, np.float32)

        # valid mask: (2B,)
        vm = np.zeros((2 * self.buffer_B,), np.float32)
        if decision is not None:
            for (bi, rot) in decision.valid_pairs:
                vm[2 * bi + rot] = 1.0

        # item mask (critic pooling용)
        item_mask = np.array(
            [1.0] * min(n_real, self.buffer_B) + [0.0] * max(0, self.buffer_B - n_real),
            np.float32,
        )

        return s_bin, s_items, rc, vm, item_mask, decision.space if decision else None

    def step(self, action: int):
        bi = int(action // 2)
        rot = int(action % 2)

        decision = self._cached_decision
        if decision is None:
            obs = self._obs()
            return obs, 0.0, True, {"util": self.get_util(), "reason": "no_current_space"}

        if bi >= len(self.buffer):
            obs = self._obs()
            return obs, -1.0, True, {"util": self.get_util(), "reason": "bad_action_index"}

        if (bi, rot) not in decision.valid_pairs:
            obs = self._obs()
            return obs, -1.0, True, {"util": self.get_util(), "reason": "invalid_action_for_space"}

        box = self.buffer[bi]
        l, w, h = box["size"]
        mass = float(box["mass"])
        if rot == 1:
            l, w = w, l

        sp = decision.space
        sx, sy = sp[0], sp[1]

        # 안정성 체크는 이미 통과했다고 가정. place 후 base(cell)를 받는다.
        pr = self.stab.place(sx, sy, (l, w, h), mass)
        bz = pr.base * self.cell


        # EMS 갱신
        self.ems.update((sx, sy, bz), (l, w, h))

        vol = float(box["size"][0]) * float(box["size"][1]) * float(box["size"][2])
        self.placed.append(
            {
                "x": float(sx), "y": float(sy), "z": float(bz),
                "l": float(l), "w": float(w), "h": float(h),
                "mass": float(mass),
                "step": int(box["step"]), "id": int(box["id"]),
                "rotation": 90 if rot else 0,
            }
        )
        self.placed_vol += vol

        # buffer pop + fill
        self.buffer.pop(bi)
        self._fill()

        # 보상: r = alpha_lr * r_LR + alpha_hd * r_HD
        pallet_vol = self.L * self.W * self.H
        cur_util_abs = self.placed_vol / pallet_vol
        r_lr = cur_util_abs - self.prev_util

        h_std = self.stab.get_height_std() / max(1, self.Hc)
        cur_imp = -h_std
        r_hd = cur_imp - self.prev_improvement

        reward = float(np.clip(self.alpha_lr * r_lr + self.alpha_hd * r_hd, -1.0, 1.0))
        self.prev_util = cur_util_abs
        self.prev_improvement = cur_imp

        obs = self._obs()
        done = (obs[3].sum() == 0.0) or (len(self.buffer) == 0 and self.ptr >= len(self.seq))
        return obs, reward, bool(done), {"util": self.get_util(), "n": len(self.placed)}

    def get_util(self) -> float:
        return float(self.placed_vol / (self.L * self.W * self.H))


# ─────────────────────────────────────────────────────────────────────────────
# 3) PPO TRAINING
# ─────────────────────────────────────────────────────────────────────────────
class GAEBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.sb, self.si, self.rc, self.vm, self.im = [], [], [], [], []
        self.acts, self.lps, self.vals, self.rews, self.dns = [], [], [], [], []

    def store(self, sb, si, rc, vm, im, a, lp, v, r, d):
        self.sb.append(sb)
        self.si.append(si)
        self.rc.append(rc)
        self.vm.append(vm)
        self.im.append(im)
        self.acts.append(a)
        self.lps.append(lp)
        self.vals.append(v)
        self.rews.append(r)
        self.dns.append(d)

    def compute(self, gamma: float, lam: float):
        T = len(self.rews)
        if T == 0:
            return None

        vs = self.vals + [0.0]
        adv = np.zeros((T,), np.float32)
        g = 0.0
        for t in reversed(range(T)):
            delta = self.rews[t] + gamma * vs[t + 1] * (1.0 - self.dns[t]) - vs[t]
            g = delta + gamma * lam * (1.0 - self.dns[t]) * g
            adv[t] = g
        ret = adv + np.array(self.vals[:T], np.float32)

        return dict(
            sb=self.sb,
            si=self.si,
            rc=self.rc,
            vm=self.vm,
            im=self.im,
            acts=np.array(self.acts, dtype=np.int64),
            lps=np.array(self.lps, dtype=np.float32),
            adv=adv,
            ret=ret,
        )


def pad_to_batch(arrays: List[np.ndarray]):
    """
    s_bin 은 (n,7)에서 n이 가변이므로 padding + padding mask 반환
    반환:
      - padded: (B, max_n, 7)
      - mask: (B, max_n) bool True=padding
    """
    mx = max(a.shape[0] for a in arrays)
    d = arrays[0].shape[1]
    out = np.zeros((len(arrays), mx, d), np.float32)
    mask = np.ones((len(arrays), mx), dtype=bool)
    for i, a in enumerate(arrays):
        n = a.shape[0]
        out[i, :n] = a
        mask[i, :n] = False
    return out, mask


def export_meta(save_root: str, buffer_B: int):
    meta_path = os.path.join(save_root, f"model_meta_B{buffer_B}.json")
    meta = {
        "onnx_path": f"O4MSP_B{buffer_B}.onnx",
        "N": int(buffer_B),
        "pos_scale": float(POS_SCALE),
        "mass_scale": float(MASS_SCALE),
        "Lc": 120,
        "Wc": 100,
        "Hc": 125,
        "cell": 0.01,
        "r_s": 0.75,             # 강화 조건
        "r_w": 2.0,              # 강화 조건
        "min_corners": 3,        # 강화 조건 (네 모서리 중 3개 이상)
        "sort_order": "z_first", # EMS 정렬: z → x → y
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"   [META] {meta_path}")


def export_onnx(net: nn.Module, buffer_B: int, onnx_path: str, meta_path: str, device: str):
    try:
        net.eval()
        # dummy inputs (dynamic axes: n for s_bin)
        dummy_sb = torch.zeros(1, 4, 7, device=device)             # n=4 임시
        dummy_si = torch.zeros(1, buffer_B, 4, device=device)
        dummy_rc = torch.zeros(1, 2 * buffer_B, 3, device=device)
        dummy_bm = torch.zeros(1, 4, dtype=torch.bool, device=device)
        dummy_im = torch.ones(1, buffer_B, device=device)

        torch.onnx.export(
            net,
            (dummy_sb, dummy_si, dummy_rc, dummy_bm, dummy_im),
            onnx_path,
            input_names=["s_bin", "s_items", "rot_cands", "bin_mask", "item_mask"],
            output_names=["logits", "value"],
            dynamic_axes={
                "s_bin": {1: "n"},
                "bin_mask": {1: "n"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        export_meta(os.path.dirname(meta_path), buffer_B)
    except Exception as e:
        print(f"   [WARN] ONNX export skipped: {e}")
    finally:
        net.train()


@torch.no_grad()
def validate(net: nn.Module, buffer_B: int, device: str, n_val: int, n_boxes: int):
    env = PackingEnv(buffer_B=buffer_B, n_boxes=n_boxes)
    rng = np.random.default_rng(SEED)

    utils = []
    for _ in range(int(n_val)):
        obs = env.reset(rng, fixed_ratio=1.0)
        done = False
        while not done:
            sb, si, rc, vm, im, _ = obs
            if vm.sum() == 0.0:
                break
            t_sb = torch.tensor(sb, dtype=torch.float32, device=device).unsqueeze(0)
            t_si = torch.tensor(si, dtype=torch.float32, device=device).unsqueeze(0)
            t_rc = torch.tensor(rc, dtype=torch.float32, device=device).unsqueeze(0)
            t_vm = torch.tensor(vm, dtype=torch.float32, device=device).unsqueeze(0)
            t_im = torch.tensor(im, dtype=torch.float32, device=device).unsqueeze(0)

            t_bin_mask = torch.zeros(1, sb.shape[0], dtype=torch.bool, device=device)
            logits, _ = net(t_sb, t_si, t_rc, bin_mask=t_bin_mask, item_mask=t_im)

            # NaN 방어: validation 중 오염 감지 시 해당 episode만 조기 종료
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("[WARN][validate] logits에서 NaN/Inf 감지 → episode 중단")
                break

            dist = safe_categorical(logits, t_vm)
            a = int(dist.probs.argmax(dim=-1).item())
            obs, _r, done, _info = env.step(a)

        utils.append(env.get_util())

    return float(np.mean(utils))


def train(
    buffer_B: int,
    max_iter: int = 10000,
    save_root: str = "./best_model_o4msp",
    fixed_ratio: float = 0.5,
    n_boxes: int = 250,
    # optim/schedule
    lr: float = 3e-5,
    min_lr: float = 1e-6,
    warmup: int = 100,
    # ppo
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    ppo_epochs: int = 4,
    # validate/early stop
    val_every: int = 20,
    val_size: int = 64,
    patience: int = 100,
):
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] B={buffer_B}, device={device}, max_iter={max_iter}")

    net = O4MSPNet(bin_dim=7, item_dim=4, rot_dim=3, d=128, layers=3, heads=4).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    def lr_lambda(it: int):
        if it < warmup:
            return float(it) / float(max(1, warmup))
        progress = float(it - warmup) / float(max(1, max_iter - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(float(min_lr / lr), float(cosine))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    os.makedirs(save_root, exist_ok=True)
    pt_path = os.path.join(save_root, f"O4MSP_B{buffer_B}.pt")
    onnx_path = os.path.join(save_root, f"O4MSP_B{buffer_B}.onnx")
    meta_path = os.path.join(save_root, f"model_meta_B{buffer_B}.json")

    env = PackingEnv(buffer_B=buffer_B, n_boxes=n_boxes)
    rng = np.random.default_rng(SEED)

    best_val = -1.0
    no_improve = 0

    for it in range(int(max_iter)):
        fr = curriculum_fixed_ratio(it / max_iter, start=fixed_ratio, end=0.9, warmup=0.1)

        obs = env.reset(rng, fixed_ratio=fr)
        buf = GAEBuffer()
        done = False

        # 한 iteration = 한 에피소드 rollout (기존 v2 구조 유지)
        while not done:
            sb, si, rc, vm, im, _ = obs
            if vm.sum() == 0.0:
                break

            t_sb = torch.tensor(sb, dtype=torch.float32, device=device).unsqueeze(0)
            t_si = torch.tensor(si, dtype=torch.float32, device=device).unsqueeze(0)
            t_rc = torch.tensor(rc, dtype=torch.float32, device=device).unsqueeze(0)
            t_vm = torch.tensor(vm, dtype=torch.float32, device=device).unsqueeze(0)
            t_im = torch.tensor(im, dtype=torch.float32, device=device).unsqueeze(0)

            t_bin_mask = torch.zeros(1, sb.shape[0], dtype=torch.bool, device=device)
            raw_logits, value = net(t_sb, t_si, t_rc, bin_mask=t_bin_mask, item_mask=t_im)

            # NaN 방어: rollout 중 네트워크 출력이 오염되면 이 episode는 즉시 중단(스킵)
            if torch.isnan(raw_logits).any() or torch.isinf(raw_logits).any() or \
               torch.isnan(value).any() or torch.isinf(value).any():
                print(f"[WARN][iter {it}] rollout logits/value에서 NaN/Inf 감지 → episode 중단")
                break

            dist = safe_categorical(raw_logits, t_vm)

            a = dist.sample()
            lp = dist.log_prob(a)

            obs2, reward, done, _info = env.step(int(a.item()))
            buf.store(sb, si, rc, vm, im, int(a.item()), float(lp.item()), float(value.item()), float(reward), float(done))
            obs = obs2

        data = buf.compute(gamma=gamma, lam=lam)
        if data is None:
            scheduler.step()
            continue

        # PPO update batch 구성
        padded_sb, bin_mask_np = pad_to_batch(data["sb"])
        b_sb = torch.tensor(padded_sb, dtype=torch.float32, device=device)
        b_bin_mask = torch.tensor(bin_mask_np, dtype=torch.bool, device=device)

        b_si = torch.tensor(np.stack(data["si"]), dtype=torch.float32, device=device)
        b_rc = torch.tensor(np.stack(data["rc"]), dtype=torch.float32, device=device)
        b_vm = torch.tensor(np.stack(data["vm"]), dtype=torch.float32, device=device)
        b_im = torch.tensor(np.stack(data["im"]), dtype=torch.float32, device=device)

        b_act = torch.tensor(data["acts"], dtype=torch.long, device=device)
        b_old_lp = torch.tensor(data["lps"], dtype=torch.float32, device=device)
        b_adv = torch.tensor(data["adv"], dtype=torch.float32, device=device)
        b_ret = torch.tensor(data["ret"], dtype=torch.float32, device=device)

        # NaN 방어 #1: advantage/return을 [-10, 10]으로 clamp (극단값 잘라내기)
        b_adv = torch.nan_to_num(b_adv, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        b_ret = torch.nan_to_num(b_ret, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        # NaN 방어 #3: std를 최소 1e-6으로 clamp (0 나눗셈 방지)
        b_adv = (b_adv - b_adv.mean()) / b_adv.std().clamp(min=1e-6)

        last_loss = None

        for _ in range(int(ppo_epochs)):
            logits, values = net(b_sb, b_si, b_rc, bin_mask=b_bin_mask, item_mask=b_im)

            # NaN 방어 #5: 네트워크 raw 출력(logits) 오염 감지 → 이번 epoch 스킵
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"[WARN][iter {it+1}] PPO logits에서 NaN/Inf 감지 → epoch 스킵")
                break

            dist = safe_categorical(logits, b_vm)

            nlp = dist.log_prob(b_act)
            ent = dist.entropy()

            # NaN 방어 #2: log_ratio clamp(-10, 10) → ratio(exp) 폭발 방지
            log_ratio = (nlp - b_old_lp).clamp(-10.0, 10.0)
            ratio = log_ratio.exp()

            s1 = ratio * b_adv
            s2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * b_adv
            loss_actor = -torch.min(s1, s2).mean()

            loss_critic = F.mse_loss(values, b_ret)
            loss = loss_actor + value_coef * loss_critic - entropy_coef * ent.mean()

            # NaN 방어 #4: loss가 NaN/Inf면 그 epoch 스킵(오염된 gradient로 업데이트하지 않음)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN][iter {it+1}] loss NaN/Inf 감지 → epoch 스킵")
                break

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()

            last_loss = float(loss.item())

        scheduler.step()
        buf.clear()

        if (it + 1) % int(val_every) == 0:
            util = validate(net, buffer_B, device, n_val=val_size, n_boxes=n_boxes)
            bonus = max(0, 20 - int(buffer_B))
            total = util * 100.0 + float(bonus)
            cur_lr = float(opt.param_groups[0]["lr"])

            loss_str = f"{last_loss:.3f}" if last_loss is not None else "nan"
            print(
                f"[iter {it+1}/{max_iter}] B={buffer_B} fr={fr:.2f} lr={cur_lr:.2e} "
                f"loss={loss_str} util={util:.4f} total={total:.2f} best={max(best_val, 0.0):.2f}"
            )

            if total > best_val:
                best_val = total
                no_improve = 0
                torch.save(
                    {
                        "state_dict": net.state_dict(),
                        "buffer_B": int(buffer_B),
                        "d_model": 128,
                        "n_layers": 3,
                        "n_heads": 4,
                        "util": float(util),
                        "total_score": float(total),
                    },
                    pt_path,
                )
                export_onnx(net, buffer_B, onnx_path, meta_path, device)
                print(f"   ↳ best 갱신 (total={total:.2f})")
            else:
                no_improve += 1
                if no_improve >= int(patience):
                    print(f"[early stopping] {patience}회 개선 없음 → 종료 (best={best_val:.2f})")
                    break

    print(f"학습 완료. best={best_val:.2f} (B={buffer_B}), 모델={pt_path}")
    return best_val


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def _parse_buffers(spec: str):
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--buffer_B", type=int, default=1)
    p.add_argument("--buffers", type=str, default=None, help="버퍼 sweep: '1,5,10' 또는 '1-20'")

    p.add_argument("--fixed_ratio", type=float, default=0.5)
    p.add_argument("--n_boxes", type=int, default=250)

    p.add_argument("--max_iter", type=int, default=10000)
    p.add_argument("--val_every", type=int, default=20)
    p.add_argument("--val_size", type=int, default=64)
    p.add_argument("--patience", type=int, default=100)

    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=100)

    p.add_argument("--save_root", default="./best_model_o4msp")
    args = p.parse_args()

    if args.buffers:
        bs = _parse_buffers(args.buffers)
        print(f"[SWEEP] 버퍼 순회: {bs}\n")
        res = {}
        for B in bs:
            print(f"\n{'='*60}\n[SWEEP] B={B}\n{'='*60}")
            res[B] = train(
                buffer_B=B,
                max_iter=args.max_iter,
                save_root=args.save_root,
                fixed_ratio=args.fixed_ratio,
                n_boxes=args.n_boxes,
                lr=args.lr,
                min_lr=args.min_lr,
                warmup=args.warmup,
                val_every=args.val_every,
                val_size=args.val_size,
                patience=args.patience,
            )
        best_B = max(res, key=res.get)
        print(f"\n>>> 최적 B={best_B} (total={res[best_B]:.2f})")
        print(f">>> 모델: {args.save_root}/O4MSP_B{best_B}.onnx")
    else:
        train(
            buffer_B=args.buffer_B,
            max_iter=args.max_iter,
            save_root=args.save_root,
            fixed_ratio=args.fixed_ratio,
            n_boxes=args.n_boxes,
            lr=args.lr,
            min_lr=args.min_lr,
            warmup=args.warmup,
            val_every=args.val_every,
            val_size=args.val_size,
            patience=args.patience,
        )