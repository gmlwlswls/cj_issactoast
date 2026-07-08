"""
model_learn_v2.py — O4M-SP 기반 PPO 학습 (9번 논문 방식)
==============================================================================
구성요소:
  1) Network: Transformer Feature Extractor (3층, 128차원)
             + Actor (상태특징 × 회전임베딩 → softmax)
             + Critic (→ V(s) 스칼라)
  2) Environment: EMS 기반 공간 관리 + Stability Checker + 가중 보상
  3) PPO 학습: AdamW, cosine annealing(1e-4→1e-6), warmup 20,
              10000 iteration, 20마다 검증, patience=100

온라인 적용:
  - S_valid_items = 버퍼 B개 (전체 목록 대신, 수량 열 제거 → k×4)
  - 동적 shape (n+2 가변) — ONNX export 시 dynamic_axes

실행:
  python model_learn_v2.py --buffer_B 1
  python model_learn_v2.py --buffers 1,5,10   # 버퍼 sweep
==============================================================================
"""
import os, sys, random, argparse, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ems import EMSManager
from stability_checker import StabilityChecker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from sequence_generator import (generate_sequence, fit_mass_law,
                                    SKU_CATALOG, curriculum_fixed_ratio)
except ImportError:
    SKU_CATALOG = [
        {"type": 0, "type_name": "1", "size": [0.195, 0.178, 0.134], "mass": 0.5},
        {"type": 1, "type_name": "2", "size": [0.245, 0.178, 0.140], "mass": 1.0},
        {"type": 2, "type_name": "3", "size": [0.245, 0.220, 0.158], "mass": 2.0},
        {"type": 3, "type_name": "4", "size": [0.310, 0.233, 0.210], "mass": 4.0},
        {"type": 4, "type_name": "5", "size": [0.315, 0.272, 0.257], "mass": 6.0},
    ]
    def fit_mass_law(c):
        vol = np.array([np.prod(s["size"]) for s in c])
        mass = np.array([s["mass"] for s in c])
        b, la = np.polyfit(np.log(vol), np.log(mass), 1)
        return float(np.exp(la)), float(b)
    def generate_sequence(*a, **kw): return []
    def curriculum_fixed_ratio(p, start=0.5, end=0.9, warmup=0.1):
        p = max(0.0, min(1.0, p))
        return start if p <= warmup else start + (end-start)*(p-warmup)/(1-warmup)

SEED = 42
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ══════════════════════════════════════════════════════════════════════════
#  1. NETWORK
# ══════════════════════════════════════════════════════════════════════════

class ManualMHA(nn.Module):
    """ONNX 동적 shape 호환 MultiheadAttention (수동 구현)."""
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads; self.dk = d // heads
        self.wq = nn.Linear(d, d); self.wk = nn.Linear(d, d); self.wv = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)

    def forward(self, q, k, v):
        B = q.size(0)
        Q = self.wq(q).reshape(B, -1, self.h, self.dk).transpose(1, 2)  # (B,h,Sq,dk)
        K = self.wk(k).reshape(B, -1, self.h, self.dk).transpose(1, 2)  # (B,h,Sk,dk)
        V = self.wv(v).reshape(B, -1, self.h, self.dk).transpose(1, 2)  # (B,h,Sk,dk)
        sc = (Q @ K.transpose(-2, -1)) / (self.dk ** 0.5)               # (B,h,Sq,Sk)
        att = F.softmax(sc, dim=-1)
        out = (att @ V).transpose(1, 2).reshape(B, -1, self.h * self.dk) # (B,Sq,d)
        return self.wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, d=128, heads=4, ff=256):
        super().__init__()
        self.sa = ManualMHA(d, heads)
        self.n1 = nn.LayerNorm(d)
        self.ca = ManualMHA(d, heads)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.n3 = nn.LayerNorm(d)

    def forward(self, sb, si):
        x = self.sa(sb, sb, sb); sb = self.n1(sb + x)
        x = self.ca(sb, si, si); sb = self.n2(sb + x)
        x = self.ff(sb);         sb = self.n3(sb + x)
        return sb


class O4MSPNet(nn.Module):
    """Transformer Feature Extractor + Actor(×rot_embed) + Critic."""

    def __init__(self, bin_dim=7, item_dim=4, d=128, layers=3, heads=4):
        super().__init__()
        self.d = d
        self.emb_bin = nn.Linear(bin_dim, d)
        self.emb_item = nn.Linear(item_dim, d)
        self.blocks = nn.ModuleList([TransformerBlock(d, heads, d*2) for _ in range(layers)])
        self.actor_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.rot_emb = nn.Linear(5, d)          # (l',w',h',mass,rot_flag) → d
        self.critic_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, s_bin, s_items, rot_cands):
        """
        s_bin:      (B, n+2, 7)   — 동적 n
        s_items:    (B, k, 4)
        rot_cands:  (B, 2k, 5)
        Returns: logits (B, 2k), value (B,)
        """
        # 입력 정규화 (LayerNorm 안정성)
        sb = s_bin / s_bin.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        si = s_items / s_items.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        hb = self.emb_bin(sb)                    # (B, n+2, d)
        hi = self.emb_item(si)                   # (B, k, d)
        for blk in self.blocks:
            hb = blk(hb, hi)
        feat = hb.mean(dim=1)                    # (B, d)  — 평균 풀링
        # Actor: 상태특징 × 회전임베딩 → 스칼라 점수
        sf = self.actor_mlp(feat)                # (B, d)
        rc_norm = rot_cands / rot_cands.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        rf = self.rot_emb(rc_norm)               # (B, 2k, d)
        logits = (sf.unsqueeze(1) * rf).sum(-1)  # (B, 2k)
        # Critic
        value = self.critic_mlp(feat).squeeze(-1) # (B,)
        return logits, value


def safe_categorical(logits, vmask):
    """mask 적용 + NaN 방어 Categorical 분포."""
    neg = torch.finfo(logits.dtype).min
    m = torch.where(vmask > 0, logits, torch.full_like(logits, neg))
    if (vmask.sum(dim=-1) == 0).any():
        m[vmask.sum(dim=-1) == 0] = 0.0
    if torch.isnan(m).any():
        m = torch.nan_to_num(m, nan=0.0)
    p = F.softmax(m, dim=-1).clamp(min=1e-8)
    return torch.distributions.Categorical(probs=p, validate_args=False)


# ══════════════════════════════════════════════════════════════════════════
#  2. ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════

class PackingEnv:
    def __init__(self, L=1.2, W=1.0, H=1.25, cell=0.01,
                 buffer_B=1, n_boxes=250, r_s=0.66, r_w=3.0,
                 alpha_lr=1.0, alpha_hd=0.5, max_placed=80):
        self.L, self.W, self.H, self.cell = L, W, H, cell
        self.Lc = int(round(L/cell)); self.Wc = int(round(W/cell)); self.Hc = int(round(H/cell))
        self.buffer_B = buffer_B
        self.n_boxes = n_boxes
        self.alpha_lr, self.alpha_hd = alpha_lr, alpha_hd
        self.max_placed = max_placed
        self.ems = EMSManager(L, W, H)
        self.stab = StabilityChecker(self.Lc, self.Wc, self.Hc, cell, r_s, r_w)
        self.mass_a, self.mass_b = fit_mass_law(SKU_CATALOG)

    def reset(self, rng, fixed_ratio=0.5, noise_pct=0.10):
        self.seq = generate_sequence(rng, self.mass_a, self.mass_b, self.n_boxes,
                                     fixed_ratio, noise_pct, "uniform", 2.0)
        self.ptr = 0
        self.ems.reset(); self.stab.reset()
        self.placed = []
        self.placed_vol = 0.0
        self.prev_util = 0.0
        self.prev_improvement = 0.0
        self.buffer = []
        self._fill()
        return self._obs()

    def _fill(self):
        while len(self.buffer) < self.buffer_B and self.ptr < len(self.seq):
            self.buffer.append(self.seq[self.ptr]); self.ptr += 1

    def _find_current_space(self):
        """XYZ 정렬된 EMS 중 버퍼 박스가 들어가고 안정한 공간을 찾는다."""
        spaces = self.ems.get_sorted_spaces()
        for sp in spaces:
            for box in self.buffer[:self.buffer_B]:
                if box is None: continue
                l, w, h = box["size"]; mass = box["mass"]
                for rot in (0, 1):
                    rl, rw = (l, w) if rot == 0 else (w, l)
                    if self.ems.can_fit(sp, (rl, rw, h)):
                        if self.stab.check(sp[0], sp[1], (rl, rw, h), mass):
                            return sp
        return None

    def _obs(self):
        cur_space = self._find_current_space()
        # S_bin: (n+2, 7)
        rows = [[self.L, self.W, self.H, 0, 0, 0, 0]]
        if cur_space:
            si = self.ems.get_space_info(cur_space)
            rows.append(list(si) + [0])
        else:
            rows.append([0]*7)
        for p in self.placed[-self.max_placed:]:
            rows.append([p["x"],p["y"],p["z"],p["l"],p["w"],p["h"],p["mass"]])
        s_bin = np.array(rows, np.float32)

        # S_items: (B, 4)
        items = []
        for b in self.buffer[:self.buffer_B]:
            items.append([b["size"][0],b["size"][1],b["size"][2],b["mass"]] if b else [0]*4)
        while len(items) < self.buffer_B: items.append([0]*4)
        s_items = np.array(items[:self.buffer_B], np.float32)

        # rot_candidates: (2B, 5) + valid_mask: (2B,)
        rc, vm = [], []
        for i, box in enumerate(self.buffer[:self.buffer_B]):
            if box is None:
                rc += [[0]*5, [0]*5]; vm += [0, 0]; continue
            l, w, h, mass = *box["size"], box["mass"]
            for rot in (0, 1):
                rl, rw = (l,w) if rot==0 else (w,l)
                rc.append([rl, rw, h, mass, float(rot)])
                if cur_space and self.ems.can_fit(cur_space, (rl,rw,h)):
                    vm.append(1 if self.stab.check(cur_space[0],cur_space[1],(rl,rw,h),mass) else 0)
                else:
                    vm.append(0)
        while len(rc) < 2*self.buffer_B: rc.append([0]*5); vm.append(0)
        return (s_bin, s_items,
                np.array(rc[:2*self.buffer_B], np.float32),
                np.array(vm[:2*self.buffer_B], np.float32),
                cur_space)

    def step(self, action):
        bi = action // 2; rot = action % 2
        box = self.buffer[bi]
        l, w, h = box["size"]; mass = box["mass"]
        if rot == 1: l, w = w, l

        sp = self._find_current_space()
        if sp is None:
            return self._obs(), 0.0, True, {"util": self.placed_vol/(self.L*self.W*self.H)}
        sx, sy = sp[0], sp[1]
        xc, yc, base = self.stab.place(sx, sy, (l,w,h), mass)
        bz = base * self.cell
        self.ems.update((sx, sy, bz), (l, w, h))

        vol = box["size"][0]*box["size"][1]*box["size"][2]
        self.placed.append({"x":sx,"y":sy,"z":bz,"l":l,"w":w,"h":h,"mass":mass,
                            "step":box["step"],"id":box["id"],"rotation":90 if rot else 0})
        self.placed_vol += vol
        self.buffer.pop(bi); self._fill()

        # 보상 r = α₁·r_LR + α₂·r_HD
        H_N = max(1, self.stab.get_max_height())
        cur_util = self.placed_vol / (self.L * self.W * H_N * self.cell)
        r_lr = cur_util - self.prev_util

        h_std = self.stab.get_height_std()
        cur_imp = H_N - h_std
        r_hd = cur_imp - self.prev_improvement

        reward = self.alpha_lr * r_lr + self.alpha_hd * r_hd
        self.prev_util = cur_util
        self.prev_improvement = cur_imp

        obs = self._obs()
        done = (obs[3].sum() == 0) or len(self.buffer) == 0
        return obs, reward, done, {"util": self.placed_vol/(self.L*self.W*self.H), "n": len(self.placed)}

    def get_util(self):
        return self.placed_vol / (self.L * self.W * self.H)


# ══════════════════════════════════════════════════════════════════════════
#  3. PPO TRAINING
# ══════════════════════════════════════════════════════════════════════════

class GAEBuffer:
    def __init__(self):
        self.clear()
    def clear(self):
        self.sb,self.si,self.rc,self.vm=[],[],[],[]
        self.acts,self.lps,self.vals,self.rews,self.dns=[],[],[],[],[]
    def store(self,sb,si,rc,vm,a,lp,v,r,d):
        self.sb.append(sb);self.si.append(si);self.rc.append(rc);self.vm.append(vm)
        self.acts.append(a);self.lps.append(lp);self.vals.append(v);self.rews.append(r);self.dns.append(d)
    def compute(self, gamma=0.99, lam=0.95):
        T=len(self.rews)
        if T==0: return None
        vs=self.vals+[0.0]; adv=np.zeros(T,np.float32); g=0.0
        for t in reversed(range(T)):
            d=self.rews[t]+gamma*vs[t+1]*(1-self.dns[t])-vs[t]
            g=d+gamma*lam*(1-self.dns[t])*g; adv[t]=g
        ret=adv+np.array(self.vals[:T],np.float32)
        return dict(sb=self.sb,si=self.si,rc=self.rc,vm=self.vm,
                    acts=np.array(self.acts),lps=np.array(self.lps),adv=adv,ret=ret)


def pad_to_batch(arrays):
    mx=max(a.shape[0] for a in arrays); d=arrays[0].shape[1]
    out=np.zeros((len(arrays),mx,d),np.float32)
    for i,a in enumerate(arrays): out[i,:a.shape[0]]=a
    return out


def export_onnx(net, buffer_B, d_model, save_path, meta_path):
    """동적 shape로 ONNX export + 메타 저장."""
    net.eval()
    dummy_sb = torch.zeros(1, 3, 7)   # n+2=3 최소
    dummy_si = torch.zeros(1, buffer_B, 4)
    dummy_rc = torch.zeros(1, 2*buffer_B, 5)
    try:
        torch.onnx.export(
            net, (dummy_sb, dummy_si, dummy_rc), save_path,
            input_names=["s_bin","s_items","rot_cands"],
            output_names=["logits","value"],
            dynamic_axes={"s_bin": {0:"batch", 1:"seq"},
                          "s_items": {0:"batch"},
                          "rot_cands": {0:"batch"},
                          "logits": {0:"batch"}, "value": {0:"batch"}},
            opset_version=17, dynamo=False)
        print(f"   [ONNX] {save_path}")
    except Exception as e:
        print(f"   [ONNX 실패] {e}")
    # 메타
    meta = {"onnx_path": os.path.basename(save_path), "buffer_B": buffer_B,
            "d_model": d_model, "n_layers": 3, "n_heads": 4, "item_dim": 4, "bin_dim": 7,
            "Lc": 120, "Wc": 100, "Hc": 125, "cell": 0.01,
            "r_s": 0.66, "r_w": 3.0}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"   [META] {meta_path}")
    net.train()


@torch.no_grad()
def validate(net, buffer_B, device, n_val=32):
    env = PackingEnv(buffer_B=buffer_B)
    rng = np.random.default_rng(SEED)
    us = []
    for _ in range(n_val):
        obs = env.reset(rng, fixed_ratio=1.0)
        done = False
        while not done:
            sb,si,rc,vm,_=obs
            if vm.sum()==0: break
            t_sb=torch.tensor(sb,dtype=torch.float32,device=device).unsqueeze(0)
            t_si=torch.tensor(si,dtype=torch.float32,device=device).unsqueeze(0)
            t_rc=torch.tensor(rc,dtype=torch.float32,device=device).unsqueeze(0)
            t_vm=torch.tensor(vm,dtype=torch.float32,device=device).unsqueeze(0)
            logits,_=net(t_sb,t_si,t_rc)
            d=safe_categorical(logits,t_vm)
            a=d.probs.argmax(dim=-1).item()
            obs,r,done,info=env.step(a)
        us.append(env.get_util())
    return float(np.mean(us))


def train(buffer_B, max_iter=10000, save_root="./best_model_o4msp",
          fixed_ratio=0.5, lr=1e-4, min_lr=1e-6, warmup=20,
          gamma=0.99, lam=0.95, clip_eps=0.2,
          entropy_coef=0.01, value_coef=0.5, ppo_epochs=4,
          val_every=20, val_size=32, patience=100):
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] B={buffer_B}, device={device}, max_iter={max_iter}")

    net = O4MSPNet(bin_dim=7, item_dim=4, d=128, layers=3, heads=4).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing with warmup
    def lr_lambda(ep):
        if ep < warmup:
            return ep / max(1, warmup)   # 선형 워밍업
        progress = (ep - warmup) / max(1, max_iter - warmup)
        return max(min_lr / lr, 0.5 * (1 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    os.makedirs(save_root, exist_ok=True)
    pt_path = os.path.join(save_root, f"O4MSP_B{buffer_B}.pt")
    onnx_path = os.path.join(save_root, f"O4MSP_B{buffer_B}.onnx")
    meta_path = os.path.join(save_root, f"model_meta_B{buffer_B}.json")

    env = PackingEnv(buffer_B=buffer_B)
    rng = np.random.default_rng(SEED)
    best_val, no_improve = -1.0, 0

    for it in range(max_iter):
        fr = curriculum_fixed_ratio(it/max_iter, start=fixed_ratio, end=0.9, warmup=0.1)
        obs = env.reset(rng, fixed_ratio=fr)
        buf = GAEBuffer()
        done = False
        while not done:
            sb,si,rc,vm,_=obs
            if vm.sum()==0: break
            t_sb=torch.tensor(sb,dtype=torch.float32,device=device).unsqueeze(0)
            t_si=torch.tensor(si,dtype=torch.float32,device=device).unsqueeze(0)
            t_rc=torch.tensor(rc,dtype=torch.float32,device=device).unsqueeze(0)
            t_vm=torch.tensor(vm,dtype=torch.float32,device=device).unsqueeze(0)
            logits,value=net(t_sb,t_si,t_rc)
            dist=safe_categorical(logits,t_vm)
            a=dist.sample(); lp=dist.log_prob(a)
            obs2,reward,done,info=env.step(a.item())
            buf.store(sb,si,rc,vm,a.item(),lp.item(),value.item(),reward,float(done))
            obs=obs2

        data=buf.compute(gamma,lam)
        if data is None: continue

        # PPO update
        b_sb=torch.tensor(pad_to_batch(data["sb"]),dtype=torch.float32,device=device)
        b_si=torch.tensor(np.stack(data["si"]),dtype=torch.float32,device=device)
        b_rc=torch.tensor(np.stack(data["rc"]),dtype=torch.float32,device=device)
        b_vm=torch.tensor(np.stack(data["vm"]),dtype=torch.float32,device=device)
        b_act=torch.tensor(data["acts"],dtype=torch.long,device=device)
        b_olp=torch.tensor(data["lps"],dtype=torch.float32,device=device)
        b_adv=torch.tensor(data["adv"],dtype=torch.float32,device=device)
        b_ret=torch.tensor(data["ret"],dtype=torch.float32,device=device)
        std=b_adv.std().clamp(min=1e-6)
        b_adv=(b_adv-b_adv.mean())/std

        for _ in range(ppo_epochs):
            logits,values=net(b_sb,b_si,b_rc)
            if torch.isnan(logits).any(): break
            dist=safe_categorical(logits,b_vm)
            nlp=dist.log_prob(b_act); ent=dist.entropy()
            ratio=(nlp-b_olp).exp()
            s1=ratio*b_adv; s2=ratio.clamp(1-clip_eps,1+clip_eps)*b_adv
            la=-torch.min(s1,s2).mean()
            lc=F.mse_loss(values,b_ret)
            loss=la+value_coef*lc-entropy_coef*ent.mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),0.5)
            opt.step()

        scheduler.step()
        buf.clear()

        # 검증
        if (it+1) % val_every == 0:
            util = validate(net, buffer_B, device, val_size)
            bonus = max(0, 20 - buffer_B)
            total = util*100 + bonus
            cur_lr = opt.param_groups[0]["lr"]
            print(f"[iter {it+1}/{max_iter}] B={buffer_B} fr={fr:.2f} lr={cur_lr:.2e} "
                  f"loss={loss.item():.3f} util={util:.4f} total={total:.2f} best={max(best_val,0):.2f}")
            if total > best_val:
                best_val, no_improve = total, 0
                torch.save({"state_dict": net.state_dict(), "buffer_B": buffer_B,
                            "d_model": 128, "n_layers": 3, "n_heads": 4,
                            "util": util, "total_score": total}, pt_path)
                export_onnx(net, buffer_B, 128, onnx_path, meta_path)
                print(f"   ↳ best 갱신 (total={total:.2f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[early stopping] {patience}회 개선 없음 → 종료 (best={best_val:.2f})")
                    break
    print(f"학습 완료. best={best_val:.2f} (B={buffer_B}), 모델={pt_path}")
    return best_val


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def _parse_buffers(s):
    s=s.strip()
    if "-" in s and "," not in s:
        lo,hi=s.split("-"); return list(range(int(lo),int(hi)+1))
    return [int(x) for x in s.split(",") if x.strip()]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--buffer_B", type=int, default=1)
    p.add_argument("--buffers", type=str, default=None, help="버퍼 sweep: '1,5,10'")
    p.add_argument("--fixed_ratio", type=float, default=0.5)
    p.add_argument("--max_iter", type=int, default=10000)
    p.add_argument("--val_every", type=int, default=20)
    p.add_argument("--val_size", type=int, default=32)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--save_root", default="./best_model_o4msp")
    args = p.parse_args()

    if args.buffers:
        bs = _parse_buffers(args.buffers)
        print(f"[SWEEP] 버퍼 순회: {bs}\n")
        res = {}
        for B in bs:
            print(f"\n{'='*60}\n[SWEEP] B={B}\n{'='*60}")
            res[B] = train(B, args.max_iter, args.save_root, args.fixed_ratio,
                           val_every=args.val_every, val_size=args.val_size,
                           patience=args.patience)
        print(f"\n{'='*60}\n[결과]\n{'='*60}")
        for B in bs: print(f"  B={B:>2} → total={res[B]:.2f}")
        best_B = max(res, key=res.get)
        print(f"\n>>> 최적 B={best_B} (total={res[best_B]:.2f})")
        print(f">>> 모델: {args.save_root}/O4MSP_B{best_B}.onnx")
    else:
        train(args.buffer_B, args.max_iter, args.save_root, args.fixed_ratio,
              val_every=args.val_every, val_size=args.val_size, patience=args.patience)
