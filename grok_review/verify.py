"""Проверка трёх утверждений из grok-ревью. Основу не трогает.

Запуск: uv run python grok_review/verify.py
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.kda_head import KDA
from model.model import K3Config, LatentMoE


def banner(t):
    print(f"\n=== {t} ===")


# ------------------------------------------------------------------
# A. RMSNorm: какой eps на самом деле
# ------------------------------------------------------------------
def probe_eps(device):
    banner("A. nn.RMSNorm eps по умолчанию")
    n = nn.RMSNorm(128).to(device)
    print(f"  n.eps = {n.eps}")
    for dt in (torch.float32, torch.bfloat16):
        print(f"  finfo({dt}).eps = {torch.finfo(dt).eps:.3e}")
    # эмпирически: где начинается пол
    for rms in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        x = torch.randn(1, 128, device=device)
        x = x / x.pow(2).mean().sqrt() * rms
        out_rms = float(n(x).pow(2).mean().sqrt())
        print(f"    вход rms={rms:.0e} -> выход rms={out_rms:.4f}")


# ------------------------------------------------------------------
# B. ShortConv init при РЕАЛЬНОМ масштабе входа слоя (после pre-norm!)
# ------------------------------------------------------------------
@torch.no_grad()
def probe_conv_init(device):
    banner("B. ShortConv init: вход слоя после pre-norm (rms=1), не эмбеддинг")
    torch.manual_seed(0)
    pre = nn.RMSNorm(512).to(device)

    def run(tag, layer, x):
        q = layer._heads(F.silu(layer.conv_q(layer.W_q(x))))
        v = layer._heads(F.silu(layer.conv_v(layer.W_v(x))))
        qn = F.normalize(q, dim=-1)
        k = layer._heads(F.silu(layer.conv_k(layer.W_k(x))))
        kn = F.normalize(k, dim=-1)
        beta = torch.sigmoid(layer.W_beta(x)).transpose(1, 2)
        z = layer._heads(layer.W_a_up(layer.W_a_down(x)) + layer.b_alpha)
        g = layer.g_min * torch.sigmoid(layer.A.exp().view(1, layer.n_heads, 1, 1) * z)
        from model.kda_head import kda_chunkwise
        o = kda_chunkwise(qn, kn, v, g, beta, chunk=layer.chunk_size).to(x.dtype)
        on = layer.o_norm(o)
        y = layer.W_o(torch.sigmoid(layer.W_g(x)) * on.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1))
        print(f"  {tag}")
        print(f"    W_v x       rms={layer.W_v(x).pow(2).mean().sqrt():.4f}")
        print(f"    v           rms={v.pow(2).mean().sqrt():.4f}")
        print(f"    õ           rms={o.pow(2).mean().sqrt():.6f}")
        print(f"    RMSNorm(õ)  rms={on.pow(2).mean().sqrt():.4f}   <- если ~1, пол RMSNorm не задет")
        print(f"    выход слоя  rms={y.pow(2).mean().sqrt():.4f}")
        # насколько q/k вообще зависят от масштаба conv (они L2-нормированы)
        return qn.clone(), kn.clone(), y.clone()

    for scale, name in ((1.0, "вход rms=1.0 (реальный: sub(norm(h)))"),
                        (0.02, "вход rms=0.02 (как в пробе grok — эмбеддинг БЕЗ pre-norm)")):
        print(f"\n  --- {name} ---")
        torch.manual_seed(0)
        layer = KDA(512, 2, 128).to(device)
        x = torch.randn(1, 256, 512, device=device)
        x = pre(x) * scale if scale != 1.0 else pre(x)
        q1, k1, y1 = run("kaiming conv (сейчас)", layer, x)
        for conv in (layer.conv_q, layer.conv_k, layer.conv_v):
            nn.init.normal_(conv.conv.weight, 0.0, 0.02)
        q2, k2, y2 = run("conv ~ N(0,0.02) (как fla)", layer, x)
        cos_q = F.cosine_similarity(q1.flatten(0, 2), q2.flatten(0, 2), dim=-1).mean()
        print(f"    cos(q_kaiming, q_0.02) = {cos_q:.4f}  (q L2-нормирован -> масштаб conv сокращается)")
        print(f"    |Δ выхода слоя| / rms = {(y1 - y2).pow(2).mean().sqrt() / y1.pow(2).mean().sqrt():.3f}")


# ------------------------------------------------------------------
# C. MoE overflow: кто выпадает при РЕАЛЬНОМ cf=1.25 и T=1536
# ------------------------------------------------------------------
@torch.no_grad()
def probe_overflow(device):
    banner("C. overflow при cf=1.25, B=1 T=1536 (не игрушечный cf=0.01)")
    cfg = K3Config()
    moe = LatentMoE(cfg).to(device).eval()
    torch.manual_seed(7)

    # стабильность argsort на этой карте
    a = torch.tensor([3, 1, 3, 1, 3, 1], device=device)
    print(f"  argsort стабилен? {torch.argsort(a).tolist()} (стабильный = [1,3,5,0,2,4])")

    for T in (256, 1536):
        x = torch.randn(1, T, cfg.d_model, device=device)
        m = T
        flat = x.reshape(m, cfg.d_model)
        with torch.autocast(device.type, enabled=False):
            s = torch.sigmoid(F.linear(flat.float(), moe.W_router.weight.float()))
        idx = torch.topk(s + moe.qb_bias, cfg.top_k, dim=-1).indices
        gathered = s.gather(-1, idx)
        p = (gathered / gathered.sum(-1, keepdim=True)).reshape(-1)

        slot_expert = idx.reshape(-1)
        order = torch.argsort(slot_expert)
        sorted_expert = slot_expert[order]
        src_tok = order // cfg.top_k
        counts = torch.zeros(cfg.n_routed, dtype=torch.long, device=device)
        counts.scatter_add_(0, slot_expert, torch.ones_like(slot_expert))
        starts = torch.cumsum(counts, 0) - counts
        pos = torch.arange(m * cfg.top_k, device=device) - starts[sorted_expert]
        cap = max(1, int(m * cfg.top_k / cfg.n_routed * cfg.capacity_factor))
        overflow = pos >= cap
        ov_tok = src_tok[overflow]
        n_ov = int(overflow.sum())
        print(f"\n  T={T}: cap={cap}, дропнуто {n_ov}/{m*cfg.top_k} слотов ({100*n_ov/(m*cfg.top_k):.2f}%)")
        if n_ov:
            frac = ov_tok.float() / m
            print(f"    позиция дропнутых токенов (доля от T): mean={float(frac.mean()):.3f} "
                  f"median={float(frac.median()):.3f}  (0.5 = равномерно, ->1 = хвост)")
            # сколько токенов потеряли хотя бы один эксперт / всех
            lost = torch.zeros(m, device=device)
            lost.scatter_add_(0, ov_tok, torch.ones_like(ov_tok, dtype=torch.float))
            print(f"    токенов потерявших >=1 эксперта: {int((lost>0).sum())}/{m}  "
                  f"всех {cfg.top_k}: {int((lost>=cfg.top_k).sum())}")
            # какой вес у выброшенных: слабые слоты или случайные?
            w_ov = p[order][overflow]
            print(f"    вес p выброшенных: mean={float(w_ov.mean()):.4f}  "
                  f"против среднего по всем {float(p.mean()):.4f}")
            first_half = float((frac < 0.5).float().mean())
            print(f"    доля дропов в первой половине последовательности: {first_half:.3f}")


# ------------------------------------------------------------------
# D. Времена жизни: наш гейт vs обе ветки fla
# ------------------------------------------------------------------
def probe_halflife():
    banner("D. полураспад: наш гейт vs обе ветки fla (не гипотетический A=1)")
    torch.manual_seed(0)
    inner = 256
    dt = torch.exp(torch.rand(inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
    z = dt + torch.log(-torch.expm1(-dt))          # inverse softplus, как у нас и у fla

    def report(tag, g):
        half = math.log(2) / (-g.clamp(max=-1e-9))
        print(f"  {tag}")
        print(f"    полураспад: min={float(half.min()):.1f} median={float(half.median()):.1f} "
              f"max={float(half.max()):.1f}")

    # наш путь = fla safe_gate=True, lower_bound=-5, A_log=0
    report("наш / fla safe_gate=True: g = -5*sigmoid(z)", -5.0 * torch.sigmoid(z))
    # fla по умолчанию: safe_gate=False, A_log = log U(1,16)
    A = torch.empty(2).uniform_(1, 16)             # 2 головы
    A_full = A.repeat_interleave(inner // 2)
    report(f"fla default: g = -A*softplus(z), A~U(1,16) (тут A={A.tolist()})",
           -A_full * F.softplus(z))
    # гипотеза grok: A=1
    report("гипотеза grok: g = -softplus(z), т.е. A=1", -F.softplus(z))
    # предложение grok: z = logit(dt/5) под sigmoid
    z2 = torch.logit((dt / 5).clamp(1e-9, 1 - 1e-9))
    report("предложение grok: g = -5*sigmoid(logit(dt/5))", -5.0 * torch.sigmoid(z2))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  torch={torch.__version__}")
    probe_eps(device)
    probe_conv_init(device)
    probe_overflow(device)
    probe_halflife()


if __name__ == "__main__":
    main()
