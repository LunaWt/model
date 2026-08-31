"""Пробы для ревью K3Model. Основу не трогает.

Запуск: uv run python grok_review/probes.py
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.kda_head import KDA
from model.model import AttnRes, K3Config, K3Model, LatentMoE, SiTUGLU

LN_V = math.log(16384)


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def situ_one(x, W_g, W_u, W_d, b1, b2):
    g = x @ W_g
    gate = b1 * torch.tanh(g / b1) * torch.sigmoid(g)
    up = b2 * torch.tanh((x @ W_u) / b2)
    return (gate * up) @ W_d


# ------------------------------------------------------------------
# 1. Init: ShortConv vs Linear, alpha, hybrid pattern
# ------------------------------------------------------------------
def probe_init() -> dict:
    banner("1. инициализация и раскладка слоёв")
    cfg = K3Config()
    m = K3Model(cfg)
    kda = m.attn[0]
    assert isinstance(kda, KDA)

    conv_std = float(kda.conv_v.conv.weight.detach().std())
    wv_std = float(kda.W_v.weight.detach().std())
    print(f"  ShortConv.conv_v std = {conv_std:.4f}   (kaiming, fan_in=kernel=4)")
    print(f"  KDA.W_v std          = {wv_std:.4f}   (N(0, 0.02) из apply)")
    print(f"  отношение conv/W_v   = {conv_std / wv_std:.1f}x")

    n_mla = sum(1 for a in m.attn if a.__class__.__name__ == "GatedMLA")
    n_kda = sum(1 for a in m.attn if isinstance(a, KDA))
    kinds = ["MLA" if a.__class__.__name__ == "GatedMLA" else "KDA" for a in m.attn]
    print(f"  слои: {n_kda} KDA + {n_mla} MLA  pattern={kinds}")
    print(f"  последний слой MLA? {kinds[-1] == 'MLA'}")
    print(f"  extra MLA хвост (два MLA подряд в конце)? {kinds[-2:] == ['MLA', 'MLA']}")

    n_dense = sum(1 for f in m.ffn if isinstance(f, SiTUGLU))
    n_moe = sum(1 for f in m.ffn if isinstance(f, LatentMoE))
    print(f"  FFN: dense={n_dense} (ожидаем 1), MoE={n_moe}")

    z = kda.b_alpha.detach()
    g = kda.g_min * torch.sigmoid(z)          # A=0
    alpha = g.exp()
    half = math.log(2) / (-g.clamp(max=-1e-8))
    print(f"  α при z=b_α, A=0:  min={float(alpha.min()):.4f}  max={float(alpha.max()):.4f}  "
          f"median={float(alpha.median()):.4f}")
    print(f"  полураспад (токенов): min={float(half.min()):.1f}  "
          f"median={float(half.median()):.1f}  max={float(half.max()):.1f}   "
          f"(контекст T={cfg.max_seq_len})")

    # тот же z=b_α, но как в GDN/Kimi Linear: g = −softplus(z) при A=0 → α=exp(−dt)
    dt_from_z = F.softplus(z)
    alpha_gdn = (-dt_from_z).exp()
    half_gdn = math.log(2) / dt_from_z.clamp(min=1e-8)
    print(f"  если бы это был GDN-softplus на том же z: α median={float(alpha_gdn.median()):.4f}  "
          f"полураспад median={float(half_gdn.median()):.1f} max={float(half_gdn.max()):.1f}")

    n = m.count_params()
    print(f"  params total={n['total']/1e6:.2f}M  active={n['active']/1e6:.2f}M")
    return {"model": m, "cfg": cfg, "conv_std": conv_std, "wv_std": wv_std}


def kda_v_and_o(layer: KDA, x: torch.Tensor):
    """Повторяет начало KDA.forward, чтобы снять v и õ до RMSNorm. Слой не меняем."""
    q = layer._heads(F.silu(layer.conv_q(layer.W_q(x))))
    k = layer._heads(F.silu(layer.conv_k(layer.W_k(x))))
    v = layer._heads(F.silu(layer.conv_v(layer.W_v(x))))
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    beta = torch.sigmoid(layer.W_beta(x)).transpose(1, 2)
    z = layer.W_a_up(layer.W_a_down(x)) + layer.b_alpha
    z = layer._heads(z)
    g = layer.g_min * torch.sigmoid(layer.A.exp().view(1, layer.n_heads, 1, 1) * z)
    from model.kda_head import kda_chunkwise, kda_recurrent
    o = kda_chunkwise(q, k, v, g, beta, chunk=layer.chunk_size) if layer.chunk_size else kda_recurrent(q, k, v, g, beta)
    o = o.to(x.dtype)
    return v, o


# ------------------------------------------------------------------
# 2. Масштаб v / õ  (текущий conv vs N(0,0.02))
# ------------------------------------------------------------------
@torch.no_grad()
def probe_v_scale(device: torch.device) -> None:
    banner("2. масштаб v и õ (один слой KDA, T=256)")
    torch.manual_seed(0)
    layer = KDA(512, 2, 128).to(device)
    x = torch.randn(1, 256, 512, device=device) * 0.02   # как эмбеддинг после init

    def stats(tag, layer, x):
        v, o = kda_v_and_o(layer, x)
        on = layer.o_norm(o)
        print(f"  {tag}")
        print(f"    |v| mean={v.abs().mean():.4f}  rms={v.pow(2).mean().sqrt():.4f}  max={v.abs().max():.3f}")
        print(f"    |õ| mean={o.abs().mean():.4f}  rms={o.pow(2).mean().sqrt():.4f}  max={o.abs().max():.3f}  finite={torch.isfinite(o).all().item()}")
        print(f"    |RMSNorm(õ)| rms={on.pow(2).mean().sqrt():.4f}")

    stats("как сейчас (kaiming conv)", layer, x)

    with torch.no_grad():
        for conv in (layer.conv_q, layer.conv_k, layer.conv_v):
            nn.init.normal_(conv.conv.weight, 0.0, 0.02)
    stats("после conv ~ N(0, 0.02), как Linear и как fla", layer, x)


# ------------------------------------------------------------------
# 3. AttnRes: RMS источников после одного forward
# ------------------------------------------------------------------
@torch.no_grad()
def probe_attnres_scales(model: K3Model, device: torch.device) -> None:
    banner("3. AttnRes: масштаб значений (эмбеддинг vs сумма блока)")
    model = model.to(device).eval()
    records: list[list[float]] = []
    orig = AttnRes.forward

    def wrapped(self, sources):
        rms = [float(s.detach().float().pow(2).mean().sqrt()) for s in sources]
        records.append(rms)
        return orig(self, sources)

    AttnRes.forward = wrapped
    try:
        tok = torch.randint(0, model.cfg.vocab_size, (1, 32), device=device)
        model(tok)
    finally:
        AttnRes.forward = orig

    # первый подслой: только h0; последний res_out: [h0, b1, b2, b3, b4]
    print(f"  вызовов AttnRes: {len(records)}  (16 слоёв × 2 подслоя + res_out = 33)")
    print(f"  первый (только embed) RMS = {records[0]}")
    print(f"  второй (embed + attn_0) RMS = {records[1]}")
    # первый подслой второго блока: layer 4 attn = index 8
    print(f"  старт блока 1 (layer 4 attn) RMS = {records[8]}  <- embed vs полный b1")
    print(f"  res_out RMS = {records[-1]}")
    if records[-1]:
        e, *blocks = records[-1]
        print(f"  embed/mean(blocks) = {e / (sum(blocks) / len(blocks) + 1e-12):.3f}  "
              f"(<1 значит эмбеддинг тонет в значениях)")


# ------------------------------------------------------------------
# 4. LatentMoE: packed bmm vs наивный цикл; overflow
# ------------------------------------------------------------------
def _expert(experts, e: int, x: torch.Tensor) -> torch.Tensor:
    return situ_one(x, experts.W_g[e], experts.W_u[e], experts.W_d[e], experts.b1, experts.b2)


def naive_routed(moe: LatentMoE, h: torch.Tensor, idx: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    m, k = idx.shape
    u = torch.zeros(m, h.shape[-1], dtype=h.dtype, device=h.device)
    for t in range(m):
        for j in range(k):
            e = int(idx[t, j])
            u[t] = u[t] + p[t, j] * _expert(moe.experts, e, h[t:t + 1]).squeeze(0)
    return u


@torch.no_grad()
def probe_moe(device: torch.device) -> None:
    banner("4. LatentMoE: packed vs naive, overflow")
    cfg = K3Config()
    moe = LatentMoE(cfg).to(device).eval()
    torch.manual_seed(1)
    B, T = 2, 16
    x = torch.randn(B, T, cfg.d_model, device=device) * 0.5
    y = moe(x)

    m = B * T
    flat = x.reshape(m, cfg.d_model)
    with torch.autocast(device.type, enabled=False):
        s = torch.sigmoid(F.linear(flat.float(), moe.W_router.weight.float()))
    idx = torch.topk(s + moe.qb_bias, cfg.top_k, dim=-1).indices
    gathered = s.gather(-1, idx)
    p = gathered / gathered.sum(-1, keepdim=True)
    h = moe.W_down(flat)
    u_naive = naive_routed(moe, h, idx, p)
    u_ref = moe.u_norm(u_naive)  # сравним routed-часть до shared через повторный forward? нет —
    # достанем u из packed: прогоним кусок dispatch ещё раз при большом cap
    dropped = int(moe.dropped.detach().cpu()) if hasattr(moe, "dropped") else -1
    print(f"  dropped на этом батче (cf={cfg.capacity_factor}): {dropped}")
    print(f"  y finite={torch.isfinite(y).all().item()}  rms={y.float().pow(2).mean().sqrt():.4f}")

    # без дропа: capacity_factor огромный
    moe.cfg.capacity_factor = 100.0
    y_big = moe(x)
    # naive полный Eq. 11
    u = naive_routed(moe, h, idx, p)
    y_naive = moe.W_up(moe.u_norm(u))
    for exp in moe.shared:
        y_naive = y_naive + exp(flat)
    y_naive = y_naive.view(B, T, cfg.d_model)
    err = (y_big.float() - y_naive.float()).abs().max().item()
    print(f"  packed (cf=100, drop=0) vs naive Eq.11: max|Δ|={err:.3e}")

    # overflow: крошечная полка
    moe.cfg.capacity_factor = 0.01
    y_drop = moe(x)
    dropped = int(moe.dropped.detach().cpu())
    print(f"  cf=0.01 dropped={dropped}/{m * cfg.top_k} слотов  y finite={torch.isfinite(y_drop).all().item()}")

    # кто выпал: хвост по индексу токена внутри эксперта
    cap = max(1, int(m * cfg.top_k / cfg.n_routed * 0.01))
    slot_expert = idx.reshape(-1)
    order = torch.argsort(slot_expert)
    sorted_expert = slot_expert[order]
    src_tok = order // cfg.top_k
    counts = torch.zeros(cfg.n_routed, dtype=torch.long, device=device)
    counts.scatter_add_(0, slot_expert, torch.ones_like(slot_expert))
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(m * cfg.top_k, device=device) - starts[sorted_expert]
    overflow = pos >= cap
    ov_tok = src_tok[overflow]
    print(f"  cap={cap}; overflow token ids: min={int(ov_tok.min()) if ov_tok.numel() else -1} "
          f"max={int(ov_tok.max()) if ov_tok.numel() else -1}  "
          f"mean={float(ov_tok.float().mean()) if ov_tok.numel() else -1:.1f}  "
          f"(при стабильном argsort это хвост батча, не слабые p)")


# ------------------------------------------------------------------
# 5. GroupedSiTUGLU vs цикл по экспертам
# ------------------------------------------------------------------
@torch.no_grad()
def probe_grouped_situ(device: torch.device) -> None:
    banner("5. GroupedSiTUGLU vs поэлементный цикл")
    cfg = K3Config()
    moe = LatentMoE(cfg).to(device)
    torch.manual_seed(2)
    n, cap, ell = 8, 5, cfg.moe_latent
    x = torch.randn(n, cap, ell, device=device)
    # возьмём первых 8 экспертов
    W_g, W_u, W_d = moe.experts.W_g[:n], moe.experts.W_u[:n], moe.experts.W_d[:n]
    b1, b2 = moe.experts.b1, moe.experts.b2
    packed = torch.bmm(
        (b1 * torch.tanh(torch.bmm(x, W_g) / b1) * torch.sigmoid(torch.bmm(x, W_g)))
        * (b2 * torch.tanh(torch.bmm(x, W_u) / b2)),
        W_d,
    )
    loop = torch.stack([situ_one(x[e], W_g[e], W_u[e], W_d[e], b1, b2) for e in range(n)])
    print(f"  max|Δ|={ (packed - loop).abs().max().item():.3e}")


# ------------------------------------------------------------------
# 6. Полная модель: стартовый CE, градиенты, причинность
# ------------------------------------------------------------------
def probe_model_run(model: K3Model, device: torch.device) -> None:
    banner("6. полный forward: CE, грады, причинность (B=1 T=48)")
    model = model.to(device)
    model.train()
    torch.manual_seed(3)
    B, T = 1, 48
    tok = torch.randint(0, model.cfg.vocab_size, (B, T), device=device)
    tgt = torch.randint(0, model.cfg.vocab_size, (B, T), device=device)
    logits = model(tok)
    acc = torch.promote_types(logits.dtype, torch.float32)
    loss = F.cross_entropy(logits.reshape(-1, model.cfg.vocab_size).to(acc), tgt.reshape(-1))
    print(f"  CE={float(loss):.3f}   ln(V)={LN_V:.3f}   Δ={float(loss) - LN_V:+.3f}")
    print(f"  logits rms={logits.float().pow(2).mean().sqrt():.4f}  max={logits.abs().max():.3f}")

    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    zero = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().max() == 0]
    print(f"  параметров без градиента: {len(missing)}  {missing[:8]}")
    print(f"  градиент точно ноль: {len(zero)}  {zero[:8]}")
    # conv_v должен иметь град — путь v живой
    conv_g = model.attn[0].conv_v.conv.weight.grad
    print(f"  |grad conv_v| mean={conv_g.abs().mean():.3e}  |grad W_v| mean={model.attn[0].W_v.weight.grad.abs().mean():.3e}")

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        base = model(tok)
        tok2 = tok.clone()
        tok2[:, -1] = (tok2[:, -1] + 1) % model.cfg.vocab_size
        other = model(tok2)
        early = (other[:, :-1] - base[:, :-1]).abs().max().item()
        last = (other[:, -1] - base[:, -1]).abs().max().item()
        print(f"  причинность: max|Δ| до последнего токена={early:.3e}  на последнем={last:.3e}")


# ------------------------------------------------------------------
# 7. QB: один апдейт, mean-center, нагрузка
# ------------------------------------------------------------------
def probe_qb(device: torch.device) -> None:
    banner("7. Quantile Balancing, один шаг")
    cfg = K3Config()
    moe = LatentMoE(cfg).to(device)
    moe.train()
    torch.manual_seed(4)
    x = torch.randn(4, 64, cfg.d_model, device=device)
    moe(x)
    b0 = moe.qb_bias.clone()
    moe.update_router_bias()
    b1 = moe.qb_bias
    print(f"  |b| до={b0.abs().max():.3e}  после mean={float(b1.mean()):.3e}  std={float(b1.std()):.4f}  max={float(b1.abs().max()):.4f}")
    # второй батч — нагрузка
    moe(x)
    s = moe._scores[-1]
    idx = torch.topk(s + moe.qb_bias, cfg.top_k, dim=-1).indices
    load = torch.zeros(cfg.n_routed, device=device)
    load.scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel(), device=device, dtype=load.dtype))
    target = x.shape[0] * x.shape[1] * cfg.top_k / cfg.n_routed
    print(f"  нагрузка после 1 апдейта: min={int(load.min())} max={int(load.max())} "
          f"target={target:.1f}  cv={float(load.std() / (load.mean() + 1e-8)):.3f}")
    moe.update_router_bias()
    moe(x)
    s = moe._scores[-1]
    idx = torch.topk(s + moe.qb_bias, cfg.top_k, dim=-1).indices
    load.zero_()
    load.scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel(), device=device, dtype=load.dtype))
    print(f"  нагрузка после 2 апдейтов: min={int(load.min())} max={int(load.max())} "
          f"cv={float(load.std() / (load.mean() + 1e-8)):.3f}")


# ------------------------------------------------------------------
# 8. Shared experts: цикл vs один bmm
# ------------------------------------------------------------------
def probe_shared_bmm(device: torch.device) -> None:
    banner("8. shared experts: 2× Linear vs сложенный bmm (микробенч)")
    cfg = K3Config()
    moe = LatentMoE(cfg).to(device).eval()
    x = torch.randn(1, 256, cfg.d_model, device=device)
    flat = x.reshape(-1, cfg.d_model)

    def loop():
        y = torch.zeros_like(flat)
        for exp in moe.shared:
            y = y + exp(flat)
        return y

    W_g = torch.stack([e.W_g.weight.t().contiguous() for e in moe.shared])
    W_u = torch.stack([e.W_u.weight.t().contiguous() for e in moe.shared])
    W_d = torch.stack([e.W_d.weight.t().contiguous() for e in moe.shared])
    b1, b2 = cfg.situ_beta_gate, cfg.situ_beta_up
    inp = flat.unsqueeze(0).expand(cfg.n_shared, -1, -1).contiguous()

    def packed():
        g = torch.bmm(inp, W_g)
        gate = b1 * torch.tanh(g / b1) * torch.sigmoid(g)
        up = b2 * torch.tanh(torch.bmm(inp, W_u) / b2)
        return torch.bmm(gate * up, W_d).sum(0)

    with torch.no_grad():
        err = (loop() - packed()).abs().max().item()
    print(f"  max|Δ| loop vs bmm = {err:.3e}")

    for _ in range(10):
        loop(); packed()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(50):
        loop()
    t1.record(); torch.cuda.synchronize()
    ms_loop = t0.elapsed_time(t1) / 50
    t0.record()
    for _ in range(50):
        packed()
    t1.record(); torch.cuda.synchronize()
    ms_bmm = t0.elapsed_time(t1) / 50
    print(f"  loop 2 экспертов: {ms_loop:.3f} мс   packed bmm: {ms_bmm:.3f} мс   "
          f"({ms_loop / max(ms_bmm, 1e-6):.2f}x)")


# ------------------------------------------------------------------
# 9. CUDA-граф только вокруг kda_chunkwise (гипотеза про запуски)
# ------------------------------------------------------------------
def probe_kda_cudagraph(device: torch.device) -> None:
    banner("9. kda_chunkwise eager vs CUDA graph (T=1536, 1 слой)")
    from model.kda_head import kda_chunkwise
    torch.manual_seed(5)
    B, H, T, D = 1, 2, 1536, 128
    q = F.normalize(torch.randn(B, H, T, D, device=device), dim=-1)
    k = F.normalize(torch.randn(B, H, T, D, device=device), dim=-1)
    v = torch.randn(B, H, T, D, device=device)
    g = -5.0 * torch.sigmoid(torch.randn(B, H, T, D, device=device))
    beta = torch.sigmoid(torch.randn(B, H, T, device=device))
    for _ in range(3):
        kda_chunkwise(q, k, v, g, beta, chunk=16)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(8):
        kda_chunkwise(q, k, v, g, beta, chunk=16)
    t1.record(); torch.cuda.synchronize()
    ms_eager = t0.elapsed_time(t1) / 8

    static_q, static_k, static_v = q.clone(), k.clone(), v.clone()
    static_g, static_beta = g.clone(), beta.clone()
    graph = torch.cuda.CUDAGraph()
    # warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            out = kda_chunkwise(static_q, static_k, static_v, static_g, static_beta, chunk=16)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(graph):
        out = kda_chunkwise(static_q, static_k, static_v, static_g, static_beta, chunk=16)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    t0.record()
    for _ in range(8):
        graph.replay()
    t1.record(); torch.cuda.synchronize()
    ms_graph = t0.elapsed_time(t1) / 8
    print(f"  eager {ms_eager:.2f} мс   CUDA graph {ms_graph:.2f} мс   ({ms_eager / max(ms_graph, 1e-6):.1f}x)")
    print("  это только ядро KDA, без проекций/свёрток; полный слой и модель — другие цифры")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    init = probe_init()
    probe_v_scale(device)
    probe_attnres_scales(init["model"], device)
    probe_moe(device)
    probe_grouped_situ(device)
    probe_model_run(init["model"], device)
    probe_qb(device)
    if device.type == "cuda":
        probe_shared_bmm(device)
        probe_kda_cudagraph(device)
    print("\nготово.")


if __name__ == "__main__":
    main()
