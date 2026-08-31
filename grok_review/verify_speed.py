"""Один харнесс для двух чисел grok, которые противоречат друг другу:
kda_chunkwise 66 мс против целого слоя KDA 43.1 мс (слой не может быть быстрее
своего же ядра). Меряем оба на одних и тех же данных.

Запуск: PYTHONPATH=. uv run python grok_review/verify_speed.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from model.kda_head import KDA, kda_chunkwise


def timeit(fn, n=8, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / n


@torch.no_grad()
def main():
    device = torch.device("cuda")
    torch.manual_seed(0)
    B, T = 1, 1536
    layer = KDA(512, 2, 128).to(device).eval()
    x = torch.randn(B, T, 512, device=device)

    # то, что слой реально скармливает ядру
    q = F.normalize(layer._heads(F.silu(layer.conv_q(layer.W_q(x)))), dim=-1)
    k = F.normalize(layer._heads(F.silu(layer.conv_k(layer.W_k(x)))), dim=-1)
    v = layer._heads(F.silu(layer.conv_v(layer.W_v(x))))
    beta = torch.sigmoid(layer.W_beta(x)).transpose(1, 2)
    z = layer._heads(layer.W_a_up(layer.W_a_down(x)) + layer.b_alpha)
    g_real = layer.g_min * torch.sigmoid(layer.A.exp().view(1, 2, 1, 1) * z)
    # то, что подставлял grok в probes.py: g из randn, разброс на весь [-5, 0]
    g_rand = -5.0 * torch.sigmoid(torch.randn_like(g_real))

    print(f"T={T}, B={B}, fwd только, eager, fp32")
    print(f"  весь слой KDA(x)                  {timeit(lambda: layer(x)):7.2f} мс")
    print(f"  kda_chunkwise, g из слоя          {timeit(lambda: kda_chunkwise(q, k, v, g_real, beta, 16)):7.2f} мс")
    print(f"  kda_chunkwise, g = -5*sigm(randn) {timeit(lambda: kda_chunkwise(q, k, v, g_rand, beta, 16)):7.2f} мс")
    print(f"  g из слоя:   min={float(g_real.min()):.3f} median={float(g_real.median()):.4f}")
    print(f"  g из randn:  min={float(g_rand.min()):.3f} median={float(g_rand.median()):.4f}")


if __name__ == "__main__":
    main()
