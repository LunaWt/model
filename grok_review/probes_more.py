"""Дополнительные пробы: причинность по чанкам, CUDA-граф на весь слой KDA."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from model.kda_head import KDA, kda_chunkwise
from model.model import K3Config, K3Model


def causality_by_chunk() -> None:
    print("=== причинность: тот же чанк vs предыдущие ===")
    device = torch.device("cuda")
    cfg = K3Config()
    m = K3Model(cfg).to(device).eval()
    torch.manual_seed(3)
    T = 48  # chunk=16 → чанки [0:16), [16:32), [32:48)
    tok = torch.randint(0, cfg.vocab_size, (1, T), device=device)
    with torch.no_grad():
        base = m(tok)
        tok2 = tok.clone()
        tok2[:, -1] = (tok2[:, -1] + 1) % cfg.vocab_size
        other = m(tok2)
        d = (other - base).abs()[0]
        print(f"  max|Δ| чанк0 [0:16)  = {d[:16].max().item():.3e}")
        print(f"  max|Δ| чанк1 [16:32) = {d[16:32].max().item():.3e}")
        print(f"  max|Δ| чанк2 [32:47) = {d[32:47].max().item():.3e}  (тот же чанк, не последний токен)")
        print(f"  max|Δ| pos 47        = {d[47].max().item():.3e}")

        # fp32-модель на том же входе — слишком дорого; снимем один KDA
        layer = m.attn[0]
        x = m.embed(tok)
        x2 = m.embed(tok2)
        y = layer(x)
        y2 = layer(x2)
        dy = (y2 - y).abs()[0]
        print(f"  только KDA[0]: чанк0={dy[:16].max().item():.3e}  "
              f"чанк1={dy[16:32].max().item():.3e}  "
              f"чанк2[:-1]={dy[32:47].max().item():.3e}  last={dy[47].max().item():.3e}")

        # ядро в fp32
        q = F.normalize(torch.randn(1, 2, T, 128, device=device), dim=-1)
        k = F.normalize(torch.randn(1, 2, T, 128, device=device), dim=-1)
        v = torch.randn(1, 2, T, 128, device=device)
        g = -5.0 * torch.sigmoid(torch.randn(1, 2, T, 128, device=device))
        beta = torch.sigmoid(torch.randn(1, 2, T, device=device))
        o1 = kda_chunkwise(q, k, v, g, beta, 16)
        v2 = v.clone(); v2[:, :, -1] += 1.0
        o2 = kda_chunkwise(q, k, v2, g, beta, 16)
        dd = (o2 - o1).abs()
        print(f"  kda_chunkwise fp32: [0:32]={dd[:, :, :32].max().item():.3e}  "
              f"[32:47]={dd[:, :, 32:47].max().item():.3e}  last={dd[:, :, 47].max().item():.3e}")


def kda_layer_cudagraph() -> None:
    print("\n=== CUDA graph на весь слой KDA (проекции+conv+chunkwise), T=1536 ===")
    device = torch.device("cuda")
    layer = KDA(512, 2, 128).to(device).eval()
    x = torch.randn(1, 1536, 512, device=device)
    for _ in range(3):
        layer(x)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(8):
        layer(x)
    t1.record(); torch.cuda.synchronize()
    ms_eager = t0.elapsed_time(t1) / 8

    static = x.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            y = layer(static)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = layer(static)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    t0.record()
    for _ in range(8):
        graph.replay()
    t1.record(); torch.cuda.synchronize()
    ms_g = t0.elapsed_time(t1) / 8
    print(f"  eager {ms_eager:.2f} мс   graph {ms_g:.2f} мс   ({ms_eager / max(ms_g, 1e-6):.1f}x)")
    print(f"  ×12 слоёв: ~{12 * ms_eager:.0f} мс eager vs ~{12 * ms_g:.0f} мс graph  (только fwd)")


if __name__ == "__main__":
    causality_by_chunk()
    kda_layer_cudagraph()
