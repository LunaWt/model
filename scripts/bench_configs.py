"""Какие конфиги вообще влезают в карту и с какой скоростью.

    uv run python -m scripts.bench_configs --configs A16 L12_4x2 --seq-len 1536

Гоняет НАСТОЯЩИЙ шаг обучения (forward + backward + Muon + AdamW + QB), а не
только forward: на этой карте всё решает память под состояние оптимизатора, и
forward-only замер про неё ничего не говорит. OOM ловится и печатается строкой,
а не роняет прогон, — иначе один не влезший конфиг убивает всю таблицу.

Пик памяти сравним между строками и НЕ сравним с числами из ledger до 2 сен:
там был AdamW на всех параметрах и полные fp32-логиты.
"""

from __future__ import annotations

import argparse
import time

import torch

from model.configs import CONFIGS
from model.data import DataConfig, MixedLoader
from model.losses import chunked_cross_entropy
from model.model import K3Model
from model.optim import build_optimizers


def bench(name: str, seq_len: int, steps: int, warmup: int, accum: int,
          device: str, capacity_factor: float | None = None) -> dict:
    cfg = CONFIGS[name]
    cfg.max_seq_len = seq_len
    if capacity_factor:
        cfg.capacity_factor = capacity_factor
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = K3Model(cfg).to(device)
    counts = model.count_params()
    muon, adamw = build_optimizers(model)
    loader = MixedLoader(DataConfig(seq_len=seq_len, batch_size=1), "train")
    model.train()

    times = []
    for step in range(warmup + steps):
        t0 = time.time()
        for micro in range(accum):
            x, y = loader.batch(step * accum + micro, device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                h = model.body(x)
            (chunked_cross_entropy(h, model.lm_head.weight, y) / accum).backward()
            model.harvest_router_scores()
        for opt in (muon, adamw):
            opt.step()
            opt.zero_grad(set_to_none=True)
        model.update_router_bias()
        torch.cuda.synchronize()
        if step >= warmup:
            times.append(time.time() - t0)

    tok = seq_len * accum
    ms = sorted(times)[len(times) // 2] * 1000
    return {
        "cf": cfg.capacity_factor,
        "exec": len(model.execution_order),
        "total": counts["total"] / 1e6,
        "executed": counts["executed"] / 1e6,
        "peak": torch.cuda.max_memory_allocated() / 2**20,
        "ms": ms,
        "tok_s": tok / (ms / 1000),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=["A16", "L12_4x2", "A24", "L16_8x2", "L12_6x3"])
    p.add_argument("--seq-len", nargs="+", type=int, default=[1536])
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--capacity-factors", nargs="+", type=float, default=[0.0],
                   help="перебрать запас ёмкости MoE; 0 — как в конфиге")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    print(f"grad_accum={a.grad_accum}, шаг обучения целиком "
          f"(fwd+bwd+Muon+AdamW+QB), медиана из {a.steps}\n")
    head = "%-10s %6s %5s %5s %9s %10s %10s %9s %9s"
    print(head % ("config", "T", "cf", "exec", "total M", "exec M",
                  "peak MiB", "ms/step", "ток/с"))
    for name in a.configs:
        for T in a.seq_len:
            for cf in a.capacity_factors:
                try:
                    r = bench(name, T, a.steps, a.warmup, a.grad_accum, a.device, cf)
                    print(head % (name, T, "%.2f" % r["cf"], r["exec"], "%.1f" % r["total"],
                                  "%.1f" % r["executed"], "%.0f" % r["peak"],
                                  "%.0f" % r["ms"], "%.0f" % r["tok_s"]))
                except torch.OutOfMemoryError:
                    print(head % (name, T, "%.2f" % cf, "-", "-", "-", "OOM", "-", "-"))
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
