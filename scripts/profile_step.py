"""Куда уходит время шага: топ ядер и разбивка по подсистемам.

    uv run python -m scripts.profile_step --config A16 --seq-len 672 --batch-size 2
    colab --auth=adc exec -s t4 -f scripts/profile_step.py

Вопрос, на который отвечает файл: почему модель на 48.7M активных параметров
выдаёт единицы процентов от потолка карты по матмулу. Считаем две вещи и кладём
их рядом:

  * полезные FLOP шага = 6 · активные_параметры · токены (2 на forward, 4 на
    backward) — это то, ради чего всё делается;
  * реальное время по ядрам из профайлера, сгруппированное по подсистемам.

Группировка по именам ядер грубая и нарочно консервативная: всё, что не опознано,
попадает в «прочее», а не размазывается по удобным корзинам.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from model.configs import get as get_config
from model.losses import chunked_cross_entropy
from model.model import K3Model

BUCKETS = [
    ("матмул/GEMM", r"gemm|sgemm|hgemm|s16816|volta_|turing_|ampere_|cutlass|bmm|magma|dot_kernel|nn_128|nt_128|tn_128"),
    ("KDA (fla)",   r"chunk_|fused_recurrent|wy_fast|solve_tril|_kda|kda_|prepare_wy|_gate"),
    ("softmax/SDPA", r"softmax|attention|sdpa|flash|mem_eff"),
    ("MoE-раскладка", r"sort|scatter|gather|index_|cumsum|topk|nonzero|unique|argsort|masked_"),
    ("нормы/поэлементно", r"norm|elementwise|vectorized_|silu|sigmoid|mul|add|copy|cast|exp|fill|zero|reduce|Cat|slice|transpose|permute|contiguous"),
]


def bucket_of(name: str) -> str:
    low = name.lower()
    for label, pattern in BUCKETS:
        if re.search(pattern, low):
            return label
    return "прочее"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="A16")
    p.add_argument("--seq-len", type=int, default=672)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--capacity-factor", type=float, default=2.0)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--top", type=int, default=18)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    dev = "cuda"
    dtype = getattr(torch, a.dtype)
    cfg = get_config(a.config)
    cfg.max_seq_len = a.seq_len
    cfg.capacity_factor = a.capacity_factor

    model = K3Model(cfg).to(dev).train()
    counts = model.count_params()
    active = counts["active"] if isinstance(counts, dict) and "active" in counts else None
    tokens = a.batch_size * a.seq_len

    x = torch.randint(0, cfg.vocab_size, (a.batch_size, a.seq_len), device=dev)
    y = torch.randint(0, cfg.vocab_size, (a.batch_size, a.seq_len), device=dev)

    def once():
        with torch.autocast(dev, dtype=dtype, enabled=dtype is not torch.float32):
            h = model.body(x)
        chunked_cross_entropy(h, model.lm_head.weight, y).backward()
        for p in model.parameters():
            p.grad = None

    for _ in range(3):
        once()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        once()
        torch.cuda.synchronize()

    events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    total_us = sum(e.self_device_time_total for e in events)

    by_bucket: dict[str, float] = defaultdict(float)
    n_launches = 0
    for e in events:
        by_bucket[bucket_of(e.key)] += e.self_device_time_total
        n_launches += e.count

    cap = torch.cuda.get_device_capability(0)
    print(f"{torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} | {a.config} | "
          f"{a.dtype} | B={a.batch_size} T={a.seq_len}")
    print(f"время на GPU {total_us / 1000:.1f} мс, запусков ядер {n_launches}, "
          f"уникальных ядер {len(events)}")
    if active:
        gflop = 6 * active * tokens / 1e9
        print(f"полезных FLOP {gflop:.1f} ГФЛОП (6 · {active / 1e6:.1f}М активных · {tokens} токенов) "
              f"→ {gflop / (total_us / 1000):.2f} ТFLOP/с")

    print(f"\n{'подсистема':<20} {'мс':>8} {'доля':>7}")
    for label, us in sorted(by_bucket.items(), key=lambda kv: -kv[1]):
        print(f"{label:<20} {us / 1000:>8.1f} {us / total_us:>6.1%}")

    print(f"\nтоп-{a.top} ядер:")
    print(f"{'ядро':<58} {'мс':>7} {'доля':>6} {'запусков':>9}")
    for e in sorted(events, key=lambda e: -e.self_device_time_total)[: a.top]:
        print(f"{e.key[:58]:<58} {e.self_device_time_total / 1000:>7.2f} "
              f"{e.self_device_time_total / total_us:>5.1%} {e.count:>9}")


if __name__ == "__main__":
    main()
