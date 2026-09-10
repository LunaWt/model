"""Скорость одного forward+backward на случайных токенах, без данных и без compile.

    uv run python -m scripts.bench_step --config A16 --seq-len 672 --batch-size 2
    colab --auth=adc exec -s t4 -f scripts/bench_step.py

Нужен, чтобы сравнивать карты между собой: датасет для этого не требуется, а
`torch.compile` только мешает — он стоит минуты и его выигрыш зависит от карты
отдельно от самой арифметики.

Типы перебираются все, которые карта вообще может исполнить. На sm_75 bf16
эмулируется и оказывается медленнее fp32, а fp16 попадает в тензорные ядра, если
они есть (T4 — есть, GTX 1660 Ti — нет), поэтому «какой тип быстрее» — вопрос к
конкретной карте, а не к коду.

fp16 здесь идёт БЕЗ `GradScaler`: замеряется время, а не сходимость. Для обучения
в fp16 масштабирование потерь обязательно, иначе мелкие градиенты схлопываются в
ноль — это отдельная правка в `scripts/train.py`, не в этом файле.
"""

from __future__ import annotations

import argparse
import time

import torch

from model.configs import get as get_config
from model.losses import chunked_cross_entropy
from model.model import K3Model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="*", default=["A16"])
    p.add_argument("--seq-len", type=int, default=672)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--capacity-factor", type=float, default=2.0)
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--dtypes", nargs="*", default=["float16", "bfloat16", "float32"])
    return p.parse_args()


def bench(cfg, dtype, a) -> dict:
    dev = "cuda"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = K3Model(cfg).to(dev)
    model.train()
    x = torch.randint(0, cfg.vocab_size, (a.batch_size, a.seq_len), device=dev)
    y = torch.randint(0, cfg.vocab_size, (a.batch_size, a.seq_len), device=dev)

    def once():
        with torch.autocast(dev, dtype=dtype, enabled=dtype is not torch.float32):
            h = model.body(x)
        loss = chunked_cross_entropy(h, model.lm_head.weight, y)
        loss.backward()
        for p in model.parameters():
            p.grad = None
        return loss

    for _ in range(2):
        once()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(a.reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        once()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)

    peak = torch.cuda.max_memory_allocated() / 2**20
    free, total = torch.cuda.mem_get_info()
    out = {"ms": best * 1000, "tok_s": a.batch_size * a.seq_len / best,
           "peak_mib": peak, "card_mib": (total - free) / 2**20}
    del model, x, y
    torch.cuda.empty_cache()
    return out


def main() -> None:
    a = parse_args()
    cap = torch.cuda.get_device_capability(0)
    print(f"{torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} | torch {torch.__version__} | "
          f"B={a.batch_size} T={a.seq_len} cf={a.capacity_factor}")
    print(f"\n{'конфиг':<10} {'dtype':>9} {'мс':>8} {'ток/с':>9} {'пик МиБ':>9} {'карта МиБ':>10}")

    for name in a.config:
        cfg = get_config(name)
        cfg.max_seq_len = a.seq_len
        cfg.capacity_factor = a.capacity_factor
        n = sum(p.numel() for p in K3Model(cfg).parameters())
        for dname in a.dtypes:
            dtype = getattr(torch, dname)
            try:
                r = bench(cfg, dtype, a)
            except torch.OutOfMemoryError:
                print(f"{name:<10} {dname:>9} {'OOM':>8}")
                torch.cuda.empty_cache()
                continue
            print(f"{name:<10} {dname:>9} {r['ms']:>8.1f} {r['tok_s']:>9.0f} "
                  f"{r['peak_mib']:>9.0f} {r['card_mib']:>10.0f}")
        print(f"{'':<10} ({n / 1e6:.1f}М параметров)")


if __name__ == "__main__":
    main()
