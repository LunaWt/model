"""Стоит ли батчить Ньютона–Шульца по параметрам одинаковой формы.

    uv run python -m scripts.bench_muon

Замер, а не рассуждение. Шаг Muon на A16 стоит 1112 мс против 314 мс у
forward+backward микро-батча, и это арифметика, а не запуски: компиляция дала
всего 1000 мс. Основная масса — 45 стопок весов экспертов формы (64, 256, 256),
5 итераций по 3 матмула каждая.

Вопрос ровно один: батчевый вызов на (g·64, 256, 256) быстрее, чем g вызовов на
(64, 256, 256)? Если да — Muon упирается в занятость GPU мелкими матрицами, и
группировка по форме окупится. Если нет — он упирается в FLOPs, и единственный
рычаг это `ns_steps`.

Стопка в fp32 стоит памяти: 45 × 64 × 256 × 256 × 4 Б = 755 МиБ, поэтому кривая
снимается по размеру группы, а не одним числом.
"""

from __future__ import annotations

import argparse
import time

import torch

from model.optim import orthogonalize


def timed(fn, n: int = 10) -> float:
    fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best * 1000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--groups", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--ns-steps", type=int, nargs="+", default=[5, 3])
    p.add_argument("--reps", type=int, default=10)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    dev = "cuda"
    n, d = a.experts, a.dim
    flop_one = 3 * 2 * n * d ** 3 / 1e9      # три матмула на итерацию

    for steps in a.ns_steps:
        print(f"\nns_steps={steps}   форма ({n}, {d}, {d}), {flop_one * steps:.1f} GFLOP на тензор")
        print(f"{'группа':>7} {'по одному мс':>13} {'батчем мс':>11} {'выигрыш':>8} {'ТFLOP/с':>8}")
        for g in a.groups:
            xs = [torch.randn(n, d, d, device=dev) for _ in range(g)]
            stacked = torch.randn(g * n, d, d, device=dev)

            one = timed(lambda: [orthogonalize(x, steps=steps) for x in xs], a.reps)
            bat = timed(lambda: orthogonalize(stacked, steps=steps), a.reps)
            print(f"{g:>7} {one:>13.1f} {bat:>11.1f} {one / bat:>7.2f}x "
                  f"{g * flop_one * steps / bat:>8.2f}")
            del xs, stacked
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
