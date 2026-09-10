"""Чем считать экспертный bmm: замер бэкендов и типов на настоящих формах.

    uv run python -m scripts.bench_bmm

Что проверяется. Эксперты MoE — это три `torch.bmm` вида
`(n, cap, ℓ) @ (n, ℓ, hidden)`. В профиле обучения A16 они дают 918 вызовов
`magma_sgemmEx_kernel<float, __nv_bfloat16, ...>` по ~180 мкс, то есть больше
половины времени forward+backward. Имя ядра означает, что bf16-bmm на sm_75
уходит не в cuBLAS, а в MAGMA, и, судя по числу вызовов, поэлементно по стопке.

Гипотезы, каждая проверяется отдельным столбцом:
  * fp32 попадёт в cuBLAS `gemmStridedBatched` и окажется быстрее bf16;
  * Triton-шаблон inductor'а (доступен только со снятым порогом `is_big_gpu`,
    см. `model/compile_patch.enable_gemm_autotune`) обгонит оба;
  * укладка всех экспертов в один большой mm вместо стопки (для случая, когда
    ёмкости равны) меняет картину.
"""

from __future__ import annotations

import argparse
import time

import torch


def timed(fn, n: int = 20) -> float:
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
    p.add_argument("--n-experts", type=int, nargs="+", default=[64, 80])
    p.add_argument("--tokens", type=int, default=1344)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--capacity-factor", type=float, default=1.5)
    p.add_argument("--latent", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--reps", type=int, default=20)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    dev = "cuda"
    print(f"{'n':>4} {'cap':>5} {'dtype':>8} {'bmm мс':>9} {'mm мс':>9} {'triton мс':>10} "
          f"{'GFLOP':>7} {'ТFLOP/с':>8}")

    for n in a.n_experts:
        cap = max(1, int(a.tokens * a.top_k / n * a.capacity_factor)) + 1
        flop = 2 * n * cap * a.latent * a.hidden / 1e9
        for dt in (torch.bfloat16, torch.float32):
            x = torch.randn(n, cap, a.latent, device=dev, dtype=dt)
            w = torch.randn(n, a.latent, a.hidden, device=dev, dtype=dt)
            ms_bmm = timed(lambda: torch.bmm(x, w), a.reps)

            # тот же объём работы, но одним mm: (n·cap, ℓ) @ (ℓ, hidden).
            # Математически это ДРУГАЯ операция (все токены через один набор
            # весов) — берётся как нижняя граница «сколько это стоило бы, если бы
            # железо считало плотно».
            xf = x.reshape(n * cap, a.latent)
            wf = w[0]
            ms_mm = timed(lambda: xf @ wf, a.reps)

            try:
                from model.compile_patch import enable_bf16_compile, enable_gemm_autotune
                enable_bf16_compile()
                enable_gemm_autotune()
                fn = torch.compile(torch.bmm, mode="max-autotune-no-cudagraphs")
                ms_tri = timed(lambda: fn(x, w), a.reps)
            except Exception as exc:  # noqa: BLE001
                ms_tri = float("nan")
                print(f"  triton не собрался: {type(exc).__name__}: {exc}"[:160])

            name = str(dt).replace("torch.", "")
            print(f"{n:>4} {cap:>5} {name:>8} {ms_bmm:>9.3f} {ms_mm:>9.3f} {ms_tri:>10.3f} "
                  f"{flop:>7.2f} {flop / ms_bmm:>8.2f}")


if __name__ == "__main__":
    main()
