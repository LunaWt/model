"""Потолок карты по матмулу: fp32 / fp16 / bf16, на любой машине.

    uv run python -m scripts.colab_ceiling                       # локально
    colab --auth=adc exec -s t4 -f scripts/colab_ceiling.py      # на Colab

Тот же замер, что дал 2.91 TFLOP/s fp32 на GTX 1660 Ti (notes/ledger.md, 2 сен),
чтобы числа с разных карт лежали в одной таблице. Формы взяты две: плотный
квадрат 4096 — верхняя граница железа, и настоящая форма экспертного bmm — то,
во что упирается наш шаг.

Зачем сравнивать три типа, а не два: у Turing (sm_75) нет аппаратного bf16, но у
T4, в отличие от TU116, есть тензорные ядра, и они работают в fp16. Поэтому
ответ «bf16 или fp16» на T4 не такой, как на карте без тензорных ядер, и его
надо мерить, а не выводить.
"""

from __future__ import annotations

import time

import torch


def timed(fn, warmup: int = 3, reps: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best * 1000


def main() -> None:
    dev = "cuda"
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"{name} sm_{cap[0]}{cap[1]} | torch {torch.__version__} | "
          f"bf16 аппаратно: {torch.cuda.is_bf16_supported(including_emulation=False)}")
    print(f"\n{'форма':<22} {'dtype':>9} {'мс':>9} {'ТFLOP/с':>9}")

    for n in (1024, 2048, 4096):
        flop = 2 * n ** 3 / 1e9
        for dt in (torch.float32, torch.float16, torch.bfloat16):
            a = torch.randn(n, n, device=dev, dtype=dt)
            b = torch.randn(n, n, device=dev, dtype=dt)
            ms = timed(lambda: a @ b)
            print(f"{f'{n}x{n} плотный':<22} {str(dt).replace('torch.',''):>9} "
                  f"{ms:>9.3f} {flop / ms:>9.2f}")
            del a, b
            torch.cuda.empty_cache()

    e, cap_tokens, latent, hidden = 64, 127, 256, 256
    flop = 2 * e * cap_tokens * latent * hidden / 1e9
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(e, cap_tokens, latent, device=dev, dtype=dt)
        w = torch.randn(e, latent, hidden, device=dev, dtype=dt)
        ms = timed(lambda: torch.bmm(x, w))
        print(f"{f'bmm {e}x{cap_tokens}x{latent}':<22} {str(dt).replace('torch.',''):>9} "
              f"{ms:>9.3f} {flop / ms:>9.2f}")
        del x, w
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
