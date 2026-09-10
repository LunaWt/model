"""Опорная точка: обычный плотный трансформер той же активной массы.

    uv run python -m scripts.bench_dense_ref
    colab --auth=adc exec -s t4 -f scripts/bench_dense_ref.py

Зачем. Профиль шага K3 (`scripts/profile_step.py`) показывает 2–3% от потолка
карты по матмулу: 16.5 тыс. запусков ядер, 37% времени в поэлементных операциях,
17% в GEMM. Само по себе это ещё не приговор архитектуре — может, столько же
выдаст любая модель такого размера на этой карте.

Поэтому здесь намеренно скучная модель: RMSNorm + GQA + SwiGLU, никакого MoE,
никакой линейной аттенции, один поток остатка. Тот же токенизатор, та же длина,
та же активная масса. Всё, что она меряет — сколько токенов в секунду даёт
железо, когда тензоры крупные, а ядер мало.

Разница между двумя числами и есть цена архитектуры.

Второе назначение, добавленное позже: сравнение с `scripts/bench_moe_jax.py` на
той же карте. Чтобы сравнение было честным, включаются флаги `--optim`,
`--checkpoint` и `--ce-chunk` — тогда меряется полный шаг обучения, а не
прямой-обратный проход. `--ddp` разносит батч по всем видимым картам через
отдельные процессы и NCCL: на 2xT4 связь идёт по PCIe, поэтому цена обмена
градиентами тут заметно выше, чем на TPU.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

try:
    from model.compile_patch import enable_bf16_compile, enable_gemm_autotune
except ImportError:  # запуск одним файлом на Kaggle, без репозитория рядом
    def enable_bf16_compile() -> bool:
        return False

    def enable_gemm_autotune() -> None:
        raise SystemExit("--gemm-autotune требует model/compile_patch.py")


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, n_kv: int, hidden: int):
        super().__init__()
        self.n_heads, self.n_kv = n_heads, n_kv
        self.d_head = d // n_heads
        self.norm1 = nn.RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, n_kv * self.d_head, bias=False)
        self.v = nn.Linear(d, n_kv * self.d_head, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.norm2 = nn.RMSNorm(d)
        self.up = nn.Linear(d, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        """`enable_gqa=True` на sm_75 отключает экономное ядро внимания.

        Проверено прямым вызовом: с `enable_gqa` доступен только backend MATH, и
        матрица T x T выезжает в память целиком — 646 МиБ против 18 на тех же формах.
        Стоит развернуть ключи повторением вручную, и включается cutlass-ядро.
        Flash на Turing нет вовсе, он с sm_80.
        """
        B, T, _ = x.shape
        h = self.norm1(x)
        q = self.q(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(h).view(B, T, self.n_kv, self.d_head).transpose(1, 2)
        v = self.v(h).view(B, T, self.n_kv, self.d_head).transpose(1, 2)
        rep = self.n_heads // self.n_kv
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            a = F.scaled_dot_product_attention(
                q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, -1))
        g, u = self.up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(g) * u)


class DenseLM(nn.Module):
    def __init__(self, vocab: int, d: int, layers: int, n_heads: int, n_kv: int, hidden: int,
                 ckpt: bool = False, ce_chunk: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, n_heads, n_kv, hidden) for _ in range(layers))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight
        self.ckpt, self.ce_chunk, self.vocab = ckpt, ce_chunk, vocab

    def forward(self, tokens, targets=None):
        """При `ce_chunk` логиты не материализуются на все токены сразу.

        (B, T, 32768) в fp32 при B=8 T=2048 — это 2 ГБ одним куском, то есть треть
        карты T4 на одну промежуточную величину. Кусок считается под `checkpoint`,
        поэтому обратный проход пересчитывает его, а не хранит.
        """
        x = self.embed(tokens)
        for b in self.blocks:
            x = checkpoint(b, x, use_reentrant=False) if self.ckpt else b(x)
        h = self.norm(x)
        if targets is None:
            return self.head(h)
        h, t = h.reshape(-1, h.shape[-1]), targets.reshape(-1)
        if not self.ce_chunk:
            return F.cross_entropy(self.head(h).float(), t)

        def piece(hi, ti):
            return F.cross_entropy(self.head(hi).float(), ti, reduction="sum")

        total = sum(checkpoint(piece, h[i:i + self.ce_chunk], t[i:i + self.ce_chunk],
                               use_reentrant=False)
                    for i in range(0, h.shape[0], self.ce_chunk))
        return total / h.shape[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=672)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--vocab", type=int, default=16384)
    p.add_argument("--dtypes", nargs="*", default=["float16", "float32"])
    p.add_argument("--compile", nargs="*", default=["none"],
                   help="none | default | reduce-overhead | max-autotune")
    p.add_argument("--gemm-autotune", action="store_true",
                   help="снять порог is_big_gpu и включить max_autotune_gemm")
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--optim", default="none", help="none | adamw | adafactor")
    p.add_argument("--checkpoint", action="store_true",
                   help="пересчитывать активации блока на обратном проходе")
    p.add_argument("--ce-chunk", type=int, default=0,
                   help="считать кросс-энтропию кусками по столько токенов; 0 — одним куском")
    p.add_argument("--ddp", action="store_true",
                   help="разнести батч по всем картам отдельными процессами")
    p.add_argument("--backend", default="nccl", help="nccl | gloo")
    p.add_argument("--ddp-timeout", type=int, default=120, help="секунд до падения коллектива")
    p.add_argument("--shapes", nargs="*",
                   default=["512:16:8:2:1365", "768:10:12:4:2048", "1024:8:16:4:2730"],
                   help="d:слоёв:голов:kv-голов:скрытая")
    return p.parse_args()


def make_optimizer(name, params):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=1e-3, fused=True)
    if name == "adafactor":
        return torch.optim.Adafactor(params, lr=1e-3)
    return None


def bench(a, rank: int, world: int) -> None:
    dev = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    local_b = a.batch_size // world
    if rank == 0:
        cap = torch.cuda.get_device_capability(rank)
        print(f"{torch.cuda.get_device_name(rank)} sm_{cap[0]}{cap[1]} | "
              f"torch {torch.__version__} | карт {world} | B={a.batch_size} "
              f"({local_b} на карту) T={a.seq_len} | опт {a.optim} | "
              f"remat {a.checkpoint} | ce-chunk {a.ce_chunk}", flush=True)
        print(f"\n{'d:слоёв:голов':<18} {'всего М':>8} {'dtype':>9} {'compile':>15} "
              f"{'сборка с':>9} {'мс':>8} {'ток/с':>9} {'ТFLOP/с':>8} {'карта МиБ':>10}",
              flush=True)

    for shape in a.shapes:
        d, layers, heads, kv, hidden = (int(v) for v in shape.split(":"))
        for dname in a.dtypes:
            dtype = getattr(torch, dname)
            for cmode in a.compile:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.compiler.reset()
                model = DenseLM(a.vocab, d, layers, heads, kv, hidden,
                                a.checkpoint, a.ce_chunk).to(dev).train()
                n = sum(p.numel() for p in model.parameters())
                fn = model if cmode == "none" else torch.compile(model, mode=cmode)
                if world > 1:
                    fn = nn.parallel.DistributedDataParallel(fn, device_ids=[rank])
                opt = make_optimizer(a.optim, model.parameters())
                x = torch.randint(0, a.vocab, (local_b, a.seq_len), device=dev)
                y = torch.randint(0, a.vocab, (local_b, a.seq_len), device=dev)

                def once(g=None):
                    torch.compiler.cudagraph_mark_step_begin()
                    with torch.autocast("cuda", dtype=dtype,
                                        enabled=dtype is not torch.float32):
                        loss = (g or fn)(x, y)
                    loss.backward()
                    if opt is not None:
                        opt.step()
                    model.zero_grad(set_to_none=True)

                head = f"{shape:<18} {n / 1e6:>8.1f} {dname:>9} {cmode:>15}"
                try:
                    once(model)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    once()
                    torch.cuda.synchronize()
                    build = time.perf_counter() - t0
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
                except torch.OutOfMemoryError:
                    if rank == 0:
                        print(f"{head} {'':>9} {'OOM':>8}", flush=True)
                    del model
                    torch.cuda.empty_cache()
                    continue
                except Exception as exc:  # noqa: BLE001
                    if rank == 0:
                        print(f"{head} {type(exc).__name__}: {str(exc)[:60]}", flush=True)
                    del model
                    torch.cuda.empty_cache()
                    continue

                tokens = a.batch_size * a.seq_len
                gflop = 6 * n * tokens / 1e9
                if rank == 0:
                    peak = torch.cuda.max_memory_allocated(rank) / 2**20
                    print(f"{head} {build:>9.1f} {best * 1000:>8.1f} "
                          f"{tokens / best:>9.0f} {gflop / (best * 1000):>8.2f} "
                          f"{peak:>10.0f}", flush=True)
                del model, fn, x, y, opt
                torch.cuda.empty_cache()


def worker(rank: int, world: int, a) -> None:
    """NCCL между двумя T4 на Kaggle зависает: первый же all-reduce на 297 млн чисел
    провисел 600 секунд и убил процесс. P2P между картами там нет, поэтому обмен
    принудительно уводится в общую память, а таймаут ставится коротким — чтобы
    падать за минуту, а не за десять."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    dist.init_process_group(a.backend, rank=rank, world_size=world,
                            timeout=timedelta(seconds=a.ddp_timeout))
    try:
        bench(a, rank, world)
    finally:
        dist.destroy_process_group()


def main() -> None:
    a = parse_args()
    print(f"патч bf16 {enable_bf16_compile()} | gemm-autotune {a.gemm_autotune}", flush=True)
    if a.gemm_autotune:
        enable_gemm_autotune()
    bench(a, 0, 1)
    if not a.ddp:
        return
    world = torch.cuda.device_count()
    if world < 2:
        print(f"--ddp пропущен: видно карт {world}", flush=True)
        return
    if a.batch_size % world:
        raise SystemExit(f"батч {a.batch_size} не делится на {world} карт")
    print(f"\n--- то же самое на {world} картах через DDP ---", flush=True)
    torch.multiprocessing.spawn(worker, args=(world, a), nprocs=world, join=True)


if __name__ == "__main__":
    main()
