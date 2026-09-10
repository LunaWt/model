"""Проверка, что torch.compile(mode="max-autotune") реально доходит до CUDA-графов.

    uv run python -m scripts.compile_check --config A16 --seq-len 1344

Проверяются четыре вещи, каждая — отдельная причина, по которой компиляция может
«сработать» и не дать ничего:

1. согласие с eager по числам — на sm_75 мы снимаем запрет inductor на bf16
   (см. model/compile_patch.py), и это надо подтверждать замером, а не верой;
2. сколько раз Dynamo рвал граф — при разрыве компилируются куски, и CUDA-граф
   в лучшем случае накрывает часть шага;
3. сколько CUDA-графов реально ЗАПИСАНО менеджером inductor'а: `cudagraph_skips == 0`
   означает лишь, что отказа не было вслух, а записанных графов может не быть вовсе;
4. время шага до и после.

Выход — JSON в stdout.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from model.compile_patch import compile_body, compile_report, cudagraph_stats
from model.configs import get as get_config
from model.losses import chunked_cross_entropy
from model.model import K3Model
from model.optim import build_optimizers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="tiny")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--mode", default="max-autotune")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--gemm-autotune", action="store_true",
                   help="снять порог is_big_gpu и дать inductor подбирать Triton-шаблоны для матмулов")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def timed(fn, n: int) -> float:
    out = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        out.append(time.time() - t0)
    return min(out)


def main() -> None:
    a = parse_args()
    cfg = get_config(a.config)
    if a.seq_len:
        cfg.max_seq_len = a.seq_len
    dev = a.device

    torch.manual_seed(0)
    model = K3Model(cfg).to(dev)
    muon, adamw = build_optimizers(model)
    x = torch.randint(0, cfg.vocab_size, (a.batch_size, cfg.max_seq_len), device=dev)
    y = torch.randint(0, cfg.vocab_size, (a.batch_size, cfg.max_seq_len), device=dev)

    def loss_of(body) -> torch.Tensor:
        with torch.autocast(dev, dtype=torch.bfloat16):
            h = body(x)
        return chunked_cross_entropy(h, model.lm_head.weight, y)

    def fwd_bwd(body):
        loss_of(body).backward()
        model.harvest_router_scores()

    def optim_step():
        for opt in (muon, adamw):
            opt.step()
            opt.zero_grad(set_to_none=False)
        model.update_router_bias()

    model.train()
    fwd_bwd(model.body)
    eager_ms = 1000 * timed(lambda: fwd_bwd(model.body), a.steps)
    optim_ms = 1000 * timed(optim_step, a.steps)

    from torch._dynamo.utils import counters
    counters.clear()
    body = compile_body(model, a.mode, gemm_autotune=a.gemm_autotune)

    t0 = time.time()
    fwd_bwd(body)
    compile_s = time.time() - t0
    # второй прогон — на нём inductor ЗАПИСЫВАЕТ CUDA-графы; без него первый
    # замеряемый шаг попадает на запись и завышает время на треть
    fwd_bwd(body)
    comp_ms = 1000 * timed(lambda: fwd_bwd(body), a.steps)
    model.zero_grad(set_to_none=False)

    model.eval()
    with torch.no_grad():
        ref = loss_of(model.body).item()
        got = loss_of(body).item()

    report = {
        "config": a.config,
        "seq_len": cfg.max_seq_len,
        "mode": a.mode,
        "gemm_autotune": a.gemm_autotune,
        "compile_s": round(compile_s, 1),
        "eager_fwd_bwd_ms": round(eager_ms, 1),
        "compiled_fwd_bwd_ms": round(comp_ms, 1),
        "optim_ms": round(optim_ms, 1),
        "speedup": round(eager_ms / comp_ms, 2),
        "loss_eager": ref,
        "loss_compiled": got,
        "loss_abs_diff": abs(ref - got),
        **compile_report(),
        **cudagraph_stats(),
        "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20),
    }
    print("REPORT " + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
