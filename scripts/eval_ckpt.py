"""Оценить любые чекпоинты на ОДНОЙ И ТОЙ ЖЕ валидации.

    uv run python -m scripts.eval_ckpt runs/*/last.pt --seq-len 1344 --batches 32

Зачем отдельно от `scripts.train`: там val считается на той длине, на которой шло
обучение, и `val_loss` прогона при T=672 несравним с прогоном при T=1344 — при
короткой длине у токенов меньше контекста, и CE выше просто из-за этого, а не
из-за модели. Как только мы начинаем менять T между конфигами, единственный
честный способ — прогнать все чекпоинты через одну сетку.

Ёмкость экспертов на время замера снимается: при фиксированной полке результат
токена зависит от того, какие ещё токены рядом (см. `K3Model.set_full_capacity`),
и модель, обученная при другом B·T, получила бы фору или штраф ни за что.
"""

from __future__ import annotations

import argparse
import json
import math

import torch

from model.configs import get as get_config
from model.data import DataConfig, MixedLoader
from model.losses import chunked_cross_entropy
from model.model import K3Config, K3Model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--seq-len", type=int, default=1344)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--batches", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--full-capacity", action="store_true", default=True)
    p.add_argument("--keep-capacity", dest="full_capacity", action="store_false")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def score(model, loader, n_batches: int, device: str) -> float:
    total = 0.0
    for i in range(n_batches):
        x, y = loader.batch(i, device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            h = model.body(x)
        total += chunked_cross_entropy(h, model.lm_head.weight, y).item()
    return total / n_batches


def main() -> None:
    a = parse_args()
    loader = MixedLoader(DataConfig(seq_len=a.seq_len, batch_size=a.batch_size,
                                    seed=a.seed), "val")
    rows = []
    for path in a.checkpoints:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = K3Config(**ck["cfg"]) if "cfg" in ck else get_config(ck["config"])
        trained_at = cfg.max_seq_len
        cfg.max_seq_len = a.seq_len
        model = K3Model(cfg)
        model.load_state_dict(ck["model"])
        model = model.to(a.device).eval()
        model.set_full_capacity(a.full_capacity)
        val = score(model, loader, a.batches, a.device)
        rows.append({"checkpoint": path, "config": ck["config"], "step": ck["step"],
                     "trained_at_T": trained_at, "val": round(val, 4),
                     "ppl": round(math.exp(min(val, 20)), 1)})
        del model
        if a.device == "cuda":
            torch.cuda.empty_cache()
        r = rows[-1]
        print(f"{r['config']:<12} шаг {r['step']:>4}  обучен при T={trained_at:<5} "
              f"val {r['val']:.4f}  ppl {r['ppl']:.1f}   {path}")

    print("\n" + json.dumps({"seq_len": a.seq_len, "batch_size": a.batch_size,
                             "batches": a.batches, "full_capacity": a.full_capacity,
                             "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
