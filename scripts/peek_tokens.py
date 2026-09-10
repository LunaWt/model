"""Посмотреть глазами, что лежит в закодированном корпусе.

    uv run python -m scripts.peek_tokens --tokens 1000 --window 250
    uv run python -m scripts.peek_tokens --group science --tokens 2000

Окна берутся тем же сэмплером, что и обучение (`MixedLoader` по одной группе),
поэтому видно ровно то, что увидит модель: те же границы, та же случайность,
включая обрывы посреди документа.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer

from model.data import DataConfig, MixedLoader, load_manifest

TOKENIZER = Path("tokenizers/bpe_16384.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--group", nargs="*", default=None, help="по умолчанию все из манифеста")
    p.add_argument("--tokens", type=int, default=1000, help="сколько токенов на группу")
    p.add_argument("--window", type=int, default=250)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--root", type=Path, default=Path("data/tokens"))
    return p.parse_args()


def main() -> None:
    a = parse_args()
    tok = Tokenizer.from_file(str(TOKENIZER))
    groups = a.group or sorted(load_manifest(a.root)["groups"])

    for g in groups:
        n = a.tokens * (2 if g == "science" and not a.group else 1)
        cfg = DataConfig(root=a.root, mix={g: 1.0}, seq_len=a.window,
                         batch_size=1, seed=a.seed)
        loader = MixedLoader(cfg, "train")
        print(f"\n{'=' * 78}\n=== {g}: {n} токенов, окна по {a.window}, "
              f"всего в группе {loader.slots[g] * (a.window + 1) / 1e9:.2f} млрд\n{'=' * 78}")
        for i in range(max(1, n // a.window)):
            x, _ = loader.batch(i)
            print(f"\n--- {g} #{i + 1} " + "-" * 60)
            print(tok.decode(x[0].tolist(), skip_special_tokens=False))


if __name__ == "__main__":
    main()
