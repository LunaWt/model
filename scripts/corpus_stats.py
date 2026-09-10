"""Из чего состоит группа корпуса: доли разметки, кода и обычного текста.

    uv run python -m scripts.corpus_stats --windows 400 --window 256

Смотреть глазами (`scripts/peek_tokens.py`) хватает, чтобы понять жанр, но не
чтобы решить, какой вес дать группе. Здесь по случайной выборке окон считаются
грубые, зато сравнимые признаки:

  latex   доля символов из `$\\{}^_&` — насколько текст является формулой;
  code    доля окон, где встречается тройной бэктик или отступ + `def`/`class`/
          `import`/`function`/`return`;
  markup  доля символов `<>|#*` — html, таблицы, заголовки markdown;
  alpha   доля букв — сколько остаётся собственно языка;
  eot     сколько документов в среднем начинается внутри окна (граница `<|endoftext|>`).

Признаки нарочно тупые: они не «понимают» текст и потому одинаково несправедливы
ко всем группам, что и делает их сравнимыми.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from tokenizers import Tokenizer

from model.data import DataConfig, MixedLoader, load_manifest

TOKENIZER = Path("tokenizers/bpe_16384.json")
CODE_RE = re.compile(r"```|^\s*(def |class |import |from \w+ import|function |return )",
                     re.MULTILINE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--group", nargs="*", default=None)
    p.add_argument("--windows", type=int, default=400)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--root", type=Path, default=Path("data/tokens"))
    return p.parse_args()


def share(text: str, chars: str) -> float:
    return sum(text.count(c) for c in chars) / max(1, len(text))


def main() -> None:
    a = parse_args()
    tok = Tokenizer.from_file(str(TOKENIZER))
    eot = load_manifest(a.root)["eot_id"]
    groups = a.group or sorted(load_manifest(a.root)["groups"])

    print(f"{a.windows} окон по {a.window} токенов на группу\n")
    print(f"{'группа':<9} {'latex':>7} {'markup':>7} {'alpha':>7} {'код':>7} "
          f"{'док/окно':>9} {'симв/ток':>9}")
    for g in groups:
        cfg = DataConfig(root=a.root, mix={g: 1.0}, seq_len=a.window,
                         batch_size=1, seed=a.seed)
        loader = MixedLoader(cfg, "train")
        latex = markup = alpha = chars = 0.0
        code = docs = 0
        for i in range(a.windows):
            ids = loader.batch(i)[0][0].tolist()
            text = tok.decode(ids, skip_special_tokens=True)
            latex += share(text, "$\\{}^_&")
            markup += share(text, "<>|#*")
            alpha += sum(c.isalpha() for c in text) / max(1, len(text))
            chars += len(text) / len(ids)
            docs += ids.count(eot)
            code += bool(CODE_RE.search(text))
        n = a.windows
        print(f"{g:<9} {latex / n:>7.3f} {markup / n:>7.3f} {alpha / n:>7.3f} "
              f"{code / n:>7.3f} {docs / n:>9.2f} {chars / n:>9.2f}")


if __name__ == "__main__":
    main()
