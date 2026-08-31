"""Build a text sample for tokenizer training + a held-out sample for evaluation.

Why a sample at all: BPE merge statistics saturate long before the full corpus.
A ~1 GB sample gives merges that are indistinguishable from full-corpus merges,
and it lets us sweep four vocab sizes in minutes instead of hours.

Why per-source files: the whole point of the sweep is to see compression
(bytes/token) *separately* for web / code / math. Code and LaTeX behave very
differently from prose, and a single averaged number would hide exactly the
tradeoff we're deciding on.

Held-out shards are reserved per source so evaluation never sees training text.

Output: data/tok_sample/{web,code,math}.{train,eval}.jsonl  ({"text": ...} per line)
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
import zstandard

RAW = Path("data/raw")
OUT = Path("data/tok_sample")

# Provisional pretrain mix. Not final — the tokenizer is only mildly sensitive
# to these ratios, but they should be in the right ballpark.
TARGETS_MB = {"web": 350, "code": 250, "math": 400}
EVAL_MB = 20


def _read_jsonl_gz(path: Path) -> Iterator[str]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def _read_jsonl_zst(path: Path) -> Iterator[str]:
    with path.open("rb") as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            yield json.loads(line)["text"]


def _read_parquet(path: Path) -> Iterator[str]:
    # iter_batches keeps memory flat; these shards are ~2.4 GB each.
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1000, columns=["text"]):
        yield from batch.column("text").to_pylist()


def _reader(path: Path) -> Iterator[str]:
    if path.suffix == ".parquet":
        return _read_parquet(path)
    if path.name.endswith(".jsonl.zst"):
        return _read_jsonl_zst(path)
    if path.name.endswith(".jsonl.gz"):
        return _read_jsonl_gz(path)
    raise ValueError(f"unknown format: {path}")


def sources() -> dict[str, tuple[list[Path], list[Path]]]:
    """Returns {group: (train_shards, eval_shards)}. Eval shards are disjoint."""
    web = sorted((RAW / "web_mix_50_30_20/data").glob("*.parquet"))
    dolma_code = sorted((RAW / "dolma35/dolma_code/python_quality_p95").glob("*.jsonl.zst"))
    swallow_code = sorted((RAW / "dolma35/swallow-code").glob("*.jsonl.gz"))
    math = sorted((RAW / "dolma35/swallow-math/stage3-qa_decon_ngram_filtered").glob("*.jsonl.gz"))

    for name, shards in (("web", web), ("dolma_code", dolma_code),
                         ("swallow_code", swallow_code), ("math", math)):
        if len(shards) < 2:
            raise SystemExit(f"{name}: need >=2 shards to hold one out, found {len(shards)}")

    return {
        # Interleave the two code corpora so neither dominates the sample.
        "code": (dolma_code[:-1] + swallow_code[:-1], [dolma_code[-1], swallow_code[-1]]),
        "web": (web[:-1], [web[-1]]),
        "math": (math[:-1], [math[-1]]),
    }


def write_sample(shards: list[Path], out_path: Path, target_bytes: int) -> tuple[int, int]:
    """Round-robin across shards so one source never dominates. Returns (docs, bytes)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    readers = [_reader(p) for p in shards]
    written = docs = 0

    with out_path.open("w", encoding="utf-8") as out:
        while readers and written < target_bytes:
            for reader in list(readers):
                try:
                    text = next(reader)
                except StopIteration:
                    readers.remove(reader)
                    continue
                if not text:
                    continue
                out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                written += len(text.encode("utf-8"))
                docs += 1
                if written >= target_bytes:
                    break
    return docs, written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply all targets (0.1 for a quick smoke run)")
    args = ap.parse_args()

    for group, (train_shards, eval_shards) in sources().items():
        for split, shards, mb in (
            ("train", train_shards, TARGETS_MB[group]),
            ("eval", eval_shards, EVAL_MB),
        ):
            path = OUT / f"{group}.{split}.jsonl"
            docs, written = write_sample(shards, path, int(mb * 1e6 * args.scale))
            print(f"{group:>5} {split:<5} {docs:>8,} docs  {written / 1e6:>7.1f} MB  -> {path}")


if __name__ == "__main__":
    main()
