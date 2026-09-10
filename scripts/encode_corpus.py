"""Encode the raw corpus into flat uint16 token shards, resumably.

Layout: data/tokens/<group>/<NNNN>.bin, plus a manifest recording exact token
counts per group. Shards are raw little-endian uint16 with no header, so
training reads them with np.memmap and zero parsing.

uint16 is safe only because V=16384 < 65536, and it halves both disk and the
page cache the training loader depends on. The manifest records the dtype so a
later vocab change fails loudly instead of reading garbage.

Documents are separated by <|endoftext|> so the model learns where text ends;
without it, concatenated shards teach the model that documents run together.

Resumability is per input file: the manifest lists files already encoded, and an
interrupted file is simply redone. Groups stay in separate directories because
the pretrain mix is decided *after* we know the real token counts, not before.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zstandard

from tokenizers import Tokenizer

RAW = Path("data/raw")
OUT = Path("data/tokens")
TOKENIZER = Path("tokenizers/bpe_16384.json")

DTYPE = np.uint16
TOKENS_PER_SHARD = 100_000_000  # ~200 MB per shard on disk
BATCH_DOCS = 1000


def groups() -> dict[str, list[Path]]:
    return {
        "web": sorted((RAW / "web_mix_50_30_20/data").glob("*.parquet")),
        "code": (
            sorted((RAW / "dolma35/dolma_code/python_quality_p95").glob("*.jsonl.zst"))
            + sorted((RAW / "dolma35/swallow-code").glob("*.jsonl.gz"))
        ),
        "math": sorted(
            (RAW / "dolma35/swallow-math/stage3-qa_decon_ngram_filtered").glob("*.jsonl.gz")
        ),
        "synth": sorted((RAW / "synth").glob("*.jsonl.gz")),
        "science": sorted((RAW / "arxiv_cp").glob("*.json.gz")),
    }


def read_docs(path: Path) -> Iterator[str]:
    if path.suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_DOCS, columns=["text"]):
            yield from batch.column("text").to_pylist()
    elif path.name.endswith(".jsonl.zst"):
        with path.open("rb") as fh:
            reader = zstandard.ZstdDecompressor().stream_reader(fh)
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                yield json.loads(line)["text"]
    elif path.name.endswith((".jsonl.gz", ".json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["text"]
    else:
        raise ValueError(f"unknown format: {path}")


def batched(it: Iterator[str], n: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for item in it:
        if item:
            batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


class ShardWriter:
    """Streams tokens into fixed-size shards, rolling over as they fill."""

    def __init__(self, out_dir: Path, tokens_per_shard: int, start_index: int = 0):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tokens_per_shard = tokens_per_shard
        self.index = start_index
        self.written_in_shard = 0
        self.total = 0
        self.shards: list[str] = []
        self._fh = None

    def _open(self) -> None:
        path = self.out_dir / f"{self.index:04d}.bin"
        self._fh = path.open("ab")
        self.shards.append(path.name)
        self.written_in_shard = path.stat().st_size // 2

    def write(self, ids: np.ndarray) -> None:
        if self._fh is None:
            self._open()
        assert self._fh is not None
        self._fh.write(ids.tobytes())
        self.written_in_shard += len(ids)
        self.total += len(ids)
        if self.written_in_shard >= self.tokens_per_shard:
            self._fh.close()
            self._fh = None
            self.index += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def open_writer(out_dir: Path, committed_tokens: int, tokens_per_shard: int) -> ShardWriter:
    """Position a writer at the end of *committed* data, discarding the rest.

    The manifest only records a file once it finishes, but shards are appended
    to continuously. So an interrupted run leaves tokens on disk that no
    manifest entry claims, and the file that produced them will be encoded
    again on resume — duplicating them. Truncate back to what was committed
    before writing anything new.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    remaining = committed_tokens
    resume_index = 0

    for path in sorted(out_dir.glob("*.bin")):
        have = path.stat().st_size // 2
        keep = min(have, remaining)
        if keep < have:
            with path.open("r+b") as fh:
                fh.truncate(keep * 2)
        remaining -= keep
        # Continue into this shard if it still has room, otherwise past it.
        resume_index = int(path.stem) if keep < tokens_per_shard else int(path.stem) + 1

    if remaining:
        raise SystemExit(
            f"{out_dir}: manifest claims {committed_tokens:,} tokens but only "
            f"{committed_tokens - remaining:,} are on disk — shards are missing"
        )
    return ShardWriter(out_dir, tokens_per_shard, start_index=resume_index)


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"vocab_size": None, "dtype": str(np.dtype(DTYPE)), "groups": {}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["web", "code", "math"])
    ap.add_argument("--limit-files", type=int, default=None,
                    help="encode only the first N files per group (smoke test)")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(TOKENIZER))
    vocab_size = tok.get_vocab_size()
    if vocab_size > np.iinfo(DTYPE).max + 1:
        raise SystemExit(f"vocab {vocab_size} does not fit in {np.dtype(DTYPE)}")
    eot = tok.token_to_id("<|endoftext|>")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = load_manifest(manifest_path)
    manifest["vocab_size"] = vocab_size
    manifest["eot_id"] = eot

    all_groups = groups()
    for group in args.groups:
        files = all_groups[group]
        if args.limit_files:
            files = files[: args.limit_files]

        state = manifest["groups"].setdefault(group, {"files_done": [], "tokens": 0, "shards": []})
        done = set(state["files_done"])
        todo = [f for f in files if f.name not in done]
        if not todo:
            print(f"{group}: nothing to do ({len(done)} files already encoded)")
            continue

        writer = open_writer(OUT / group, state["tokens"], TOKENS_PER_SHARD)
        print(f"{group}: {len(todo)} files to encode ({len(done)} already done)")

        for path in todo:
            start = time.perf_counter()
            before = writer.total
            for batch in batched(read_docs(path), BATCH_DOCS):
                encodings = tok.encode_batch_fast(batch)
                flat: list[int] = []
                for enc in encodings:
                    flat.extend(enc.ids)
                    flat.append(eot)
                writer.write(np.asarray(flat, dtype=DTYPE))

            produced = writer.total - before
            state["files_done"].append(path.name)
            state["tokens"] += produced
            state["shards"] = sorted({*state["shards"], *writer.shards})
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"  {path.name}: {produced / 1e6:8.1f}M tokens "
                  f"in {time.perf_counter() - start:5.1f}s  (group total {state['tokens'] / 1e9:.3f}B)")

        writer.close()
        manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\nитог:")
    for group, state in manifest["groups"].items():
        print(f"  {group:>5}: {state['tokens'] / 1e9:6.3f}B токенов, "
              f"{len(state['shards'])} шардов, {len(state['files_done'])} файлов")
    total = sum(s["tokens"] for s in manifest["groups"].values())
    print(f"  всего: {total / 1e9:.3f}B токенов")


if __name__ == "__main__":
    main()
