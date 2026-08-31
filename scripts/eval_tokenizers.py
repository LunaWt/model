"""Compare trained tokenizers on held-out text and turn compression into cost.

Compression alone does not decide anything: a bigger vocabulary always
compresses better, and always makes the (never-sparse) output head more
expensive. This script measures both sides and reports the product, so the
vocab-size choice comes out of numbers rather than folklore.

Reported per tokenizer:

* bytes/token per source — web vs code vs math, because they diverge sharply
* vocabulary utilisation on held-out text — how many entries are effectively
  dead weight, which is the question `min_frequency` is usually asked to answer
* longest learned tokens — a check that max_token_length is not amputating
  legitimate code identifiers
* relative training cost — tokens needed for the corpus x per-token FLOPs
* logits VRAM at the planned shapes
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

SAMPLE = Path("data/tok_sample")
TOKENIZERS = Path("tokenizers")

# Planned model shape; the head is V*d and is active for every token.
D_MODEL = 640
ACTIVE_TRANSFORMER = 30_000_000
BASELINE_VOCAB = 16384

# Attention is not a weight, so it misses a pure parameter count -- but at 2k
# context it is a quarter of the per-token cost, and leaving it out inflates the
# apparent penalty of a big vocabulary. Only the full-attention (MLA) layers are
# quadratic; KDA layers are linear in T and already priced in their weights.
CONTEXT = 2048
N_FULL_ATTN_LAYERS = 5  # 3 KDA : 1 MLA over ~20 layers
ATTN_UNITS = N_FULL_ATTN_LAYERS * 2 * CONTEXT * D_MODEL


def load_docs(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def measure(tok: Tokenizer, docs: list[str]) -> tuple[float, Counter]:
    """Returns (bytes per token, token-id histogram)."""
    hist: Counter = Counter()
    total_bytes = total_tokens = 0
    for i in range(0, len(docs), 512):
        batch = docs[i : i + 512]
        for text, enc in zip(batch, tok.encode_batch_fast(batch), strict=True):
            hist.update(enc.ids)
            total_bytes += len(text.encode("utf-8"))
            total_tokens += len(enc.ids)
    return total_bytes / total_tokens, hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-gb", type=float, default=12.0,
                    help="raw text size of the planned pretrain corpus")
    args = ap.parse_args()

    eval_sets = {p.name.split(".")[0]: load_docs(p) for p in sorted(SAMPLE.glob("*.eval.jsonl"))}
    print("held-out sets: " + ", ".join(f"{k} ({len(v):,} docs)" for k, v in eval_sets.items()))

    # Sort by vocab size, then by name so ablation variants land next to their
    # baseline instead of in filesystem order.
    paths = sorted(TOKENIZERS.glob("bpe_*.json"), key=lambda p: (int(p.stem.split("_")[1]), p.stem))
    results = []

    for path in paths:
        tok = Tokenizer.from_file(str(path))
        V = tok.get_vocab_size()

        per_source = {}
        merged: Counter = Counter()
        for name, docs in eval_sets.items():
            bpt, hist = measure(tok, docs)
            per_source[name] = bpt
            merged.update(hist)

        vocab = tok.get_vocab()
        by_id = {i: s for s, i in vocab.items()}
        unused = V - len(merged)
        rare = sum(1 for i in range(V) if merged.get(i, 0) < 10)
        longest = sorted(vocab, key=len, reverse=True)[:8]

        # Cost model: fewer tokens (better compression) but a fatter head.
        overall_bpt = sum(per_source.values()) / len(per_source)
        tokens = args.corpus_gb * 1e9 / overall_bpt
        per_token_flops = ACTIVE_TRANSFORMER + V * D_MODEL + ATTN_UNITS
        results.append({
            "name": path.stem.removeprefix("bpe_"),
            "V": V, "per_source": per_source, "overall_bpt": overall_bpt,
            "tokens": tokens, "cost": tokens * per_token_flops,
            "head_M": V * D_MODEL / 1e6, "unused": unused, "rare": rare,
            "longest": longest,
        })

    base = next(r for r in results if r["name"] == str(BASELINE_VOCAB))["cost"]

    print(f"\n{'токенизатор':>20} | {'web':>6} {'code':>6} {'math':>6} | {'токенов':>9} | "
          f"{'голова':>8} | {'стоимость':>9} | {'мёртвых':>8} | {'<10 раз':>8}")
    print("-" * 106)
    for r in results:
        s = r["per_source"]
        print(f"{r['name']:>20} | {s['web']:>6.2f} {s['code']:>6.2f} {s['math']:>6.2f} | "
              f"{r['tokens'] / 1e9:>7.2f}B | {r['head_M']:>6.1f}M | "
              f"{r['cost'] / base:>8.3f}x | {r['unused']:>8,} | {r['rare']:>8,}")
    print("\n(bytes/token — больше значит лучше сжатие; стоимость нормирована на V=16384)")

    print("\nлогиты, GB (bf16 + fp32 копия):")
    print(f"{'V':>6} | " + " | ".join(f"T={T},B={B}" for T, B in ((2048, 2), (2048, 4), (4096, 2))))
    for r in results:
        cells = [f"{r['V'] * T * B * 6 / 1e9:9.2f}" for T, B in ((2048, 2), (2048, 4), (4096, 2))]
        print(f"{r['V']:>6} | " + " | ".join(cells))

    print("\nсамые длинные выученные токены:")
    for r in results:
        print(f"  V={r['V']:>6}: {r['longest']}")

    probe = "Solve 31415926 + 2024. def train(model, lr=3e-4):\n        return model"
    print(f"\nпробная токенизация: {probe!r}")
    for path in paths:
        tok = Tokenizer.from_file(str(path))
        enc = tok.encode(probe)
        print(f"  V={tok.get_vocab_size():>6} ({len(enc.ids):>3} токенов): {enc.tokens}")


if __name__ == "__main__":
    main()
