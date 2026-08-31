"""Train a BPE tokenizer sweep over vocab sizes.

Design decisions baked in here (each one is a real choice, not a default):

* **No normalizer.** Byte-level BPE round-trips text exactly only if nothing
  rewrites the bytes first. Our corpus is a third code, where an NFC rewrite
  inside a string literal silently changes the data. `my_model` used NFC; this
  is a deliberate departure.

* **Digits split individually**, via a `Split` regex rather than a `Digits`
  pre-tokenizer. Otherwise BPE glues frequent numbers into single tokens: `2024`
  becomes one symbol while `2025` becomes two, and arithmetic degrades into
  memorising number-shaped strings. We pay in sequence length and buy the
  ability to actually learn to count — which matters for a math/reasoning model.

  The regex form matters for a second reason: `Sequence([Digits, ByteLevel])`
  cannot be loaded by gigatoken at all ("Unsupported pre_tokenizer type:
  Sequence (no Split regex found)"), while the `Split`-regex form loads and
  produces byte-identical ids. Same behaviour, one fewer door closed. This is
  also how GPT-4 and Llama-3 do it.

* **`max_token_length=16`.** Caps the `======` / `----------` class of junk that
  web boilerplate otherwise donates to the vocabulary. The eval script prints
  the longest learned tokens so we can see whether the cap is cutting anything
  legitimate out of code.

* **32 special-token slots.** Resizing embeddings on a trained model to bolt on
  a chat template later is painful; reserving slots now costs V*d parameters we
  were spending anyway.

`min_frequency` is left at 2 on purpose. It is a threshold on *merge-pair*
frequency, and greedy BPE consumes pairs in descending frequency order, so with
a corpus this large the threshold cannot bind. The real question it is meant to
answer — "did we learn tokens the corpus does not support?" — is answered by
eval_tokenizers.py, which counts how many vocabulary entries are actually used.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors, trainers

SAMPLE = Path("data/tok_sample")
OUT = Path("tokenizers")

# cl100k's split pattern with one deliberate change: `\p{N}` instead of
# `\p{N}{1,3}`, so every digit becomes its own pre-token and BPE can never merge
# digits. The "grouped" variant restores GPT-2 behaviour (`\p{N}+`, free merging
# inside a number) purely so the ablation can price what we gave up.
_HEAD = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
_TAIL = r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
DIGIT_PATTERNS = {"single": _HEAD + r"\p{N}" + _TAIL, "grouped": _HEAD + r"\p{N}+" + _TAIL}

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|bos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|think|>",
    "<|/think|>",
    "<|response|>",
    "<|/response|>",
    "<|tool|>",
    "<|/tool|>",
    "<|end_of_msg|>",
]
SPECIAL_TOKENS += [f"<|reserved_{i}|>" for i in range(32 - len(SPECIAL_TOKENS))]

MAX_TOKEN_LENGTH = 16
MIN_FREQUENCY = 2


def iter_corpus(max_mb: float | None = None) -> Iterator[str]:
    """Yields documents from every training split, interleaved by file."""
    for path in sorted(SAMPLE.glob("*.train.jsonl")):
        written = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["text"]
                yield text
                if max_mb is not None:
                    written += len(text)
                    if written > max_mb * 1e6:
                        break


def build_tokenizer(split_digits: bool = True) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    pattern = DIGIT_PATTERNS["single" if split_digits else "grouped"]
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
        # use_regex=False: the Split above already did the splitting; leaving
        # ByteLevel's own GPT-2 regex on would split a second time.
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=True)
    return tok


def train_one(vocab_size: int, max_mb: float | None, split_digits: bool = True) -> Path:
    tok = build_tokenizer(split_digits)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=MIN_FREQUENCY,
        max_token_length=MAX_TOKEN_LENGTH,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    start = time.perf_counter()
    tok.train_from_iterator(iter_corpus(max_mb), trainer=trainer)
    elapsed = time.perf_counter() - start

    OUT.mkdir(exist_ok=True)
    suffix = "" if split_digits else "_nodigitsplit"
    path = OUT / f"bpe_{vocab_size}{suffix}.json"
    tok.save(str(path), pretty=False)
    print(f"  vocab={tok.get_vocab_size():>6}  {elapsed / 60:5.1f} min  -> {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-sizes", type=int, nargs="+",
                    default=[12288, 16384, 24576, 32768])
    ap.add_argument("--max-mb", type=float, default=None,
                    help="cap text per source file; BPE training is RAM-hungry")
    ap.add_argument("--no-split-digits", action="store_true",
                    help="ablation: let BPE glue digits together, to price what "
                         "individual-digit splitting costs us in sequence length")
    args = ap.parse_args()

    for vocab_size in args.vocab_sizes:
        print(f"training vocab_size={vocab_size} split_digits={not args.no_split_digits} ...")
        train_one(vocab_size, args.max_mb, split_digits=not args.no_split_digits)


if __name__ == "__main__":
    main()
