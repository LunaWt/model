"""Guards on the properties we deliberately chose for the tokenizer.

Each of these encodes a decision from the vocab sweep, and each fails silently
if broken: text that no longer round-trips corrupts the corpus at encode time,
digits that re-glue quietly turn arithmetic back into memorisation, and special
tokens that shift ids invalidate every checkpoint trained before the change.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer

TOKENIZER_PATH = Path("tokenizers/bpe_16384.json")

pytestmark = pytest.mark.skipif(
    not TOKENIZER_PATH.exists(),
    reason="run scripts/train_tokenizers.py first",
)


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.from_file(str(TOKENIZER_PATH))


@pytest.mark.parametrize("text", [
    "Solve 31415926 + 2024",
    "def f(n):\n    return n * 2  # комментарий",
    "\t mixed\ttabs \r\n and CRLF",
    "emoji 🙂 and mathematics ∫ x² dx = x³/3",
    '{"json": [1, 2.5, null], "esc": "quote\\"inside"}',
])
def test_roundtrip_is_lossless(tok: Tokenizer, text: str) -> None:
    """No normalizer, so byte-level BPE must reproduce the input exactly.

    Adding an NFC/NFKC normalizer would break this for composed Unicode, which
    is why we left it out: a third of the corpus is code, where a rewrite inside
    a string literal silently changes the data.
    """
    assert tok.decode(tok.encode(text).ids) == text


def test_digits_are_split_individually(tok: Tokenizer) -> None:
    """Every digit must be its own token, so numbers are composed not memorised."""
    tokens = tok.encode("value 31415926 and 2024").tokens
    for run in ("31415926", "2024"):
        assert all(d in tokens for d in run)
        assert not any(len(t.strip("Ġ")) > 1 and t.strip("Ġ").isdigit() for t in tokens)


def test_no_multi_digit_tokens_in_vocab(tok: Tokenizer) -> None:
    """Stronger than the encode check: the vocabulary itself must not contain them."""
    multi = [t for t in tok.get_vocab() if len(t.lstrip("Ġ")) > 1 and t.lstrip("Ġ").isdigit()]
    assert multi == []


def test_special_tokens_have_stable_low_ids(tok: Tokenizer) -> None:
    """Ids are frozen by any trained checkpoint; reordering them is a silent break."""
    vocab = tok.get_vocab()
    for expected_id, name in enumerate(["<|endoftext|>", "<|pad|>", "<|bos|>"]):
        assert vocab[name] == expected_id
    for name in ["<|think|>", "<|/think|>", "<|end_of_msg|>"]:
        assert vocab[name] < 32


def test_reserved_slots_exist(tok: Tokenizer) -> None:
    """Spare slots let us add chat-template tokens without resizing embeddings."""
    reserved = [t for t in tok.get_vocab() if t.startswith("<|reserved_")]
    assert len(reserved) >= 10


def test_max_token_length_respected(tok: Tokenizer) -> None:
    """Caps the `======` class of web-boilerplate junk. Specials are exempt."""
    longest = max(
        (t for t in tok.get_vocab() if not t.startswith("<|")),
        key=len,
    )
    assert len(longest) <= 16, longest


def test_every_byte_is_representable(tok: Tokenizer) -> None:
    """ByteLevel initial alphabet means no input can ever produce UNK."""
    raw = bytes(range(256)).decode("latin-1")
    assert tok.decode(tok.encode(raw).ids) == raw


def test_gigatoken_loads_and_agrees(tok: Tokenizer) -> None:
    """gigatoken only accepts pre-tokenizers it has a fast path for.

    `Sequence([Digits, ByteLevel])` is rejected outright, and even among Split
    regexes it matches against a known set — `\\p{N}+` is refused while our
    single-digit `\\p{N}` is accepted. So this is fragile in exactly one way:
    editing the split pattern can silently cost us the fast encoder. Fail here
    instead of at corpus-encoding time.
    """
    gigatoken = pytest.importorskip("gigatoken")
    fast = gigatoken.Tokenizer(str(TOKENIZER_PATH))
    for text in [
        "Solve 31415926 + 2024 exactly",
        "def f(n):\n    return n * 2\n",
        "∫ x² dx and some prose to merge",
    ]:
        assert fast.encode(text).tolist() == tok.encode(text).ids, text
