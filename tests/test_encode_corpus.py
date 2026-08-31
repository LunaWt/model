"""Guards on the token-shard writer.

The training loader will np.memmap these files and trust their shape blindly,
so a silent bug here is a bug that only surfaces as a model that will not learn.
Two things are worth pinning: shards roll over at the configured size (not
somewhere else), and appending to an existing shard continues it rather than
truncating a previous run's work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from encode_corpus import DTYPE, ShardWriter, batched, open_writer  # noqa: E402


def test_shards_roll_over_at_the_configured_size(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, tokens_per_shard=100)
    for _ in range(25):
        writer.write(np.arange(10, dtype=DTYPE))
    writer.close()

    sizes = [p.stat().st_size // 2 for p in sorted(tmp_path.glob("*.bin"))]
    assert sizes[:-1] == [100, 100], sizes
    assert sum(sizes) == 250
    assert writer.total == 250


def test_resuming_appends_instead_of_truncating(tmp_path: Path) -> None:
    first = ShardWriter(tmp_path, tokens_per_shard=1000)
    first.write(np.arange(10, dtype=DTYPE))
    first.close()

    second = ShardWriter(tmp_path, tokens_per_shard=1000, start_index=0)
    second.write(np.arange(10, 20, dtype=DTYPE))
    second.close()

    data = np.fromfile(tmp_path / "0000.bin", dtype=DTYPE)
    assert data.tolist() == list(range(20))


def test_written_tokens_survive_a_memmap_roundtrip(tmp_path: Path) -> None:
    """uint16 is only valid while the vocabulary fits; assert the boundary holds."""
    writer = ShardWriter(tmp_path, tokens_per_shard=1_000_000)
    ids = np.array([0, 1, 16383, 42], dtype=DTYPE)
    writer.write(ids)
    writer.close()

    loaded = np.memmap(tmp_path / "0000.bin", dtype=DTYPE, mode="r")
    assert loaded.tolist() == ids.tolist()
    assert 16383 <= np.iinfo(DTYPE).max


def test_resume_discards_tokens_the_manifest_never_committed(tmp_path: Path) -> None:
    """The real failure mode: a run dies mid-file, so its tokens are on disk but
    unclaimed, and the same input file gets encoded again on resume."""
    crashed = ShardWriter(tmp_path, tokens_per_shard=1000)
    crashed.write(np.arange(100, dtype=DTYPE))  # committed: file finished
    crashed.write(np.arange(100, 180, dtype=DTYPE))  # written, then the run died
    crashed.close()

    writer = open_writer(tmp_path, committed_tokens=100, tokens_per_shard=1000)
    writer.write(np.arange(100, 180, dtype=DTYPE))  # the redone file
    writer.close()

    data = np.fromfile(tmp_path / "0000.bin", dtype=DTYPE)
    assert data.tolist() == list(range(180)), "uncommitted tokens were duplicated"


def test_resume_continues_past_a_full_shard(tmp_path: Path) -> None:
    first = ShardWriter(tmp_path, tokens_per_shard=100)
    for _ in range(10):
        first.write(np.arange(10, dtype=DTYPE))
    first.close()
    assert (tmp_path / "0000.bin").stat().st_size // 2 == 100

    writer = open_writer(tmp_path, committed_tokens=100, tokens_per_shard=100)
    writer.write(np.arange(5, dtype=DTYPE))
    writer.close()
    assert (tmp_path / "0000.bin").stat().st_size // 2 == 100
    assert (tmp_path / "0001.bin").stat().st_size // 2 == 5


def test_resume_refuses_to_run_when_shards_are_missing(tmp_path: Path) -> None:
    """Better to stop than to silently train on a corpus with a hole in it."""
    with pytest.raises(SystemExit, match="shards are missing"):
        open_writer(tmp_path, committed_tokens=500, tokens_per_shard=1000)


def test_batched_drops_empty_documents_and_flushes_remainder() -> None:
    batches = list(batched(iter(["a", "", "b", "c"]), 2))
    assert batches == [["a", "b"], ["c"]]


@pytest.mark.parametrize("n", [1, 3, 7])
def test_batched_never_exceeds_batch_size(n: int) -> None:
    for batch in batched(iter([str(i) for i in range(20)]), n):
        assert len(batch) <= n
