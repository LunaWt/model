"""Загрузчик токенов: склейка шардов, детерминизм, разделение train/val."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model.data import DataConfig, MixedLoader, SlotPermutation, TokenStream


def make_group(root, name: str, shards: list[np.ndarray]) -> None:
    d = root / name
    d.mkdir(parents=True)
    for i, arr in enumerate(shards):
        (d / f"{i:04d}.bin").write_bytes(arr.astype(np.uint16).tobytes())


def test_stream_reads_across_shard_boundaries(tmp_path):
    make_group(tmp_path, "g", [np.arange(0, 100), np.arange(100, 250), np.arange(250, 300)])
    st = TokenStream(tmp_path / "g")
    assert st.total == 300
    assert np.array_equal(st.read(0, 300), np.arange(300))
    assert np.array_equal(st.read(95, 20), np.arange(95, 115))
    assert np.array_equal(st.read(240, 30), np.arange(240, 270))
    with pytest.raises(IndexError):
        st.read(290, 20)


def test_targets_are_inputs_shifted_by_one(tmp_path):
    make_group(tmp_path, "g", [np.arange(1000, dtype=np.uint16)])
    cfg = DataConfig(root=tmp_path, mix={"g": 1.0}, seq_len=16, batch_size=2, val_frac=0.1)
    x, y = MixedLoader(cfg, "train").batch(0)
    assert x.shape == y.shape == (2, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_same_step_gives_the_same_batch_and_different_steps_do_not(tmp_path):
    make_group(tmp_path, "g", [np.arange(5000, dtype=np.uint16) % 60000])
    cfg = DataConfig(root=tmp_path, mix={"g": 1.0}, seq_len=32, batch_size=2, val_frac=0.1)
    a, b = MixedLoader(cfg, "train"), MixedLoader(cfg, "train")
    assert torch.equal(a.batch(7)[0], b.batch(7)[0])
    assert not torch.equal(a.batch(7)[0], a.batch(8)[0])


def test_val_never_overlaps_train(tmp_path):
    make_group(tmp_path, "g", [np.arange(10000, dtype=np.uint16)])
    cfg = DataConfig(root=tmp_path, mix={"g": 1.0}, seq_len=8, batch_size=4, val_frac=0.2)
    train, val = MixedLoader(cfg, "train"), MixedLoader(cfg, "val")
    (t_lo, t_hi), (v_lo, v_hi) = train.ranges["g"], val.ranges["g"]
    assert t_hi == v_lo and t_lo == 0 and v_hi == 10000
    # содержимое: токены = индексы, значит по значениям видно, откуда взято
    assert train.batch(0)[0].max().item() < t_hi
    assert val.batch(0)[0].min().item() >= v_lo


def test_mix_weights_are_respected_over_many_draws(tmp_path):
    make_group(tmp_path, "a", [np.zeros(4000, dtype=np.uint16)])
    make_group(tmp_path, "b", [np.ones(4000, dtype=np.uint16)])
    cfg = DataConfig(root=tmp_path, mix={"a": 0.75, "b": 0.25}, seq_len=8,
                     batch_size=64, val_frac=0.1)
    loader = MixedLoader(cfg, "train")
    xs = torch.cat([loader.batch(s)[0] for s in range(20)])
    share_b = xs[:, 0].float().mean().item()
    assert 0.20 < share_b < 0.30, share_b


def test_slot_permutation_is_a_bijection():
    for n in (1, 2, 3, 17, 64, 1000):
        perm = SlotPermutation(n, key=12345)
        assert sorted(perm(i) for i in range(n)) == list(range(n)), n


def test_slot_permutation_reorders_and_depends_on_the_key():
    n = 4096
    a = [SlotPermutation(n, key=1)(i) for i in range(n)]
    b = [SlotPermutation(n, key=2)(i) for i in range(n)]
    assert a != list(range(n)) and a != b


def test_epoch_covers_every_slot_exactly_once(tmp_path):
    # 40 слотов по 9 токенов: значение первого токена = номер слота
    make_group(tmp_path, "g", [(np.arange(360, dtype=np.uint16) // 9)])
    cfg = DataConfig(root=tmp_path, mix={"g": 1.0}, seq_len=8, batch_size=4,
                     val_frac=0.0, block=8)
    loader = MixedLoader(cfg, "train")
    assert loader.slots["g"] == 40
    seen = [int(x[0]) for s in range(10) for x in loader.batch(s)[0]]
    assert sorted(seen) == list(range(40))


def test_next_epoch_repeats_the_slots_in_a_different_order(tmp_path):
    make_group(tmp_path, "g", [(np.arange(360, dtype=np.uint16) // 9)])
    cfg = DataConfig(root=tmp_path, mix={"g": 1.0}, seq_len=8, batch_size=4,
                     val_frac=0.0, block=8)
    loader = MixedLoader(cfg, "train")
    first = [int(x[0]) for s in range(10) for x in loader.batch(s)[0]]
    second = [int(x[0]) for s in range(10, 20) for x in loader.batch(s)[0]]
    assert sorted(second) == list(range(40))
    assert first != second


def test_group_shares_are_exact_on_each_block(tmp_path):
    make_group(tmp_path, "a", [np.zeros(4000, dtype=np.uint16)])
    make_group(tmp_path, "b", [np.ones(4000, dtype=np.uint16)])
    cfg = DataConfig(root=tmp_path, mix={"a": 0.75, "b": 0.25}, seq_len=8,
                     batch_size=10, val_frac=0.1, block=100)
    loader = MixedLoader(cfg, "train")
    xs = torch.cat([loader.batch(s)[0] for s in range(10)])   # ровно один блок
    assert int(xs[:, 0].sum().item()) == 25


def test_too_short_a_group_fails_loudly(tmp_path):
    make_group(tmp_path, "g", [np.arange(20, dtype=np.uint16)])
    cfg = DataConfig(root=tmp_path, mix={"g": 1.0}, seq_len=64, batch_size=1)
    with pytest.raises(ValueError, match="меньше одной последовательности"):
        MixedLoader(cfg, "train")
