"""Загрузчик токенов: memmap по шардам, смесь групп, детерминированный порядок.

Шарды лежат как `data/tokens/<group>/NNNN.bin` — плоский little-endian uint16 без
заголовка (см. `scripts/encode_corpus.py`). Внутри группы шарды склеиваются в один
виртуальный поток; окно длины T+1 может пересекать границу шарда.

# Как выбирается последовательность

Поток группы нарезан на непересекающиеся слоты по T+1 токенов. Порядок обхода —
случайная перестановка слотов: до конца эпохи каждый слот выдаётся ровно один раз,
повторы начинаются только со следующей эпохи и в другом порядке. Перестановка не
материализуется, а считается на лету сетью Фейстеля (`SlotPermutation`), поэтому
её стоимость не зависит от числа слотов, а их у нас ~15 млн.

Группа выбирается не броском монеты, а расписанием с точной пропорцией на каждом
блоке из `block` последовательностей (`GroupSchedule`). Разница видна на коротких
прогонах: при B=2 и 250 шагах это 500 последовательностей, и мультиномиальный
розыгрыш промахивается мимо заданной смеси на единицы процентов, что уже сравнимо
с эффектами, которые мы измеряем.

# Детерминизм

`batch(step)` — чистая функция от (seed, step): номер слота и группа считаются из
глобального индекса последовательности j = step·B + b. Ничего не надо сохранять в
чекпоинт, кроме номера шага, и возобновление с шага N даёт ровно тот же поток.

Валидация — хвост каждой группы (`val_frac`), в обучение он не попадает никогда.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

DTYPE = np.uint16
M64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def _mix64(x: int) -> int:
    """splitmix64: перемешивает 64 бита. Нужна только как раундовая функция Фейстеля."""
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & M64
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & M64
    return (x ^ (x >> 31)) & M64


class SlotPermutation:
    """Псевдослучайная биекция [0, n) -> [0, n) за O(1) без памяти.

    Сеть Фейстеля на 2b битах — биекция на [0, 2^2b) при любой раундовой функции,
    потому что каждый раунд обратим. Домен округлён вверх до степени двойки, лишние
    значения отсеиваются «прогулкой по циклу»: пока результат >= n, прогоняем ещё
    раз. Так как отображение биективно, орбита конечна и обязательно содержит
    значение < n; при n > 2^2b / 2 в среднем нужно меньше двух проходов.

    Альтернатива — хранить перестановку целиком: 15 млн слотов это 120 МБ на группу
    и пересборка при каждой смене T.
    """

    def __init__(self, n: int, key: int, rounds: int = 4):
        if n < 1:
            raise ValueError(f"пустой домен перестановки: n={n}")
        self.n = n
        self.key = key & M64
        self.rounds = rounds
        self.half = max(1, ((n - 1).bit_length() + 1) // 2)
        self.mask = (1 << self.half) - 1

    def _round_trip(self, x: int) -> int:
        left, right = x >> self.half, x & self.mask
        for k in range(self.rounds):
            f = _mix64(right ^ self.key ^ (k * GOLDEN)) & self.mask
            left, right = right, left ^ f
        return (left << self.half) | right

    def __call__(self, i: int) -> int:
        y = i % self.n
        for _ in range(64):
            y = self._round_trip(y)
            if y < self.n:
                return y
        raise RuntimeError(f"прогулка по циклу не сошлась: n={self.n}, i={i}")


def _largest_remainder(probs: np.ndarray, total: int) -> np.ndarray:
    """Разложить `total` мест по весам так, чтобы сумма была ровно `total`."""
    exact = probs * total
    counts = np.floor(exact).astype(np.int64)
    for idx in np.argsort(-(exact - counts))[: total - int(counts.sum())]:
        counts[idx] += 1
    return counts


class GroupSchedule:
    """Из какой группы брать j-ю последовательность и какая она в этой группе по счёту.

    Блок из `block` последовательностей содержит ровно `round(w_g · block)` мест
    каждой группы, перемешанных ключом (seed, номер блока). Отсюда порядковый номер
    внутри группы считается точно: полные блоки до текущего плюс сколько таких мест
    встретилось в текущем блоке левее j.
    """

    def __init__(self, n_groups: int, probs: np.ndarray, block: int, seed: int):
        self.counts = _largest_remainder(probs, block)
        if (self.counts == 0).any():
            raise ValueError(
                f"группа получила 0 мест в блоке из {block}: веса {probs.tolist()} "
                f"слишком неравные, увеличь block"
            )
        self.base = np.repeat(np.arange(n_groups), self.counts)
        self.block = block
        self.seed = seed
        self.n_groups = n_groups
        self._cached_block = -1
        self._pattern = np.empty(0, dtype=np.int64)
        self._prefix = np.empty((0, 0), dtype=np.int64)

    def _build(self, b: int) -> None:
        if b == self._cached_block:
            return
        gen = torch.Generator().manual_seed(int(_mix64(self.seed ^ (b * GOLDEN)) >> 1))
        order = torch.randperm(self.block, generator=gen).numpy()
        pattern = self.base[order]
        onehot = np.zeros((self.block, self.n_groups), dtype=np.int64)
        onehot[np.arange(self.block), pattern] = 1
        self._pattern = pattern
        self._prefix = np.cumsum(onehot, axis=0) - onehot
        self._cached_block = b

    def __call__(self, j: int) -> tuple[int, int]:
        b, p = divmod(j, self.block)
        self._build(b)
        g = int(self._pattern[p])
        return g, b * int(self.counts[g]) + int(self._prefix[p, g])


class TokenStream:
    """Одна группа: список шардов как один непрерывный поток uint16."""

    def __init__(self, group_dir: Path):
        self.paths = sorted(group_dir.glob("*.bin"))
        if not self.paths:
            raise FileNotFoundError(f"нет шардов в {group_dir}")
        self.sizes = [p.stat().st_size // 2 for p in self.paths]
        self.starts = np.cumsum([0] + self.sizes)
        self.total = int(self.starts[-1])
        self._maps: list[np.memmap | None] = [None] * len(self.paths)

    def _shard(self, i: int) -> np.memmap:
        m = self._maps[i]
        if m is None:
            m = np.memmap(self.paths[i], dtype=DTYPE, mode="r")
            self._maps[i] = m
        return m

    def read(self, offset: int, length: int) -> np.ndarray:
        if offset + length > self.total:
            raise IndexError(f"{offset}+{length} > {self.total}")
        out = np.empty(length, dtype=DTYPE)
        written = 0
        i = int(np.searchsorted(self.starts, offset, side="right") - 1)
        while written < length:
            local = offset + written - int(self.starts[i])
            take = min(length - written, self.sizes[i] - local)
            out[written:written + take] = self._shard(i)[local:local + take]
            written += take
            i += 1
        return out


@dataclass
class DataConfig:
    root: Path = Path("data/tokens")
    mix: dict[str, float] = field(default_factory=lambda: {"web": 0.6, "math": 0.25, "code": 0.15})
    seq_len: int = 1536
    batch_size: int = 1
    val_frac: float = 0.002
    seed: int = 1234
    block: int = 1000


def natural_mix(root: Path = Path("data/tokens")) -> dict[str, float]:
    """Веса пропорционально числу токенов в группе — «всё подряд, но вперемешку»."""
    groups = load_manifest(root)["groups"]
    tokens = {g: int(v["tokens"] if isinstance(v, dict) else v) for g, v in groups.items()}
    total = sum(tokens.values())
    return {g: n / total for g, n in sorted(tokens.items(), key=lambda kv: -kv[1])}


class MixedLoader:
    """Батчи (x, y) формы (B, T) int64 из смеси групп, без повторов внутри эпохи.

    `split="train"` читает голову каждого потока, `split="val"` — хвост длиной
    `val_frac`. Границу считаем один раз по реальным размерам шардов.
    """

    def __init__(self, cfg: DataConfig, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.names = list(cfg.mix)
        self.streams = {g: TokenStream(cfg.root / g) for g in self.names}

        self.ranges: dict[str, tuple[int, int]] = {}
        self.slots: dict[str, int] = {}
        for g, st in self.streams.items():
            cut = int(st.total * (1.0 - cfg.val_frac))
            lo, hi = (0, cut) if split == "train" else (cut, st.total)
            n = (hi - lo) // (cfg.seq_len + 1)
            if n < 1:
                raise ValueError(f"группа {g}: в срезе {split} меньше одной последовательности")
            self.ranges[g] = (lo, hi)
            self.slots[g] = n

        w = np.array([cfg.mix[g] for g in self.names], dtype=np.float64)
        self.probs = w / w.sum()
        self._offset = 0 if split == "train" else 1 << 40
        self.schedule = GroupSchedule(len(self.names), self.probs, cfg.block,
                                      cfg.seed + self._offset)
        self.perms = {
            g: SlotPermutation(self.slots[g], _mix64(cfg.seed + self._offset + i * GOLDEN))
            for i, g in enumerate(self.names)
        }

    def batch(self, step: int, device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        rows = np.empty((cfg.batch_size, cfg.seq_len + 1), dtype=np.int64)
        for b in range(cfg.batch_size):
            g, seen = self.schedule(step * cfg.batch_size + b)
            name = self.names[g]
            n = self.slots[name]
            epoch, within = divmod(seen, n)
            # новая эпоха — другая перестановка тех же слотов
            perm = self.perms[name] if epoch == 0 else SlotPermutation(
                n, _mix64(self.perms[name].key ^ ((epoch + 1) * GOLDEN)))
            off = self.ranges[name][0] + perm(within) * (cfg.seq_len + 1)
            rows[b] = self.streams[name].read(off, cfg.seq_len + 1).astype(np.int64)

        t = torch.from_numpy(rows)
        x = t[:, :-1].contiguous().to(device, non_blocking=True)
        y = t[:, 1:].contiguous().to(device, non_blocking=True)
        return x, y

    def token_counts(self) -> dict[str, int]:
        return {g: hi - lo for g, (lo, hi) in self.ranges.items()}

    def epoch_steps(self) -> dict[str, float]:
        """Сколько шагов обучения до того, как группа пойдёт на второй круг."""
        per_block = dict(zip(self.names, self.schedule.counts.tolist()))
        return {g: self.slots[g] / per_block[g] * self.cfg.block / self.cfg.batch_size
                for g in self.names}


def load_manifest(root: Path = Path("data/tokens")) -> dict:
    return json.loads((root / "manifest.json").read_text())
