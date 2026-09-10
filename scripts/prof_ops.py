"""Разобрать xplane от `bench_moe_jax.py --profile`: куда ушёл шаг, по операциям HLO.

    uv run python -m scripts.prof_ops ~/.claude/scratch/kg/<прогон>/out/prof

Дифференциальный замер («убрали оптимизатор — стало на столько быстрее») стоит
компиляции на каждую точку и не различает то, что XLA слил в один fusion. Здесь
берётся готовая разбивка по инструкциям.

Три ловушки этого формата, каждая из которых даёт неверную сумму:

  * планов восемь, по одному на чип, и складывать их нельзя — время у них одно и
    то же, параллельное. Считается один план, печатается разброс между ними: он и
    есть цена дисбаланса маршрутизации;
  * линий несколько, и `Steps`, `XLA Modules`, `XLA Ops` описывают один и тот же
    шаг с разной подробностью. Берётся только `XLA Ops`;
  * `while` и `conditional` — обёртки: их длительность включает всё, что внутри, а
    внутренние операции лежат на той же линии отдельно. В сумму листьев они не
    идут, печатаются отдельной строкой.

⚠ Группировать по корню имени можно, но верить ей как «столько ушло на матмулы»
нельзя: XLA называет fusion по корневой операции, и матмул, слитый со сложением,
станет `convolution_add_fusion`, а слитый с чем-то ещё — просто `fusion.NNN`.
Коллективы и scatter/gather по имени видны честно, арифметика — нет; её приходится
опознавать по формам операндов в списке инструкций ниже.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

from jax.profiler import ProfileData

NEST = ("while", "conditional")


def _kind(name: str) -> str:
    m = re.match(r"%?([A-Za-z0-9_.\-]+?)(?:\.\d+)?\s*=", name)
    return m.group(1) if m else name.split()[0].lstrip("%")


def collect(path: Path):
    """(план → {инструкция: [вызовов, нс, категория]}, разное) по одному файлу."""
    out, nest, async_ms, seen = {}, {}, {}, []
    pd = ProfileData.from_file(str(path))
    for plane in pd.planes:
        low = plane.name.lower()
        if not any(t in low for t in ("device", "tpu", "gpu")):
            continue
        acc, wrap = {}, {}
        for line in plane.lines:
            seen.append((plane.name, line.name, sum(1 for _ in line.events)))
            if line.name == "Async XLA Ops":
                async_ms[plane.name] = sum(e.duration_ns for e in line.events) / 1e6
            if line.name != "XLA Ops":
                continue
            for ev in line.events:
                kind = _kind(ev.name)
                tgt = wrap if kind in NEST else acc
                rec = tgt.setdefault(ev.name, [0, 0.0, kind])
                rec[0] += 1
                rec[1] += ev.duration_ns
        if acc:
            out[plane.name] = acc
            nest[plane.name] = wrap
    return out, nest, async_ms, seen


def report(path: Path, top: int) -> None:
    planes, nest, async_ms, seen = collect(path)
    if not planes:
        print(f"{path.name}: линии `XLA Ops` не нашлось. Что есть в файле:")
        for pl, ln, n in seen[:40]:
            print(f"  {pl} | {ln} | событий {n}")
        return
    names = sorted(planes)
    busy = {n: sum(v[1] for v in planes[n].values()) / 1e6 for n in names}
    lo, hi = min(busy.values()), max(busy.values())
    first = planes[names[0]]
    total = busy[names[0]]
    print(f"\n=== {path.parent.parent.parent.name} ===")
    print(f"планов {len(names)}, листья {lo:.1f}–{hi:.1f} мс на план "
          f"(разброс {100 * (hi - lo) / max(hi, 1e-9):.1f}%), "
          f"обёртки while/cond {sum(v[1] for v in nest[names[0]].values()) / 1e6:.1f} мс, "
          f"асинхронные {async_ms.get(names[0], 0.0):.1f} мс")

    kinds = collections.defaultdict(lambda: [0, 0.0])
    for cnt, ns, kind in first.values():
        kinds[kind][0] += cnt
        kinds[kind][1] += ns
    print(f"\n{'корень инструкции':<40}{'мс':>9}{'%':>7}{'вызовов':>9}")
    for key, (cnt, ns) in sorted(kinds.items(), key=lambda kv: -kv[1][1])[:20]:
        print(f"{key[:39]:<40}{ns / 1e6:9.1f}"
              f"{100 * ns / 1e6 / max(total, 1e-9):7.1f}{cnt:9d}")

    print(f"\n{'инструкция':<64}{'мс':>9}{'%':>7}{'вызовов':>9}")
    for name, (cnt, ns, _) in sorted(first.items(), key=lambda kv: -kv[1][1])[:top]:
        short = re.sub(r"\s+", " ", name)[:62]
        print(f"{short:<64}{ns / 1e6:9.1f}"
              f"{100 * ns / 1e6 / max(total, 1e-9):7.1f}{cnt:9d}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    p.add_argument("--top", type=int, default=25)
    a = p.parse_args(argv)
    files = sorted(a.path.rglob("*.xplane.pb")) if a.path.is_dir() else [a.path]
    if not files:
        sys.exit(f"в {a.path} нет ни одного *.xplane.pb")
    for f in files:
        report(f, a.top)


if __name__ == "__main__":
    main()
