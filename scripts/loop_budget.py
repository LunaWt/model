"""Бюджеты конфигов для трека looped (notes/looped.md).

Печатает таблицу из notes/looped.md: total-параметры, «исполненные» параметры
(прокси FLOPs на токен), число KV-слотов и проверку, что последний слой — MLA.
Числа в заметке взяты отсюда; если конфиг меняется, перегенерировать:

    uv run python -m scripts.loop_budget
"""

from __future__ import annotations

from model.configs import CONFIGS
from model.model import GatedMLA, K3Model


def analyse(cfg) -> dict:
    m = K3Model(cfg)
    n = m.count_params()
    return {
        "exe": len(m.execution_order),
        "routed": cfg.n_routed,
        "span": cfg.loop_span,
        "total": n["total"],
        "executed": n["executed"],
        "kv": sum(1 for i in m.execution_order if isinstance(m.attn[i], GatedMLA)),
        "last_is_mla": isinstance(m.attn[cfg.n_layers - 1], GatedMLA),
    }


TRACK = ["A16", "L12_4x2", "A24", "L16_8x2", "L12_6x3"]


def main() -> None:
    rows = [(name, analyse(CONFIGS[name])) for name in TRACK]
    ref = {d["exe"]: d for _, d in rows if d["span"] is None}
    head = "%-16s %4s %7s %10s %8s %10s %8s %4s %8s"
    print(head % ("config", "exe", "routed", "total M", "d%", "exec M", "d%", "KV", "lastMLA"))
    for name, d in rows:
        b = ref[d["exe"]]
        print(head % (
            name, d["exe"], d["routed"],
            "%.1f" % (d["total"] / 1e6), "%+.2f" % (100 * (d["total"] / b["total"] - 1)),
            "%.1f" % (d["executed"] / 1e6), "%+.2f" % (100 * (d["executed"] / b["executed"] - 1)),
            d["kv"], d["last_is_mla"]))


if __name__ == "__main__":
    main()
