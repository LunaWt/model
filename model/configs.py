"""Именованные конфиги модели: то, что можно запустить как `--config <имя>`.

Пять из notes/looped.md §4: два незациклённых контроля (A16, A24) и три
зациклённых варианта, сматченных с ними по параметрам, исполняемым параметрам и
KV-слотам. `tiny` существует только чтобы прогонять код за секунды.

Группа `M*` — «средний размер»: больше A16, но всё ещё в карте. Пространство тут
не свободное, а зажато замером 2 сен: при 16 исполнениях и T=1536 пик A16 —
5098 МиБ, из них ~2.3 ГиБ это веса + градиенты + момент Muon (10 байт на
параметр) и ~2.8 ГиБ активации (~173 МиБ на одно исполнение слоя). Отсюда два
следствия:

  * **глубину трогать нельзя** — каждый слой стоит и параметров, и активаций,
    +4 слоя это больше гигабайта, и пара #3–#5 уже не влезает (замер: 7.5 ГиБ,
    подкачка в хост, 43–166 ток/с вместо 1146);
  * **ширина эксперта и их число почти бесплатны по активациям** — растёт только
    состояние оптимизатора, 10 МиБ на миллион параметров.

Поэтому оба M-конфига держат 16 слоёв и d_model=512 и добавляют размер только в
MoE: `M_experts` — числом экспертов (чистый sparse-рост, активные параметры не
меняются), `M_mix` — ещё и шириной эксперта с top_k, то есть растут и FLOPs.

⚠️ У них СВОЙ `max_seq_len`, и это не стиль, а замер. Обрыв по памяти резкий:
M_experts при T=1344 — 5412 МиБ и 805 ток/с, при T=1408 — 5502 МиБ и 453 ток/с,
то есть аллокатор начал сливать в хост-память. Практический потолок карты лежит
между 5.4 и 5.5 ГиБ, а не там, где кончаются 6 ГБ по паспорту: остальное держит
рабочий стол Windows. Так что T тут подобран под размер модели, а не наоборот, и
кратен 16 (шаг чанка KDA), хотя ни модель, ни fla этого не требуют.
"""

from __future__ import annotations

from model.model import K3Config

CONFIGS: dict[str, K3Config] = {
    "tiny": K3Config(
        d_model=128, n_layers=4, n_heads=2, d_head=32, max_seq_len=256,
        mla_kv_latent=32, mla_q_latent=48, moe_latent=64, n_routed=8,
        n_shared=1, top_k=2, expert_hidden=64, shared_hidden=64,
        dense_hidden=256, n_blocks=2,
    ),
    "tiny_loop": K3Config(
        d_model=128, n_layers=4, n_heads=2, d_head=32, max_seq_len=256,
        mla_kv_latent=32, mla_q_latent=48, moe_latent=64, n_routed=8,
        n_shared=1, top_k=2, expert_hidden=64, shared_hidden=64,
        dense_hidden=256, n_blocks=2, loop_span=(1, 2), loop_r=2,
    ),
    "A16": K3Config(n_layers=16, n_routed=64, n_blocks=4),
    "M_experts": K3Config(n_layers=16, n_routed=80, n_blocks=4, max_seq_len=1344),
    "M_mix": K3Config(n_layers=16, n_routed=72, n_blocks=4, expert_hidden=288,
                      top_k=5, max_seq_len=1024),
    "L12_4x2": K3Config(n_layers=12, n_routed=90, n_blocks=4, loop_span=(4, 7), loop_r=2),
    "A24": K3Config(n_layers=24, n_routed=64, n_blocks=4),
    "L16_8x2": K3Config(n_layers=16, n_routed=103, n_blocks=4, loop_span=(4, 11), loop_r=2),
    "L12_6x3": K3Config(n_layers=12, n_routed=143, n_blocks=4, loop_span=(4, 9), loop_r=3),
}


def get(name: str) -> K3Config:
    if name not in CONFIGS:
        raise SystemExit(f"нет конфига {name!r}; есть: {', '.join(CONFIGS)}")
    return CONFIGS[name]
