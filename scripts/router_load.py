"""Насколько ровно чекпоинт раскладывает токены по экспертам.

    uv run python -m scripts.router_load runs/A16_first/last.pt

Считает реальные решения роутера на нескольких батчах и печатает, какая доля
слотов вылетела бы при разных `capacity_factor`. Смысл в том, чтобы выбирать
запас ёмкости по замеру на обученном роутере, а не по вкусу: у необученного
роутера скоры почти равны и перекоса нет вообще, так что вопрос имеет смысл
только на чекпоинте.

Отдельно печатается перекос ВНУТРИ микро-батча против перекоса ПО ВСЕМ батчам
сразу. Расхождение между ними и есть цена того, что B=1 и микро-батч — это один
непрерывный кусок одного документа: QB выравнивает среднее, а ёмкость
проверяется на каждом микро-батче отдельно.

`--batch-size B --seq-len T` при том же произведении B·T отвечает ровно на
вопрос «поможет ли смешивать документы»: загрузчик тянет группу НА КАЖДУЮ строку
батча, так что B=2 это два независимых документа из двух (возможно) разных
доменов в одном forward, при той же цене по токенам.
"""

from __future__ import annotations

import argparse

import torch

from model.configs import get as get_config
from model.data import DataConfig, MixedLoader
from model.model import K3Config, K3Model, LatentMoE


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = K3Config(**ck["cfg"]) if "cfg" in ck else get_config(ck["config"])
    if a.seq_len:
        cfg.max_seq_len = a.seq_len
    model = K3Model(cfg).to(a.device)
    model.load_state_dict(ck["model"])
    model.train()

    loader = MixedLoader(DataConfig(seq_len=cfg.max_seq_len, batch_size=a.batch_size), "train")
    moes = [m for m in model.modules() if isinstance(m, LatentMoE)]

    scores: list[list[torch.Tensor]] = []
    with torch.no_grad():
        for i in range(a.batches):
            x, _ = loader.batch(i, a.device)
            with torch.autocast(a.device, dtype=torch.bfloat16):
                model.body(x)
            scores.append([m._visit_scores[0] for m in moes])
            for m in moes:
                m._visit_scores.clear()

    m_tokens = cfg.max_seq_len * a.batch_size
    k, n = cfg.top_k, cfg.n_routed
    ideal = m_tokens * k / n
    print(f"{ck['config']}, шаг {ck['step']}, {len(moes)} MoE-слоёв, "
          f"{a.batches} батчей по B={a.batch_size} x T={cfg.max_seq_len} = {m_tokens} токенов")
    print(f"идеально ровная нагрузка на эксперта: {ideal:.0f} слотов из {m_tokens * k}\n")

    old = torch.stack([torch.stack([counts(m, s) for m, s in zip(moes, row)])
                       for row in scores])
    report("биас из чекпоинта (решатель в 1 итерацию)", old, ideal, a.batches)

    refit(moes, scores, iters=cfg.qb_iters)
    new = torch.stack([torch.stack([counts(m, s) for m, s in zip(moes, row)])
                       for row in scores])
    report(f"биас пересчитан на этих же батчах, {cfg.qb_iters} итерации",
           new, ideal, a.batches)

    print("выброшено слотов при разном capacity_factor:")
    print("  cf     старый   пересчитанный")
    for cf in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0):
        cap = max(1, int(ideal * cf))
        d_old = (old - cap).clamp(min=0).sum() / old.sum()
        d_new = (new - cap).clamp(min=0).sum() / new.sum()
        print(f"{cf:>5.2f}  {d_old:>9.3f}  {d_new:>13.3f}")


def refit(moes, scores, iters: int) -> None:
    """Подобрать qb_bias по ВСЕМ собранным батчам — верхняя граница возможного.

    Так балансировщик видит то же распределение, на котором его потом мерят;
    остаток перекоса после этого — уже не про сходимость решателя, а про то,
    что нагрузка не выравнивается одним сдвигом на эксперта в принципе.
    """
    for j, m in enumerate(moes):
        m._scores = [row[j] for row in scores]
        m.cfg.qb_iters = iters
        m.update_router_bias()


def report(tag: str, c: torch.Tensor, ideal: float, batches: int) -> None:
    within = c.max(-1).values / ideal
    pooled = c.sum(0).max(-1).values / (ideal * batches)
    print(f"{tag}:")
    print(f"  пик/идеал внутри микро-батча    медиана {within.median():.2f}  "
          f"максимум {within.max():.2f}")
    print(f"  пик/идеал по всем батчам сразу  медиана {pooled.median():.2f}  "
          f"максимум {pooled.max():.2f}\n")


def counts(moe: LatentMoE, s: torch.Tensor) -> torch.Tensor:
    idx = torch.topk(s + moe.qb_bias, moe.cfg.top_k, dim=-1).indices
    return torch.bincount(idx.reshape(-1), minlength=moe.cfg.n_routed)


if __name__ == "__main__":
    main()
