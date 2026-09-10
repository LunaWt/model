"""Сэмплирование с кэшем состояния.

Кэш здесь — не только KV. У гибрида три разных вида памяти, и все три надо нести:

  * **MLA** — латент `c_t` (B, T, kv_latent). Это и есть KV-кэш; K и V
    восстанавливаются из него up-проекциями, поэтому хранится он в
    `d_model / kv_latent` = 8 раз компактнее обычного.
  * **KDA** — рекуррентное состояние S (B, H, d_k, d_v). Оно НЕ растёт с длиной:
    64 КиБ на слой при любом контексте. Отсюда следует вещь, которую легко
    упустить: у нас 12 слоёв из 16 имеют кэш постоянного размера, и с длиной
    растёт только четыре MLA-слоя.
  * **ShortConv** — K−1 предыдущих входов на каждый из трёх путей. Мелочь на
    3·(K−1)·H·D чисел, но без неё свёртка на каждом шаге декодирования видела бы
    слева нули, и результат разошёлся бы с прогоном всей последовательности.

Словарь кэша ключуется НОМЕРОМ ИСПОЛНЕНИЯ, а не номером слоя: у зациклённой
модели один и тот же слой стоит в стеке дважды и памяти у этих двух проходов
разные (см. `K3Model.body`).

Проверка эквивалентности — `tests/test_generate.py`: жадная генерация с кэшем
обязана совпасть с полным пересчётом токен в токен.
"""

from __future__ import annotations

import torch
from torch import nn


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
            gen: torch.Generator | None) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k:
        kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = logits.softmax(-1)
    if top_p < 1.0:
        srt, idx = probs.sort(dim=-1, descending=True)
        cum = srt.cumsum(-1)
        # оставляем минимальный префикс с суммой >= top_p: сдвиг на 1 нужен,
        # чтобы токен, ПЕРЕСЁКШИЙ порог, сам остался
        srt = srt.masked_fill((cum - srt) > top_p, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, idx, srt)
        probs = probs / probs.sum(-1, keepdim=True)
    return torch.multinomial(probs, 1, generator=gen)


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    eot_id: int | None = None,
    seed: int | None = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """prompt (B, T0) int64 -> (B, T0 + n) int64.

    `use_cache=False` — полный forward на каждый токен, без состояния. Медленно и
    существует только как образец правды для теста эквивалентности.
    """
    was_training = model.training
    model.eval()
    # без этого префилл и пошаговое декодирование считают разное: см.
    # K3Model.set_full_capacity
    if hasattr(model, "set_full_capacity"):
        model.set_full_capacity(True)
    gen = torch.Generator(device=prompt.device).manual_seed(seed) if seed is not None else None

    out = prompt
    if use_cache:
        cache: dict = {}
        logits = model(prompt, cache=cache)[:, -1].float()
        for i in range(max_new_tokens):
            nxt = _sample(logits, temperature, top_k, top_p, gen)
            out = torch.cat([out, nxt], dim=1)
            if eot_id is not None and (nxt == eot_id).all():
                break
            if i + 1 < max_new_tokens:
                logits = model(nxt, cache=cache)[:, -1].float()
    else:
        max_len = getattr(model.cfg, "max_seq_len", 1024)
        for _ in range(max_new_tokens):
            logits = model(out[:, -max_len:])[:, -1].float()
            nxt = _sample(logits, temperature, top_k, top_p, gen)
            out = torch.cat([out, nxt], dim=1)
            if eot_id is not None and (nxt == eot_id).all():
                break

    if hasattr(model, "set_full_capacity"):
        model.set_full_capacity(False)
    if was_training:
        model.train()
    return out


def sample_text(model, tokenizer, prompt: str, device, **kw) -> str:
    ids = tokenizer.encode(prompt).ids if prompt else []
    x = torch.tensor([ids or [0]], dtype=torch.long, device=device)
    out = generate(model, x, eot_id=tokenizer.token_to_id("<|endoftext|>"), **kw)
    return tokenizer.decode(out[0].tolist())
