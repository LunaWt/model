"""Цикл предобучения.

    uv run python -m scripts.train --config A16 --steps 200

Что здесь принципиального, кроме «сделать шаг»:

* **Логиты не материализуются.** `model.body` возвращает (B, T, d) до lm_head, а
  `chunked_cross_entropy` считает CE кусками, пересчитывая логиты в backward.
  При T=1536 и V=16384 обычный путь стоил бы ~350 МиБ на ровном месте.
* **Muon вместо AdamW на матрицах.** Одно состояние вместо двух, и bf16 вместо
  fp32 — иначе Config F не помещается в 6 ГБ (замер 2 сен, notes/ledger.md).
* **QB обновляется раз в шаг оптимизатора**, после step(), по скорам, собранным
  со всех микро-батчей: квантиль должен считаться по батчу шага, а не по
  микро-батчу.
* **Данные детерминированы по номеру шага**, поэтому возобновление не требует
  ничего, кроме `step` в чекпоинте.
* **Обрезка градиента здесь — предохранитель, а не регулятор.** И Muon
  (ортогонализация выбрасывает масштаб), и Adam (деление на sqrt(v)) почти
  инвариантны к общему множителю на градиенте, так что clip меняет не силу шага,
  а только реакцию на выброс. Замер 2 сен: типичная норма на A16 — 5–7, редкие
  выбросы до 15; поэтому порог стоит выше типичного, иначе он срабатывает каждый
  шаг и ничего при этом не делает.
* **CUDA-графы требуют стабильных буферов градиента.** `zero_grad(set_to_none=True)`
  освобождает `.grad` и на следующем шаге аллоцирует заново — уже ВНУТРИ
  захваченного графа, после чего накопление читает перезаписанную память и torch
  падает с «accessing gradient tensor output of CUDAGraphs that has been
  overwritten». Поэтому: один eager-прогон до компиляции (он и создаёт буферы) и
  дальше только `set_to_none=False`. Память от этого не растёт — при accum > 1
  градиенты и так живут весь шаг.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from model.compile_patch import compile_body, compile_report, cudagraph_stats
from model.configs import get as get_config
from model.data import DataConfig, MixedLoader, natural_mix
from model.generate import sample_text
from model.losses import chunked_cross_entropy
from model.model import K3Model, LatentMoE
from model.optim import build_optimizers, lr_multiplier
from tokenizers import Tokenizer

TOKENIZER = Path("tokenizers/bpe_16384.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="A16")
    p.add_argument("--run-name", default=None)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=None, help="по умолчанию cfg.max_seq_len")
    p.add_argument("--lr-muon", type=float, default=6e-4)
    p.add_argument("--lr-adam", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-frac", type=float, default=0.01)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=10.0)
    p.add_argument("--mix", default="web=0.6,math=0.25,code=0.15",
                   help="веса групп или \"natural\" — пропорционально числу токенов")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--sample-every", type=int, default=0, help="0 — только в конце")
    p.add_argument("--sample-tokens", type=int, default=80)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--out", default="runs")
    p.add_argument("--resume", default=None)
    p.add_argument("--max-minutes", type=float, default=0.0, help="0 — без ограничения")
    p.add_argument("--compile", dest="compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--compile-mode", default="max-autotune")
    p.add_argument("--gemm-autotune", action="store_true",
                   help="дать inductor подбирать Triton-шаблоны для матмулов (см. compile_patch)")
    p.add_argument("--capacity-factor", type=float, default=None,
                   help="перекрыть cfg.capacity_factor")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def parse_mix(s: str, root=Path("data/tokens")) -> dict[str, float]:
    if s.strip() == "natural":
        return natural_mix(root)
    out = {}
    for part in s.split(","):
        g, _, w = part.partition("=")
        out[g.strip()] = float(w)
    return out


def evaluate(model, loader, n_batches: int, device: str) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(n_batches):
            x, y = loader.batch(i, device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                h = model.body(x)
            total += chunked_cross_entropy(h, model.lm_head.weight, y).item()
    model.train()
    return total / max(1, n_batches)


def dropped_fraction(model, slots_per_layer: int) -> float:
    """Доля слотов, не попавших к своему эксперту, в среднем по MoE-слоям.

    `dropped` — счётчик последнего микро-батча внутри слоя; делим на число слотов
    ОДНОГО слоя и на число слоёв, иначе метрика растёт с глубиной модели.
    """
    vals = [m.dropped for m in model.modules()
            if isinstance(m, LatentMoE) and hasattr(m, "dropped")]
    if not vals:
        return 0.0
    return float(torch.stack(vals).sum().item()) / (slots_per_layer * len(vals))


def card_used_mib(dev: str) -> int:
    if dev != "cuda":
        return 0
    free, total = torch.cuda.mem_get_info()
    return round((total - free) / 2**20)


def warm_and_compile(model, loader, opts, a, dev: str):
    """Прогреть буферы, скомпилировать `body`, записать CUDA-графы.

    Три прогона по одному микро-батчу, и каждый нужен по своей причине:
      1. eager — создаёт `.grad` вне захвата (см. шапку модуля);
      2. первый компилированный — собственно компиляция;
      3. второй — inductor записывает CUDA-графы только на ВТОРОМ вызове, первый
         идёт «прогревочным» в обычном режиме. Без него первый шаг обучения
         оказался бы в 12 раз дороже остальных (замер 2 сен: 7.9 с, 4.0 с, затем
         0.31 с).
    Скоры роутера после прогонов сбрасываются: это не данные шага оптимизатора.
    """
    x, y = loader.batch(0, dev)

    def once(fn):
        with torch.autocast(dev, dtype=torch.bfloat16):
            h = fn(x)
        chunked_cross_entropy(h, model.lm_head.weight, y).backward()
        model.harvest_router_scores()
        for ffn in model.ffn:
            if isinstance(ffn, LatentMoE):
                ffn._scores.clear()
        for opt in opts:
            opt.zero_grad(set_to_none=False)

    once(model.body)
    t0 = time.time()
    compiled = compile_body(model, a.compile_mode, gemm_autotune=a.gemm_autotune)
    once(compiled)
    once(compiled)
    torch.cuda.synchronize()
    info = {"mode": a.compile_mode, "gemm_autotune": a.gemm_autotune, "compile_s": round(time.time() - t0, 1),
            **compile_report(), **cudagraph_stats()}
    if info["cudagraph_skips"] or not info["captured_nodes"]:
        raise RuntimeError(f"CUDA-графы не записаны: {info}")
    return compiled, info


def main() -> None:
    a = parse_args()
    torch.manual_seed(a.seed)
    dev = a.device

    cfg = get_config(a.config)
    if a.seq_len:
        cfg.max_seq_len = a.seq_len
    if a.capacity_factor:
        cfg.capacity_factor = a.capacity_factor
    model = K3Model(cfg).to(dev)
    counts = model.count_params()

    data_cfg = DataConfig(mix=parse_mix(a.mix), seq_len=cfg.max_seq_len,
                          batch_size=a.batch_size, seed=a.seed)
    train_loader = MixedLoader(data_cfg, "train")
    val_loader = MixedLoader(data_cfg, "val")

    muon, adamw = build_optimizers(model, lr_muon=a.lr_muon, lr_adam=a.lr_adam,
                                   weight_decay=a.weight_decay)

    run = Path(a.out) / (a.run_name or f"{a.config}_{time.strftime('%m%d_%H%M')}")
    run.mkdir(parents=True, exist_ok=True)
    log_path = run / "log.jsonl"
    (run / "config.json").write_text(json.dumps(
        {"model": asdict(cfg), "args": vars(a), "params": counts}, indent=2, default=str))

    start_step = 0
    if a.resume:
        # строго на CPU: `map_location=dev` кладёт рядом с уже созданной моделью
        # ВТОРУЮ копию весов и состояний оптимизаторов, а на 6 ГБ это выталкивает
        # рабочий набор в хост-память — замер 2 сен: 5652 МиБ и 86 ток/с вместо 3484 и 1500.
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        muon.load_state_dict(ck["muon"])
        adamw.load_state_dict(ck["adamw"])
        start_step = ck["step"]
        del ck
        print(f"возобновление с шага {start_step} из {a.resume}")

    tokens_per_step = a.batch_size * a.grad_accum * cfg.max_seq_len
    print(f"конфиг {a.config}: total {counts['total'] / 1e6:.1f}M, "
          f"active {counts['active'] / 1e6:.1f}M, executed {counts['executed'] / 1e6:.1f}M, "
          f"исполнений {len(model.execution_order)}")
    print(f"данные: {' '.join(f'{g} {n / 1e9:.2f}B' for g, n in train_loader.token_counts().items())}")
    print(f"шаг = {tokens_per_step} токенов, всего {a.steps * tokens_per_step / 1e6:.1f}M\n")

    model.train()
    params = [p for p in model.parameters() if p.requires_grad]

    body = model.body
    if a.compile and dev == "cuda":
        body, info = warm_and_compile(model, train_loader, (muon, adamw), a, dev)
        print("компиляция: " + json.dumps(info, ensure_ascii=False))
        with log_path.open("a") as fh:
            fh.write(json.dumps({"step": start_step, "compile": info}) + "\n")

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    deadline = time.time() + a.max_minutes * 60 if a.max_minutes else math.inf
    t_start = time.time()
    stopped = "конец"

    for step in range(start_step, a.steps):
        t0 = time.time()
        mult = lr_multiplier(step, a.steps, a.warmup_frac, a.min_lr_frac)
        for opt, base in ((muon, a.lr_muon), (adamw, a.lr_adam)):
            for g in opt.param_groups:
                g["lr"] = base * mult

        loss_sum = torch.zeros((), device=dev)
        for micro in range(a.grad_accum):
            x, y = train_loader.batch(step * a.grad_accum + micro, dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                h = body(x)
            loss = chunked_cross_entropy(h, model.lm_head.weight, y) / a.grad_accum
            loss.backward()
            model.harvest_router_scores()
            loss_sum += loss.detach()

        gnorm = torch.nn.utils.clip_grad_norm_(params, a.clip).item()
        for opt in (muon, adamw):
            opt.step()
            opt.zero_grad(set_to_none=False)
        model.update_router_bias()

        dt = time.time() - t0
        rec = {
            "step": step, "loss": loss_sum.item(), "lr_mult": mult,
            "grad_norm": gnorm, "s_per_step": round(dt, 3),
            "tok_s": round(tokens_per_step / dt, 1),
            # `max_memory_allocated` НЕ видит пул CUDA-графов: inductor держит
            # активации в приватном пуле, и после компиляции эта цифра падает
            # вдвое, хотя карта занята так же. Смотреть надо на устройство целиком.
            "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20) if dev == "cuda" else 0,
            "card_mib": card_used_mib(dev),
            "dropped": dropped_fraction(model, a.batch_size * cfg.max_seq_len * cfg.top_k),
        }
        with log_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{step:5d}] loss {rec['loss']:.4f}  lr×{mult:.3f}  |g| {gnorm:6.2f}  "
              f"{rec['tok_s']:7.1f} ток/с  {rec['card_mib']} МиБ карты  drop {rec['dropped']:.3f}")

        if a.eval_every and (step + 1) % a.eval_every == 0:
            v = evaluate(model, val_loader, a.eval_batches, dev)
            print(f"        val {v:.4f}  (ppl {math.exp(min(v, 20)):.1f})")
            with log_path.open("a") as fh:
                fh.write(json.dumps({"step": step, "val_loss": v}) + "\n")

        if a.sample_every and (step + 1) % a.sample_every == 0:
            show_sample(model, dev, a.sample_tokens)

        if a.save_every and (step + 1) % a.save_every == 0:
            save(run / "last.pt", model, muon, adamw, step + 1, a.config)

        if time.time() > deadline:
            stopped = f"лимит {a.max_minutes} мин"
            a.steps = step + 1
            break

    save(run / "last.pt", model, muon, adamw, a.steps, a.config)
    print(f"\n{stopped}: {a.steps - start_step} шагов за {(time.time() - t_start) / 60:.1f} мин")
    v = evaluate(model, val_loader, a.eval_batches, dev)
    print(f"итоговый val {v:.4f}  (ppl {math.exp(min(v, 20)):.1f})")
    show_sample(model, dev, a.sample_tokens)


def show_sample(model, dev: str, n_tokens: int) -> None:
    tok = Tokenizer.from_file(str(TOKENIZER))
    for prompt in ("", "The answer is", "def solve(n):"):
        text = sample_text(model, tok, prompt, dev, max_new_tokens=n_tokens,
                           temperature=0.8, top_k=50, top_p=0.95)
        print(f"  ─ {prompt!r} → {text!r}")


def save(path: Path, model, muon, adamw, step: int, config: str) -> None:
    torch.save({"model": model.state_dict(), "muon": muon.state_dict(),
                "adamw": adamw.state_dict(), "step": step, "config": config,
                "cfg": asdict(model.cfg)}, path)


if __name__ == "__main__":
    main()
