"""Маленький бенчмарк на 30 промптов: генерация с кэшем + вынужденный выбор.

    uv run python -m scripts.bench_prompts runs/A16_cmp/last.pt runs/M_experts_cmp/last.pt

Две части, потому что на нашем масштабе одной мало.

**Генерация** показывает, что модель вообще выучила: связность, формат, обрывы.
Читается глазами, числа из неё не выжать.

**Вынужденный выбор** даёт число. У каждого промпта есть два продолжения:
осмысленное и переставленное/бессмысленное. Считаем средний NLL на токен по
каждому и смотрим, какому модель дала меньше. Метрика работает и на очень слабой
модели, в отличие от точности ответа, и сравнима между конфигами.

⚠️ 30 промптов — это ±9 процентных пунктов стандартной ошибки у доли около 0.5.
Разница меньше 15 п.п. между конфигами здесь ничего не значит.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from model.configs import get as get_config
from model.generate import generate
from model.model import K3Config, K3Model
from tokenizers import Tokenizer

TOKENIZER = Path("tokenizers/bpe_16384.json")

# (тема, промпт, осмысленное продолжение, бессмысленное продолжение)
PROMPTS: list[tuple[str, str, str, str]] = [
    ("факты", "The capital of France is", " Paris, a city on the river Seine.",
     " Paris, a river on the city Seine."),
    ("факты", "Water freezes at a temperature of", " zero degrees Celsius.",
     " zero degrees of Celsius temperature freezes."),
    ("факты", "The largest planet in the solar system is", " Jupiter, a gas giant.",
     " Jupiter, a giant gas the."),
    ("факты", "There are seven days in", " a week, from Monday to Sunday.",
     " a week, to Sunday from Monday."),
    ("факты", "The human heart pumps", " blood through the body.",
     " body through the blood."),
    ("здравый смысл", "She opened the umbrella because it started to",
     " rain heavily outside.", " rain heavily the outside started."),
    ("здравый смысл", "He put the milk back in the", " refrigerator to keep it cold.",
     " refrigerator to keep it hot."),
    ("здравый смысл", "The students were quiet because the teacher",
     " asked them to stop talking.", " asked them to talking stop the."),
    ("здравый смысл", "After running for an hour she was very",
     " tired and needed water.", " tired and needed a stone."),
    ("здравый смысл", "You need a key to", " open a locked door.",
     " open a door locked to."),
    ("математика", "The sum of 2 and 3 is", " 5.", " 23."),
    ("математика", "If x = 4, then 2 * x equals", " 8.", " 42."),
    ("математика", "Ten divided by two equals", " five.", " twenty."),
    ("математика", "The square of 6 is", " 36.", " 12."),
    ("математика", "A triangle has", " three sides and three angles.",
     " three sides and four angles."),
    ("математика", "The next number in the sequence 1, 2, 4, 8 is", " 16.", " 9."),
    ("математика", "The derivative of x squared is", " 2x.", " x squared over two."),
    ("код", "def add(a, b):\n    return", " a + b\n", " a + b the return of\n"),
    ("код", "for i in range(10):\n    print(", "i)\n", "i(\n"),
    ("код", "x = [1, 2, 3]\nlen(x) is", " 3", " [1, 2, 3] of length x"),
    ("код", "if x > 0:\n    print('positive')\nelse:", "\n    print('negative')\n",
     "\n    print positive negative else\n"),
    ("код", "import numpy as np\narr = np.", "zeros((3, 3))\n", "((3, 3))zeros np\n"),
    ("код", "# a function that returns the maximum of two numbers\ndef max2(a, b):",
     "\n    return a if a > b else b\n", "\n    return a if a b > else\n"),
    ("язык", "The opposite of hot is", " cold.", " warm hot the."),
    ("язык", "One, two, three, four,", " five, six, seven.", " seven, five, four."),
    ("язык", "A group of wolves is called a", " pack.", " pack called group of."),
    ("язык", "The past tense of go is", " went.", " goed."),
    ("наука", "Plants use sunlight to make food in a process called",
     " photosynthesis.", " photosynthesis called process a."),
    ("наука", "The chemical symbol for water is", " H2O.", " O2H."),
    ("наука", "Objects fall to the ground because of", " gravity.",
     " gravity because of the ground."),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="runs/bench_prompts.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load(path: str, device: str) -> tuple[K3Model, str, int]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = K3Config(**ck["cfg"]) if "cfg" in ck else get_config(ck["config"])
    model = K3Model(cfg)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck["config"], ck["step"]


@torch.no_grad()
def continuation_nll(model, tok, prompt: str, cont: str, device: str) -> float:
    """Средний NLL на токен продолжения при данном промпте.

    Считается по ПОЛНОЙ последовательности одним forward, ёмкость экспертов
    снята — иначе результат зависел бы от длины варианта (см.
    `K3Model.set_full_capacity`), а варианты у нас разной длины.
    """
    p_ids = tok.encode(prompt).ids or [0]
    c_ids = tok.encode(cont).ids
    ids = torch.tensor([p_ids + c_ids], dtype=torch.long, device=device)
    model.set_full_capacity(True)
    logits = model(ids).float()
    model.set_full_capacity(False)
    logp = logits[0, len(p_ids) - 1:-1].log_softmax(-1)
    tgt = torch.tensor(c_ids, device=device)
    return -logp.gather(-1, tgt[:, None]).mean().item()


def main() -> None:
    a = parse_args()
    tok = Tokenizer.from_file(str(TOKENIZER))
    eot = tok.token_to_id("<|endoftext|>")
    report: dict = {"prompts": len(PROMPTS), "models": {}}

    for path in a.checkpoints:
        model, name, step = load(path, a.device)
        key = f"{name}@{step}"
        hits, by_topic, rows = 0, {}, []
        t0 = time.time()
        n_tokens = 0
        for topic, prompt, good, bad in PROMPTS:
            ids = tok.encode(prompt).ids or [0]
            x = torch.tensor([ids], dtype=torch.long, device=a.device)
            out = generate(model, x, max_new_tokens=a.max_new_tokens,
                           temperature=a.temperature, top_k=a.top_k, top_p=a.top_p,
                           eot_id=eot, seed=a.seed)
            n_tokens += out.shape[1] - x.shape[1]
            text = tok.decode(out[0].tolist())

            n_good = continuation_nll(model, tok, prompt, good, a.device)
            n_bad = continuation_nll(model, tok, prompt, bad, a.device)
            ok = n_good < n_bad
            hits += ok
            t = by_topic.setdefault(topic, [0, 0])
            t[0] += ok
            t[1] += 1
            rows.append({"topic": topic, "prompt": prompt, "text": text,
                         "nll_good": round(n_good, 4), "nll_bad": round(n_bad, 4),
                         "ok": bool(ok)})
        dt = time.time() - t0
        report["models"][key] = {
            "checkpoint": path,
            "accuracy": round(hits / len(PROMPTS), 4),
            "by_topic": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_topic.items())},
            "gen_tok_s": round(n_tokens / dt, 1),
            "rows": rows,
        }
        del model
        torch.cuda.empty_cache()

        print(f"\n=== {key} ({path}) ===")
        print(f"вынужденный выбор: {hits}/{len(PROMPTS)} = {hits / len(PROMPTS):.2f}   "
              f"генерация {n_tokens / dt:.1f} ток/с")
        for k, v in sorted(by_topic.items()):
            print(f"  {k:14s} {v[0]}/{v[1]}")
        for r in rows[:6]:
            print(f"  ─ {r['prompt']!r}\n    → {r['text']!r}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nполный отчёт: {a.out}")


if __name__ == "__main__":
    main()
