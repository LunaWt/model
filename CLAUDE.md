# model — small-scale LM research 🧪

Sandbox for real experiments on small LMs on one 6 GB GPU. **Not a portfolio piece** —
breaking things and abandoning an experiment halfway is allowed. Two lines run in parallel:
reimplementing the **Kimi K3** architecture at this scale (main), and a **looped-transformer
comparison** on top of it (`notes/looped.md`) that must never block the main one.

## Where things live

| Path | What |
|---|---|
| `model/` | the model itself: `kda.py` / `kda_head.py` (Kimi Delta Attention), `model.py`, `losses.py`, `compile_patch.py` |
| `scripts/` | tokenizer training and evaluation, corpus sampling and encoding |
| `tokenizers/` | trained BPE vocabularies (`bpe_16384.json` is the one in use) |
| `tests/` | pytest suite — `uv run pytest` |
| `grok_review/` | one-off probes written to check claims from an external review |
| `notes/` | **untracked.** Plan, dated ledger, architecture reading, working agreement, `looped.md` (parallel experiment track) |

Plan and ledger are updated in-session when a step closes, not "later".

## Environment

- GPU **GTX 1660 Ti, 6 GB, sm_75** — no tensor cores. **bf16, never fp16** (measured 5×
  slower on this card). Runs are sized to minutes-to-hours.
- **`uv run <cmd>`** always; dependencies via `uv add`. Never activate `.venv`, never
  `uv pip install`.
- The repo lives on the Linux filesystem, not under `/mnt/c` — the cross-filesystem I/O
  penalty is large enough to distort timings.

## Testing

Logic worth trusting — masking, a data loader, a custom attention variant — gets a pytest
that **stays in the repo**. Experiment scripts do not.

## Working rules for an agent here

- **Explain → run → interpret.** What do we expect and why → what came out → what it means.
  A run nobody interpreted is a wasted run.
- Direction belongs to the repo owner: what to try, what to measure, when to move on.
  Finishing a step is not permission to start the next one — propose it in one or two
  lines and stop.
- The local working agreement (which pieces are written by hand rather than by the agent)
  lives in `notes/`, which is untracked. If it is missing, ask before writing a core
  mechanism from scratch.
- Numbers go into `notes/ledger.md` with the run that produced them. A figure without its
  derivation beside it gets cut, not kept.
