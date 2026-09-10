"""Что вообще работает на TPU в том JAX, который стоит на Kaggle.

    uv run python -m scripts.kaggle_run scripts/tpu_probe.py \
        --name tpu-probe --accelerator tpuV5e8

Смысл в очереди. Полный прогон сетки ждёт 20–50 минут и падает целиком, если
одна библиотечная функция на этом железе не реализована — так уже случилось с
`ragged_dot` («rhs_contracting_dim != 1 - NYI») и с `expert`-шардингом. Дешевле
один короткий прогон, который каждую спорную функцию вызывает на игрушечных
формах и печатает, что вышло.

Каждая проверка независима и ловит своё исключение: цель — таблица «работает /
не работает», а не первый попавшийся стектрейс.
"""

from __future__ import annotations

import subprocess
import sys

if "--pip" in sys.argv:
    specs = sys.argv[sys.argv.index("--pip") + 1].split(",")
    print(f"ставлю {specs}", flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *specs],
                       capture_output=True, text=True)
    print(f"pip код {r.returncode}: {(r.stdout + r.stderr).strip()[-900:]}", flush=True)
    v = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
    print(" | ".join(l for l in v.stdout.splitlines()
                     if l.split(" ")[0] in ("jax", "jaxlib", "libtpu", "optax", "flax")),
          flush=True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import random  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

OK, BAD = [], []


def check(name):
    def deco(fn):
        try:
            out = fn()
            print(f"[ДА ] {name}: {out}", flush=True)
            OK.append(name)
        except Exception as exc:  # noqa: BLE001
            msg = " ".join(str(exc).split())[:200] or type(exc).__name__
            print(f"[НЕТ] {name}: {type(exc).__name__}: {msg}", flush=True)
            BAD.append(name)
        return fn

    return deco


def main() -> None:
    n = jax.device_count()
    try:
        import libtpu
        ver = getattr(libtpu, "__version__", "?")
    except ImportError:
        ver = "нет пакета"
    print(f"{jax.local_devices()[0].device_kind} | jax {jax.__version__} | "
          f"libtpu {ver} | чипов {n}", flush=True)

    key = random.PRNGKey(0)
    N, E, d, h = 512, 8, 256, 128
    lhs32 = random.normal(key, (N, d), jnp.float32)
    sizes = jnp.full((E,), N // E, jnp.int32)

    for dt in (jnp.float32, jnp.bfloat16):
        nm = dt.__name__

        @check(f"ragged_dot прямой (N,d)x(E,d,h) {nm}")
        def _(dt=dt):
            w = random.normal(key, (E, d, h), jnp.float32).astype(dt)
            return jax.jit(lambda a, b: jax.lax.ragged_dot(a, b, sizes))(
                lhs32.astype(dt), w).shape

        @check(f"ragged_dot прямой (N,h)x(E,h,d) {nm}")
        def _(dt=dt):
            w = random.normal(key, (E, h, d), jnp.float32).astype(dt)
            a = random.normal(key, (N, h), jnp.float32).astype(dt)
            return jax.jit(lambda a, b: jax.lax.ragged_dot(a, b, sizes))(a, w).shape

        @check(f"ragged_dot градиент {nm}")
        def _(dt=dt):
            w = random.normal(key, (E, d, h), jnp.float32).astype(dt)
            g = jax.jit(jax.grad(lambda a, b: jax.lax.ragged_dot(a, b, sizes).sum(),
                                 argnums=(0, 1)))(lhs32.astype(dt), w)
            return [x.shape for x in g]

    @check("megablox.gmm импортируется")
    def _():
        from jax.experimental.pallas.ops.tpu.megablox import gmm
        return gmm.__module__

    @check("megablox.gmm считает")
    def _():
        from jax.experimental.pallas.ops.tpu.megablox import gmm
        w = random.normal(key, (E, d, h), jnp.bfloat16)
        return jax.jit(lambda a, b: gmm(a, b, sizes))(lhs32.astype(jnp.bfloat16), w).shape

    @check("splash_attention импортируется")
    def _():
        from jax.experimental.pallas.ops.tpu.splash_attention import (
            splash_attention_kernel as sk)
        return [x for x in dir(sk) if "make_splash" in x]

    def splash(window):
        from jax.experimental.pallas.ops.tpu.splash_attention import (
            splash_attention_kernel as sk, splash_attention_mask as sm)
        T, H, D = 1024, 2, 128
        one = sm.LocalMask((T, T), (window - 1, 0), 0) if window else sm.CausalMask((T, T))
        kern = sk.make_splash_mha(sm.MultiHeadMask([one] * H), head_shards=1, q_seq_shards=1)
        k = random.split(key, 3)
        q, kk, vv = (random.normal(k[i], (1, H, T, D), jnp.bfloat16) for i in range(3))
        return kern, (q, kk, vv)

    @check("splash причинный прямой")
    def _():
        kern, arg = splash(0)
        return jax.jit(jax.vmap(kern))(*arg).shape

    @check("splash окно 256 прямой")
    def _():
        kern, arg = splash(256)
        return jax.jit(jax.vmap(kern))(*arg).shape

    @check("splash окно 256 градиент")
    def _():
        kern, arg = splash(256)
        g = jax.jit(jax.grad(lambda *a: jax.vmap(kern)(*a).sum(), argnums=(0, 1, 2)))(*arg)
        return [x.shape for x in g]

    @check("политики remat есть")
    def _():
        return [p for p in ("dots_saveable", "dots_with_no_batch_dims_saveable")
                if hasattr(jax.checkpoint_policies, p)]

    @check("optax стоит")
    def _():
        import optax
        from optax.contrib import muon
        return f"{optax.__version__}, muon {muon is not None}"

    @check("шардинг только по экспертам: шаг повторяется без перекомпиляции")
    def _():
        """Ровно то, на чём упал `moe-shard`: вход разложен по чипам, выход обязан
        лечь так же, иначе второй вызов компилируется заново и не влезает."""
        mesh = Mesh(np.array(jax.devices()).reshape(n), ("x",))
        rep = NamedSharding(mesh, P())
        shd = NamedSharding(mesh, P(None, "x", None, None))
        w = jax.device_put(random.normal(key, (2, E * n, d, h), jnp.bfloat16), shd)
        x = jax.device_put(random.normal(key, (64, d), jnp.bfloat16), rep)

        @jax.jit
        def step(w, x):
            y = jnp.einsum("ledh,nd->lenh", w, x)
            return w - jnp.bfloat16(1e-3) * jnp.mean(y), y.sum()

        w2, _ = step(w, x)
        return f"{w.sharding == w2.sharding}, {w2.shape}"

    print(f"\nработает: {len(OK)}, не работает: {len(BAD)}", flush=True)
    if BAD:
        print("не работает: " + "; ".join(BAD), flush=True)


if __name__ == "__main__":
    main()
