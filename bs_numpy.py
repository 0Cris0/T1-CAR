"""
bs_numpy.py  --  Paralelismo con joblib.Parallel, ajuste resolviendo
                 (X^T X) beta = X^T y con NumPy puro.

Version mas rapida de las tres. El costo dominante es formar la matriz de
Gram: ~2 * n * (k+1)^2 flops, con n = filas efectivas. Resolver el sistema
(k+1)^3/3 flops es despreciable en comparacion (k+1 = 301 vs n ~ 63.000).

Optimizaciones respecto de la version ingenua:
  1. `solve` en vez de `inv(A) @ b`.
  2. X^(b) nunca se materializa: se usa X^T diag(w) X con w = conteos del
     sorteo multinomial equivalente.
  3. Se descartan las filas con w_i = 0 (~37% de N) -> ~37% menos flops.
  4. Escalado por sqrt(w) in-place: una sola copia de memoria por resample.
  5. X se comparte por memmap; los workers reciben rutas, no arreglos.
  6. 1 hilo BLAS por proceso (--inner-threads 1) para evitar sobresuscripcion.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

import common

VERSION = "bs_numpy"


def _resample_fit(xpath, ypath, ss, dtype):
    X, y = common.load_shared(xpath, ypath)

    # Paso 2(I): N indices con reemplazo, expresados como conteos w
    nz, sw = common.bootstrap_weights(np.random.default_rng(ss), X.shape[0])

    # Paso 2(II): (sqrt(w) X_nz, sqrt(w) y_nz) equivale a (X^(b), y^(b))
    Xw, yw = common.weighted_design(X, y, nz, sw, dtype)

    # Paso 2(III): ecuaciones normales
    return np.linalg.solve(Xw.T @ Xw, Xw.T @ yw)


def bootstrap(P, B, p, dtype, batch_size=1):
    seeds = common.child_seeds(B)
    return Parallel(n_jobs=p, backend="loky", batch_size=batch_size,
                    pre_dispatch="2*n_jobs")(
        delayed(_resample_fit)(P["X"], P["y"], s, dtype) for s in seeds
    )


def main():
    args = common.base_parser(__doc__.splitlines()[1]).parse_args()
    P = common.paths(args.tag)
    dtype = np.dtype(args.dtype)

    X, y = common.load_shared(P["X"], P["y"])
    beta_star = np.load(P["beta_star"])

    # Paso 1
    with threadpool_limits(args.inner_threads):
        beta_hat = common.fit_normal_equations(np.asarray(X), np.asarray(y))

    # Pasos 2 y 3
    mejor = np.inf
    for _ in range(args.repeats):
        with parallel_config(backend="loky",
                             inner_max_num_threads=args.inner_threads):
            with common.cronometro() as t:
                betas = bootstrap(P, args.n_resamples, args.n_jobs, dtype)
        mejor = min(mejor, t.dt)

    common.reportar(VERSION, args, mejor, betas, beta_hat, beta_star)


if __name__ == "__main__":
    main()
