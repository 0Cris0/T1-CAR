"""
bs_sklearn.py  --  Paralelismo explicito con joblib.Parallel, ajuste con
                   LinearRegression de sklearn.

Cada tarea = un resample completo (sorteo + ajuste). Las tareas son
independientes, asi que no hay comunicacion ni sincronizacion entre procesos:
paralelismo de tareas puro.

Diferencia clave con bs_auto: aca controlamos como se arma X^(b), y aplicamos
el truco de pesos + descarte de filas no sorteadas. El ajuste sigue siendo el
de sklearn (scipy.linalg.lstsq, basado en SVD), que es el cuello de botella
de esta version.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

import common

VERSION = "bs_sklearn"


def _resample_fit(xpath, ypath, ss, dtype):
    """Un resample. Recibe RUTAS, no arreglos: X se abre por memmap una sola
    vez por worker (ver common.load_shared)."""
    X, y = common.load_shared(xpath, ypath)
    nz, sw = common.bootstrap_weights(np.random.default_rng(ss), X.shape[0])
    Xw, yw = common.weighted_design(X, y, nz, sw, dtype)

    lr = LinearRegression(fit_intercept=False, copy_X=False)
    lr.fit(Xw, yw)
    return lr.coef_


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
