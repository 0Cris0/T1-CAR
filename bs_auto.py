"""
bs_auto.py  --  Bootstrapping y paralelismo COMPLETAMENTE internos.

BaggingRegressor(bootstrap=True, max_samples=1.0) hace exactamente el Paso 2
del enunciado: por cada uno de los n_estimators sortea N indices con reemplazo
y ajusta el estimador base. Con n_jobs=p reparte esos ajustes entre p procesos
(usa joblib internamente).

Notas de implementacion:
  * fit_intercept=False es OBLIGATORIO: X ya trae la columna de unos. Con
    fit_intercept=True el diseno queda rank-deficient y los coeficientes
    dejan de ser comparables con beta*.
  * bootstrap_features=False y max_features=1.0 -> se usan las k+1 columnas
    en todos los estimadores (bagging solo sobre filas, como pide el enunciado).
  * No hay control sobre como se construye cada resample: internamente sklearn
    pasa sample_weight sobre las N filas completas, sin descartar las filas de
    peso cero. Ese es el costo de la version "automatica".
"""

from __future__ import annotations

import numpy as np
from joblib import parallel_config
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

import common

VERSION = "bs_auto"


def bootstrap(X, y, B: int, p: int, seed: int = common.SEED):
    bag = BaggingRegressor(
        estimator=LinearRegression(fit_intercept=False, copy_X=False),
        n_estimators=B,
        max_samples=1.0,        # N muestras por resample
        bootstrap=True,         # con reemplazo
        max_features=1.0,
        bootstrap_features=False,
        n_jobs=p,
        random_state=seed,
    )
    bag.fit(X, y)
    return np.vstack([est.coef_ for est in bag.estimators_])


def main():
    args = common.base_parser(__doc__.splitlines()[1]).parse_args()
    P = common.paths(args.tag)

    X, y = common.load_shared(P["X"], P["y"])
    beta_star = np.load(P["beta_star"])

    # Paso 1: beta_hat sobre el dataset completo (serial)
    with threadpool_limits(args.inner_threads):
        beta_hat = common.fit_normal_equations(np.asarray(X), np.asarray(y))

    # Pasos 2 y 3
    mejor = np.inf
    for _ in range(args.repeats):
        with parallel_config(backend="loky",
                             inner_max_num_threads=args.inner_threads):
            with common.cronometro() as t:
                betas = bootstrap(X, y, args.n_resamples, args.n_jobs)
        mejor = min(mejor, t.dt)

    common.reportar(VERSION, args, mejor, betas, beta_hat, beta_star)


if __name__ == "__main__":
    main()
