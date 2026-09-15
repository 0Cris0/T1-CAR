"""
(a) Generacion de los datos sinteticos con semilla aleatoria fija.

    (I)   beta* in R^{k+1}, iid N(0,1)
    (II)  X in R^{N x (k+1)}: columna de unos + N x k entradas iid N(0,1)
    (III) y = X beta* + ruido,  ruido iid N(0,1)

Los arreglos se guardan en .npy para que las tres versiones del bootstrapping
usen EXACTAMENTE los mismos datos (comparacion justa) y puedan abrirlos con
mmap_mode="r" sin duplicar memoria entre procesos.

Uso:
    python gen_data.py                      # N=100000, k=300
    python gen_data.py -N 20000 -k 50 --tag mini
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import common


def generar(N: int, k: int, seed: int = common.SEED, chunk: int = 10_000):
    rng = np.random.default_rng(seed)

    # (I) coeficientes verdaderos
    beta_star = rng.standard_normal(k + 1)

    # (II) matriz de diseno con la columna de unos al inicio
    X = np.empty((N, k + 1), dtype=np.float64)
    X[:, 0] = 1.0
    # Se llena por bloques de filas: rng.standard_normal((N,k)) crearia un
    # temporal de 240 MB ADEMAS de X, duplicando el peak de memoria.
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        X[i:j, 1:] = rng.standard_normal((j - i, k))

    # (III) salida con ruido
    y = X @ beta_star + rng.standard_normal(N)
    return beta_star, X, y


def main():
    ap = argparse.ArgumentParser(description="Genera X, y y beta* sinteticos.")
    ap.add_argument("-N", type=int, default=common.N_DEF)
    ap.add_argument("-k", type=int, default=common.K_DEF)
    ap.add_argument("--seed", type=int, default=common.SEED)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.makedirs(common.DATA_DIR, exist_ok=True)
    P = common.paths(args.tag)

    with common.cronometro() as t:
        beta_star, X, y = generar(args.N, args.k, args.seed)

    np.save(P["beta_star"], beta_star)
    np.save(P["X"], X)
    np.save(P["y"], y)
    with open(P["meta"], "w") as f:
        json.dump(
            {"N": args.N, "k": args.k, "seed": args.seed,
             "dtype": "float64", "X_MB": round(X.nbytes / 2**20, 1)},
            f, indent=2,
        )

    print(f"X {X.shape} ({X.nbytes / 2**20:.1f} MB)   y {y.shape}   "
          f"beta* {beta_star.shape}")
    print(f"generado en {t.dt:.2f} s -> {common.DATA_DIR}/")


if __name__ == "__main__":
    main()
