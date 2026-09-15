"""
bench.py  --  Barrido de p para las tres versiones + graficos.

Cada corrida se lanza como subproceso nuevo para que ninguna version herede
el estado (pool de loky, page cache calentado, hilos BLAS) de la anterior.

Uso:
    python bench.py                          # barrido completo + graficos
    python bench.py --versions bs_numpy      # solo una version
    python bench.py --solo-graficos          # regenera graficos desde el CSV
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import common

VERSIONES = ["bs_auto", "bs_sklearn", "bs_numpy"]
ETIQUETAS = {
    "bs_auto": "bs_auto (BaggingRegressor)",
    "bs_sklearn": "bs_sklearn (joblib + LinearRegression)",
    "bs_numpy": "bs_numpy (joblib + ecuaciones normales)",
}
COLORES = {"bs_auto": "#C44E52", "bs_sklearn": "#DD8452", "bs_numpy": "#4C72B0"}
MARCAS = {"bs_auto": "s", "bs_sklearn": "^", "bs_numpy": "o"}


# ---------------------------------------------------------------------------
def valores_de_p(maximo: int, B: int) -> list[int]:
    """1..maximo. Se incluyen los p que NO dividen a B a proposito: el
    desbalance de carga se ve como escalones en la curva de speedup."""
    return list(range(1, maximo + 1))


def correr(version, p, args):
    cmd = [
        sys.executable, f"{version}.py",
        "-p", str(p),
        "-B", str(args.B),
        "--inner-threads", str(args.inner_threads),
        "--dtype", args.dtype,
        "--repeats", str(args.repeats),
        "--csv", args.csv,
    ]
    if args.tag:
        cmd += ["--tag", args.tag]
    if p == 1 and version == "bs_numpy":
        cmd += ["--save-betas", os.path.join(common.RESULTS_DIR, "betas.npy")]
    print("  $ " + " ".join(cmd[1:]))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
def graficos(csv_path, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
        "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.grid": True, "grid.alpha": 0.3,
    })

    df = pd.read_csv(csv_path)
    df = df.sort_values(["version", "p"])
    host = df["host"].iloc[0]
    cpu = df["cpu"].iloc[0]
    B = int(df["B"].iloc[0])
    sub = f"{cpu}  |  B={B} resamples  |  {host}"

    os.makedirs(common.RESULTS_DIR, exist_ok=True)

    # --- 1. tiempo vs p -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for v, g in df.groupby("version"):
        ax.plot(g["p"], g["t_bootstrap_s"], marker=MARCAS.get(v, "o"),
                color=COLORES.get(v), label=ETIQUETAS.get(v, v), lw=2, ms=7)
    ax.set_yscale("log")
    ax.set_xlabel("Numero de procesos $p$")
    ax.set_ylabel("Tiempo del bootstrapping $T(p)$  [s]")
    ax.set_title("Tiempo de ejecucion vs. procesos\n" + sub, fontsize=13)
    ax.set_xticks(sorted(df["p"].unique()))
    ax.legend(title="Implementacion")
    fig.savefig(os.path.join(common.RESULTS_DIR, f"tiempo_vs_p{tag}.png"))
    plt.close(fig)

    # --- 2. speedup ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    pmax = int(df["p"].max())
    ax.plot([1, pmax], [1, pmax], "k--", lw=1.5, label="Ideal $S_p = p$")
    for v, g in df.groupby("version"):
        t1 = g.loc[g["p"] == 1, "t_bootstrap_s"]
        if t1.empty:
            continue
        ax.plot(g["p"], float(t1.iloc[0]) / g["t_bootstrap_s"],
                marker=MARCAS.get(v, "o"), color=COLORES.get(v),
                label=ETIQUETAS.get(v, v), lw=2, ms=7)
    ax.set_xlabel("Numero de procesos $p$")
    ax.set_ylabel("Speedup $S_p = T(1)/T(p)$")
    ax.set_title("Speedup vs. procesos\n" + sub, fontsize=13)
    ax.set_xticks(sorted(df["p"].unique()))
    ax.legend()
    fig.savefig(os.path.join(common.RESULTS_DIR, f"speedup{tag}.png"))
    plt.close(fig)

    # --- 3. eficiencia ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(1.0, color="k", ls="--", lw=1.5, label="Ideal $E_p = 1$")
    for v, g in df.groupby("version"):
        t1 = g.loc[g["p"] == 1, "t_bootstrap_s"]
        if t1.empty:
            continue
        ax.plot(g["p"], float(t1.iloc[0]) / (g["t_bootstrap_s"] * g["p"]),
                marker=MARCAS.get(v, "o"), color=COLORES.get(v),
                label=ETIQUETAS.get(v, v), lw=2, ms=7)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Numero de procesos $p$")
    ax.set_ylabel("Eficiencia $E_p = S_p / p$")
    ax.set_title("Eficiencia paralela vs. procesos\n" + sub, fontsize=13)
    ax.set_xticks(sorted(df["p"].unique()))
    ax.legend()
    fig.savefig(os.path.join(common.RESULTS_DIR, f"eficiencia{tag}.png"))
    plt.close(fig)

    # --- 4. intervalos de confianza ----------------------------------------
    bpath = os.path.join(common.RESULTS_DIR, "betas.npy")
    if os.path.exists(bpath):
        betas = np.load(bpath)
        beta_star = np.load(common.paths(tag.lstrip("_"))["beta_star"])
        lo, hi = common.ci_percentil(betas)
        cob = common.cobertura(lo, hi, beta_star)
        m = min(25, betas.shape[1])
        j = np.arange(m)
        # Se grafica centrado en beta*: los IC (ancho ~1e-2) son invisibles
        # frente al rango de beta* (~+-3). Asi se ve directamente la cobertura.
        dentro = (beta_star[:m] >= lo[:m]) & (beta_star[:m] <= hi[:m])
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axhline(0, color="#C44E52", lw=1.5, ls="--",
                   label=r"$\beta^*_j$ verdadero")
        ax.vlines(j[dentro], (lo - beta_star)[:m][dentro],
                  (hi - beta_star)[:m][dentro],
                  color="#4C72B0", lw=4, alpha=0.7, label="IC 95% que cubre $\\beta^*_j$")
        if (~dentro).any():
            ax.vlines(j[~dentro], (lo - beta_star)[:m][~dentro],
                      (hi - beta_star)[:m][~dentro],
                      color="#C44E52", lw=4, alpha=0.7, label="IC 95% que NO cubre")
        ax.plot(j, (betas.mean(0) - beta_star)[:m], "o", color="#2A3F5F", ms=5,
                label=r"$\bar{\beta}^{(b)}_j - \beta^*_j$")
        ax.set_xlabel("Indice del coeficiente $j$")
        ax.set_ylabel(r"Desviacion respecto de $\beta^*_j$")
        ax.set_title(f"Intervalos de confianza bootstrap al 95% "
                     f"(primeros {m} de {betas.shape[1]} coeficientes)\n"
                     f"B = {betas.shape[0]} resamples  |  "
                     f"cobertura global = {cob:.1%}", fontsize=13)
        ax.legend(loc="best", fontsize=10)
        fig.savefig(os.path.join(common.RESULTS_DIR, f"intervalos{tag}.png"))
        plt.close(fig)

    # --- tabla resumen ------------------------------------------------------
    tabla = df.pivot_table(index="p", columns="version",
                           values="t_bootstrap_s").round(3)
    tabla.to_csv(os.path.join(common.RESULTS_DIR, f"tabla_tiempos{tag}.csv"))
    print("\nTiempos [s]:\n", tabla.to_string())
    print(f"\nGraficos en {common.RESULTS_DIR}/")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Barrido de p y graficos.")
    ap.add_argument("-B", type=int, default=common.B_DEF)
    ap.add_argument("--pmax", type=int, default=None,
                    help="p maximo (default: nucleos fisicos)")
    ap.add_argument("--versions", nargs="+", default=VERSIONES)
    ap.add_argument("--inner-threads", type=int, default=1)
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--solo-graficos", action="store_true")
    args = ap.parse_args()

    os.makedirs(common.RESULTS_DIR, exist_ok=True)
    info = common.machine_info()
    if args.csv is None:
        args.csv = os.path.join(common.RESULTS_DIR, f"tiempos_{info['host']}.csv")

    if not args.solo_graficos:
        if os.path.exists(args.csv):
            os.remove(args.csv)
        for clave, valor in info.items():
            print(f"  {clave:18s}: {valor}")
        print(f"  {'BLAS':18s}: {common.blas_info()}\n")
        pmax = args.pmax or info["nucleos_fisicos"] or 1
        for v in args.versions:
            print(f"== {v} ==")
            for p in valores_de_p(pmax, args.B):
                correr(v, p, args)

    graficos(args.csv, f"_{args.tag}" if args.tag else "")


if __name__ == "__main__":
    main()
