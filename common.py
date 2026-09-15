"""
Utilidades compartidas para la Tarea 1 (IIC3533).

Contiene: rutas de datos, carga compartida por memmap, generacion de semillas
independientes por resample, ecuaciones normales, intervalo de confianza
bootstrap y registro de tiempos / informacion de la maquina.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time

import numpy as np

# ----------------------------------------------------------------------------
# Configuracion del experimento (enunciado)
# ----------------------------------------------------------------------------
SEED = 20262          # semilla fija -> datos y resamples reproducibles
N_DEF = 100_000       # observaciones
K_DEF = 300           # variables de entrada
B_DEF = 48            # resamples bootstrap
ALPHA = 0.05          # 95% de confianza

DATA_DIR = "data"
RESULTS_DIR = "results"


def paths(tag: str = "") -> dict:
    """Rutas de los .npy. `tag` permite tener datasets reducidos para pruebas."""
    suf = f"_{tag}" if tag else ""
    return {
        "X": os.path.join(DATA_DIR, f"X{suf}.npy"),
        "y": os.path.join(DATA_DIR, f"y{suf}.npy"),
        "beta_star": os.path.join(DATA_DIR, f"beta_star{suf}.npy"),
        "meta": os.path.join(DATA_DIR, f"meta{suf}.json"),
    }


# ----------------------------------------------------------------------------
# Carga compartida (una sola vez por proceso)
# ----------------------------------------------------------------------------
_CACHE: dict = {}


def load_shared(xpath: str, ypath: str):
    """Abre X e y como memmap, cacheando por proceso.

    Por que memmap y no pasar el ndarray como argumento a Parallel:
      * X pesa N*(k+1)*8 = 240 MB. Enviarlo como argumento obliga a joblib a
        serializarlo / volcarlo a disco antes de cada lote de tareas.
      * Con memmap el kernel mantiene UNA sola copia en page cache y todos los
        procesos la comparten: la RAM no se multiplica por p.
      * Los workers de loky se reutilizan entre tareas, asi que el open() se
        paga una vez por worker, no una vez por resample.
    """
    key = (os.path.abspath(xpath), os.path.abspath(ypath))
    if key not in _CACHE:
        _CACHE[key] = (
            np.load(xpath, mmap_mode="r"),
            np.load(ypath, mmap_mode="r"),
        )
    return _CACHE[key]


# ----------------------------------------------------------------------------
# Semillas independientes por resample
# ----------------------------------------------------------------------------
def child_seeds(B: int, seed: int = SEED):
    """B semillas hijas estadisticamente independientes.

    Usar `default_rng(seed + b)` NO garantiza independencia entre streams.
    SeedSequence.spawn() si, y ademas hace que el resultado sea identico
    para cualquier p (el resample b siempre usa la misma semilla).
    """
    return np.random.SeedSequence(seed).spawn(B)


# ----------------------------------------------------------------------------
# Ajuste por ecuaciones normales
# ----------------------------------------------------------------------------
def fit_normal_equations(X, y):
    """Resuelve (X^T X) beta = X^T y.

    `solve` (factorizacion LU) en vez de `inv(A) @ b`: la inversa explicita
    cuesta ~3x mas flops y es numericamente peor.
    """
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)


def bootstrap_weights(rng, N: int):
    """Sortea N indices con reemplazo y los devuelve como vector de conteos.

    Equivale exactamente al Paso 2(I) del enunciado, pero en lugar de
    materializar X^(b) (240 MB por resample) se usa la identidad

        X^(b)T X^(b) = X^T diag(w) X = (sqrt(w) X)^T (sqrt(w) X)
        X^(b)T y^(b) = X^T diag(w) y = (sqrt(w) X)^T (sqrt(w) y)

    Ademas ~37% de las filas tienen w_i = 0 (prob. 1/e de no ser sorteada),
    y esas filas no aportan nada al producto: se pueden descartar.
    """
    idx = rng.integers(0, N, N)
    w = np.bincount(idx, minlength=N)
    nz = np.flatnonzero(w)
    return nz, np.sqrt(w[nz])


def weighted_design(X, y, nz, sw, dtype=None):
    """Construye (sqrt(w) X_nz, sqrt(w) y_nz) con una sola copia de memoria."""
    Xw = np.asarray(X[nz], dtype=dtype)   # gather: unica copia (~0.63 N filas)
    Xw *= sw[:, None]                     # escalado in-place, sin buffer extra
    yw = np.asarray(y[nz], dtype=dtype) * sw
    return Xw, yw


# ----------------------------------------------------------------------------
# Intervalo de confianza bootstrap
# ----------------------------------------------------------------------------
def ci_percentil(betas, alpha: float = ALPHA):
    """Paso 3 tal como lo define el enunciado.

    Ordena los B valores de cada coeficiente, descarta floor(alpha/2 * B) por
    cada extremo y toma los valores extremos restantes.
    Con B = 48: floor(0.025*48) = 1 -> se descarta 1 por lado, IC = [s[1], s[46]].
    """
    betas = np.asarray(betas)
    B = betas.shape[0]
    s = np.sort(betas, axis=0)
    m = int(np.floor(alpha / 2 * B))
    return s[m], s[B - 1 - m]


def ci_interpolado(betas, alpha: float = ALPHA):
    """Variante estandar (percentiles interpolados). Util como contraste."""
    return tuple(
        np.percentile(betas, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    )


def cobertura(lo, hi, beta_star):
    """Fraccion de coeficientes verdaderos que cae dentro de su IC."""
    return float(np.mean((beta_star >= lo) & (beta_star <= hi)))


# ----------------------------------------------------------------------------
# Maquina y registro de resultados
# ----------------------------------------------------------------------------
def _cpu_model() -> str:
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            import subprocess
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
    except Exception:
        pass
    return platform.processor() or "desconocido"


def machine_info() -> dict:
    import joblib
    try:
        fisicos = joblib.cpu_count(only_physical_cores=True)
    except TypeError:
        fisicos = joblib.cpu_count()
    return {
        "host": platform.node(),
        "so": f"{platform.system()} {platform.release()}",
        "cpu": _cpu_model(),
        "nucleos_fisicos": fisicos,
        "nucleos_logicos": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }


def blas_info() -> str:
    try:
        from threadpoolctl import threadpool_info
        return ", ".join(
            f"{d.get('internal_api')} {d.get('version')} ({d.get('num_threads')} hilos)"
            for d in threadpool_info()
        )
    except Exception:
        return "desconocido"


def append_csv(path: str, fila: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    nuevo = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(fila))
        if nuevo:
            wtr.writeheader()
        wtr.writerow(fila)


class cronometro:
    """with cronometro() as t: ...   ->  t.dt en segundos (wall clock)."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self.t0
        return False


# ----------------------------------------------------------------------------
# CLI comun a las tres versiones
# ----------------------------------------------------------------------------
def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("-p", "--n-jobs", type=int, default=1,
                    help="procesos paralelos (p)")
    ap.add_argument("-B", "--n-resamples", type=int, default=B_DEF)
    ap.add_argument("--tag", default="",
                    help="sufijo del dataset (ej: 'mini' para pruebas)")
    ap.add_argument("--inner-threads", type=int, default=1,
                    help="hilos BLAS DENTRO de cada proceso "
                         "(1 evita sobresuscripcion)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeticiones; se reporta el minimo")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--csv", default=None, help="archivo CSV donde acumular tiempos")
    ap.add_argument("--save-betas", default=None, help=".npy con los B coeficientes")
    ap.add_argument("--quiet", action="store_true")
    return ap


def reportar(version: str, args, t_boot: float, betas, beta_hat, beta_star):
    betas = np.asarray(betas)
    lo, hi = ci_percentil(betas, ALPHA)
    info = machine_info()
    fila = {
        "version": version,
        "host": info["host"],
        "cpu": info["cpu"],
        "nucleos_fisicos": info["nucleos_fisicos"],
        "p": args.n_jobs,
        "B": args.n_resamples,
        "inner_threads": args.inner_threads,
        "dtype": args.dtype,
        "t_bootstrap_s": round(t_boot, 4),
        "ancho_ic_medio": round(float(np.mean(hi - lo)), 6),
        "cobertura_beta_star": round(cobertura(lo, hi, beta_star), 4),
        "err_rel_beta_hat": round(
            float(np.linalg.norm(betas.mean(0) - beta_hat)
                  / np.linalg.norm(beta_hat)), 8),
    }
    if args.csv:
        append_csv(args.csv, fila)
    if args.save_betas:
        os.makedirs(os.path.dirname(args.save_betas) or ".", exist_ok=True)
        np.save(args.save_betas, betas)
    if not args.quiet:
        print(f"[{version}] p={args.n_jobs} B={args.n_resamples} "
              f"hilos_internos={args.inner_threads} dtype={args.dtype}")
        print(f"  tiempo bootstrap : {t_boot:8.3f} s")
        print(f"  ancho IC medio   : {fila['ancho_ic_medio']:.6f}")
        print(f"  cobertura beta*  : {fila['cobertura_beta_star']:.3f}")
        print(f"  ||mean(b)-bhat|| : {fila['err_rel_beta_hat']:.2e}")
    return fila
