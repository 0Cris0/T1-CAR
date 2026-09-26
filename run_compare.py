#!/usr/bin/env python3
"""
run_compare.py -- version en Python de run_compare.sh. Corre directo en
                  Windows (CMD/PowerShell), Linux o macOS, sin bash ni WSL.

Corre bs_numpy.py con todas las combinaciones (p, t) tales que p * t <= P_MAX,
anexa la salida a <hostname>.log y ademas la muestra en pantalla.

Uso:
    python run_compare.py P_MAX [args extra para bs_numpy.py...]
    (en Windows, si "python" no se reconoce, usa "py" en su lugar)

Ejemplo:
    python run_compare.py 16 --n-resamples 2000 --repeats 3
"""
import os
import socket
import subprocess
import sys
from datetime import datetime

# bs_numpy.py debe estar en la misma carpeta que este archivo
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bs_numpy.py")


def ahora():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    # Si la consola no puede mostrar algun caracter, lo reemplaza en vez de caerse
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    try:
        p_max = int(sys.argv[1])
    except (IndexError, ValueError):
        print("Uso: python run_compare.py P_MAX [args extra para bs_numpy.py...]",
              file=sys.stderr)
        sys.exit(1)

    extra_args = sys.argv[2:]

    # Nombre corto del equipo (equivalente a `hostname -s`)
    machine_name = socket.gethostname().split(".")[0]
    logfile = f"{machine_name}.log"

    # El hijo escribe sin buffer (salida en tiempo real) y en UTF-8
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}

    # newline="\n": el log queda con saltos de linea Unix, igual que con bash
    with open(logfile, "a", encoding="utf-8", newline="\n") as log:

        def tee(linea=""):
            """Muestra la linea en pantalla y la anexa al log (como `tee -a`)."""
            print(linea, flush=True)
            log.write(linea + "\n")
            log.flush()

        tee(f"Sweep iniciado {ahora()} -- host={machine_name} p_max={p_max}")

        for p in range(1, p_max + 1):
            t_max = p_max // p
            for t in range(1, t_max + 1):
                tee()
                tee(f"=== {ahora()} p={p} inner-threads={t} (p*t={p * t}) ===")

                # sys.executable: el mismo Python (con su numpy) que corre este script
                cmd = [sys.executable, SCRIPT, "-p", str(p),
                       "--inner-threads", str(t), *extra_args]

                # stdout y stderr juntos, como `2>&1 | tee -a`
                with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, env=env,
                                      encoding="utf-8", errors="replace") as proc:
                    for linea in proc.stdout:
                        tee(linea.rstrip("\n"))

                # Igual que `set -e` + `pipefail`: si una corrida falla, se detiene todo
                if proc.returncode != 0:
                    print(f"ERROR: bs_numpy.py termino con codigo {proc.returncode} "
                          f"(p={p}, t={t}). El detalle quedo en {logfile}",
                          file=sys.stderr)
                    sys.exit(1)

        tee()
        tee(f"Sweep finalizado {ahora()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSweep interrumpido (Ctrl+C).", file=sys.stderr)
        sys.exit(130)
