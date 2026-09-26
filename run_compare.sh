#!/usr/bin/env bash
# sweep.sh -- corre bs_numpy.py con todas las combinaciones (p, t) tales
#             que p * t <= p_max, y vuelca la salida a <hostname>.log
#
# Uso:
#   ./sweep.sh P_MAX [args extra para bs_numpy.py...]
#
# Ejemplo:
#   ./sweep.sh 16 --n-resamples 2000 --repeats 3

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Uso: $0 P_MAX [args extra para bs_numpy.py...]" >&2
    exit 1
fi

P_MAX="$1"
shift
EXTRA_ARGS=("$@")

SCRIPT="bs_numpy.py"
MACHINE_NAME="$(whoami -s 2>/dev/null || whoami)"
LOGFILE="${MACHINE_NAME}.log"

echo "Sweep iniciado $(date '+%Y-%m-%d %H:%M:%S') -- host=${MACHINE_NAME} p_max=${P_MAX}" | tee -a "$LOGFILE"

for (( p=1; p<=P_MAX; p++ )); do
    t_max=$(( P_MAX / p ))
    for (( t=1; t<=t_max; t++ )); do
        {
            echo ""
            echo "=== $(date '+%Y-%m-%d %H:%M:%S') p=${p} inner-threads=${t} (p*t=$((p*t))) ==="
        } | tee -a "$LOGFILE"

        # Se captura tanto stdout como stderr y se anexan al log,
        # y ademas se muestran en pantalla en tiempo real.
        python "$SCRIPT" -p "$p" --inner-threads "$t" "${EXTRA_ARGS[@]}" \
            2>&1 | tee -a "$LOGFILE"
    done
done

echo "" | tee -a "$LOGFILE"
echo "Sweep finalizado $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOGFILE"