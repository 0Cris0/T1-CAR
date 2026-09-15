# Tarea 1 — Bootstrapping para regresión lineal (IIC3533)

## Requisitos
```bash
pip install numpy scikit-learn joblib threadpoolctl matplotlib pandas
```

## Uso
```bash
# (a) generar los datos:  X (100000 x 301, 240 MB), y, beta*
python gen_data.py

# (b) cada versión por separado
python bs_auto.py    -p 8
python bs_sklearn.py -p 8
python bs_numpy.py   -p 8

# barrido completo de p + gráficos + tabla  (repetir en cada computador)
python bench.py

# prueba rápida con datos reducidos
python gen_data.py -N 20000 -k 60 --tag mini
python bench.py --tag mini -B 12 --pmax 4
```

Flags útiles: `--inner-threads N` (hilos BLAS por proceso; 1 evita
sobresuscripción), `--dtype float32`, `--repeats 3` (reporta el mínimo),
`--csv archivo.csv`.

Ver `NOTAS.md` para la bitácora de optimización y qué reportar.
