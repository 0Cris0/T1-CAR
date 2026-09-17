# Informe

## Respuesta a cada inciso/ítem

### b) **Sobre iteraciones de cada versión de bootstrapping**

(INSERTAR)

### c) **Sobre correctitud y reproductibilidad**

#### c.1) Sobre la similitud y consistentica de los intervalos de confianza

Respecto a los IC, a nivel general, se usa una función *ci_percentil()* para poder calcularlos de igual manera al final de la ejecución de cada una de las variantes del algoritmo, entregando los *betas* calculados y el *alpha=0.05*, resultando en descartar 1 valor por lado (descarta el 2.5% superior o inferior, equivalente al multiplicar por B=48 --> 1) y lograr un nivel de confianza de más menos 95% aproximadamente. Para verificar esto, se utiliza la función *cobertura()*, para poder calcular la fracción de los coeficientes verdaderos (BETA_STAR) que quedan dentro del IC, la cual se espera que rondee ese 95% en los resultados empíricos

En cuanto a los resultados obtenidos, al revisar el reporte obtenido en *tiempos_XXXX.csv* con XXXX = computador en el que fue ejecutado los algoritmos, se puede apreciar que los IC tienden a tener características idénticas (ancho del IC promedio, cobertura de BETA_STAR y error relativo de BETA_STAR) para las variantes *sklearn* y *numpy*, y ligeramente distintas para las de la versión *auto* (tiene ligeramente peor cobertura, error relativo y es menos ancho que las otras 2)

En concreto, en el computador 2 se logran valores de ancho alrededor de *0.011812*, cobertura cercana al *95%* y un error relativo de *0.00042929*, en cambio en el computador 1 se logran valores de *0.024227* (+ ancho), *88,5%* (- cobertura) y *0.00220527* (+ error relativo).

#### c.2) Sobre la equivalencia entre los resultados de las distintas versiones: bs_auto, bs_sklearn y bs_numpy

En cuanto a equivalencia, tanto *bs_numpy* como *bs_sklearn* usan ```common.child.seeds(B)``` (que usa SeedSequence) y ```common.bootstrap_weights(rng, N)```bajo la misma secuencia de semillas generados por ella, lo que significa que ambas versionas sortean los mismos N índices para el resample *b*, por ende armando los mismos X^(b) e y^(b) ponderado, siendo la única diferencia entre ambas el algoritmo usado para resolver el álgebra lineal del problema:
- bs_numpy: resuelve mediante ecuaciones normales con ```np.linalg.solve```, que hace uso de factorización LU en el producto de X con su transpuesta
- bs_sklearn: resuelve mediante ```LinearRegression```, que se basa en SVD (Descomposición de Valores Singulares)
Como ambos métodos resulelven el mismo sistema, las diferencias entre sus resultades probablemente se deban a errores de redondeo de puntos flotantes, lo cuál tiene ordenes de magnitud muy bajos (1*10^{10} o {12} aprox), lo que se traduce en resultados casi idénticos, como se puede ver en las métricas antes mencionadas, donde para ambos computadores dan = valores.

Por otro lado, *bs_auto* es comparable pero no tan idéntido, de ahí la ligera diferencia entre valores vistos previamente, pues utiliza ```BaggingRegressor```el cual hace su propio muestreo bootstrap interno con su propio generador de índices distntos al SeedSequence utilizado por los otros 2. Así, aunque se esté usando la misma semilla, al tener un mecanismo de generación de números aleatorios distinto al ```common.child.seeds(B)```, los índices que termina sorteando no coinciden resample a resample con los de *bs_numpy* y *bd_sklearn*
- Aquí cabe mencionar que la diferencia radica principalmente en ese muestreo distintos del conjunto de B coeficientes, pero el mecanismo de bootstrap se mantiene el los 3, por lo que siempre es comparable
- Otro aspecto a considerar, aunque del lado computacional, no de resultados, es que *bs_auto* no descarta la fila de peso 0 a diferencia de la implementación en ```common.weighted_design```, lo que si las descarta, lo que lo hace más lento como se puede ver en los resultados de los .csv

#### c.3) Respecto a la reproductibilidad

Para que los resultados sean reproducibles se necesita:
- **= semilla en la generación de datos**: Por ello en ```gen_data.py```se usa una semilla fija al momento de ejecutar ```np.random.default_rng(seed)``` al crear los datos sintéticos aleatorios, lo que logra que X y BETA_STAR sean ideńtidos entre ejecuciones y computadores.

**NOTA**: Además, se guardan los arreglos en un *.npy* para que las 3 variantes de bootstrapping puedan usar los mismos datos y puedan mediante *mmap mode="r"* poder abrirlos sin duplicar la memoria entre procesos.
- Esto es lo que queda en la carpeta **/data** en forma de `beta_star.npy, X.npy, y.npy`

- **semillas independientes por resample y desacopladas de p**: Como ```default_rng(seed + b)```no da garantías de independencias entre streams, pudiendo haber colisiones, si se usara podría hacer que el resultado de las operaciones dependa del orden de ejecución y por ende de *p*. Por ello se plantea usar `p.random.SeedSequence().spawn(B)` en ```common.child_seeds```, para generar B streams estadísticamente independientes (lo garantiza al alejarlos mucho mediante el uso de hashing criptográfico) y generar streams que provoquen que el mismo resample *b* (*b=i*) siempre reciba la misma semilla hija, sin importar cuantos procesos p se usen o cómo *joblib* reparta las tareas entre ellos
- - Esto es la base que permite comparar tiempos de ejecución para sitinto *p* sin que el valor del resultado cambie


- **= numero de hilos internos**: Como el producto de X con su transpuesta se calcula con BLAS y este puede llegar a usar múltiples hilos, la suma de los productos internos puede hacerse en distintos orden según esos distintos hilos, afectando el resultado, por ello para lograr una reproductibilidad más exacta conviene usar el mismo número de hilos internos

- **= versión de las librerías**: como la implementación depende mucho del funcionamiento de las librerías (Numpy, Sklearn, ...), variaciones de versiones en estas pueden generar algoritmos ligeramente distintos, lo que puede generar cierta variabilidad aunque sea menor en los resultados, por ello conviene usar = version para lograr una buena reproductibilidad 

- **= dtype**: La cossa cambia si uno usa puntos flotantes de tipo *float32* que *float64*, por ello para fomentar la reproducibilidad, conviene usar = tipo de dato

- **no reusar estado entre versiones/subprocesos**: así evitando que el pool del backend utilizado pueda contaminar la siguiente medición, por eso ```bench.py```hace que cada subproceso sea una nueva versión. Esto más que nada respecto a reproducibilidad temporal que resultados


### d) **Sobre Backend *multiprocessing* de *joblib***

(INSERTAR)

### e) **Sobre resultados de la actividad del SO basado en *bs_numpy.py***

(INSERTAR)

### f) **Análisis y comparación de tiempos de ejecución en bae a *p* para los 3 algoritmos
**
(INSERTAR)

### g) **Cálculo, graficación y comparación de T(p), S(p) y E(p)**

(INSERTAR)

### h) **Cálculo de oberhead y sus principales fuentes**

(INSERTAR)

### i) **Experimentación con distinto número de procesos *p* y threads internos *t*, graficación y análisis**

(INSERTAR)

### j) **Comparación de resultados entre distintos computadores**

(INSERTAR)

## Uso de IA

(INSERTAR)

Se utilizó Gemini para comprender los conceptos y librerías a utilizar