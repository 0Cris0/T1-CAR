# Informe

# OJO: Rehacer informe en base a ejecutar Sebas u otro computador con B = 48 como pide el enunciado


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

**NOTA**: Además, se guardan los arreglos en un *.npy* para que las 3 variantes de bootstrapping puedan usar los mismos datos y puedan mediante *mmap mode="r"* poder abrirlos sin duplicar la memoria entre procesos. Esto es lo que queda en la carpeta **/data** en forma de `beta_star.npy, X.npy, y.npy`

- **semillas independientes por resample y desacopladas de p**: Como ```default_rng(seed + b)```no da garantías de independencias entre streams, pudiendo haber colisiones, si se usara podría hacer que el resultado de las operaciones dependa del orden de ejecución y por ende de *p*. Por ello se plantea usar `p.random.SeedSequence().spawn(B)` en ```common.child_seeds```, para generar B streams estadísticamente independientes (lo garantiza al alejarlos mucho mediante el uso de hashing criptográfico) y generar streams que provoquen que el mismo resample *b* (*b=i*) siempre reciba la misma semilla hija, sin importar cuantos procesos p se usen o cómo *joblib* reparta las tareas entre ellos
     - Esto es la base que permite comparar tiempos de ejecución para sitinto *p* sin que el valor del resultado cambie


- **= numero de hilos internos**: Como el producto de X con su transpuesta se calcula con BLAS y este puede llegar a usar múltiples hilos, la suma de los productos internos puede hacerse en distintos orden según esos distintos hilos, afectando el resultado, por ello para lograr una reproductibilidad más exacta conviene usar el mismo número de hilos internos

- **= versión de las librerías**: como la implementación depende mucho del funcionamiento de las librerías (Numpy, Sklearn, ...), variaciones de versiones en estas pueden generar algoritmos ligeramente distintos, lo que puede generar cierta variabilidad aunque sea menor en los resultados, por ello conviene usar = version para lograr una buena reproductibilidad 

- **= dtype**: La cossa cambia si uno usa puntos flotantes de tipo *float32* que *float64*, por ello para fomentar la reproducibilidad, conviene usar = tipo de dato

- **no reusar estado entre versiones/subprocesos**: así evitando que el pool del backend utilizado pueda contaminar la siguiente medición, por eso ```bench.py```hace que cada subproceso sea una nueva versión. Esto más que nada respecto a reproducibilidad temporal que resultados


### d) **Sobre Backend *multiprocessing* de *joblib***

#### d.1) Sobre asignación de memoria al crear un proceso
El backend de **multiprocessing** utiliza por defecto **loky** (además se setea **loky** a lo largo de la tarea) y gestiona la creación de los **p** procesos hijo. Para la creación de procesos normalmente se usa **fork** en Linux y **spawn** en macOS y Windows
- **fork:** aquí el SO clona el proceso padre, creando un proceso hijo que tiene una copia ideńtica del espacio de direcciones y la tabla de paǵinas de la memoria virtual del padre
- **spawn:** inicia un ejecutable de Python desde 0, completamente limpio: nuevo interprete, reimporta módulos y reconstruye el contexto de ejecución pasando los argumentos vía serialización

En cuanto a la asignación de memoria, **fork** intenta utilizar **Copy on Write (CoW)**, evitando asignar nueva memoria RAM física, por ende duplicando los datos del programa, sino que hace que el proceso hijo comparta las mismas páginas físicas de la RAM que su padre bajo solo permisos de lectura. Así, solo cuando el padre o el hijo intenten escribir en esa página, el sistema hage un **page fault**, duplicando esa página específica y le asigna memoria RAM física dedicada al hijo. Pero, existe el gran problema que el Garbage Collector y el contador de referencias modifican los encabezados (contadores de referencia) de los objetos en memoria constantemente, incluso para operaciones de lectura, lo que rompe la optimización del **COW**, forzando al SO a duplicar las páginas de RAM innecesariamente.
- Si se utilizara **spawn**, se asigna un bloque de RAM independiente desde el inicio, por lo que eso no pasa


#### d.2) Lo que ocurre con X e y al lanzar los p procesos
En cuanto al manejo de **X** e **y** en *Joblib*, para arreglos grandes de NumPy (umbral de 1 MB aprox) *joblib* usa **Memory Mapping (memmap)**, almacenando termporalmente los arreglos en una RAM compartida (/dev/shm) antes de lanzar las tareas paralelas, entregando solo punteros de lectura mapeados en memoria a cada uno de los **p** procesos (no copias de los datos). Así los procesos leen directamente de los bloques de la matriz original de la RAM, generando un acceso concurrente sin duplicación.

Esto se hace para evitar caer en usar los pipes estandar de **multiprocessing**, en los cuales se tendría que aplicar  **pickling (serialización)** a los arreglos y transmitirlos mediante los canales (sockets/pipes) de IPC (comunicación entre procesos) a cada uno de los **p** procesos, lo cual generaría un consumo masivo de memoria al duplicar **p** veces los arreglos en la RAM, además de añadir la penalización temporal relacionada a la serialización


### e) **Sobre resultados de la actividad del SO basado en *bs_numpy.py***

(INSERTAR)

### f) **Análisis y comparación de tiempos de ejecución en bae a *p* para los 3 algoritmos**
Para este informe se consideran 2 computadores:
- Sebas: 4 cores físicos
- Kintsugi: 4 cores físicos
como ambos tienen un máximo de cores = 4, se considera la ejecución de las 3 versiones con **p** = {1, 2, 3, 4} procesos

Los resultados son los siguientes:
# TODO: Actualizar en base a nueva run de Sebas con B=48
#### f.1) Sebas 
(INSERTAR tabla de TIEMPOS_SEBAS.csv)
(INSERTAR grafico de [results/tiempo_vs_p_sebas.png] )

#### f.2) Kintsugi
(INSERTAR de tiempos TIEMPOS_SEBAS.csv)
(INSERTAR grafico de [results/tiempo_vs_p_kintsugi.png] )

#### f.3) Comparación de resultados de las 3 variantes

De ellos, se puede apreciar la tendencia de **bs_auto** y **bs_sklearn** a tener mayor tiempo de ejecución que **bs_numpy**, donde **bs_auto** es el de peor desempeño, mientras que **bs_numpy** es muchísimo menor.

Otro aspecto a considerar, es la tendencia de las variantes a intentar disminuir sus tiempos de ejecución a medida que se aumenta el número de procesos, pero llegar a una especie de "meseta", donde no pareciera seguir mejorando considerablemente al acercarse a los **p = p_max**

Todo lo anterior se puede atribuir a que:
- **bs_numpy** construye cada muestra usando pesos, con los cuales decide descartar aquellas observaciones de peso 0, y resuelve las operaciones algebraicas con ecuaciones normales dadas por ```solve```, así ahorrando tiempo y cómputo.
- **bs_sklearn** utiliza el mismo sistema de muestreo que el anterior, pero resuelve mediante ```LinearRegression```el cual se basa en el método de mínimos cuadrados y SVD, lo cua les mas costoso que resolver con las operaciones normales y por ende tiene un peor desempeño
- **bs_auto** en cambio, deja que ```BagginRegressor``` se encargue del muestreo bootstrap y ajuste interno, trabajando con las **N** filas completas, incluyendo las filas de peso 0, por ende requiriendo más operaciones y teniendo el peor tiempo de todos.
 


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