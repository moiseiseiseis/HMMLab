
# Diseño de Experimentos: Análisis Latente de EEG mediante HMM

Este documento detalla la hoja de ruta para la exploración de dinámicas cerebrales utilizando Modelos Ocultos de Markov (HMM). El diseño se divide en dos fases: optimización de la estructura matemática y validación de hipótesis biológicas.

---

## 1. Grupo I: Estructura y Arquitectura del Modelo

Esta fase busca optimizar cómo el HMM representa el espacio latente (complejidad, granularidad y geometría estadística).

### A. Parámetros de Granularidad y Geometría

| Dimensión | Hipótesis | Experimentos Sugeridos |
| --- | --- | --- |
| **Número de Estados (K)** | ¿Cuántas dinámicas latentes robustas existen en la tarea? | `K=3`, `K=4`, `K=5` |
| **Tipo de Covarianza** | ¿Los estados requieren modelar correlaciones entre features? | `diag`, `full`, `tied` |
| **Retención de Varianza (PCA)** | ¿Qué nivel de compresión permite el mejor balance señal/ruido? | `PCA90`, `PCA95`, `PCA99`, `No PCA` |

**Detalle de Covarianza:**

* `diag`: Asume features independientes (modelo más simple/robusto).
* `full`: Modela correlaciones explícitas entre canales/métricas.
* `tied`: Todos los estados comparten la misma matriz de covarianza global.

### B. Resolución Temporal

* **Ventana Temporal (Window):** Evaluación de la dinámica según la resolución (100ms, 200ms, 400ms).
* **Paso de Ventana (Step):** Impacto del solapamiento en la suavidad de las transiciones de Viterbi.

### C. Selección de Biomarcadores (Feature Engineering)

Se evalúa qué combinación de métricas organiza mejor los estados:

1. **Bandas de Frecuencia:** Theta, Alpha, Beta.
2. **Parámetros de Hjorth:** Mobility, Complexity.
3. **Entropía:** Shannon/Sample Entropy.
4. **Combinaciones:** Integración total (Bands + Hjorth + Entropy).

---

## 2. Grupo II: Hipótesis Neurocientíficas

Una vez fijado el modelo óptimo, se procede a responder preguntas biológicas sin modificar la estructura del HMM.

### A. Paradigma de Control Inhibitorio (GO vs NO-GO)

**Pregunta:** ¿Cómo altera la inhibición motora la probabilidad de transición y la ocupación de estados?

* **Métricas:** Fractional Occupancy (FO), Dwell Time (ms), Transition Probability.

### B. Análisis del Desarrollo (Adultos vs Adolescentes)

**Pregunta:** ¿La maduración cerebral altera la estabilidad de las redes latentes?

* **Comparativas:**
* Adultos (Control) vs. Adolescentes.
* Interacción: `Condition (GO/NOGO)` × `Development (Adult/Adolescent)`.



### C. Dinámica Evocada y Estabilidad

* **Evoked Dynamics:** Análisis de la probabilidad de estado con *locking* temporal al estímulo.
* **Split-half Stability:** Validación de la reproducibilidad dividiendo la muestra por sujetos y sesiones.
* **Cross-model Reproducibility:** Evaluación de si un $K$ mayor descubre estados nuevos o simplemente fragmenta redes existentes (Similitud de Coseno / Hungarian Matching).

---

## 3. Convención de Nomenclatura y Pipeline

Para evitar la explosión combinatoria, se sigue un orden jerárquico de ejecución.

### Estructura de Archivos YAML

Los experimentos se nombran siguiendo el patrón: `feat_[task]_[k]_[cov]_[pca]`.

| Experimento (Ejemplo) | Variable Principal | Propósito |
| --- | --- | --- |
| `feat_task_k3_diag_pca90` | Baseline | Punto de partida estándar. |
| `feat_task_k4_diag_pca90` | Solo K | Evaluar incremento de granularidad. |
| `feat_task_k4_full_pca90` | K + Covariance | Evaluar impacto de correlaciones en K=4. |
| `feat_task_k3_diag_pca95` | Solo PCA | Evaluar impacto de mayor retención de datos. |

### Flujo de Trabajo (Roadmap)

1. **Fase de Estructura:** Identificar la arquitectura matemática más estable (K, Cov, PCA).
2. **Fase de Selección:** Fijar el "Mejor Modelo" basado en criterios de convergencia y interpretabilidad.
3. **Fase de Inferencia:** Aplicar el modelo fijo para resolver las preguntas de investigación (Diferencias clínicas y por tarea).

> **Nota:** Se prioriza la robustez estadística sobre la cantidad de modelos. El objetivo es evitar el sobreajuste y la generación de "YAMLs oscuros" sin justificación fisiológica.

---







# GRUPO 1 — Experimentos de ESTRUCTURA DEL MODELO

Estos modifican:

```text id="2v8i7z"
cómo el HMM representa el espacio latente
```

o sea:

* complejidad
* granularidad
* geometría estadística
* compresión

Son los experimentos más “machine learning”.

---

# A. Número de estados (K)

Hipótesis:

```text id="4z6w4z"
¿cuántas dinámicas latentes robustas existen?
```

Experimentos:

```text id="m1x4o3"
K=3
K=4
K=5
```

---

# B. Tipo de covarianza

Hipótesis:

```text id="9i0g2m"
¿los estados requieren relaciones entre features?
```

Experimentos:

```text id="m5kp3q"
diag
full
tied
```

Interpretación:

| Tipo | Qué modela                   |
| ---- | ---------------------------- |
| diag | features independientes      |
| full | correlaciones entre features |
| tied | misma covarianza global      |

---

# C. PCA variance retained

Hipótesis:

```text id="9n61aa"
¿cuánta dimensionalidad necesita realmente el HMM?
```

Experimentos:

```text id="t7txde"
PCA90
PCA95
PCA99
No PCA
```

---

# D. Ventana temporal

Hipótesis:

```text id="9a9crs"
¿la dinámica depende de la resolución temporal?
```

Experimentos:

```text id="7v6goz"
100 ms
200 ms
400 ms
```

y también:

```text id="sx8pji"
step sizes distintos
```

---

# E. Features usadas

Hipótesis:

```text id="rv2c0m"
¿qué biomarcadores organizan los estados?
```

Experimentos:

## Solo bandas

```text id="fkrgxa"
Theta
Alpha
Beta
```

---

## Solo Hjorth

```text id="c4r8gr"
Mobility
Complexity
```

---

## Solo Entropía

```text id="e2j8rq"
Entropy
```

---

## Combinaciones

```text id="jlwm17"
Bands + Hjorth
Bands + Entropy
Todo junto
```

---

# F. Seeds / reproducibilidad

Hipótesis:

```text id="jlwm17"
¿el modelo converge a estados similares?
```

Experimentos:

```text id="jlwm17"
seed=1
seed=42
seed=123
```

---

# G. Inicialización

Hipótesis:

```text id="jlwm17"
¿el óptimo es estable o depende del init?
```

Experimentos:

```text id="jlwm17"
kmeans init
random init
multiple restarts
```

---

# GRUPO 2 — Experimentos de HIPÓTESIS NEUROCIENTÍFICA

Estos NO cambian el modelo.

Cambian:

```text id="jlwm17"
qué pregunta biológica haces
```

---

# A. GO vs NOGO

Hipótesis:

```text id="jlwm17"
¿la dinámica cerebral cambia con inhibición?
```

Comparar:

* FO
* lifetime
* transition probability
* switching rate

---

# B. Adultos vs adolescentes

Hipótesis:

```text id="jlwm17"
¿maduración cerebral altera dinámicas latentes?
```

Comparar:

* estabilidad
* dwell time
* complejidad dinámica
* ocupación

---

# C. GO adultos vs GO adolescentes

Interacción:

```text id="jlwm17"
condition × development
```

---

# D. NO-GO adultos vs NO-GO adolescentes

Posible:

```text id="jlwm17"
maduración de control inhibitorio
```

---

# E. Transiciones específicas

Hipótesis:

```text id="jlwm17"
¿algunas transiciones aumentan/disminuyen?
```

Ejemplo:

```text id="jlwm17"
State 1 → State 2
```

---

# F. Evoked dynamics

Hipótesis:

```text id="jlwm17"
¿los estados tienen locking temporal al estímulo?
```

Medir:

* probabilidad temporal post-estímulo
* state occupancy curves

---

# G. Split-half stability

Hipótesis:

```text id="jlwm17"
¿los estados son reproducibles?
```

Dividir:

```text id="jlwm17"
half subjects
half epochs
half sessions
```

---

# H. Cross-model reproducibility

Hipótesis:

```text id="jlwm17"
¿K mayores descubren nuevos estados o fragmentan?
```

Aquí entra:

* cosine similarity
* Hungarian matching
* profile correlations

---

# Y LO MÁS IMPORTANTE

Los experimentos NO eran totalmente independientes.

Habíamos hablado de:

```text id="jlwm17"
combinar parámetros de estructura
```

por ejemplo:

| Experimento                | Cambios        |
| -------------------------- | -------------- |
| feat_task_k3_diag_pca90    | baseline       |
| feat_task_k4_diag_pca90    | solo K         |
| feat_task_k4_full_pca90    | K + covariance |
| feat_task_k3_diag_pca95    | PCA            |
| feat_task_k3_diag_smallwin | ventana        |
| feat_task_k3_alphaonly     | features       |

---

# La lógica que habíamos establecido

## Primero

Explorar:

```text id="jlwm17"
estructura matemática estable
```

con:

* K
* covariance
* PCA

---

## Luego

Fijar:

```text id="jlwm17"
el mejor modelo
```

---

## Después

Hacer:

```text id="jlwm17"
preguntas neurocientíficas
```

GO/NOGO, edad, etc.

---

