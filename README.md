# HMMLab

Pipeline para el análisis de dinámica cerebral con Hidden Markov Models sobre EEG.
Implementa tres variantes metodológicas: Featured-HMM, TDE-HMM y AE-HMM (en desarrollo).

Datos: paradigma Go/NoGo, 59 sujetos (30 adultos, 29 adolescentes), 19 canales.

---

## Estructura del proyecto

```
configs/          configuración de experimentos y preprocesamiento
scripts/          scripts ejecutables por etapa
src/              librería interna (features, decoding, utils)
orchestration/    CLI y orquestador del pipeline
notebooks/        análisis exploratorio y reportes
slurm/            scripts para HPC
```

---

## Pipelines disponibles

### Featured-HMM

Extrae features espectrales y de complejidad por ventana temporal y entrena un GaussianHMM sobre ese espacio.

Etapas: `preprocess → extract → pca → hmm → decode`

Features canónicas: envolventes alpha/theta/beta + Hjorth mobility + Hjorth complexity (95D → ~56 PCs).

### TDE-HMM

Opera sobre la señal raw con lags temporales concatenados (Time-Delay Embedding), capturando estructura espacio-temporal sin features manuales.

Etapas: `preprocess → embed → hmm → decode`

Embedding canónico: n_lags=7, lag_step=1, 19 canales → 152D → ~50 PCs.

---

## Uso

```bash
# Instalar
pip install -e .

# Correr un experimento completo
hmm run configs/experiments/canonical/canonical_k4_full_95d_v2.yaml

# Reanudar desde una etapa específica
hmm run configs/experiments/canonical/canonical_k4_full_95d_v2.yaml --from pca
hmm run configs/experiments/canonical/tde_k4_t7.yaml --from embed

# Ver etapas del pipeline
hmm stages
hmm stages configs/experiments/canonical/tde_k4_t7.yaml

# Inspeccionar el último run de un experimento
hmm inspect configs/experiments/canonical/canonical_k4_full_95d_v2.yaml
```

El tipo de pipeline (Featured vs TDE) se detecta automáticamente desde el campo `pipeline_type` del YAML.

---

## Experimentos canónicos

### Featured-HMM

| Experimento | K | Cov | Features | Score |
|---|---|---|---|---|
| canonical_k4_full_95d_v2 | 4 | full | Bands+Hjorth | 4/4 |
| canonical_k3_full_95d | 3 | full | Bands+Hjorth | 4/4 |
| canonical_k5_full_95d | 5 | full | Bands+Hjorth | 3/4 |
| canonical_k4_full_hjorth | 4 | full | Hjorth only | 1/4 |

### TDE-HMM

| Experimento | K | n_lags | D_raw | Estado |
|---|---|---|---|---|
| tde_k4_t7 | 4 | 7 | 152D | pendiente HPC |
| tde_k4_t15 | 4 | 15 | 304D | pendiente HPC |

---

## Configuración de un experimento

Cada experimento es un archivo YAML en `configs/experiments/canonical/`:

```yaml
experiment:
  name: "canonical_k4_full_95d_v2"
  pipeline_type: "feature"   # "feature" o "tde"
  seed: 42

paths:
  features_dir: "data/interim/features/task_hjorthonly_temporalqc/"
  output_dir:   "outputs/experiments/canonical/"

features:
  use_alpha:  true
  use_theta:  true
  use_beta:   true
  use_hjorth: true

pipeline:
  pca:
    variance_retained: 0.90
  hmm:
    k_states:        4
    covariance_type: "full"
    n_iter:          500
    tol:             0.001
```

Para TDE, se añade la sección `tde`:

```yaml
tde:
  n_lags:   7
  lag_step: 1
  sfreq:    250.0
```

---

## Outputs por experimento

Cada experimento genera su directorio en `outputs/experiments/canonical/{nombre}/`:

```
hmm_model_k{K}.pkl          modelo entrenado
viterbi_paths_k{K}.npy      secuencia de estados por ventana
X_pca.npy                   datos en espacio PCA
scaler.pkl                  StandardScaler ajustado
lengths.npy                 longitud de cada trial
state_stats.csv             FO, dwell time, self-transition por estado
fo_by_subject.csv           FO por sujeto y condición
feature_manifest.json       MD5 de los archivos de entrada
run.json                    metadatos del run (status, timestamps, convergencia)
```

---

## Criterios de calidad (score 4/4)

Un experimento se considera apto para reporte si pasa los 4 criterios:

- FO mínima > 10% — ningún estado marginal
- Dwell time máximo < 2,000ms — sin pseudo-attractors
- Rango de FO < 0.30 — distribución equilibrada entre estados
- Confianza media > 0.60 — decoding no ambiguo

---

## Entorno

```bash
conda env create -f environment.yml
conda activate eeg-hmm
pip install -e .
```

Dependencias principales: `mne`, `hmmlearn`, `scikit-learn`, `numpy`, `pandas`, `joblib`, `typer`.


