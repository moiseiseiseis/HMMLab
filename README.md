### 1. Capa de Preprocesamiento
* **Archivo de Configuración:** `configs/preprocessing/task_feature_hmm.yaml`
* **Script de Ejecución:** `scripts/preprocessing/preprocess_task_batch.py`
* **Entrada:** Datos de EEG crudos (`.vhdr`, `.edf`, etc.).
* **Procesos Clave:** Resampling (fs target), filtrado de frecuencias, y segmentación en épocas (GO/NO-GO).
* **Salida:** Archivos limpios y segmentados (`data/interim/preprocessed/task/*_epo.fif`)

### 2. Capa de Configuración (Modelo HMM)
* **Archivo:** `configs/experiments/feat_task_k3_diag_pca90.yaml`
* **Parámetros Clave:**
    * `K=3`
    * `covariance_type: "diag"`
    * `pca_variance: 0.90`
    * `metrics: [Theta, Alpha, Beta, Mobility, Complexity, Entropy]`

### 3. Capa de Procesamiento (Extracción y Entrenamiento)

El pipeline completo se ejecuta con: `hmm run configs/experiments/<experimento>.yaml`

Para reanudar desde una etapa específica: `hmm run configs/experiments/<experimento>.yaml --from extract`

Etapas disponibles (`hmm stages`):

| Etapa | Script | Descripción |
|-------|--------|-------------|
| `preprocess` | `scripts/preprocessing/preprocess_task_batch.py` | Raw `.txt` → épocas `.fif` limpias (CAR, bandpass, resample, artefactos) |
| `extract` | `scripts/features/01_extract_features.py` | Ventanas deslizantes + features modulares (parámetros en el YAML del experimento) |
| `pca` | `scripts/features/02_fit_pca.py` | StandardScaler + PCA con varianza retenida configurable |
| `hmm` | `scripts/training/03_train_hmm.py` | GaussianHMM + decodificación Viterbi |
| `decode` | `scripts/decoding/decode_feature_hmm.py` | Fractional Occupancy, Dwell Time y Transition Rate por sujeto → `state_stats.csv` |

### 4. Capa de Reportes y Análisis
* **`notebooks/reports/05_state_visualization_k3.ipynb`**: Retro-proyección topográfica de los microestados (Mapas MNE).
* **`notebooks/reports/06_clinical_statistics_k3.ipynb`**: Cálculo de la Trinidad Clínica (Fractional Occupancy, Dwell Time, Transition Rate) y comparación Adultos vs Adolescentes.