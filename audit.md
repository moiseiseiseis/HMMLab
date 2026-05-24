# Pipeline Audit — EEG-HMM Platform
**Fecha:** 2026-05-15

---

## 1. Dataset

### Participantes
| Grupo | Subgrupo | n sujetos | Sufijo ID | Ejemplo |
|---|---|---|---|---|
| ADULTO | Hombres | 15 | SC o SIN | AAELSC, EDGSSIN |
| ADULTO | Mujeres | 15 | SC o SIN | AVLLSC, CGMSIN |
| ADOLESCENTE | Hombres (TS) | 15 | TS | CJTS, DCTS, OPTS |
| ADOLESCENTE | Mujeres | 14 | SIN | ACBSIN, FVGSIN |
| **Total** | | **59** | | |

- **118 archivos** de features = 59 sujetos × 2 condiciones (GO / NOGO)
- Excepción: `DATS` (solo NOGO) y `DATSN` (solo GO) son sujetos distintos dentro del grupo TS

### Etiquetas canónicas
`configs/subject_labels.csv` — fuente de verdad para condición y grupo.
- Columnas: `npy_file`, `subject_id`, `condition`, `group`
- Generado por `notebooks/exploratory/subject_labeling.ipynb`
- Verdad de terreno: estructura de carpetas en `Desktop/lab 2026/EEG/EEG/{ADULTOS,ADOLESCENTES}/`
- **Advertencia:** re-ejecutar celda 2 del notebook sobreescribe el CSV; celda 1 usa el conjunto `ADULTS` correcto para no perder la corrección

### Convención de nombres en archivos .npy
```
{SUBJECT_ID}{COND_SUFFIX}_sin_contexto_features.npy
```
Donde `COND_SUFFIX` = `GO` o `NG` al final del ID (sin token explícito `_GO_`).
Para detección robusta usar `subject_labels.csv` como fallback; regex `re.search(r'_(NOGO|GO)_', fpath.name)` aplica solo a `.fif` (que sí llevan token explícito).

**Edge case documentado:** `DATSNGO` termina en `NGO` pero condición = GO (sujeto `DATSN`). El CSV lo tiene correcto.

---

## 2. Preprocesamiento

**Script:** `scripts/preprocessing/preprocess_task_batch.py`
**Config:** `configs/preprocessing/task_feature_hmm.yaml`

| Paso | Parámetro |
|---|---|
| Referencia | Promedio (CAR) |
| Filtro | Lowpass 30 Hz, FIR zero-phase |
| Resampleo | 500 Hz → 250 Hz |
| Rechazo artefactos | Drop epochs > 500 µV |
| Baseline | Corrección aplicada |
| ICA | Desactivado |

**Salida:** `data/interim/preprocessed/task/` — 118 archivos `*_epo.fif`

---

## 3. Arquitectura del pipeline

```
preprocess → extract → pca → hmm → decode
```

Orquestación vía CLI:
```
hmm run configs/experiments/<exp>.yaml
hmm run configs/experiments/<exp>.yaml --from pca
hmm stages
hmm inspect configs/experiments/<exp>.yaml
```

### Etapa: extract (`scripts/features/01_extract_features.py`)
- Motor modular: `src/features/extractor_engine.py`
- Registro automático via `@REGISTRY.register` y `src/features/__init__.py`
- Ventanas deslizantes sobre cada época; edge trim configurable
- **Parámetros canónicos:** window=300 ms, step=100 ms, edge_trim=1

### Etapa: pca (`scripts/features/02_fit_pca.py`)
- StandardScaler → PCA (varianza retenida configurable)
- Guarda: `X_pca.npy`, `lengths.npy`, `pca_model.pkl`, `scaler.pkl`

### Etapa: hmm (`scripts/training/03_train_hmm.py`)
- `hmmlearn.GaussianHMM`, n_iter=500, tol=0.001, seed=42
- Guarda: `hmm_model_k{K}.pkl`, `viterbi_paths_k{K}.npy`

### Etapa: decode (`scripts/decoding/decode_feature_hmm.py`)
- Métricas por sujeto: Fractional Occupancy, Dwell Time (s), Transition Rate (trans/s)
- Guarda: `state_stats.csv`, `transition_matrix_k{K}.npy`, `decoding_summary.json`

---

## 4. Features implementadas

| Nombre canónico | Categoría | Dimensiones | HMM suitability |
|---|---|---|---|
| `theta_envelope` | spectral | n_channels | high |
| `alpha_envelope` | spectral | n_channels | high |
| `beta_envelope` | spectral | n_channels | high |
| `hjorth_mobility` | temporal | n_channels | high |
| `hjorth_complexity` | temporal | n_channels | high |
| `temporal_entropy` | temporal | n_channels | medium |

**Dimensiones brutas pre-PCA** (19 canales):

| Set | Features activas | Dims | Etiqueta canonical |
|---|---|---|---|
| Hjorth only | mobility + complexity | 38 | `38d` |
| Bands + Hjorth | theta + alpha + beta + hjorth | 95 | `95d` |
| Full | bands + hjorth + entropy | 114 | `114d` |

---

## 5. Sets de features extraídos

| Directorio (`data/interim/features/`) | Features | Archivos | Notas |
|---|---|---|---|
| `task_hjorthonly_temporalqc/` | Hjorth only | 118 | **Principal canónico** — QC temporal activo |
| `task_fullfeatures_v2/` | Bands+Hjorth+Entropy | 118 | 25 949 ventanas (menos restricciones QC) |
| `feat_task_k4_diag_pca90_hjorthonly/` | Hjorth only | 118 | 25 949 ventanas (sin edge trim adicional) |
| `task_hjorthonly/` | Hjorth only | 118 | Versión previa sin QC temporal |
| `task_bandsonly/` | Theta+Alpha+Beta | 118 | Exploración |
| `task/` | (varios) | 118 | Legacy |

---

## 6. Experimentos

### 6.1 Canónicos (`outputs/experiments/canonical/`)
Todos los experimentos canónicos incluyen las 5 etapas completas (preprocess → decode).

| Experimento | K | Cov | Features | Dims | Ventanas | FO global | Dwell medio (s) | TR (trans/s) | Estado |
|---|---|---|---|---|---|---|---|---|---|
| `canonical_k3_diag_95d` | 3 | diag | Bands+Hjorth | 95 | 21 231 | S0=13%, S1=16%, S2=70% | S2=7.6 | 2.87 | COMPLETO |
| `canonical_k4_diag_95d` | 4 | diag | Bands+Hjorth | 95 | 21 231 | S0=45%, S1=14%, S2=12%, S3=29% | ~1.7 | 0.74 | COMPLETO |
| `canonical_k4_full_95d` | 4 | full | Bands+Hjorth | 95 | 21 231 | S0=27%, S1=24%, S2=19%, S3=31% | ~0.7 | 1.75 | COMPLETO |
| `canonical_k5_diag_95d` | 5 | diag | Bands+Hjorth | 95 | 21 231 | S0=42%, S1=12%, S2=5%, S3=27%, S4=14% | ~1.1 | 0.79 | COMPLETO |
| `canonical_k4_diag_114d` | 4 | diag | Full (c/ entropy) | 114 | 25 949 | S0=5%, S1=35%, S2=21%, S3=39% | ~1.6 | 0.76 | COMPLETO |
| `canonical_k4_diag_38d` | 4 | diag | Hjorth only | 38 | 25 949 | S0=30%, S1=42%, S2=7%, S3=21% | ~0.7 | 1.32 | COMPLETO |

> **Nota K=3:** S2 domina con FO=70% y dwell=7.6 s — señal de degeneración de estados o estado "reposo" inducido por la codificación de ventanas. Revisión pendiente.

### 6.2 Legado (`outputs/processed/experiments/feature_hmm/task/`)
Experimentos previos a la infraestructura canónica. Entrenados pero sin decode (excepto uno).

| Experimento | K | Cov | Features dir | Decode | Notas |
|---|---|---|---|---|---|
| `feat_task_k3_diag_pca90_hjorthonly` | 3 | diag | hjorthonly | **SÍ** | Único legado con `state_stats.csv` |
| `feat_task_k3_diag_pca90_hjorthonly_temporalqc` | 3 | diag | hjorthonly_temporalqc | No | — |
| `feat_task_k3_diag_pca90` | 3 | diag | task | No | `all_lengths.npy` en lugar de `lengths.npy` |
| `feat_task_k3_diag_pca90_bandsonly` | 3 | diag | task_bandsonly | No | — |
| `feat_task_k3_diag_hjorth` | 3 | diag | hjorthonly | No | — |
| `feat_task_k4_diag_pca90_hjorthonly_temporalqc` | 4 | diag | hjorthonly_temporalqc | No | — |
| `feat_task_k4_diag_pca90_hjorthonly` | 4 | diag | hjorthonly | No | — |
| `feat_task_k4_diag_pca90` | 4 | diag | task | No | Sin `scaler.pkl` |
| `feat_task_k4_diag_pca90_fullfeatures_v2` | 4 | diag | fullfeatures_v2 | No | — |
| `feat_task_k4_full_pca90_fullfeatures_v2` | 4 | full | fullfeatures_v2 | No | — |
| `feat_task_k4_diag_hjorth` | 4 | diag | hjorthonly | No | — |
| `feat_task_k5_diag_pca90` | 5 | diag | task | No | Sin `scaler.pkl` |
| `feat_task_k6_diag_hjorth` | 6 | diag | hjorthonly | No | Tiene directorio `diagnostics/` |
| `feat_task_k5_diag_hjorth` | 5 | diag | hjorthonly | No | **INCOMPLETO** — solo `run.json` |

---

## 7. Módulos fuente (`src/`)

| Módulo | Archivo | Estado |
|---|---|---|
| Preprocesamiento | `src/preprocessing/core.py` | Implementado |
| Feature extractor | `src/features/extractor_engine.py` | Implementado |
| Registry de features | `src/features/registry.py` | Implementado |
| Hjorth | `src/features/temporal/hjorth.py` | Implementado |
| Entropy | `src/features/temporal/entropy.py` | Implementado |
| Spectral envelopes | `src/features/spectral/{alpha,beta,theta}_envelope.py` | Implementado |
| Métricas decoding | `src/decoding/metrics.py` | Implementado |
| Diagnósticos | `src/features/diagnostics/{distribution,redundancy,temporality,inter_subject,hmm_score}.py` | Implementado |
| TDE-HMM | `src/` (stub) | **Sin implementar** |
| AE-HMM | `src/` (stub) | **Sin implementar** |

---

## 8. Notebooks

### Exploratorios
| Notebook | Propósito | Estado |
|---|---|---|
| `subject_labeling.ipynb` | Genera `configs/subject_labels.csv` | Actualizado — regex correcto para condición y grupo |
| `deep_analysis.ipynb` | Análisis profundo de estados HMM | Actualizado — CSV lookup para condición/grupo |
| `master_report_v2.ipynb` | Reporte consolidado multi-experimento | Actualizado — `load_experiment()` usa regex + CSV fallback |

### Reportes (`notebooks/reports/`)
Notebooks de visualización e inferencia clínica para cada experimento. Versiones para K=3, K=4, K=5, bandas, hjorth, full.

---

## 9. Problemas conocidos y pendientes

| # | Problema | Severidad | Notas |
|---|---|---|---|
| 1 | K=3 canónico: S2 con FO=70%, dwell=7.6s — probable colapso de estados | Alta | Revisar convergencia y transmat |
| 2 | TDE-HMM y AE-HMM sin implementar | Media | Scripts stub en `scripts/` y `src/` |
| 3 | Experimentos legado sin decode | Baja | Se reemplazaron con experimentos canónicos |
| 4 | `feat_task_k5_diag_hjorth` incompleto (solo `run.json`) | Baja | No tiene modelo ni Viterbi |
| 5 | Inconsistencia `all_lengths.npy` vs `lengths.npy` en algunos legados | Baja | Solo afecta experimentos pre-canónicos |
| 6 | `canonical_k3_diag_95d` incluye bandas espectrales pero el nombre sugiere `95d` | Informativo | 95 dims = 19ch × (theta+alpha+beta+mobility+complexity) |

---

## 10. Referencia rápida de comandos

```bash
# Correr experimento completo
hmm run configs/experiments/canonical/canonical_k4_diag_95d.yaml

# Reanudar desde decode
hmm run configs/experiments/canonical/canonical_k4_diag_95d.yaml --from decode

# Ver etapas registradas
hmm stages

# Ver último run de un experimento
hmm inspect configs/experiments/canonical/canonical_k4_diag_95d.yaml

# Regenerar etiquetas de sujetos
# (ejecutar celda 1 + celda 2 de subject_labeling.ipynb)
```

---

## 11. Fuentes de verdad

| Artefacto | Fuente |
|---|---|
| Grupo ADULTO/ADOLESCENTE | `configs/subject_labels.csv` (validado contra estructura de carpetas del Desktop) |
| Condición GO/NOGO | `configs/subject_labels.csv` + token `_(NOGO|GO)_` en `.fif` |
| Parámetros de experimento | YAML en `configs/experiments/canonical/` |
| Resultados por sujeto | `state_stats.csv` dentro del directorio del experimento |
| Resumen global | `decoding_summary.json` dentro del directorio del experimento |
