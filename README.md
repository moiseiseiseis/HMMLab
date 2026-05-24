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


Correr en la workstation remota
Esta sección documenta el procedimiento completo para el día que tengas acceso SSH a la máquina de cómputo.
1. Conectarse
bashssh usuario@ip-de-la-maquina
Si es la primera vez, acepta la huella del servidor cuando pregunte yes/no.

2. Clonar el repo
bashcd ~
git clone https://github.com/moiseiseiseis/HMMLab.git
cd HMMLab

3. Verificar que Python y conda están disponibles
bashpython --version
conda --version
Si conda no está, instala miniconda:
bashwget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# cierra y vuelve a abrir la terminal

4. Crear el entorno
bashconda env create -f environment.yml
conda activate eeg-hmm
pip install -e .
Verifica que el CLI funciona:
bashhmm stages
Debe mostrar las etapas del pipeline Featured-HMM sin errores.

5. Transferir los datos FIF
Los archivos .fif preprocesados están en Google Drive. Descárgalos a la máquina:
Opción A — gdown (si tienen IDs de Drive):
bashpip install gdown
gdown --folder https://drive.google.com/drive/folders/TU_FOLDER_ID -O data/fif/
Opción B — SCP desde tu laptop:
bash# Corre esto en tu laptop, no en la máquina remota
scp -r "C:/Users/pokem/OneDrive/Desktop/lab 2026/EEG_Preprocesado_GoNoGo/" usuario@ip:/home/usuario/HMMLab/data/fif/

6. Actualizar las rutas en los YAMLs
Los YAMLs tienen la ruta de tu laptop hardcodeada. Actualiza fif_dir en todos los experimentos TDE:
bash# Reemplaza la ruta en todos los YAMLs TDE de una vez
sed -i 's|C:/Users/pokem/OneDrive/Desktop/lab 2026/EEG_Preprocesado_GoNoGo|/home/usuario/HMMLab/data/fif|g' \
    configs/experiments/canonical/tde_k4_t7.yaml \
    configs/experiments/canonical/tde_k4_t15.yaml \
    configs/experiments/canonical/tde_k3_t7.yaml \
    configs/experiments/canonical/tde_k5_t7.yaml \
    configs/experiments/canonical/tde_k3_t15.yaml \
    configs/experiments/canonical/tde_k5_t15.yaml
Verifica que quedó bien:
bashgrep fif_dir configs/experiments/canonical/tde_k4_t7.yaml

7. Verificar que los FIF están accesibles
bashpython -c "
from pathlib import Path
fif_dir = Path('data/fif')
files = sorted(fif_dir.glob('*_clean-epo.fif'))
print(f'archivos encontrados: {len(files)}')
for f in files[:3]:
    print(f'  {f.name}')
"
Debe mostrar 118 archivos.

8. Correr los experimentos TDE
Orden recomendado — de menor a mayor costo computacional:
bash# experimento principal
hmm run configs/experiments/canonical/tde_k4_t7.yaml

# comparativa de lags
hmm run configs/experiments/canonical/tde_k4_t15.yaml

# k sweep
hmm run configs/experiments/canonical/tde_k3_t7.yaml
hmm run configs/experiments/canonical/tde_k5_t7.yaml
Cada experimento tarda aproximadamente:

embed: 15-30 min (depende de la RAM)
hmm: 30-90 min (depende de K y convergencia)
decode: 2-5 min


9. Si un experimento se interrumpe
El pipeline guarda el estado en run.json. Para reanudar desde la etapa que falló:
bash# si falló en embed
hmm run configs/experiments/canonical/tde_k4_t7.yaml --from embed

# si falló en hmm (embed ya está listo)
hmm run configs/experiments/canonical/tde_k4_t7.yaml --from hmm

# si falló en decode
hmm run configs/experiments/canonical/tde_k4_t7.yaml --from decode

10. Verificar resultados
bash# ver el estado de un experimento
hmm inspect configs/experiments/canonical/tde_k4_t7.yaml

# ver métricas rápidas
cat outputs/experiments/canonical/tde_k4_t7/state_stats.csv

# ver score de calidad
python -c "
import json
data = json.load(open('outputs/experiments/canonical/tde_k4_t7/decoding_summary.json'))
print(f'score: {data[\"score_4\"]}/4 — {data[\"verdict\"]}')
print(f'FO min: {data[\"FO_min\"]} | FO max: {data[\"FO_max\"]}')
print(f'dwell max: {data[\"dwell_max_ms\"]}ms')
print(f'absorbentes: {data[\"n_absorbing\"]}')
"

11. Transferir outputs de vuelta a tu laptop
Una vez terminados los experimentos, descarga los outputs para analizarlos en el notebook:
bash# desde tu laptop
scp -r usuario@ip:/home/usuario/HMMLab/outputs/experiments/canonical/tde_k4_t7/ \
    "C:/Proyectos/eeg_hmm_plattform/outputs/experiments/canonical/"
O comprime primero para transferir más rápido:
bash# en la máquina remota
tar -czf tde_outputs.tar.gz outputs/experiments/canonical/tde_k4_t7/

# en tu laptop
scp usuario@ip:/home/usuario/HMMLab/tde_outputs.tar.gz .
tar -xzf tde_outputs.tar.gz