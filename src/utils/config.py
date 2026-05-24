

import yaml
from pathlib import Path


def load_yaml(path):

    with open(path, "r") as f:
        return yaml.safe_load(f)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"

paths_config = load_yaml(
    CONFIG_DIR / "paths.yaml"
)

preprocessing_config = load_yaml(
    CONFIG_DIR / "preprocessing.yaml"
)

tde_config = load_yaml(
    CONFIG_DIR / "tde_hmm.yaml"
)

feature_config = load_yaml(
    CONFIG_DIR / "feature_hmm.yaml"
)

ae_config = load_yaml(
    CONFIG_DIR / "ae_hmm.yaml"
)