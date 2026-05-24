# ============================================
# LOGGER
# ============================================

import logging
import logging.config
import yaml

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOGGING_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "logging.yaml"
)

with open(LOGGING_CONFIG, "r") as f:
    config = yaml.safe_load(f)

# ============================================
# FIX ABSOLUTE LOG PATH
# ============================================

config["handlers"]["file"]["filename"] = str(
    LOG_DIR / "project.log"
)

logging.config.dictConfig(config)

logger = logging.getLogger("EEG_HMM")