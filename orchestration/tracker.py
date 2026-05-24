# orchestration/tracker.py
#
# Genera y actualiza run.json en el output_dir del experimento.
#
# Estructura de run.json:
# {
#   "run_id": "abc123",
#   "experiment_name": "...",
#   "config_path": "...",
#   "config_hash": "sha256:...",
#   "started_at": "ISO8601",
#   "finished_at": "ISO8601 | null",
#   "status": "running | completed | failed",
#   "stages": {
#     "extract": {
#       "status": "completed | failed | skipped | pending",
#       "started_at": "...",
#       "finished_at": "...",
#       "returncode": 0,
#       "error": null
#     }
#   }
# }

import json
import hashlib
import datetime
from pathlib import Path


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _hash_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    sha256.update(path.read_bytes())
    return f"sha256:{sha256.hexdigest()}"


def _make_run_id(config_path: Path) -> str:
    raw = f"{config_path.resolve()}_{_now_iso()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


class RunTracker:

    def __init__(
        self,
        output_dir: Path,
        config_path: Path,
        experiment_name: str,
        stage_names: list[str],
    ):
        self.output_dir     = output_dir
        self.config_path    = config_path
        self.experiment_name = experiment_name
        self.run_json_path  = output_dir / "run.json"
        self._state: dict   = {}
        self._stage_names   = stage_names

    def start_run(self, skipped_stages: list[str] | None = None) -> str:
        skipped = set(skipped_stages or [])
        run_id  = _make_run_id(self.config_path)
        self._state = {
            "run_id":          run_id,
            "experiment_name": self.experiment_name,
            "config_path":     str(self.config_path.resolve()),
            "config_hash":     _hash_file(self.config_path),
            "started_at":      _now_iso(),
            "finished_at":     None,
            "status":          "running",
            "stages": {
                name: {
                    "status":      "skipped" if name in skipped else "pending",
                    "started_at":  None,
                    "finished_at": None,
                    "returncode":  None,
                    "error":       None,
                }
                for name in self._stage_names
            },
        }
        self._write()
        return run_id

    def start_stage(self, stage_name: str) -> None:
        self._state["stages"][stage_name]["status"]     = "running"
        self._state["stages"][stage_name]["started_at"] = _now_iso()
        self._write()

    def finish_stage(self, stage_name: str, returncode: int, error: str | None = None) -> None:
        stage = self._state["stages"][stage_name]
        stage["status"]      = "completed" if returncode == 0 else "failed"
        stage["finished_at"] = _now_iso()
        stage["returncode"]  = returncode
        stage["error"]       = error
        self._write()

    def finish_run(self, success: bool) -> None:
        self._state["finished_at"] = _now_iso()
        self._state["status"]      = "completed" if success else "failed"
        self._write()

    def load_existing(self) -> dict | None:
        if self.run_json_path.exists():
            return json.loads(self.run_json_path.read_text(encoding="utf-8"))
        return None

    @property
    def run_id(self) -> str:
        return self._state.get("run_id", "unknown")

    @property
    def run_json_location(self) -> Path:
        return self.run_json_path

    def _write(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.run_json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.run_json_path)