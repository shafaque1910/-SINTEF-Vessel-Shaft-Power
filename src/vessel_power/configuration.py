import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = json.loads((ROOT / "config/parameters.json").read_text(encoding="utf-8"))
CHANNELS = json.loads((ROOT / "config/channels.json").read_text(encoding="utf-8"))


def project_path(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value
