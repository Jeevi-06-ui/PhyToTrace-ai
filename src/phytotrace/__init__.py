"""PhytoTrace AI core package."""

import os
from pathlib import Path

# Dependencies are provisioned at install time, never during a photograph upload.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

# Automatically load .env from project root if present
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    try:
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _v = _line.split("=", 1)
                _k = _k.strip()
                _v = _v.strip().strip('"').strip("'")
                if _k and _k not in os.environ:
                    os.environ[_k] = _v
    except Exception:
        pass

