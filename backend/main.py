"""PhytoTrace AI - Backend Subdirectory Entrypoint for Render / Cloud Services."""
import sys
import os
from pathlib import Path

# Resolve repo root directory
backend_dir = Path(__file__).resolve().parent
repo_root = backend_dir.parent

# Change working directory to repo root so config and model checkpoint paths resolve
os.chdir(repo_root)

# Ensure src/ and repo root are in python path
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from phytotrace.api.main import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("PHYTOTRACE_API_HOST", "0.0.0.0")
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)
