"""PhytoTrace AI - Backend Application Entrypoint for Cloud & Local Deployment."""
import sys
import os
from pathlib import Path

# Add src to Python module path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import FastAPI application
from phytotrace.api.main import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("PHYTOTRACE_API_HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=False)
