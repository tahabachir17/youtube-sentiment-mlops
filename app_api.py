"""
Entrypoint for Hugging Face Spaces & Docker container.
Runs the FastAPI sentiment analysis application on port 7860.
"""

import os
import sys
import uvicorn

# Ensure repository root is in python search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.api.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    print("\n" + "=" * 60)
    print(f"[*] Local Web Browser:   http://localhost:{port}/")
    print(f"[*] Interactive Swagger: http://localhost:{port}/docs")
    print(f"[*] Health Check:        http://localhost:{port}/health")
    print("=" * 60 + "\n")
    uvicorn.run("app_api:app", host=host, port=port, reload=False, workers=1)
