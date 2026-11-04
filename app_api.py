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
    print(f"[*] Launching YouTube Sentiment API on {host}:{port}")
    uvicorn.run("app_api:app", host=host, port=port, reload=False, workers=1)
