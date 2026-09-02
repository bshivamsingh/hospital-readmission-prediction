"""Pytest config: make `app/` importable as a plain module dir, the same way
Streamlit itself imports it (Streamlit adds the entrypoint script's own
directory to sys.path — it does not treat `app/` as a package)."""
import os
import sys

APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
