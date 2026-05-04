"""
Compatibility entry point for the Streamlit dashboard.

The main dashboard lives in ui/dashboard.py, but some IDE/run configurations
look for dashboard.py at the project root.
"""

from pathlib import Path
import runpy


DASHBOARD_PATH = Path(__file__).resolve().parent / "ui" / "dashboard.py"

if __name__ == "__main__":
    runpy.run_path(str(DASHBOARD_PATH), run_name="__main__")
