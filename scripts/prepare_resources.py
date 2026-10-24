"""Prepare resources for CarryOn experiments."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from carryon.experiments.resources import main

if __name__ == "__main__":
    main([*sys.argv[1:], "--release-root", str(ROOT)])
