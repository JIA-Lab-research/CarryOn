"""Run a configured CarryOn experiment in a new output directory."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from carryon.experiments.runner import main

if __name__ == "__main__":
    main(release_root=ROOT)
