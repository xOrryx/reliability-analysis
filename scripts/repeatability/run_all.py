# -----------------------------------------------------------------------------
# Run every stage in order, from the raw logs to the tables and the figures.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
#          (supplementary repeatability measurement)
# -----------------------------------------------------------------------------
"""Run every stage in order."""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ("01_passes.py", "02_drift.py", "03_jumps.py", "04_figures.py",
          "05_controllers.py", "06_confound.py")


def main():
    for s in STAGES:
        print(f"\n{'='*72}\n{s}\n{'='*72}")
        sys.argv = [str(HERE / s)]
        runpy.run_path(str(HERE / s), run_name="__main__")


if __name__ == "__main__":
    main()
