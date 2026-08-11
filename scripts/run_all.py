# -----------------------------------------------------------------------------
# Runs the whole analysis, from the raw logs to the tables and figures.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Run the whole analysis, from the raw logs to the tables and figures."""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ["01_build.py", "02_tables.py", "03_intext.py", "04_sweep.py",
          "05_figures.py", "06_sensitivity.py"]

for s in STAGES:
    print(f"\n{'='*70}\n{s}\n{'='*70}", flush=True)
    sys.argv = [str(HERE / s)]
    runpy.run_path(str(HERE / s), run_name="__main__")
