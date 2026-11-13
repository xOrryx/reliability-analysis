# -----------------------------------------------------------------------------
# Runs the whole analysis, from the raw logs to the tables and figures.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Run the whole analysis, from the raw logs to the tables and figures.

Two measurements live in this repository and they are run one after the other:
the two winter sessions, which are what the paper is mostly about, and the
supplementary repeatability measurement of Section 3.8. The second is skipped
when its recordings are absent, and the skip is announced rather than silent,
because a run that quietly produced half the outputs would be worse than one
that failed.
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ["01_build.py", "02_tables.py", "03_intext.py", "04_sweep.py",
          "05_figures.py", "06_sensitivity.py", "07_supplement.py",
          "08_detector_cv.py", "09_jitter_vs_brightness.py", "10_controller_loss_scene.py",
          "11_controller_vs_annotations.py", "12_excursions_v2.py", "13_jr_vector.py",
          "15_descriptors_preloss.py", "16_manual_vs_auto_and_bounds.py", "17_speed_regimes.py",
          "18_frame_summary.py", "19_ctrl_rel_piecewise.py", "20_ctrl_rel_ratios.py",
          "21_fig_brightness_preloss.py"]

for s in STAGES:
    print(f"\n{'='*70}\n{s}\n{'='*70}", flush=True)
    sys.argv = [str(HERE / s)]
    runpy.run_path(str(HERE / s), run_name="__main__")

sys.path.insert(0, str(HERE.parent))
from repeatability import config as RC  # noqa: E402

if any(RC.DATA_DIR.glob("measHMD_*.csv")):
    print(f"\n{'='*70}\nrepeatability\n{'='*70}", flush=True)
    sys.argv = [str(HERE / "repeatability" / "run_all.py")]
    runpy.run_path(str(HERE / "repeatability" / "run_all.py"), run_name="__main__")
    sys.argv = [str(HERE / "14_jitter_repeatability.py")]
    runpy.run_path(str(HERE / "14_jitter_repeatability.py"), run_name="__main__")
else:
    print(f"\n{'='*70}\nrepeatability: skipped\n{'='*70}")
    print(f"no measHMD_*.csv under {RC.DATA_DIR}, so Section 3.8 was not rebuilt.")
