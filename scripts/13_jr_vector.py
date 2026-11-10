# -----------------------------------------------------------------------------
# Stage 13: rotational indicator from the angular velocity vector.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Stage 13: J_r as RMS of the high-passed angular velocity VECTOR vs. MAGNITUDE.

Reviewer 2 (round 2): filtering the magnitude of the angular rate rectifies the
gait oscillation and creates harmonics above 5 Hz; filtering the vector avoids
this. windows.py now computes both (hp_rot = vector, hp_rotmag = magnitude, as
in the first revision). February, tracked throughout, panel closed; medians
with 95 % block-bootstrap intervals over segments, standing and walking.
Writes out/jr_vector.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from reliability import config as C
from reliability.stats import block_bootstrap_median


def main():
    W = pd.read_pickle(C.OUT / "windows.pkl")
    w = W[(W.month == "February") & (W.ui < 0.5) & (W.f_track_hmd > 0.99)]
    rows = []
    for cond, g0 in (("standing", w[w.speed < C.STILL_MAX]), ("walking", w[w.speed > C.WALK_MIN])):
        for dev in C.DEVICES:
            g = g0[g0.dev == dev]
            r = dict(condition=cond, dev=dev, n=int(g.hp_rot_hmd.notna().sum()))
            for col, lab in (("hp_rot_hmd", "vector"), ("hp_rotmag_hmd", "magnitude")):
                m, lo, hi = block_bootstrap_median(g, col, group=("seg",))
                r[lab] = m; r[lab + "_lo"] = lo; r[lab + "_hi"] = hi
            rows.append(r)
    T = pd.DataFrame(rows)
    T["ratio_walk_to_stand"] = np.nan
    for dev in C.DEVICES:
        for lab in ("vector", "magnitude"):
            st = T[(T.dev == dev) & (T.condition == "standing")][lab].iloc[0]
            wk = T[(T.dev == dev) & (T.condition == "walking")][lab].iloc[0]
            T.loc[(T.dev == dev) & (T.condition == "walking"), f"ratio_{lab}"] = wk / st
    T.to_csv(C.OUT / "jr_vector.csv", index=False)
    pd.set_option("display.width", 220, "display.precision", 2)
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
