# -----------------------------------------------------------------------------
# Stage 17: where the speed thresholds sit in the distribution of window speeds.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Stage 17: justification of the 0.15 and 0.8 m/s regime thresholds.

Histogram of the mean headset speed of every tracked window (both sessions, all
devices), its modes and the trough between them, and the share of windows in the
standing (< 0.15), transitional (0.15-0.8) and walking (> 0.8) regimes.
Writes out/speed_regimes.csv and out/figures/fig_speed_hist.png.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reliability import config as C


def main():
    W = pd.read_pickle(C.OUT / "windows.pkl")
    w = W[(W.ui < 0.5) & (W.f_track_hmd > 0.99)]
    v = w.speed.to_numpy()
    edges = np.arange(0, 2.05, 0.05)
    h, _ = np.histogram(v, bins=edges)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    lo = ctr < 0.5
    mode_still = float(ctr[lo][np.argmax(h[lo])])
    mode_walk = float(ctr[~lo][np.argmax(h[~lo])])
    between = (ctr > mode_still) & (ctr < mode_walk)
    trough = float(ctr[between][np.argmin(h[between])])
    rows = [dict(quantity="mode_standing_m_s", value=mode_still),
            dict(quantity="mode_walking_m_s", value=mode_walk),
            dict(quantity="trough_m_s", value=trough)]
    for name, m in (("standing <0.15", v < C.STILL_MAX), ("transition 0.15-0.8", (v >= C.STILL_MAX) & (v <= C.WALK_MIN)),
                    ("walking >0.8", v > C.WALK_MIN)):
        rows.append(dict(quantity=f"share {name}", value=float(m.mean())))
    for q in (5, 25, 50, 75, 95):
        rows.append(dict(quantity=f"walking speed P{q}", value=float(np.percentile(v[v > C.WALK_MIN], q))))
    T = pd.DataFrame(rows)
    T.to_csv(C.OUT / "speed_regimes.csv", index=False)
    (C.OUT / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 2.6))
    ax.bar(ctr, h, width=0.05, color="0.7")
    for x in (C.STILL_MAX, C.WALK_MIN):
        ax.axvline(x, color="k", ls="--", lw=0.8)
    ax.set_xlabel("mean headset speed of window [m/s]"); ax.set_ylabel("windows")
    fig.tight_layout(); fig.savefig(C.OUT / "figures" / "fig_speed_hist.png", dpi=200)
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
