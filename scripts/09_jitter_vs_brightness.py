# -----------------------------------------------------------------------------
# Stage 9: does darkness degrade the pose while tracking holds?
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Stage 9: positional jitter against scene brightness, across recordings.

04_sweep tests scene effects inside one recording, which cannot see an effect
that differs only between recordings (a night segment against a day segment).
Here the comparison is made across recordings of one device, February session,
on windows with tracking reported throughout and the panel closed:

  bins     brightness < 5, 5-15, 15-40, 40-80, >= 80 (8-bit mean intensity),
           separately for walking (0.8-1.6 m/s) and standing (< 0.15 m/s);
           median J_p with a 95 % interval from resampling whole segments.
  near     the same, excluding windows within 10 s of an episode of loss, so
           that the margin of a loss does not drive the darkest bin.
  speed    median walking speed per bin, to see whether the bins are comparable.
  ratio    per device, median J_p in the darkest bin that has >= 30 windows
           divided by the median in brightness >= 40.

Writes out/jitter_brightness/*.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from reliability import config as C
from reliability.stats import bootstrap_median_arrays

OUT = C.OUT / "jitter_brightness"
OUT.mkdir(parents=True, exist_ok=True)
EDGES = [0, 5, 15, 40, 80, 1e9]
LABELS = ["<5", "5-15", "15-40", "40-80", ">=80"]
MM = 10.0


def main():
    S = pd.read_pickle(C.OUT / "scene.pkl")
    E = pd.read_csv(C.OUT / "episodes.csv")
    S = S[(S.f_track_hmd > 0.99) & (S.ui < 0.5) & S.hp_pos_hmd.notna()].copy()
    S["jp"] = S.hp_pos_hmd * MM
    S["bin"] = pd.cut(S.Brightness, EDGES, labels=LABELS, right=False)
    near = np.zeros(len(S), bool)
    for _, e in E[E.month == "February"].iterrows():
        m = ((S.dev == e.dev) & (S.seg == e.seg) & (S.t >= e.t0 - 10)
             & (S.t <= e.t0 + e.dur_s + 10)).to_numpy()
        near |= m
    S["near_loss"] = near

    rows = []
    for regime, sel in (("walking", (S.speed >= C.WALK_MIN) & (S.speed <= C.WALK_MAX)),
                        ("standing", S.speed < C.STILL_MAX)):
        for subset, sub in (("all", S[sel]), ("away_from_loss", S[sel & ~S.near_loss])):
            for (dev, b), g in sub.groupby(["dev", "bin"], observed=True):
                if len(g) < 30:
                    continue
                m, lo, hi = bootstrap_median_arrays([s.jp.to_numpy() for _, s in g.groupby("seg")])
                rows.append(dict(regime=regime, subset=subset, dev=dev, brightness=b,
                                 windows=len(g), segments=g.seg.nunique(),
                                 median_jp_mm=m, ci_lo=lo, ci_hi=hi,
                                 median_speed=float(g.speed.median())))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "jitter_by_brightness.csv", index=False)

    R = []
    for (regime, subset, dev), g in T.groupby(["regime", "subset", "dev"]):
        bright = g[g.brightness.isin(["40-80", ">=80"])]
        dark = g[g.brightness.isin(["<5", "5-15"])]
        if bright.empty or dark.empty:
            continue
        sb = S[(S.dev == dev) & (S.Brightness >= 40)]
        sb = sb[(sb.speed >= C.WALK_MIN) & (sb.speed <= C.WALK_MAX)] if regime == "walking" else sb[sb.speed < C.STILL_MAX]
        if subset == "away_from_loss":
            sb = sb[~sb.near_loss]
        d0 = dark.iloc[0]
        R.append(dict(regime=regime, subset=subset, dev=dev, dark_bin=d0.brightness,
                      dark_median=d0.median_jp_mm, dark_ci=(round(d0.ci_lo, 2), round(d0.ci_hi, 2)),
                      bright_median=float(sb.jp.median()), ratio=d0.median_jp_mm / float(sb.jp.median())))
    R = pd.DataFrame(R)
    R.to_csv(OUT / "dark_vs_bright_ratio.csv", index=False)
    pd.set_option("display.width", 220, "display.precision", 2, "display.max_rows", 200)
    print(T.to_string(index=False))
    print()
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
