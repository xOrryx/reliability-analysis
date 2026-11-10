# -----------------------------------------------------------------------------
# Stage 11: controller loss and controller jumps against the manual annotations.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Stage 11: what the annotator saw when the controllers failed.

The automatic segmentation almost never assigns the stone or rock classes, so
"looking at a stone wall" cannot be tested on the descriptors. The manual
annotation can: it records the dominant geometry (walls), texture richness,
repeating patterns, surface type and the controllers' own visibility. Both
sessions, headset tracked throughout, panel closed.

Two events per window:
  CTRL_LOST  a controller reported untracked for > 80 % of the window
  CTRL_JUMP  a controller excursion event (frames.controller_excursions) starts
             in the window
For every annotated attribute present (window mean > 0.5) in at least 100
windows of a device: event rate with / without it, their ratio, and the number
of recordings (device x month x segment) in which the rate is higher with it.
A within-recording comparison is what guards against the attribute merely
marking a place or a time of day.

Writes out/controller_annotations/*.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from reliability import config as C

OUT = C.OUT / "controller_annotations"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    W = pd.read_pickle(C.OUT / "windows.pkl")
    A = pd.read_pickle(C.OUT / "annotations.pkl")
    X = pd.read_csv(C.OUT / "controller_events.csv")
    keys = ["dev", "month", "seg", "t"]
    M = A.merge(W[keys + ["f_track_hmd", "f_track_left", "f_track_right", "hop_s"]], on=keys, how="left")
    M = M[(M.f_track_hmd > 0.99) & (M.ui < 0.5)].reset_index(drop=True)
    M["CTRL_LOST"] = ((M.f_track_left <= C.FAIL_FRAC) | (M.f_track_right <= C.FAIL_FRAC)).astype(int)
    M["CTRL_JUMP"] = 0
    for (dev, mon, seg), g in M.groupby(["dev", "month", "seg"]):
        ev = X[(X.dev == dev) & (X.month == mon) & (X.seg == seg)].t.to_numpy()
        if not len(ev):
            continue
        t = g.t.to_numpy()
        hit = (np.abs(ev[None, :] - t[:, None]) <= 0.25).any(axis=1)   # hop is ~0.5 s
        M.loc[g.index[hit], "CTRL_JUMP"] = 1
    attrs = [c for c in A.columns if c not in ("t", "dev", "month", "seg", "FAIL", "ui")]
    print(M.groupby(["dev", "month"])[["CTRL_LOST", "CTRL_JUMP"]].agg(["size", "sum"]))

    rows = []
    for ev in ("CTRL_LOST", "CTRL_JUMP"):
        for dev, g in M.groupby("dev"):
            for a in attrs:
                on = g[a] > 0.5
                if on.sum() < 100 or (~on).sum() < 100:
                    continue
                r1, r0 = g.loc[on, ev].mean(), g.loc[~on, ev].mean()
                higher, tested = 0, 0
                for _, h in g.groupby(["month", "seg"]):
                    o = h[a] > 0.5
                    if o.sum() >= 20 and (~o).sum() >= 20 and h[ev].sum() > 0:
                        tested += 1
                        higher += int(h.loc[o, ev].mean() > h.loc[~o, ev].mean())
                p = binomtest(higher, tested, 0.5).pvalue if tested else np.nan
                rows.append(dict(event=ev, dev=dev, attribute=a, windows_with=int(on.sum()),
                                 rate_with=r1, rate_without=r0,
                                 ratio=r1 / r0 if r0 > 0 else np.nan,
                                 recordings_tested=tested, recordings_higher=higher, p_sign=p))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "rates.csv", index=False)
    pd.set_option("display.width", 230, "display.precision", 3, "display.max_rows", 200)
    keyw = ("geom_walls", "tex_", "repeat_", "sur_", "cont_in_view", "cont_occ", "cont_sun",
            "cont_shade", "illum", "scene_", "sun_in_fov", "exposure", "refl")
    for ev in ("CTRL_LOST", "CTRL_JUMP"):
        S = T[(T.event == ev) & T.attribute.str.contains("|".join(keyw))]
        S = S[(S.recordings_tested >= 4)].sort_values(["dev", "ratio"], ascending=[True, False])
        print(f"\n=== {ev}: attributes tested in >= 4 recordings, sorted by ratio")
        print(S[["dev", "attribute", "windows_with", "rate_with", "rate_without", "ratio",
                 "recordings_tested", "recordings_higher", "p_sign"]].to_string(index=False))


if __name__ == "__main__":
    main()
