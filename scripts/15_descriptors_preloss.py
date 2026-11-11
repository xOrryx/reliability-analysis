# -----------------------------------------------------------------------------
# Stage 15: the scene BEFORE a loss of tracking.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Stage 15: scene descriptors before a loss vs. during normal operation.

During a loss of tracking the headset shows a black passthrough view with only
the rendered controllers (Reviewer 2, minor comment on the median brightness of
3.4 in loss windows), so the descriptors of loss windows describe the headset's
reaction rather than the scene. The scene that the device was in is described
by the windows shortly BEFORE the loss. Here three classes of February windows
(tracking held throughout, panel closed), on the measured video-to-pose clock:

  pre      window centre 1-10 s before the onset of an episode
  normal   not within 10 s before an onset, not inside a loss
  in_loss  inside a loss (for reference only)

For each descriptor: medians of the three classes and the univariate AUC of
pre vs. normal (oriented >= 0.5), pooled and per device. Also P(pre-loss window)
by brightness band.
Writes out/preloss/descriptors.csv, per_device.csv, by_brightness.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from reliability import config as C, scene

OUT = C.OUT / "preloss"
OUT.mkdir(parents=True, exist_ok=True)
LABEL = {"UnderexposedFrac": "Underexposed pixel fraction", "Brightness": "Brightness",
         "Entropy": "Entropy", "Contrast": "Contrast", "DynamicRangeP": "Dynamic range (P99-P1)",
         "Corners": "Harris corners", "SegFrac_sky_other": "Sky fraction",
         "SegFrac_snow": "Snow fraction", "OverexposedFrac": "Overexposed pixel fraction",
         "SegFrac_light": "Artificial light fraction", "Focus_LapVar": "Laplacian variance"}


def classes(S, E):
    cls = np.where(S.FAIL == 1, "in_loss", "normal").astype(object)
    for (dev, seg), g in S.groupby(["dev", "seg"]):
        o = E[(E.month == "February") & (E.dev == dev) & (E.seg == seg)].t0.to_numpy()
        if not len(o):
            continue
        dt = o[None, :] - g.t.to_numpy()[:, None]
        pre = ((dt >= 1) & (dt <= 10)).any(axis=1)
        near = ((dt > 0) & (dt <= 10)).any(axis=1)
        idx = g.index.to_numpy()
        c = cls[idx]
        c[(c == "normal") & near] = "excluded"
        c[(cls[idx] != "in_loss") & pre] = "pre"
        cls[idx] = c
    return cls


def auc(y, x):
    a = roc_auc_score(y, x)
    return max(a, 1 - a), ("lower before loss" if a < 0.5 else "higher before loss")


def main():
    S = pd.read_pickle(C.OUT / "scene.pkl").reset_index(drop=True)
    E = pd.read_csv(C.OUT / "episodes.csv")
    S = S[S.ui < 0.5].reset_index(drop=True)
    S["cls"] = classes(S, E)
    print(S.groupby(["dev", "cls"]).size().unstack(fill_value=0))
    D = S[S.cls.isin(["pre", "normal"])]
    y = (D.cls == "pre").astype(int).to_numpy()
    rows = []
    for col, lab in LABEL.items():
        a, d = auc(y, D[col].to_numpy(float))
        rows.append(dict(descriptor=lab, column=col,
                         pre=float(S.loc[S.cls == "pre", col].median()),
                         normal=float(S.loc[S.cls == "normal", col].median()),
                         in_loss=float(S.loc[S.cls == "in_loss", col].median()),
                         auc_pre_vs_normal=a, direction=d))
    T = pd.DataFrame(rows).sort_values("auc_pre_vs_normal", ascending=False)
    T.to_csv(OUT / "descriptors.csv", index=False)
    per = []
    for dev, g in D.groupby("dev"):
        yy = (g.cls == "pre").astype(int).to_numpy()
        if yy.sum() < 10:
            continue
        for col in ("Brightness", "UnderexposedFrac", "Entropy", "Corners"):
            a, d = auc(yy, g[col].to_numpy(float))
            per.append(dict(dev=dev, descriptor=col, n_pre=int(yy.sum()),
                            pre=float(g.loc[g.cls == "pre", col].median()),
                            normal=float(g.loc[g.cls == "normal", col].median()), auc=a, direction=d))
    P = pd.DataFrame(per)
    P.to_csv(OUT / "per_device.csv", index=False)
    b = D.Brightness.to_numpy(float)
    B = []
    for lo, hi in ((0, 5), (5, 15), (15, 40), (40, 1e9)):
        m = (b >= lo) & (b < hi)
        B.append(dict(brightness=f"{lo}-{hi if hi < 1e9 else 'max'}", windows=int(m.sum()),
                      pre_share=float(y[m].mean()) if m.any() else np.nan))
    B = pd.DataFrame(B)
    B.to_csv(OUT / "by_brightness.csv", index=False)
    pd.set_option("display.width", 220, "display.precision", 3)
    print(T.to_string(index=False)); print(); print(P.to_string(index=False)); print(); print(B.to_string(index=False))


if __name__ == "__main__":
    main()
