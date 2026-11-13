# -----------------------------------------------------------------------------
# Stage 10: when do the controllers stop being tracked, and what is in view?
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Stage 10: loss of controller tracking against the scene, February.

The scene analyses so far concern the headset. Here the event is a window in
which the headset is tracked throughout but a controller is reported untracked
for more than 80 % of the samples (the same criterion as for the headset).
For every scene descriptor: median during controller loss and otherwise, the
univariate AUC (oriented >= 0.5) and its direction, per device and pooled.
Windows with the panel open are excluded. A cross-validated detector (one recording
or one whole segment held out) on the full descriptor set is reported as well, to see whether the scene predicts
controller loss beyond darkness (brightness alone vs. full set).

Writes out/controller_loss/*.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from reliability import config as C, scene, detect

OUT = C.OUT / "controller_loss"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    W = pd.read_pickle(C.OUT / "windows.pkl")
    S = pd.read_pickle(C.OUT / "scene.pkl")
    S = S.merge(W[["dev", "month", "seg", "t", "f_track_left", "f_track_right"]],
                on=["dev", "month", "seg", "t"], how="left")
    S = S[(S.f_track_hmd > 0.99) & (S.ui < 0.5)].reset_index(drop=True)
    lost = ((S.f_track_left <= C.FAIL_FRAC) | (S.f_track_right <= C.FAIL_FRAC)).astype(int)
    S["CTRL_LOST"] = lost
    print("windows with headset tracked:", len(S), " controller lost:", int(lost.sum()),
          f"({100 * lost.mean():.1f} %)")
    print(S.groupby("dev").CTRL_LOST.agg(["size", "sum", "mean"]))

    desc = scene.feature_sets(S)["full"] + ["DynamicRange"]
    desc = [c for c in dict.fromkeys(desc) if c in S.columns]
    rows = []
    for dev, g in [("all", S)] + list(S.groupby("dev")):
        y = g.CTRL_LOST.to_numpy()
        if y.sum() < 20 or (1 - y).sum() < 20:
            continue
        for c in desc:
            x = g[c].to_numpy(float)
            if np.nanstd(x) == 0:
                continue
            a = roc_auc_score(y, x)
            rows.append(dict(dev=dev, descriptor=c, lost=float(np.median(x[y == 1])),
                             mean_lost=float(np.mean(x[y == 1])), mean_normal=float(np.mean(x[y == 0])),
                             normal=float(np.median(x[y == 0])), auc=max(a, 1 - a),
                             direction="higher when lost" if a > 0.5 else "lower when lost",
                             n_lost=int(y.sum())))
    T = pd.DataFrame(rows).sort_values(["dev", "auc"], ascending=[True, False])
    T.to_csv(OUT / "descriptors.csv", index=False)

    y = S.CTRL_LOST.to_numpy()
    res = []
    for name, cols in (("brightness", ["Brightness"]), ("full", scene.feature_sets(S)["full"])):
        for gname, groups in (("recording", (S.dev + S.seg.astype(str)).to_numpy()),
                              ("segment", S.seg.astype(str).to_numpy())):
            X = S[cols].to_numpy(float)
            p = np.zeros(len(S))
            for k in np.unique(groups):
                m = groups == k
                clf = detect._model().fit(X[~m], y[~m])
                p[m] = clf.predict_proba(X[m])[:, 1]
            res.append(dict(features=name, cv=gname, auc=float(roc_auc_score(y, p))))
    R = pd.DataFrame(res)
    R.to_csv(OUT / "detector.csv", index=False)

    stone = [c for c in ("SegFrac_stone", "SegFrac_rock", "SegFrac_wall_stone", "SegFrac_wall_brick",
                         "SegFrac_wall_concrete", "SegFrac_gravel", "SegFrac_wall_other") if c in S.columns]
    pd.set_option("display.width", 220, "display.precision", 3, "display.max_rows", 80)
    print("\nTop descriptors, pooled:")
    print(T[T.dev == "all"].head(15).to_string(index=False))
    print("\nStone / wall related, per device:")
    print(T[T.descriptor.isin(stone)].to_string(index=False))
    print("\nDetector of controller loss:")
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
