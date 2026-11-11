# -----------------------------------------------------------------------------
# Stage 16: manual vs. automatic description under segment hold-out; sweep bounds.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Stage 16.

(a) Table 7 again, with both cross-validation groupings: one device-segment
    recording held out (as before) and one whole segment held out (all devices,
    Reviewer 2), on the windows where both descriptions exist (February).
(b) The effect sizes of the scene sweep (out/sweep.csv) that bound the negative
    result in Section 3.2: the largest |median partial rho| among testable
    automatic descriptors, the same among the texture descriptors, and the most
    extreme median ratio among testable annotated attributes.
Writes out/manual_vs_auto.csv and out/sweep_bounds.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from reliability import config as C, scene, detect
from reliability.stats import spec_at_sens

TEXTURE = ["Corners", "Focus_LapVar", "Entropy", "Contrast", "DynamicRangeP"]


def cv(D, cols, groups):
    X, y = D[cols].to_numpy(float), D.FAIL.to_numpy()
    p = np.zeros(len(D))
    for k in np.unique(groups):
        m = groups == k
        p[m] = detect._model().fit(X[~m], y[~m]).predict_proba(X[m])[:, 1]
    return dict(auc=float(roc_auc_score(y, p)), spec_at_80=spec_at_sens(y, p, 0.8),
                spec_at_90=spec_at_sens(y, p, 0.9))


def main():
    S = pd.read_pickle(C.OUT / "scene.pkl")
    A = pd.read_pickle(C.OUT / "annotations.pkl")
    attrs = [c for c in A.columns if c not in ("t", "dev", "month", "seg", "FAIL", "ui")]
    M = pd.merge(S, A[attrs + ["t", "dev", "seg", "month"]], on=["dev", "seg", "month", "t"], how="inner")
    desc = scene.feature_sets(S)["full"]
    rows = []
    for gname, g in (("recording", (M.dev + M.seg.astype(str)).to_numpy()),
                     ("segment", M.seg.astype(str).to_numpy())):
        for name, cols in (("Manual annotations", attrs), ("Automatic descriptors", desc),
                           ("Both combined", attrs + desc), ("Brightness alone", ["Brightness"])):
            rows.append(dict(cv=gname, feature_set=name, n_features=len(cols), n=len(M),
                             fail=int(M.FAIL.sum()), **cv(M, cols, g)))
    T = pd.DataFrame(rows)
    T.to_csv(C.OUT / "manual_vs_auto.csv", index=False)

    W = pd.read_csv(C.OUT / "sweep.csv")
    t = W[W.testable]
    obj, sub = t[t.family == "objective"], t[t.family == "subjective"]
    tex = obj[obj.attribute.isin(TEXTURE)]
    B = pd.DataFrame([
        dict(quantity="testable objective", n=len(obj),
             max_abs_median_rho=float(obj.median_rho.abs().max())),
        dict(quantity="testable texture descriptors", n=len(tex),
             max_abs_median_rho=float(tex.median_rho.abs().max())),
        dict(quantity="testable subjective", n=len(sub),
             min_median_ratio=float(sub.median_ratio.min()), max_median_ratio=float(sub.median_ratio.max())),
        dict(quantity="all testable", n=len(t), passed=int((t.p_sign < 0.05).sum()),
             expected=0.05 * len(t)),
    ])
    B.to_csv(C.OUT / "sweep_bounds.csv", index=False)
    pd.set_option("display.width", 220, "display.precision", 3)
    print(T.to_string(index=False)); print(); print(B.to_string(index=False))


if __name__ == "__main__":
    main()
