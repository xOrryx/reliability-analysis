# -----------------------------------------------------------------------------
# Leave-one-segment-out detection of tracking loss from the scene.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Detection of tracking loss from the scene description.

The association between the scene and the loss of tracking is evaluated as a
detection problem rather than as a correlation, because the windows are not
independent and a correlation over them would be optimistic. All windows of one
device-segment recording are held out together, so no window is ever scored by a
model that has seen any part of its own recording.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from . import config as C
from .stats import spec_at_sens


def groups(D: pd.DataFrame) -> np.ndarray:
    return (D.dev.astype(str) + "_" + D.month.astype(str) + "_"
            + D.seg.astype(str)).to_numpy()


def _model():
    return HistGradientBoostingClassifier(random_state=C.SEED)


def leave_one_segment_out(D: pd.DataFrame, cols, target: str = "FAIL") -> np.ndarray:
    """Cross-validated probability for every window."""
    y = D[target].to_numpy()
    g = groups(D)
    X = D[cols].to_numpy(float)
    p = np.zeros(len(D))
    for k in np.unique(g):
        m = g == k
        clf = _model()
        clf.fit(X[~m], y[~m])
        p[m] = clf.predict_proba(X[m])[:, 1]
    return p


def summarise(y, p) -> dict:
    return dict(auc=float(roc_auc_score(y, p)),
                **{f"spec_at_{int(s*100)}": spec_at_sens(y, p, s)
                   for s in C.SENS_LEVELS})


def transfer(D: pd.DataFrame, cols, target: str = "FAIL") -> pd.DataFrame:
    """Train on every device but one, evaluate on the one held out."""
    y = D[target].to_numpy()
    X = D[cols].to_numpy(float)
    rows = []
    for dev in sorted(D.dev.unique()):
        m = (D.dev == dev).to_numpy()
        if y[m].sum() == 0 or (y[m] == 0).sum() == 0:
            continue
        clf = _model()
        clf.fit(X[~m], y[~m])
        rows.append(dict(dev=dev, n=int(m.sum()), fail=int(y[m].sum()),
                         auc=float(roc_auc_score(y[m], clf.predict_proba(X[m])[:, 1]))))
    return pd.DataFrame(rows)


def dark_subset(D: pd.DataFrame, p: np.ndarray):
    """The same probabilities restricted to the darkest quarter of windows."""
    b = D.Brightness.to_numpy(float)
    thr = float(np.nanquantile(b, C.DARK_QUANTILE))
    m = np.isfinite(b) & (b < thr)
    y = D.FAIL.to_numpy()[m]
    return thr, int(m.sum()), float(y.mean()), float(roc_auc_score(y, p[m]))
