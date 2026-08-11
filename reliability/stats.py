# -----------------------------------------------------------------------------
# Block bootstrap, partial correlation, sign test and ROC helpers.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Inference helpers.

The one thing worth reading here is why the bootstrap resamples segments. The
windows overlap by one half and consecutive windows within a recording are
strongly dependent, so resampling at the window level treats correlated
observations as independent and understates the interval by roughly a factor of
eight on these data. Whole recording segments are resampled instead.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata, binomtest
from sklearn.metrics import roc_auc_score, roc_curve

from . import config as C


def block_bootstrap_median(df: pd.DataFrame, col: str,
                           group=("dev", "seg", "month"),
                           n: int = None, seed: int = None):
    """Median with a 95 % interval from resampling whole segments."""
    rng = np.random.default_rng(C.SEED if seed is None else seed)
    keys = [k for k in df.groupby(list(group)).groups.values()
            if np.isfinite(df.loc[k, col]).any()]
    if len(keys) < 2:
        return np.nan, np.nan, np.nan
    vals = [df.loc[k, col].to_numpy() for k in keys]
    point = float(np.nanmedian(np.concatenate(vals)))
    out = np.empty(n or C.BOOTSTRAP_N)
    for i in range(len(out)):
        pick = rng.integers(0, len(vals), len(vals))
        out[i] = np.nanmedian(np.concatenate([vals[p] for p in pick]))
    return point, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def bootstrap_median_arrays(arrays, n: int = None, seed: int = None):
    """Same, when the per-segment arrays are already assembled."""
    rng = np.random.default_rng(C.SEED if seed is None else seed)
    vals = [a[np.isfinite(a)] for a in arrays]
    vals = [a for a in vals if len(a)]
    if len(vals) < 2:
        return np.nan, np.nan, np.nan
    point = float(np.median(np.concatenate(vals)))
    out = np.empty(n or C.BOOTSTRAP_N)
    for i in range(len(out)):
        pick = rng.integers(0, len(vals), len(vals))
        out[i] = np.median(np.concatenate([vals[p] for p in pick]))
    return point, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def partial_spearman(x, y, z):
    """Spearman correlation of x and y with z partialled out."""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    c = np.corrcoef(np.vstack([rx, ry, rz]))
    d = np.sqrt((1 - c[0, 2] ** 2) * (1 - c[1, 2] ** 2))
    return np.nan if d == 0 else float((c[0, 1] - c[0, 2] * c[1, 2]) / d)


def sign_test(values):
    """Two-sided sign test on the direction of an effect across groups.

    Returns (k, positives, p). With fewer than six groups the smallest attainable
    p is above 0.05, which is why config.MIN_GROUPS_SIGN_TEST exists: an attribute
    with fewer groups is not testable at that level and must not be counted as a
    null result.
    """
    v = np.asarray([x for x in values if np.isfinite(x)])
    k = len(v)
    if k == 0:
        return 0, 0, np.nan
    pos = int((v > 0).sum())
    return k, pos, float(binomtest(pos, k, 0.5).pvalue)


def auc(y, score):
    """Area under the ROC curve, oriented so that it is at least 0.5."""
    a = roc_auc_score(y, score)
    return float(max(a, 1 - a))


def spec_at_sens(y, p, sens):
    fpr, tpr, _ = roc_curve(y, p)
    return float(1 - fpr[min(np.searchsorted(tpr, sens), len(tpr) - 1)])


def precision_at_sens(y, p, sens):
    fpr, tpr, _ = roc_curve(y, p)
    i = min(np.searchsorted(tpr, sens), len(tpr) - 1)
    b = float(np.mean(y))
    return float(b * tpr[i] / (b * tpr[i] + (1 - b) * fpr[i]))


def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
        den = 1 + z ** 2 / n
        cen = (p + z ** 2 / (2 * n)) / den
        hw = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, np.clip(cen - hw, 0, 1), np.clip(cen + hw, 0, 1)
