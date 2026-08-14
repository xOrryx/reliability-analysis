# -----------------------------------------------------------------------------
# Discontinuities of the delivered pose, and the cadence at which they occur.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
#          (supplementary repeatability measurement)
# -----------------------------------------------------------------------------
"""Step changes of the reported headset position.

A walking wearer moves a few centimetres between two consecutive samples. A step
an order of magnitude larger is not locomotion. Two explanations have to be told
apart before such a step may be called a correction of the estimate:

* samples lost by the logger, in which case the step is the motion that happened
  while nothing was written, and it points along the direction of travel;
* a correction applied by the runtime to its own solution, which has no reason to
  point along the direction of travel.

The direction of the step relative to the heading established from the preceding
samples separates the two, and is reported for every step so that the separation
can be checked rather than assumed.
"""
import numpy as np
import pandas as pd

from . import config as C


def find_jumps(dev: str, t, P, ok, ui) -> pd.DataFrame:
    """One row per step larger than JUMP_MIN_M, with its direction and context."""
    D = np.diff(P, axis=0)
    step = np.linalg.norm(D, axis=1)
    dt = np.diff(t)
    valid = ok[:-1] & ok[1:] & ~ui[:-1] & ~ui[1:]
    idx = np.where((step > C.JUMP_MIN_M) & valid)[0]

    rows = []
    for k in idx:
        if k < C.HEADING_SAMPLES or k >= len(D) - 1:
            continue
        h = D[k - C.HEADING_SAMPLES:k].sum(axis=0)
        nh = np.linalg.norm(h)
        cos = float(h @ D[k] / (nh * step[k])) if nh > 0 else np.nan
        rows.append(dict(
            device=dev, i=int(k), t_s=float(t[k]),
            step_m=float(step[k]), dt_s=float(dt[k]),
            implied_speed_ms=float(step[k] / dt[k]),
            dX=float(D[k, 0]), dY=float(D[k, 1]), dZ=float(D[k, 2]),
            vertical_share=float(abs(D[k, 1]) / step[k]),
            cos_heading=cos,
            along_travel=bool(np.isfinite(cos) and cos > C.HEADING_COS_MAX),
        ))
    return pd.DataFrame(rows)


def walking_heading_cos(t, P, ok, ui, rng=(0.03, 0.06), limit=5000) -> float:
    """Median cos(step, heading) for ordinary walking steps, as a control.

    Locomotion continues in the direction it was already going, so this value is
    close to one. It is the reference against which the jumps are read.
    """
    D = np.diff(P, axis=0)
    step = np.linalg.norm(D, axis=1)
    valid = ok[:-1] & ok[1:] & ~ui[:-1] & ~ui[1:]
    idx = np.where((step > rng[0]) & (step < rng[1]) & valid)[0]
    idx = idx[idx >= C.HEADING_SAMPLES][:limit]
    out = []
    for k in idx:
        h = D[k - C.HEADING_SAMPLES:k].sum(axis=0)
        nh = np.linalg.norm(h)
        if nh > 0:
            out.append(h @ D[k] / (nh * step[k]))
    return float(np.median(out)) if out else np.nan


def periodicity(J: pd.DataFrame) -> dict:
    """Test whether jumps recur at multiples of PERIOD_S.

    Under a uniform null the fraction of intervals falling within PERIOD_TOL_S of
    a multiple of PERIOD_S is 2*PERIOD_TOL_S / PERIOD_S, which is the number the
    observed fraction has to be read against.
    """
    lo, hi = C.PERIOD_GAP_RANGE_S
    g = np.diff(np.sort(J["t_s"].to_numpy(float)))
    g = g[(g > lo) & (g < hi)]
    null = 2.0 * C.PERIOD_TOL_S / C.PERIOD_S
    if len(g) < 6:
        return dict(n_gaps=int(len(g)), hit_rate=np.nan, null_rate=null, ratio=np.nan)
    hit = float((np.abs(g - np.round(g / C.PERIOD_S) * C.PERIOD_S) < C.PERIOD_TOL_S).mean())
    return dict(n_gaps=int(len(g)), hit_rate=hit, null_rate=null, ratio=hit / null)


def size_table(J: pd.DataFrame) -> pd.DataFrame:
    """Count and cumulative magnitude of the jumps at each reporting threshold."""
    rows = []
    for dev, g in J.groupby("device", sort=False):
        for thr in C.JUMP_REPORT_M:
            s = g[g.step_m > thr]
            rows.append(dict(device=dev, threshold_m=thr, n=int(len(s)),
                             cumulative_m=float(s.step_m.sum()),
                             median_m=float(s.step_m.median()) if len(s) else np.nan,
                             max_m=float(s.step_m.max()) if len(s) else np.nan))
    return pd.DataFrame(rows)
