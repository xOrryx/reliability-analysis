# -----------------------------------------------------------------------------
# Splitting an out-and-back recording into legs, and locating the two route ends.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
#          (supplementary repeatability measurement)
# -----------------------------------------------------------------------------
"""Segmenting the recording.

The wearer walks repeatedly between two marked points. The recording is reduced
to an ordered list of visits to those two points; every reported quantity is
built from that list.

The along-route coordinate is the projection of the horizontal trajectory onto
its first principal axis. Using a principal axis rather than a reported axis
means the segmentation does not depend on how the route happens to be oriented
in the frame the runtime chose, and it survives the frame drifting during the
recording, because the drift is small compared with the length of the route.
"""
import numpy as np
import pandas as pd

from . import config as C
from .io import smooth


def along_route(P: np.ndarray) -> np.ndarray:
    """Project the horizontal trajectory onto its first principal axis."""
    H = P[:, [0, 2]]
    Hc = H - H.mean(axis=0)
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    s = Hc @ Vt[0]
    # The sign of the principal axis is arbitrary, so the labels A and B of the two
    # marks follow it; visits.csv shows which mark each recording starts at.
    return s


def speed(t: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Instantaneous horizontal speed, one value per sample."""
    d = np.linalg.norm(np.diff(P[:, [0, 2]], axis=0), axis=1) / np.diff(t)
    return np.r_[d[0], d]


def find_stands(t, v, rate):
    """Contiguous intervals of near-zero speed, as (i0, i1) index pairs."""
    still = smooth(v, int(round(rate))) < C.STILL_MAX
    d = np.diff(np.r_[0, still.astype(int), 0])
    out = []
    for i0, i1 in zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1):
        if t[min(i1, len(t) - 1)] - t[i0] >= C.MIN_STAND_S:
            out.append((int(i0), int(min(i1, len(t) - 1))))
    return out


def find_turnarounds(t, s, rate):
    """Indices at which the along-route coordinate reverses direction."""
    ss = smooth(s, int(round(C.SMOOTH_S * rate)))
    sign = np.sign(np.gradient(ss))
    cand = np.where(np.diff(sign) != 0)[0]
    keep = []
    for i in cand:
        if keep and t[i] - t[keep[-1]] < C.MIN_TURN_GAP_S:
            # Keep the more extreme of the two: a stand at the end of the route
            # otherwise splits one turnaround into two.
            lo = s[keep[-1]] < s.mean()
            if (s[i] < s[keep[-1]]) == lo:
                keep[-1] = int(i)
            continue
        keep.append(int(i))
    return keep


def build_visits(dev: str, t, P, ok, ui, rate) -> pd.DataFrame:
    """One row per visit to a route end.

    The reported position of a visit is the median position over a window around
    the visit, which averages the high-frequency variation away without changing
    the location. When the wearer stood still at the mark, the median is taken
    over the stand, which is the cleaner measurement; otherwise it is taken over
    a fixed window around the turnaround, and the row is flagged accordingly.
    """
    s = along_route(P)
    v = speed(t, P)
    turns = find_turnarounds(t, s, rate)
    stands = find_stands(t, v, rate)
    half = int(round(C.END_WINDOW_S * rate))
    mid = 0.5 * (s.max() + s.min())

    rows = []
    for i in turns:
        stand = None
        for i0, i1 in stands:
            if t[i0] - C.STAND_NEAR_TURN_S <= t[i] <= t[i1] + C.STAND_NEAR_TURN_S:
                stand = (i0, i1)
                break
        j0, j1 = stand if stand else (max(0, i - half), min(len(t) - 1, i + half))
        sl = slice(j0, j1 + 1)
        rows.append(dict(
            device=dev, t_s=float(t[i]), i=int(i),
            end="B" if s[i] > mid else "A",
            source="stand" if stand else "turnaround",
            window_s=float(t[j1] - t[j0]),
            X=float(np.median(P[sl, 0])), Y=float(np.median(P[sl, 1])),
            Z=float(np.median(P[sl, 2])),
            along_m=float(np.median(s[sl])),
            track_frac=float(ok[sl].mean()), ui_frac=float(ui[sl].mean()),
        ))

    V = pd.DataFrame(rows)
    if V.empty:
        return V
    V = V.sort_values("t_s").reset_index(drop=True)

    # Path length actually walked up to each visit, the quantity against which
    # accumulated drift is expected to grow.
    dist = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    V["path_m"] = dist[V["i"].to_numpy()]
    V["visit"] = V.groupby("end").cumcount() + 1
    V["leg_m"] = V["path_m"].diff()
    return V
