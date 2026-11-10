# -----------------------------------------------------------------------------
# Per-frame scene descriptors joined onto the pose windows.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Scene description attached to pose windows.

Input is the per-frame descriptor table produced by the MATLAB stage, one row per
video frame, with Time_s = 0 at the first decoded frame of the cut video. Video
time is placed on the pose clock by sync.video_to_log, using the offset measured
for that segment (sync/video_offsets.csv). Each pose window then takes the aggregate of every
descriptor over the video frames whose timestamp falls inside that window, using
the median for all descriptors except the overexposed fraction, for which the
mean is used, following Eq. (7) of the manuscript.

Descriptors exist for the February session only; the January recordings carry
manual annotations but no video descriptors.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from . import config as C, sync


def header(root: Path = None):
    root = Path(root or C.DATA_ROOT)
    cols = list(pd.read_csv(root / "Data_processed" / "time_sync" / "header_video.csv",
                            nrows=0).columns)
    return cols, [c for c in cols if c != "Time_s"]


def _aggregate(X: np.ndarray, names, lo, hi):
    """Median per descriptor over [lo, hi), with the mean for the listed ones."""
    mean_idx = [names.index(c) for c in C.MEAN_AGGREGATED if c in names]
    out = np.empty((len(lo), X.shape[1]))
    for r, (a, b) in enumerate(zip(lo, hi)):
        blk = X[a:b]
        out[r] = np.median(blk, axis=0)
        for k in mean_idx:
            out[r, k] = blk[:, k].mean()
    return out


def build(W: pd.DataFrame, root: Path = None, month: str = "February") -> pd.DataFrame:
    """Join the scene description onto the pose windows of one session."""
    root = Path(root or C.DATA_ROOT)
    cols, desc = header(root)
    half = 0.5 * C.WIN_S
    W = W[(W.month == month) & (W.ui < 0.5)]
    frames = []
    for (dev, seg), g in W.groupby(["dev", "seg"]):
        f = root / "Data_processed" / "time_sync" / dev / month / f"sync_video_{seg}.csv"
        if not f.exists():
            continue
        g = g.sort_values("t").reset_index(drop=True)
        v = pd.read_csv(f, header=None, names=cols)
        ts = sync.video_to_log(dev, month, int(seg), v.Time_s.to_numpy(float), root)
        X = v[desc].to_numpy(float)
        lo = np.searchsorted(ts, g.t.to_numpy() - half)
        hi = np.searchsorted(ts, g.t.to_numpy() + half)
        keep = (hi - lo) >= C.MIN_FRAMES_PER_WINDOW
        d = pd.DataFrame(_aggregate(X, desc, lo[keep], hi[keep]), columns=desc)
        gg = g[keep].reset_index(drop=True)
        for c in ("t", "speed", "f_track_hmd", "hp_pos_hmd", "hp_rot_hmd", "ui"):
            d[c] = gg[c].to_numpy()
        d["FAIL"] = (gg.f_track_hmd <= C.FAIL_FRAC).astype(int).to_numpy()
        d["n_frames"] = (hi - lo)[keep]
        d["dev"], d["seg"], d["month"] = dev, seg, month
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def feature_sets(D: pd.DataFrame):
    seg = [c for c in D.columns if c.startswith("SegFrac_")]
    return {"brightness": ["Brightness"],
            "photometric": list(C.PHOTOMETRIC),
            "full": list(C.PHOTOMETRIC) + seg}
