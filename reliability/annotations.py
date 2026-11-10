# -----------------------------------------------------------------------------
# Manual scene annotations on the pose-window grid.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Manual scene annotations on the pose-window grid.

The annotations are per video frame and exist for both sessions, unlike the
automatic descriptors, which exist only for February. Each attribute is a binary
flag per frame; a window takes the mean of the flag over its frames, so an
attribute that holds for the whole window is 1 and one that changes inside the
window takes an intermediate value.

Input is what timeSynchronization.m writes, one file per recording:
    Data_processed/time_sync/<device>/<month>/sync_anotation_<segment>.csv
first column the video time of the cut segment (the same time base as the scene
descriptors), then the manual attributes, with the three-row header in
Data_processed/time_sync/header_annotation.csv. The attribute name is the third
header row (HMD_..., LEFT_..., RIGHT_...); a name that repeats is numbered.
Video time is placed on the pose clock by sync.video_to_log(kind="annotations"),
which accounts for the annotation having been made on the time of the cut file.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from . import config as C, sync

RESERVED = ("t", "time", "time [s]", "time_video [s]", "c1", "c113")


def _attribute_columns(df: pd.DataFrame):
    return [c for c in df.columns
            if str(c).strip().lower() not in RESERVED
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].notna().any()]


def window_one(raw: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """Mean of every attribute over the frames inside each pose window."""
    attrs = _attribute_columns(raw)
    ts = raw["t"].to_numpy(float)
    X = raw[attrs].to_numpy(float)
    half = 0.5 * C.WIN_S
    w = w.sort_values("t").reset_index(drop=True)
    lo = np.searchsorted(ts, w.t.to_numpy() - half)
    hi = np.searchsorted(ts, w.t.to_numpy() + half)
    keep = (hi - lo) >= C.MIN_FRAMES_PER_WINDOW
    cs = np.vstack([np.zeros(X.shape[1]), np.nancumsum(X, axis=0)])
    M = (cs[hi[keep]] - cs[lo[keep]]) / (hi[keep] - lo[keep])[:, None]
    out = pd.DataFrame(M, columns=attrs)
    g = w[keep].reset_index(drop=True)
    out["t"] = g.t.to_numpy()
    out["ui"] = g.ui.to_numpy()
    out["FAIL"] = (g.f_track_hmd <= C.FAIL_FRAC).astype(int).to_numpy()
    return out


def attribute_names(root: Path = None):
    """Column names of the annotation files: third header row, repeats numbered."""
    root = Path(root or C.DATA_ROOT)
    h = pd.read_csv(root / "Data_processed" / "time_sync" / "header_annotation.csv",
                    header=None, nrows=3, dtype=str).fillna("")
    names, seen = [], {}
    for j in range(h.shape[1]):
        v = h.iat[2, j].strip() or h.iat[0, j].strip() or f"c{j + 1}"
        seen[v] = seen.get(v, 0) + 1
        names.append(v if seen[v] == 1 else f"{v}_{seen[v]}")
    names[0] = "t"
    return names


def build(W: pd.DataFrame, root: Path = None) -> pd.DataFrame:
    root = Path(root or C.DATA_ROOT)
    names = attribute_names(root)
    frames = []
    for (dev, mon, seg), g in W[W.ui < 0.5].groupby(["dev", "month", "seg"]):
        d0 = root / "Data_processed" / "time_sync" / dev / mon
        f = d0 / f"sync_anotation_{seg}.csv"
        if not f.exists():
            f = d0 / f"sync_annotation_{seg}.csv"
        if not f.exists():
            continue
        raw = pd.read_csv(f, header=None)
        raw.columns = names[:raw.shape[1]]
        raw["t"] = sync.video_to_log(dev, mon, int(seg), raw["t"].to_numpy(float),
                                     root, kind="annotations")
        d = window_one(raw, g)
        d["dev"], d["month"], d["seg"] = dev, mon, seg
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
