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

Expected input, one file per recording, first column a timestamp already on the
pose clock:
    Data_processed/time_sync/<device>/<month>/sync_annotation_<segment>.csv
with the column names given in Data_processed/time_sync/header_annotation.csv.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from . import config as C

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


def build(W: pd.DataFrame, root: Path = None) -> pd.DataFrame:
    root = Path(root or C.DATA_ROOT)
    head = pd.read_csv(root / "Data_processed" / "time_sync" / "header_annotation.csv",
                       header=[0, 1, 2], nrows=0)
    names = [" ".join(str(x) for x in c if "Unnamed" not in str(x)).strip()
             for c in head.columns]
    frames = []
    for (dev, mon, seg), g in W[W.ui < 0.5].groupby(["dev", "month", "seg"]):
        f = (root / "Data_processed" / "time_sync" / dev / mon
             / f"sync_annotation_{seg}.csv")
        if not f.exists():
            continue
        raw = pd.read_csv(f, header=None, names=names)
        raw = raw.rename(columns={names[0]: "t"})
        d = window_one(raw, g)
        d["dev"], d["month"], d["seg"] = dev, mon, seg
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
