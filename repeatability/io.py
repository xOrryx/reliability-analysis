# -----------------------------------------------------------------------------
# Reading the raw pose logs written by the Unity logging application.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
#          (supplementary repeatability measurement)
# -----------------------------------------------------------------------------
"""Reading the raw pose logs.

The log layout is the one produced by the Unity logging application and is
identical to the one used by the main analysis repository: semicolon separated,
two header lines, one row per runtime update.
"""
from pathlib import Path

import numpy as np

from . import config as C


def load_log(path) -> np.ndarray:
    """Return the numeric body of a measHMD_*.csv as an (n, N_COLS) array."""
    a = np.genfromtxt(path, delimiter=";", skip_header=2, usecols=range(C.N_COLS))
    a = a[np.isfinite(a[:, C.COL_TIME])]
    return a[np.argsort(a[:, C.COL_TIME])]


def device_logs(data_dir: Path = None):
    """Yield (device, path) for every measHMD_<DEV>.csv found, in DEVICES order."""
    root = Path(data_dir or C.DATA_DIR)
    for dev in C.DEVICES:
        p = root / f"measHMD_{dev}.csv"
        if p.exists():
            yield dev, p


def load_device(dev: str, path):
    """Load a log and keep only the samples that belong to the measurement.

    Every stage reads its data through this function, so all of them see exactly
    the same samples and a truncation cannot be applied in one place and
    forgotten in another.
    """
    from .trim import route_mask
    a = load_log(path)
    mask, info = route_mask(dev, a[:, C.COL_TIME], a[:, C.PARTS["hmd"]["pos"]])
    return a[mask], info, a


def unpack(a: np.ndarray):
    """Split the raw array into time, position, tracking flag and panel flag.

    Position is returned in metres in the reported frame, with no re-referencing:
    every quantity in this package is either a difference between two poses of the
    same recording or a distance, so the origin never enters a reported number.
    """
    t = a[:, C.COL_TIME]
    P = a[:, C.PARTS["hmd"]["pos"]]
    ok = a[:, C.PARTS["hmd"]["track"]] > 0.5
    ui = a[:, C.COL_UI] != 0
    return t, P, ok, ui


def head_pitch_deg(a: np.ndarray) -> np.ndarray:
    """Elevation of the headset's forward axis, in degrees, negative downwards.

    Where the wearer's head points decides what the cameras see, and therefore
    which part of the scene the tracking solution is built on. Two recordings
    walked with different head pitch are not directly comparable, so the pitch is
    reported for every recording rather than assumed to be the same.
    """
    q = a[:, C.PARTS["hmd"]["rot"]]
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    forward_y = 2.0 * (y * z - w * x)          # Unity: local +Z, world Y component
    return np.degrees(np.arcsin(np.clip(forward_y, -1.0, 1.0)))


def native_rate(t: np.ndarray) -> float:
    """Median sampling rate of a recording, in hertz."""
    return 1.0 / float(np.median(np.diff(t)))


def smooth(x: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average of odd width n, with the ends held constant."""
    n = max(1, int(n) | 1)
    if n == 1:
        return x.astype(float)
    pad = n // 2
    xp = np.r_[np.full(pad, x[0]), x, np.full(pad, x[-1])]
    return np.convolve(xp, np.ones(n) / n, mode="valid")
