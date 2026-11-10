# -----------------------------------------------------------------------------
# Pose logs to windowed jitter indicators and episodes of tracking loss.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Pose logs to windowed indicators.

The two quantities the manuscript reports are produced here: the residual
variation of the delivered pose after a high-pass filter, and the binary loss of
tracking reported by the device.

Why the filter and not a detrend. Within a one-second window the wearer's head
does not move in a straight line: it oscillates vertically at the step cadence,
about 1.7 to 1.9 Hz, so the window holds roughly two full periods and a fitted
straight line removes almost none of it. A 5 Hz high-pass lies above that band
and below the band the estimator noise occupies.

Why the native timeline. The runtime delivers poses at a non-uniform period.
Resampling onto a common grid would alter exactly the high-frequency content the
indicator is meant to measure, so the pose stream is windowed on its own grid,
derived from the measured rate of each recording.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from . import config as C


# ----------------------------------------------------------------- loading
def load_segment(path) -> np.ndarray:
    df = pd.read_csv(path, sep=";", skiprows=2, header=None, engine="c")
    return df.iloc[:, :C.N_COLS].astype(float).to_numpy()


def segment_files(root: Path = None):
    """Yield (device, month, segment index, path) for every raw pose log."""
    root = Path(root or C.DATA_ROOT)
    for dev in C.DEVICES:
        for mon in C.MONTHS:
            d = root / "Data_raw" / dev / "Data" / mon
            for f in sorted(d.glob("measHMD_*.csv")):
                yield dev, mon, int(f.stem.split("_")[-1]), f


# ----------------------------------------------------------------- helpers
def native_rate(t: np.ndarray) -> float:
    return 1.0 / float(np.median(np.diff(t)))


def quat_rate_deg(q: np.ndarray, dt: float) -> np.ndarray:
    """Magnitude of the angular rate in deg/s between successive samples.

    Expressed as a rate rather than as a per-frame displacement so that
    recordings acquired at different sampling rates stay comparable.
    """
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    d = np.abs(np.sum(qn[:-1] * qn[1:], axis=1))
    ang = np.degrees(2.0 * np.arccos(np.clip(d, -1.0, 1.0))) / dt
    return np.r_[ang[0], ang]


def continuous_hemisphere(q: np.ndarray) -> np.ndarray:
    """Flip quaternion signs so that consecutive samples lie in one hemisphere.

    q and -q are the same rotation, but interpolating component-wise between
    them passes through a near-zero quaternion. Making the sequence continuous
    before interpolation removes that artefact.
    """
    q = q.copy()
    d = np.sum(q[1:] * q[:-1], axis=1)
    flip = np.cumsum(np.r_[0, (d < 0).astype(int)]) % 2 == 1
    q[flip] *= -1
    return q


def quat_rate_vec_deg(q_xyzw: np.ndarray, dt: float) -> np.ndarray:
    """Angular velocity VECTOR in deg/s between successive samples (body frame).

    The relative rotation between consecutive quaternions is converted to a
    rotation vector and divided by the interval. High-pass filtering the
    components of this vector, rather than its magnitude, avoids rectifying the
    gait oscillation: the magnitude of an oscillating rate is a rectified signal
    whose harmonics reach above the cut-off (Reviewer 2, round 2).
    """
    from scipy.spatial.transform import Rotation
    q = q_xyzw / np.linalg.norm(q_xyzw, axis=1, keepdims=True)
    r = Rotation.from_quat(q)
    w = np.degrees((r[:-1].inv() * r[1:]).as_rotvec()) / dt
    return np.vstack([w[:1], w])


def contiguous(mask: np.ndarray, min_len: int):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def highpass(x: np.ndarray, rate: float, fc: float = None, order: int = C.ORDER):
    sos = butter(order, (fc or C.FC_HZ) / (rate / 2.0), "high", output="sos")
    return sosfiltfilt(sos, x, axis=0)


def highpass_blocks(x: np.ndarray, ok: np.ndarray, rate: float, fc: float = None):
    """High-pass each contiguous valid block separately, trimming the transient."""
    out = np.full(x.shape, np.nan)
    edge = int(round(C.EDGE_S * rate))
    for i, j in contiguous(ok, int(round(C.MIN_BLOCK_S * rate))):
        out[i + edge:j - edge] = highpass(x[i:j], rate, fc)[edge:j - i - edge]
    return out


# ----------------------------------------------------------------- core
def segment_windows(path, fc: float = None) -> pd.DataFrame:
    """One row per window, for the headset and both controllers.

    Columns: t, speed, ui, and for each tracked body f_track_*, hp_pos_* in cm
    and hp_rot_* in deg/s. Windows that do not lie inside a long enough block of
    valid tracking carry NaN in the jitter columns but keep t, speed and ui, so
    that analyses conditioning on speed do not silently lose them.
    """
    a = load_segment(path)
    t = a[:, C.COL_TIME]
    rate = native_rate(t)
    dt = 1.0 / rate
    w = int(round(rate * C.WIN_S))
    hop = max(1, int(round(w * C.OVERLAP)))
    if len(t) < 3 * w:
        return pd.DataFrame()

    tt = np.arange(t[0], t[-1], dt)
    n = len(tt)
    ui = np.interp(tt, t, (a[:, C.COL_UI] != 0).astype(float)) > 0.5
    H = np.column_stack([np.interp(tt, t, a[:, 4 + k]) for k in range(3)])
    d = np.linalg.norm(np.diff(H, axis=0), axis=1) * rate
    speed = np.r_[d[0], d]

    cols = {"t": tt, "speed": speed, "ui": ui.astype(float)}
    for name, s in C.PARTS.items():
        ok_raw = (a[:, s["track"]] == 1) & np.isfinite(a[:, s["pos"]]).all(axis=1)
        ok = np.interp(tt, t, ok_raw.astype(float)) > 0.999
        pos = np.column_stack([np.interp(tt, t, a[:, s["pos"]][:, k]) for k in range(3)])
        qraw = continuous_hemisphere(a[:, s["rot"]])
        rot = np.column_stack([np.interp(tt, t, qraw[:, k]) for k in range(4)])
        ang = quat_rate_deg(rot[:, [3, 0, 1, 2]], dt)
        angv = quat_rate_vec_deg(rot, dt)
        cols[f"ok_{name}"] = ok.astype(float)
        cols[f"_p_{name}"] = highpass_blocks(pos, ok, rate, fc)
        cols[f"_rm_{name}"] = highpass_blocks(ang[:, None], ok, rate, fc)[:, 0]
        cols[f"_r_{name}"] = highpass_blocks(angv, ok, rate, fc)

    rows = []
    for s0 in np.arange(0, max(0, n - w), hop, dtype=int):
        i = slice(s0, s0 + w)
        r = {"t": float(tt[i].mean()), "speed": float(speed[i].mean()),
             "ui": float(ui[i].mean())}
        for name in C.PARTS:
            r[f"f_track_{name}"] = float(cols[f"ok_{name}"][i].mean())
            p = cols[f"_p_{name}"][i]
            q = cols[f"_r_{name}"][i]
            qm = cols[f"_rm_{name}"][i]
            good = np.isfinite(p).all(axis=1)
            if good.mean() >= C.MIN_VALID:
                r[f"hp_pos_{name}"] = 100.0 * float(np.sqrt(np.nanmean(np.sum(p[good] ** 2, axis=1))))
                r[f"hp_rot_{name}"] = float(np.sqrt(np.nanmean(np.sum(q[good] ** 2, axis=1))))
                r[f"hp_rotmag_{name}"] = float(np.sqrt(np.nanmean(qm[good] ** 2)))
            else:
                r[f"hp_pos_{name}"] = np.nan
                r[f"hp_rot_{name}"] = np.nan
                r[f"hp_rotmag_{name}"] = np.nan
        rows.append(r)

    out = pd.DataFrame(rows)
    out.attrs["rate"] = rate
    out.attrs["hop_s"] = hop / rate
    return out


def build(root: Path = None, fc: float = None) -> pd.DataFrame:
    """The window table for every recording. Jitter columns are in centimetres."""
    frames = []
    for dev, mon, seg, f in segment_files(root):
        df = segment_windows(f, fc)
        if not len(df):
            continue
        df["dev"], df["month"], df["seg"] = dev, mon, seg
        df["rate"], df["hop_s"] = df.attrs["rate"], df.attrs["hop_s"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------- episodes
def failure_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Contiguous runs of windows in which the device reported no tracking.

    Windows with the monitoring panel open are excluded, since the panel occludes
    the view and its windows are not evaluated anywhere in this work.
    """
    rows = []
    for (dev, mon, seg), g in df.groupby(["dev", "month", "seg"]):
        g = g.sort_values("t")
        fail = (g.f_track_hmd <= C.FAIL_FRAC).to_numpy() & (g.ui.to_numpy() < 0.5)
        if not fail.any():
            continue
        hop = float(g.hop_s.iloc[0])
        e = np.flatnonzero(np.diff(fail.astype(int)) != 0)
        a, b = np.r_[0, e + 1], np.r_[e, len(fail) - 1]
        t0seg, t1seg = float(g.t.min()), float(g.t.max())
        for i, j in zip(a, b):
            if not fail[i]:
                continue
            dur = (j - i + 1) * hop
            if dur < C.MIN_EPISODE_S:
                continue
            t0 = float(g.t.iloc[i])
            rows.append(dict(dev=dev, month=mon, seg=seg, t0=t0, dur_s=dur,
                             start_pct=(t0 - t0seg) / (t1seg - t0seg) * 100,
                             end_pct=(t0 + dur - t0seg) / (t1seg - t0seg) * 100))
    return pd.DataFrame(rows)
