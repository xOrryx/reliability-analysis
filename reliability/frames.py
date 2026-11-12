# -----------------------------------------------------------------------------
# Controller frame correction, differential jitter and excursions.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Controller poses: coordinate-frame correction, differential jitter, excursions.

The headset pose is read through the Unity transform hierarchy and is expressed
in world coordinates; the controller poses are read through the input interface
and are expressed in tracking-space coordinates. The two frames coincide in the
scene configuration used for the recordings, yet the logs show a constant
rotation about the vertical axis between them, fixed within a recording and
differing between recordings. It is estimated per recording by a Procrustes fit
of the controller trajectory onto the headset trajectory in the horizontal plane
and removed before any joint analysis.

With the frames aligned, the controller can be reported as a difference with
respect to the headset. Body translation cancels in that difference, as does the
component of the estimation noise the two share, so what remains reflects the
controller estimate rather than the walker.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C
from .windows import load_segment, native_rate, highpass_blocks, segment_files

HANDS = {"left": "L", "right": "R"}


# ----------------------------------------------------------------- yaw offset
def procrustes_yaw(src_xz: np.ndarray, dst_xz: np.ndarray):
    """Rotation about the vertical axis that best maps src onto dst.

    Returns the angle in degrees and the median per-sample residual in metres,
    both after centring the two trajectories. The median is used rather than the
    RMS because a handful of recordings contain controller excursions of tens of
    metres, which would dominate a squared measure. Only the rotation is applied
    downstream: a constant translation between the two frames is present in some
    recordings but is removed by the high-pass filter and by referencing the
    excursions to each recording's own baseline.
    """
    a = src_xz - src_xz.mean(axis=0)
    b = dst_xz - dst_xz.mean(axis=0)
    num = float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
    den = float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
    th = np.arctan2(num, den)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    resid = float(np.median(np.linalg.norm(a @ R.T - b, axis=1)))
    return float(np.degrees(th)), resid


def rotate_y(xyz: np.ndarray, deg: float) -> np.ndarray:
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    out = xyz.copy()
    out[:, 0] = c * xyz[:, 0] - s * xyz[:, 2]
    out[:, 2] = s * xyz[:, 0] + c * xyz[:, 2]
    return out


def _valid(a, part):
    s = C.PARTS[part]
    return (a[:, s["track"]] == 1) & np.isfinite(a[:, s["pos"]]).all(axis=1)


def _pos(a, part):
    return a[:, C.PARTS[part]["pos"]]


# ----------------------------------------------------------------- per recording
def recording_frames(path):
    """Yaw offset and fit residual for both hands of one recording."""
    a = load_segment(path)
    out = {}
    okh = _valid(a, "hmd")
    H = _pos(a, "hmd")
    for hand in ("left", "right"):
        m = okh & _valid(a, hand)
        if m.sum() < 100:
            out[hand] = (np.nan, np.nan, np.nan, np.nan)
            continue
        Cq = _pos(a, hand)
        yaw, resid = procrustes_yaw(Cq[m][:, [0, 2]], H[m][:, [0, 2]])
        # The vertical offset is a real geometric quantity: how far the headset
        # sits above the controller. The horizontal separation is taken from the
        # fit residual, since a constant horizontal translation between the two
        # reported frames is present in some recordings and is not part of the
        # body geometry. The 3-D separation combines the two.
        Cr = rotate_y(Cq, yaw)
        d = Cr[m] - H[m]
        vert = float(np.median(np.abs(d[:, 1])))
        dist = float(np.hypot(resid, vert))
        out[hand] = (yaw, resid, vert, dist)
    return out


def controller_relative(path, fc: float = None, piecewise: bool = True) -> pd.DataFrame:
    """Windowed differential jitter of each controller with respect to the head.

    The differential signal is filtered and reduced exactly like a pose: the
    high-pass runs on each contiguous block in which both the headset and that
    controller were tracked, the transient is trimmed, and the window value is
    the RMS over the window. Units are centimetres.

    piecewise=True aligns the controller frame separately in every piece
    between re-anchorings of the headset world (excursions.aligned_controller),
    as described in Section 2.4.1 of the paper. piecewise=False uses one yaw per
    recording (the first version of the analysis).
    """
    from .excursions import aligned_controller
    a = load_segment(path)
    t = a[:, C.COL_TIME]
    rate = native_rate(t)
    dt = 1.0 / rate
    w = int(round(rate * C.WIN_S))
    hop = max(1, int(round(w * C.OVERLAP)))
    if len(t) < 3 * w:
        return pd.DataFrame()

    tt = np.arange(t[0], t[-1], dt)
    ui = np.interp(tt, t, (a[:, C.COL_UI] != 0).astype(float)) > 0.5
    H = np.column_stack([np.interp(tt, t, _pos(a, "hmd")[:, k]) for k in range(3)])
    okh = np.interp(tt, t, _valid(a, "hmd").astype(float)) > 0.999
    d = np.linalg.norm(np.diff(H, axis=0), axis=1) * rate
    speed = np.r_[d[0], d]

    yaws = None if piecewise else recording_frames(path)
    cols = {"t": tt, "speed": speed, "ui": ui.astype(float)}
    for hand in ("left", "right"):
        if piecewise:
            Pr = aligned_controller(a, hand)
            fin = np.isfinite(Pr).all(axis=1)
            Pr = np.where(fin[:, None], Pr, 0.0)
            P = np.column_stack([np.interp(tt, t, Pr[:, k]) for k in range(3)])
            ok = (np.interp(tt, t, (_valid(a, hand) & fin).astype(float)) > 0.999) & okh
            cols[f"_d_{hand}"] = P - H
            cols[f"_ok_{hand}"] = ok
            continue
        yaw = yaws[hand][0]
        if not np.isfinite(yaw):
            cols[f"_d_{hand}"] = np.full((len(tt), 3), np.nan)
            cols[f"_ok_{hand}"] = np.zeros(len(tt), bool)
            continue
        P = np.column_stack([np.interp(tt, t, _pos(a, hand)[:, k]) for k in range(3)])
        ok = (np.interp(tt, t, _valid(a, hand).astype(float)) > 0.999) & okh
        cols[f"_d_{hand}"] = rotate_y(P, yaw) - H
        cols[f"_ok_{hand}"] = ok

    for hand in ("left", "right"):
        cols[f"_hp_{hand}"] = highpass_blocks(cols[f"_d_{hand}"], cols[f"_ok_{hand}"],
                                              rate, fc)

    rows = []
    for s0 in np.arange(0, max(0, len(tt) - w), hop, dtype=int):
        i = slice(s0, s0 + w)
        r = {"t": float(tt[i].mean()), "speed": float(speed[i].mean()),
             "ui": float(ui[i].mean())}
        for hand in ("left", "right"):
            p = cols[f"_hp_{hand}"][i]
            good = np.isfinite(p).all(axis=1)
            r[f"rel_{hand}"] = (100.0 * float(np.sqrt(np.nanmean(np.sum(p[good] ** 2, axis=1))))
                                if good.mean() >= C.MIN_VALID else np.nan)
        rows.append(r)
    out = pd.DataFrame(rows)
    out.attrs["rate"] = rate
    out.attrs["hop_s"] = hop / rate
    return out


# ----------------------------------------------------------------- excursions
def controller_excursions(path):
    """Single-sample discontinuities of the controller estimate, merged into events.

    A candidate is a sample at which the controller position moves by more than
    JUMP_MIN_M while the headset moves by less than HEAD_MAX_M. Candidates are
    merged into one event while the controller-to-headset offset stays further
    than RECOVER_TOL_M from that recording's median offset; the event ends when
    the offset returns inside that tolerance. Reported per event are the size of
    the triggering jump, the largest departure of the offset from the baseline,
    the time taken to return, and the fraction of samples the device reported the
    controller as tracked during the event.
    """
    a = load_segment(path)
    t = a[:, C.COL_TIME]
    H = _pos(a, "hmd")
    okh = _valid(a, "hmd")
    jumps, events = [], []
    for hand in ("left", "right"):
        okc = _valid(a, hand)
        m = okh & okc
        if m.sum() < 100:
            continue
        P = rotate_y(_pos(a, hand), recording_frames(path)[hand][0])
        off = np.full(len(t), np.nan)
        off[m] = np.linalg.norm(P[m] - H[m], axis=1)
        base = float(np.nanmedian(off))

        dP = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1)]
        dH = np.r_[0.0, np.linalg.norm(np.diff(H, axis=0), axis=1)]
        dt = np.r_[np.median(np.diff(t)), np.diff(t)]
        cand = np.flatnonzero(m & (dP > C.JUMP_MIN_M) & (dH < C.HEAD_MAX_M))
        for i in cand:
            jumps.append(dict(hand=HANDS[hand], t=float(t[i]), jump_m=float(dP[i]),
                              dt_s=float(dt[i]), speed_implied=float(dP[i] / dt[i])))

        i = 0
        while i < len(cand):
            k = cand[i]
            j = k
            end = min(len(t) - 1, np.searchsorted(t, t[k] + C.MAX_RECOVER_S))
            while j < end and (not np.isfinite(off[j]) or
                               abs(off[j] - base) > C.RECOVER_TOL_M):
                j += 1
            seg = slice(k, j + 1)
            dev_off = np.abs(off[seg] - base)
            exc = float(np.nanmax(dev_off))
            if exc >= C.MIN_EXCURSION_M:
                events.append(dict(hand=HANDS[hand], t=float(t[k]),
                                   jump_m=float(dP[k]), excursion_m=exc,
                                   recover_s=float(t[j] - t[k]),
                                   tracked_during=100.0 * float(np.mean(okc[seg]))))
            i = int(np.searchsorted(cand, j, side="right"))
    return pd.DataFrame(jumps), pd.DataFrame(events)


# ----------------------------------------------------------------- drivers
def build_relative(root=None, fc: float = None) -> pd.DataFrame:
    frames = []
    for dev, mon, seg, f in segment_files(root):
        df = controller_relative(f, fc)
        if not len(df):
            continue
        df["dev"], df["month"], df["seg"] = dev, mon, seg
        df["hop_s"] = df.attrs["hop_s"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_frames(root=None) -> pd.DataFrame:
    rows = []
    for dev, mon, seg, f in segment_files(root):
        for hand, (yaw, resid, vert, dist) in recording_frames(f).items():
            rows.append(dict(dev=dev, month=mon, seg=seg, hand=HANDS[hand],
                             yaw_deg=yaw, resid_horizontal_m=resid,
                             vertical_m=vert, distance_3d_m=dist))
    return pd.DataFrame(rows)


def build_excursions(root=None):
    J, E = [], []
    for dev, mon, seg, f in segment_files(root):
        j, e = controller_excursions(f)
        for df, acc in ((j, J), (e, E)):
            if len(df):
                df["dev"], df["month"], df["seg"] = dev, mon, seg
                acc.append(df)
    return (pd.concat(J, ignore_index=True) if J else pd.DataFrame(),
            pd.concat(E, ignore_index=True) if E else pd.DataFrame())
