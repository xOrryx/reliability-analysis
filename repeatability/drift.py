# -----------------------------------------------------------------------------
# Repeatability and drift of the reported pose across repeated visits to a mark.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
#          (supplementary repeatability measurement)
# -----------------------------------------------------------------------------
"""Three quantities, deliberately kept apart.

1. Deviation of each visit from the first visit to the same mark, against the
   path length walked in between. These values are not independent: the wearer
   returns to the same physical point, so a monotone growth of the deviation is
   accumulated drift of the reported frame, not scatter about a fixed value.
   A standard deviation over the visits is only a meaningful summary once this
   growth has been shown to be absent.

2. Increment between consecutive visits to the same mark. Consecutive increments
   are approximately independent, so this is the quantity from which a
   repeatability figure may be quoted, and n is the number of round trips.

3. Reported distance between the two marks, one value per leg. A slow drift of
   the whole frame moves both ends together and cancels here, so this isolates
   the local geometry, and against a tape measurement it gives a scale error.
"""
import numpy as np
import pandas as pd

from . import config as C


def deviations(V: pd.DataFrame) -> pd.DataFrame:
    """Deviation of every visit from the first visit to the same mark."""
    out = []
    for (dev, end), g in V.groupby(["device", "end"], sort=False):
        g = g.sort_values("t_s")
        ref = g.iloc[0][["X", "Y", "Z"]].to_numpy(float)
        ref_path = float(g.iloc[0]["path_m"])
        for _, r in g.iterrows():
            d = r[["X", "Y", "Z"]].to_numpy(float) - ref
            out.append(dict(
                device=dev, end=end, visit=int(r["visit"]), t_s=r["t_s"],
                path_m=r["path_m"],
                # Distance walked since the visit the deviation is measured from.
                # The drift is expected to grow with this, not with the distance
                # walked since the recording began, and the two differ whenever
                # the reference visit is not the first sample of the recording.
                path_since_ref_m=float(r["path_m"]) - ref_path,
                source=r["source"],
                dX=d[0], dY=d[1], dZ=d[2],
                horizontal_m=float(np.hypot(d[0], d[2])),
                vertical_m=float(abs(d[1])),
                total_m=float(np.linalg.norm(d)),
            ))
    return pd.DataFrame(out)


def increments(V: pd.DataFrame) -> pd.DataFrame:
    """Change of the reported position between consecutive visits to a mark."""
    out = []
    for (dev, end), g in V.groupby(["device", "end"], sort=False):
        g = g.sort_values("t_s")
        Q = g[["X", "Y", "Z"]].to_numpy(float)
        for k in range(1, len(g)):
            d = Q[k] - Q[k - 1]
            out.append(dict(
                device=dev, end=end, step=k, t_s=float(g.iloc[k]["t_s"]),
                path_m=float(g.iloc[k]["path_m"] - g.iloc[k - 1]["path_m"]),
                dX=d[0], dY=d[1], dZ=d[2],
                horizontal_m=float(np.hypot(d[0], d[2])),
                vertical_m=float(abs(d[1])),
                total_m=float(np.linalg.norm(d)),
            ))
    return pd.DataFrame(out)


def leg_distances(V: pd.DataFrame) -> pd.DataFrame:
    """Reported straight-line distance between the two marks, one row per leg."""
    out = []
    for dev, g in V.groupby("device", sort=False):
        g = g.sort_values("t_s").reset_index(drop=True)
        for k in range(1, len(g)):
            a, b = g.iloc[k - 1], g.iloc[k]
            if a["end"] == b["end"]:
                continue
            d = b[["X", "Y", "Z"]].to_numpy(float) - a[["X", "Y", "Z"]].to_numpy(float)
            out.append(dict(
                device=dev, leg=k, t_s=float(b["t_s"]),
                direction=f'{a["end"]}->{b["end"]}',
                path_m=float(b["path_m"] - a["path_m"]),
                straight_m=float(np.linalg.norm(d)),
                horizontal_m=float(np.hypot(d[0], d[2])),
            ))
    D = pd.DataFrame(out)
    if not D.empty and C.TAPE_DISTANCE_M:
        D["error_m"] = D["straight_m"] - C.TAPE_DISTANCE_M
        D["error_pct"] = 100.0 * D["error_m"] / C.TAPE_DISTANCE_M
    return D


def drift_rate(dev_dev: pd.DataFrame) -> dict:
    """Least-squares growth of the total deviation with path length.

    Reported as metres per kilometre and as a percentage of distance travelled.
    The fit is forced through the origin, since the deviation of the first visit
    from itself is zero by construction.
    """
    x = dev_dev["path_since_ref_m"].to_numpy(float)
    y = dev_dev["total_m"].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or not np.any(x > 0):
        return dict(slope_m_per_km=np.nan, pct_of_distance=np.nan, n=int(len(x)))
    slope = float((x @ y) / (x @ x))
    return dict(slope_m_per_km=1000.0 * slope, pct_of_distance=100.0 * slope, n=int(len(x)))


def frame_transform(V: pd.DataFrame) -> pd.DataFrame:
    """Decompose the drift into a rigid motion of the reported frame.

    Each traversal supplies the reported position of both marks. Fitting the
    horizontal similarity transform that carries the first traversal's pair onto
    a later one separates three things that a single deviation figure confounds:
    a translation of the frame, a rotation of it about the vertical, and a change
    of scale. Two point correspondences determine the four parameters exactly.

    A frame that translates and rotates while keeping its scale leaves every
    distance measured inside it intact; only positions expressed in it move.
    """
    out = []
    for dev, g in V.groupby("device", sort=False):
        g = g.sort_values("t_s").reset_index(drop=True)
        pairs = []
        for k in range(1, len(g)):
            a, b = g.iloc[k - 1], g.iloc[k]
            if a["end"] == b["end"]:
                continue
            A, B = (a, b) if a["end"] == "A" else (b, a)
            pairs.append(dict(
                t_s=float(b["t_s"]), path_m=float(b["path_m"]),
                A=np.array([A["X"], A["Z"]], float), B=np.array([B["X"], B["Z"]], float),
                Ay=float(A["Y"]), By=float(B["Y"])))
        if len(pairs) < 2:
            continue
        p0 = pairs[0]
        v0 = p0["B"] - p0["A"]
        c0 = 0.5 * (p0["A"] + p0["B"])
        for k, p in enumerate(pairs):
            v = p["B"] - p["A"]
            c = 0.5 * (p["A"] + p["B"])
            scale = float(np.linalg.norm(v) / np.linalg.norm(v0))
            yaw = float(np.degrees(np.arctan2(v[1], v[0]) - np.arctan2(v0[1], v0[0])))
            yaw = (yaw + 180.0) % 360.0 - 180.0
            shift = c - c0
            out.append(dict(
                device=dev, traversal=k + 1, t_s=p["t_s"], path_m=p["path_m"],
                translation_m=float(np.linalg.norm(shift)),
                shift_X=float(shift[0]), shift_Z=float(shift[1]),
                yaw_deg=yaw, scale=scale, scale_error_pct=100.0 * (scale - 1.0),
                vertical_m=float(0.5 * (p["Ay"] + p["By"]) - 0.5 * (p0["Ay"] + p0["By"])),
            ))
    return pd.DataFrame(out)


def summary(V: pd.DataFrame, DEV: pd.DataFrame, INC: pd.DataFrame) -> pd.DataFrame:
    """One row per device and mark."""
    out = []
    for (dev, end), g in DEV.groupby(["device", "end"], sort=False):
        g = g.sort_values("visit")
        inc = INC[(INC.device == dev) & (INC.end == end)]
        last = g.iloc[-1]
        out.append(dict(
            device=dev, end=end, visits=int(len(g)),
            path_total_m=float(g["path_since_ref_m"].max()),
            final_horizontal_m=float(last["horizontal_m"]),
            final_vertical_m=float(last["vertical_m"]),
            final_total_m=float(last["total_m"]),
            max_total_m=float(g["total_m"].max()),
            incr_median_m=float(inc["total_m"].median()) if len(inc) else np.nan,
            incr_sd_m=float(inc["total_m"].std(ddof=1)) if len(inc) > 1 else np.nan,
            incr_max_m=float(inc["total_m"].max()) if len(inc) else np.nan,
            **drift_rate(g),
        ))
    return pd.DataFrame(out)
