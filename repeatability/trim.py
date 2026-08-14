# -----------------------------------------------------------------------------
# Truncating a recording at the point where the wearer left the route.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
#          (supplementary repeatability measurement)
# -----------------------------------------------------------------------------
"""Keeping only the part of a recording that traverses the route.

A recording may continue after the measurement has ended, because the logging
cannot always be stopped at the moment the wearer wants to stop it. Whatever is
recorded after the wearer leaves the route is not a traversal, and including it
would corrupt every quantity built on the two marked points.

The departure is found rather than entered by hand. The route is a corridor: the
wearer stays within a few metres of the line joining the two marks for the whole
measurement, so a sustained excursion beyond that corridor which is never
followed by a return marks the end of the measurement. The corridor half-width
is derived from the recording's own offsets through a median absolute deviation,
so it adapts to how straight the route actually is instead of assuming a width.

The route axis itself is fitted with the excursion excluded, since an excursion
of tens of metres would otherwise rotate the axis it is being measured against.
"""
import numpy as np

from . import config as C


def route_axes(H: np.ndarray, iterations: int = 3):
    """Principal and perpendicular horizontal axes of the route, robustly.

    Returns (centre, along_axis, perpendicular_axis). Samples lying far off the
    corridor are dropped from the fit and the fit repeated, so that a departure
    from the route does not define the direction of the route.
    """
    keep = np.ones(len(H), bool)
    centre = H.mean(axis=0)
    Vt = np.eye(2)
    for _ in range(iterations):
        Hc = H[keep] - H[keep].mean(axis=0)
        _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
        centre = H[keep].mean(axis=0)
        perp = (H - centre) @ Vt[1]
        mad = np.median(np.abs(perp - np.median(perp)))
        width = max(C.CORRIDOR_MIN_M, C.CORRIDOR_MAD_K * 1.4826 * mad)
        new = np.abs(perp - np.median(perp)) <= width
        if new.sum() < 0.5 * len(H) or (new == keep).all():
            break
        keep = new
    return centre, Vt[0], Vt[1]


def route_frame(P: np.ndarray):
    """Express the trajectory in the route's own frame.

    Returns the along-route and across-route coordinates of every sample, and a
    mapping that places any other horizontal point into the same frame. The
    mapping is what lets the origin of the reported coordinate system be drawn
    alongside the trajectory: the origin is a point in the same frame, and where
    it sits relative to the route is not otherwise obvious.
    """
    H = P[:, [0, 2]]
    centre, along_ax, perp_ax = route_axes(H)
    along_m = (H - centre) @ along_ax
    perp_raw = (H - centre) @ perp_ax
    shift = float(np.median(perp_raw))

    def to_frame(xz):
        xz = np.atleast_2d(np.asarray(xz, float))
        return np.column_stack([(xz - centre) @ along_ax,
                                (xz - centre) @ perp_ax - shift])

    return along_m, perp_raw - shift, to_frame


def offset_from_route(P: np.ndarray):
    """Perpendicular distance of every sample from the fitted route line."""
    along_m, perp_m, _ = route_frame(P)
    return along_m, perp_m


def corridor_width(perp_m: np.ndarray) -> float:
    mad = np.median(np.abs(perp_m - np.median(perp_m)))
    return max(C.CORRIDOR_MIN_M, C.CORRIDOR_MAD_K * 1.4826 * mad)


def find_departure(t: np.ndarray, along_m: np.ndarray, perp_m: np.ndarray,
                   width: float):
    """Index at which the wearer leaves the corridor and stops traversing.

    A departure has to last at least MIN_OFFROUTE_S to be one at all. What ends
    the recording is not the departure itself but the absence of any further
    traversal: the measurement consists of walks between the two marks, so if the
    wearer never again covers MIN_LEG_M along the route, nothing after the
    departure is measurement. Stepping around an obstacle in the middle of a run
    therefore truncates nothing, while a walk away from the route at the end does.

    Testing for a further traversal rather than for a return to the corridor also
    covers the case where the reported position itself jumps back into the
    corridor while the wearer is elsewhere, which a return test would accept.
    """
    outside = np.abs(perp_m) > width
    d = np.diff(np.r_[0, outside.astype(int), 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    for i0, i1 in zip(starts, ends):
        i1 = min(i1, len(t) - 1)
        if t[i1] - t[i0] < C.MIN_OFFROUTE_S:
            continue
        rest = ~outside[i1:]
        if rest.sum() < 2:
            return int(i0)
        span = float(np.ptp(along_m[i1:][rest]))
        if span < C.MIN_LEG_M:
            return int(i0)
    return None


def find_frame_resets(P: np.ndarray) -> np.ndarray:
    """Indices at which the reported frame is redefined.

    Recentring the view, and a relocalisation after the tracking has been lost,
    both move the reported position by far more than the wearer moved. Positions
    on either side of such an event are expressed in different frames, so no
    quantity may be computed across one.
    """
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return np.where(step > C.RESET_MIN_M)[0] + 1


def longest_span_without_reset(P: np.ndarray):
    """The longest run of samples containing no reset of the reported frame."""
    cuts = np.r_[0, find_frame_resets(P), len(P)]
    lengths = np.diff(cuts)
    k = int(np.argmax(lengths))
    return int(cuts[k]), int(cuts[k + 1])


def route_mask(dev: str, t: np.ndarray, P: np.ndarray):
    """Boolean mask of the samples that belong to the measurement, plus a report.

    A manual bound in config.TRIM_S overrides the detection for that device, so
    that a decision taken by a person is recorded in the configuration rather
    than applied silently somewhere in a script.
    """
    info = dict(device=dev, corridor_half_width_m=np.nan,
                t_start=float(t[0]), t_end=float(t[-1]),
                resets=int(len(find_frame_resets(P))),
                removed_head_s=0.0, removed_tail_s=0.0,
                removed_samples=0, removed_max_offset_m=0.0,
                source="none")

    manual = C.TRIM_S.get(dev)
    if manual is not None:
        lo = t[0] if manual[0] is None else manual[0]
        hi = t[-1] if manual[1] is None else manual[1]
        mask = (t >= lo) & (t <= hi)
        info["source"] = "config.TRIM_S"
    else:
        # First keep the longest stretch that contains no redefinition of the
        # frame, then, inside it, drop whatever follows the wearer leaving the
        # route. The order matters: the route can only be fitted once every
        # sample is expressed in one frame.
        i0, i1 = longest_span_without_reset(P)
        mask = np.zeros(len(t), bool)
        mask[i0:i1] = True
        along_in, perp_in = offset_from_route(P[i0:i1])
        width = corridor_width(perp_in)
        info["corridor_half_width_m"] = float(width)
        j = find_departure(t[i0:i1], along_in, perp_in, width)
        if j is not None:
            mask[i0 + j:] = False
            info["removed_max_offset_m"] = float(np.abs(perp_in[j:]).max())
        info["source"] = "detected" if (~mask).any() else "none"

    if not mask.any():
        raise ValueError(f"{dev}: nothing left after trimming")
    idx = np.where(mask)[0]
    info["i_start"], info["i_stop"] = int(idx[0]), int(idx[-1]) + 1
    info["t_start"] = float(t[mask][0])
    info["t_end"] = float(t[mask][-1])
    info["removed_head_s"] = float(t[mask][0] - t[0])
    info["removed_tail_s"] = float(t[-1] - t[mask][-1])
    info["removed_samples"] = int((~mask).sum())
    return mask, info
