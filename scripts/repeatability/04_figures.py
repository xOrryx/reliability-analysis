# -----------------------------------------------------------------------------
# Stage 4: figures. Print sizes, greyscale-legible, no colour carries meaning.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
#          (supplementary repeatability measurement)
#
# Reads   out/visits.csv, out/deviations.csv, out/jumps.csv
# Writes  out/figures/fig_repeat_drift.pdf
#         out/figures/fig_repeat_route.pdf
#         out/figures/fig_repeat_origin.pdf
#         out/figures/fig_repeat_trim.pdf
# -----------------------------------------------------------------------------
"""Figures for the repeatability measurement."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from repeatability import config as C
from repeatability.io import device_logs, load_device, unpack
from repeatability.passes import along_route
from repeatability.trim import corridor_width, offset_from_route, route_frame

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "pdf.fonttype": 42,
})

MARK = {"MQ2": ("o", "-"), "MQ3": ("s", "--"), "MQPro": ("^", ":")}
GREY = {"MQ2": "0.15", "MQ3": "0.40", "MQPro": "0.62"}


def fig_drift(DEV: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharex=True)
    for ax, comp, lab in ((axes[0], "horizontal_m", "Horizontal deviation [m]"),
                          (axes[1], "vertical_m", "Vertical deviation [m]")):
        for (dev, end), g in DEV.groupby(["device", "end"], sort=False):
            g = g.sort_values("path_since_ref_m")
            m, ls = MARK.get(dev, ("o", "-"))
            ax.plot(g.path_since_ref_m / 1000.0, g[comp], ls, marker=m, ms=3.2,
                    color=GREY.get(dev, "0.3"),
                    markerfacecolor="white" if end == "B" else GREY.get(dev, "0.3"),
                    label=f"{C.DEVICE_LABEL.get(dev, dev)}, mark {end}")
        ax.set_xlabel("Distance walked [km]")
        ax.set_ylabel(lab)
        ax.grid(True, color="0.9", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_repeat_drift.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_route(V: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    for dev, path in device_logs():
        t, P, ok, ui = unpack(load_device(dev, path)[0])
        s = along_route(P)
        ax.plot((t - t[0]) / 60.0, s, "-", lw=0.7,
                color=GREY.get(dev, "0.3"), label=C.DEVICE_LABEL.get(dev, dev))
        g = V[V.device == dev]
        if len(g):
            ax.plot((g.t_s - t[0]) / 60.0, g.along_m, "o", ms=3.0,
                    color="black", markerfacecolor="white", zorder=3)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Along-route position [m]")
    ax.grid(True, color="0.9", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fig_repeat_route.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_origin(out: Path):
    """Where the origin of the reported frame sits, for every device.

    The origin is wherever the device placed itself when the recording began.
    Left: the trajectory in the route's own frame, with the start of the
    recording and the origin marked. Right: the distance of the reported
    position from that origin against time, one full cycle per traversal.

    Read the right panel for its shape, not for its detail: on an axis that
    spans the whole route, a return that misses the origin by a few metres is
    invisible. That miss is the drift, and it is what fig_repeat_drift shows.
    """
    devs = list(device_logs())
    if not devs:
        return
    zoom = 15.0   # metres, the range in which the returns can be told apart
    fig, axes = plt.subplots(len(devs), 2, figsize=(7.6, 2.5 * len(devs)),
                             squeeze=False)
    for row, (dev, path) in enumerate(devs):
        kept, info, _ = load_device(dev, path)
        t = kept[:, C.COL_TIME]
        P = kept[:, C.PARTS["hmd"]["pos"]]
        tm = (t - t[0]) / 60.0
        r = np.hypot(P[:, 0], P[:, 2])
        label = C.DEVICE_LABEL.get(dev, dev)

        for col, (lo, hi, title) in enumerate((
                (None, None, "full range"),
                (-1.0, zoom, f"first {zoom:.0f} m, where the returns separate"))):
            ax = axes[row][col]
            ax.plot(tm, r, "-", lw=0.6, color="0.25")
            ax.axhline(0.0, color="0.55", lw=0.8, ls=":")
            ax.plot(tm[0], r[0], "s", ms=5, color="black",
                    markerfacecolor="white", zorder=5,
                    label="start of the recording")
            if lo is not None:
                ax.set_ylim(lo, hi)
            ax.set_xlabel("Time [min]")
            ax.set_ylabel("Distance from the reported origin [m]")
            ax.set_title(f"{label}: {title}")
            ax.grid(True, color="0.93", lw=0.5)
            ax.set_axisbelow(True)
            if row == 0 and col == 0:
                ax.legend(frameon=False, loc="upper right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "fig_repeat_origin.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_trim(out: Path):
    """What was removed from each recording, and why.

    Left: the horizontal trajectory in the route's own frame, with the discarded
    tail drawn separately, and with both the start of the recording and the
    origin of the reported coordinate system marked, so that it is visible where
    the frame the device reports against actually sits.
    Middle: distance from the fitted route line against time, with the corridor
    the detector uses and the point at which the recording is cut.
    Right: distance of the reported position from that origin against time. The
    origin is where the device placed itself when the recording began, so this
    panel says when the reported pose was back at its own starting point — which
    is not the same question as whether the wearer was.
    """
    devs = [(dev, path) for dev, path in device_logs()]
    devs = [(d, p) for d, p in devs
            if load_device(d, p)[1]["removed_samples"] > 0]
    if not devs:
        return
    fig, axes = plt.subplots(len(devs), 3, figsize=(9.4, 2.9 * len(devs)),
                             squeeze=False)
    for row, (dev, path) in enumerate(devs):
        kept, info, raw = load_device(dev, path)
        t = raw[:, C.COL_TIME]
        P = raw[:, C.PARTS["hmd"]["pos"]]
        i0, i1 = int(info["i_start"]), int(info["i_stop"])
        # The route frame is fitted on the kept samples only. The removed parts
        # are then mapped into that same frame: they lie outside the measurement
        # and must not be allowed to define the axis they are judged against.
        _, perp_kept, to_frame = route_frame(P[i0:i1])
        w = corridor_width(perp_kept)
        F = to_frame(P[:, [0, 2]])
        along, perp = F[:, 0], F[:, 1]
        o_along, o_perp = to_frame([0.0, 0.0])[0]
        tm = (t - t[0]) / 60.0

        # Drawn in the route's own frame, so the route runs horizontally and the
        # departure from it is visible at true scale rather than as a thin sliver.
        ax = axes[row][0]
        ax.plot(along[i0:i1], perp[i0:i1], "-", lw=0.6, color="0.25",
                label=f"kept ({t[i1-1]-t[i0]:.0f} s)")
        for j0, j1 in ((0, i0 + 1), (i1 - 1, len(t))):
            if j1 - j0 > 1:
                ax.plot(along[j0:j1], perp[j0:j1], "-", lw=1.4, color="0.62")
        ax.plot([], [], "-", lw=1.4, color="0.62",
                label=f"removed ({info['removed_head_s']+info['removed_tail_s']:.0f} s)")
        ax.plot(o_along, o_perp, "+", ms=9, mew=1.4, color="black", zorder=5,
                label="reported origin")
        ax.plot([along[i0], along[i1-1]], [perp[i0], perp[i1-1]], "o", ms=4,
                color="black", markerfacecolor="white", zorder=5, ls="none",
                label="kept span")
        ax.set_xlabel("Along the route [m]")
        ax.set_ylabel("Across the route [m]")
        ax.set_title(f"{C.DEVICE_LABEL.get(dev, dev)}: trajectory, top view")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False, loc="upper right", fontsize=6.0,
                  handlelength=1.6, borderaxespad=0.2)

        ax = axes[row][1]
        ax.axhspan(-w, w, color="0.90", zorder=0)
        ax.plot(tm[i0:i1], perp[i0:i1], "-", lw=0.6, color="0.25")
        for j0, j1 in ((0, i0 + 1), (i1 - 1, len(t))):
            if j1 - j0 > 1:
                ax.plot(tm[j0:j1], perp[j0:j1], "-", lw=1.4, color="0.62")
        for x in (tm[i0], tm[i1 - 1]):
            ax.axvline(x, color="black", lw=0.9, ls="--")
        ax.annotate(f"corridor $\\pm${w:.0f} m", xy=(0.02, 0.94),
                    xycoords="axes fraction", fontsize=6.5, color="0.35")
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Distance from the route line [m]")
        ax.set_title("Departure from the route")
        ax.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)

        ax = axes[row][2]
        r = np.hypot(P[:, 0], P[:, 2])
        ax.plot(tm[i0:i1], r[i0:i1], "-", lw=0.6, color="0.25")
        for j0, j1 in ((0, i0 + 1), (i1 - 1, len(t))):
            if j1 - j0 > 1:
                ax.plot(tm[j0:j1], r[j0:j1], "-", lw=1.4, color="0.62")
        for x in (tm[i0], tm[i1 - 1]):
            ax.axvline(x, color="black", lw=0.9, ls="--")
        ax.axhline(0.0, color="0.55", lw=0.8, ls=":")
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Distance from the reported origin [m]")
        ax.set_title("Where the origin is in time")
        ax.grid(True, color="0.93", lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "fig_repeat_trim.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    out = C.OUT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    DEV = pd.read_csv(C.OUT / "deviations.csv")
    V = pd.read_csv(C.OUT / "visits.csv")
    fig_drift(DEV, out)
    fig_route(V, out)
    fig_origin(out)
    fig_trim(out)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
