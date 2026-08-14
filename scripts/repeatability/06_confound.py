# -----------------------------------------------------------------------------
# Stage 6: can the device comparison be separated from how the headset sat?
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
#          (supplementary repeatability measurement)
#
# Reads   data/measHMD_<DEV>.csv, out/visits.csv, out/increments.csv
# Writes  out/roundtrips.csv       one row per round trip: drift and head pitch
#         out/tab_overlap.csv      the pitch range each device covers
#         out/figures/fig_repeat_common_frame.pdf
# -----------------------------------------------------------------------------
"""Where each device pointed its cameras, and whether that can be controlled for.

The elevation of the headset's forward axis differs between the devices even
though the wearer carried all three the same way, because each model sits on the
head at its own angle. Where the cameras point decides which features the
tracking solution is built on, so this is not a nuisance to be ignored.

It also cannot be removed after the fact. Rotating a finished trajectory changes
the coordinates the answer is written in, not the imagery the answer was computed
from; the drift has already happened, and the near-field parallax the estimator
did or did not receive cannot be conjured by a coordinate transform. Redoing it
properly would need the raw camera streams and the tracking stack, and the
platform exposes neither.

What can be tested is whether the devices can be compared at matched pitch. That
requires their pitch ranges to overlap, and this stage reports whether they do
rather than assuming it. It also draws all three recordings in one common frame,
which is a fair thing to do: the route is the same physical route, so expressing
every deviation in along-route and across-route components makes them
commensurable without changing any of them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from repeatability import config as C
from repeatability.io import device_logs, head_pitch_deg, load_device
from repeatability.trim import route_frame

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "pdf.fonttype": 42,
})
MARK = {"MQ2": ("o", "-"), "MQ3": ("s", "--"), "MQPro": ("^", ":")}
GREY = {"MQ2": "0.15", "MQ3": "0.40", "MQPro": "0.62"}


def round_trips(V: pd.DataFrame, I: pd.DataFrame) -> pd.DataFrame:
    """One row per round trip: how far the estimate moved, and where it looked."""
    rows = []
    for dev, path in device_logs():
        a = load_device(dev, path)[0]
        t = a[:, C.COL_TIME]
        pitch = head_pitch_deg(a)
        vv = V[V.device == dev].sort_values("t_s")
        for _, r in I[I.device == dev].iterrows():
            prev = vv[(vv.end == r["end"]) & (vv.t_s < r["t_s"])].t_s.max()
            m = (t >= prev) & (t < r["t_s"])
            if m.sum() < 100:
                continue
            rows.append(dict(device=dev, end=r["end"], t_s=float(r["t_s"]),
                             path_m=float(r["path_m"]), drift_m=float(r["total_m"]),
                             pitch_deg=float(np.median(pitch[m]))))
    return pd.DataFrame(rows)


def overlap(R: pd.DataFrame) -> pd.DataFrame:
    """The pitch range each device covers, and whether any range is shared."""
    rows = []
    for dev, g in R.groupby("device", sort=False):
        rows.append(dict(device=dev, n=len(g), pitch_min=g.pitch_deg.min(),
                         pitch_median=g.pitch_deg.median(), pitch_max=g.pitch_deg.max()))
    T = pd.DataFrame(rows)
    T["common_lo"] = T.pitch_min.max()
    T["common_hi"] = T.pitch_max.min()
    T["overlaps"] = T.common_lo < T.common_hi
    return T


def fig_common_frame(V: pd.DataFrame, out: Path):
    """All three recordings in one frame: where each thought the marks were.

    Every deviation is resolved along and across the route, which is the same
    physical route for all three, so the three clouds are directly comparable.
    Points are joined in the order they were visited, so the path each estimate
    took away from its own starting value is visible rather than only its spread.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
    for ax, end in zip(axes, ("A", "B")):
        for dev, path in device_logs():
            g = V[(V.device == dev) & (V.end == end)].sort_values("t_s")
            if len(g) < 2:
                continue
            P = np.column_stack([g.X, g.Y, g.Z])
            a = load_device(dev, path)[0]
            _, _, to_frame = route_frame(a[:, C.PARTS["hmd"]["pos"]])
            F = to_frame(P[:, [0, 2]])
            F = F - F[0]
            m, ls = MARK.get(dev, ("o", "-"))
            ax.plot(F[:, 0], F[:, 1], ls, marker=m, ms=3.4, lw=0.9,
                    color=GREY.get(dev, "0.3"), markerfacecolor="white",
                    label=C.DEVICE_LABEL.get(dev, dev))
            ax.plot(0, 0, m, ms=5, color=GREY.get(dev, "0.3"), zorder=5)
        ax.axhline(0, color="0.75", lw=0.6, ls=":")
        ax.axvline(0, color="0.75", lw=0.6, ls=":")
        ax.set_xlabel("Deviation along the route [m]")
        ax.set_ylabel("Deviation across the route [m]")
        ax.set_title(f"Reported position of mark {end}, relative to the first visit")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, color="0.93", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out / "fig_repeat_common_frame.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    V = pd.read_csv(C.OUT / "visits.csv")
    I = pd.read_csv(C.OUT / "increments.csv")
    R = round_trips(V, I)
    T = overlap(R)
    R.to_csv(C.OUT / "roundtrips.csv", index=False)
    T.to_csv(C.OUT / "tab_overlap.csv", index=False)
    fig_common_frame(V, C.OUT / "figures")

    print("--- head pitch covered by each device, one value per round trip")
    print(T.round(2).to_string(index=False))
    lo, hi = float(T.common_lo.iloc[0]), float(T.common_hi.iloc[0])
    print()
    if lo < hi:
        n = int(((R.pitch_deg >= lo) & (R.pitch_deg <= hi)).sum())
        print(f"The devices share the pitch band {lo:.1f} to {hi:.1f} deg, in which "
              f"{n} round trips fall. A comparison restricted to that band separates "
              f"the device from the pitch.")
    else:
        print("The pitch ranges of the devices do not overlap at all. Device and head "
              "pitch are therefore completely confounded in this data set: there is no "
              "round trip in which two devices were carried at a comparable angle, so "
              "no adjustment, matching or regression can separate them. Any model that "
              "appeared to do so would be extrapolating into a region where nothing was "
              "measured.")
    print("\nRanges, for the record:")
    for _, r in T.iterrows():
        print(f"  {C.DEVICE_LABEL.get(r.device, r.device):10s} "
              f"{r.pitch_min:+6.1f} to {r.pitch_max:+6.1f} deg  (n = {int(r.n)})")


if __name__ == "__main__":
    main()
