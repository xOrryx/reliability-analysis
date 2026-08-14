# -----------------------------------------------------------------------------
# Stage 2: repeatability and drift of the reported pose across repeated visits.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
#          (supplementary repeatability measurement)
#
# Reads   out/visits.csv
# Writes  out/deviations.csv  deviation of each visit from the first
#         out/increments.csv  change between consecutive visits to a mark
#         out/legs.csv        reported distance between the marks, per leg
#         out/tab_drift.csv   summary, one row per device and mark
#         out/tex/tab_drift.tex
# -----------------------------------------------------------------------------
"""Repeatability and drift across repeated visits to the same marked point."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from repeatability import config as C, drift


def tex_table(S: pd.DataFrame, R: pd.DataFrame) -> str:
    """The table as it appears in the manuscript, generated, not transcribed.

    The head pitch is carried in this table rather than in the text alone,
    because it differs between the devices, is not under the wearer's control,
    and is what stops the device comparison from being read on its own.
    """
    order = {d: i for i, d in enumerate(C.DEVICES)}
    S = S.assign(_o=S.device.map(order)).sort_values(["_o", "end"])
    pitch = R.set_index("device").pitch_median_deg.to_dict()

    L = [r"\begin{table}[h!]",
         r"\caption{Repeatability of the reported headset position over a route "
         r"walked repeatedly out and back, one row per device and marked point. "
         r"The deviation is measured against the first visit to that mark and "
         r"accumulates; the increment is the change between consecutive visits, "
         r"with the largest one in brackets, and is the quantity a repeatability "
         r"figure may be quoted from, since consecutive increments are "
         r"approximately independent. The drift is the growth of the deviation "
         r"with the distance walked. The last column is the measured elevation of "
         r"the headset's forward axis, negative downwards, recorded because it "
         r"differs between the devices and is not under the wearer's "
         r"control.}\label{tab:repeatability}",
         r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"Device & Mark & Visits & Path & Deviation & Increment & Drift & "
         r"Elevation \\",
         r" &  &  & [km] & [m] & [m] & [\% of path] & [deg] \\", r"\midrule"]
    prev = None
    for _, r in S.iterrows():
        if prev is not None and r.device != prev:
            L.append(r"\addlinespace")
        prev = r.device
        L.append(
            f"{C.DEVICE_LABEL.get(r.device, r.device)} & {r.end} & "
            f"{int(r.visits)} & {r.path_total_m/1000:.2f} & "
            f"{r.final_total_m:.2f} & "
            f"{r.incr_median_m:.2f} ({r.incr_max_m:.2f}) & "
            f"{r.pct_of_distance:.2f} & ${pitch.get(r.device, float('nan')):+.1f}$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    V = pd.read_csv(C.OUT / "visits.csv")
    DEV = drift.deviations(V)
    INC = drift.increments(V)
    LEG = drift.leg_distances(V)
    FR = drift.frame_transform(V)
    S = drift.summary(V, DEV, INC)

    (C.OUT / "tex").mkdir(parents=True, exist_ok=True)
    DEV.to_csv(C.OUT / "deviations.csv", index=False)
    INC.to_csv(C.OUT / "increments.csv", index=False)
    LEG.to_csv(C.OUT / "legs.csv", index=False)
    FR.to_csv(C.OUT / "frame_transform.csv", index=False)
    S.to_csv(C.OUT / "tab_drift.csv", index=False)
    R = pd.read_csv(C.OUT / "recordings.csv")
    (C.OUT / "tex" / "tab_drift.tex").write_text(tex_table(S, R), encoding="utf-8")

    print("--- deviation from the first visit to the same mark")
    for (dev, end), g in DEV.groupby(["device", "end"], sort=False):
        print(f"\n{dev} mark {end}")
        print(g[["visit", "path_since_ref_m", "dX", "dY", "dZ",
                 "horizontal_m", "vertical_m", "total_m"]]
              .round(2).to_string(index=False))

    print("\n--- summary")
    print(S.round(3).to_string(index=False))

    print("\n--- increments between consecutive visits")
    for (dev, end), g in INC.groupby(["device", "end"], sort=False):
        q = g["total_m"]
        print(f"{dev} mark {end}: n={len(q)}  median {q.median():.2f} m  "
              f"sd {q.std(ddof=1):.2f} m  max {q.max():.2f} m")

    print("\n--- reported distance between the marks")
    if LEG.empty:
        print("  no complete leg found")
    else:
        for dev, g in LEG.groupby("device", sort=False):
            q = g["straight_m"]
            line = (f"{dev}: n={len(q)}  median {q.median():.2f} m  "
                    f"sd {q.std(ddof=1):.2f} m  range {q.min():.2f}-{q.max():.2f} m")
            if C.TAPE_DISTANCE_M:
                line += (f"  |  tape {C.TAPE_DISTANCE_M:.2f} m, "
                         f"median error {g['error_m'].median():+.2f} m "
                         f"({g['error_pct'].median():+.2f} %)")
            print(line)
        if not C.TAPE_DISTANCE_M:
            print("  TAPE_DISTANCE_M is not set, so no scale error is reported.")

    print("\n--- drift decomposed into a rigid motion of the reported frame")
    if FR.empty:
        print("  too few traversals")
    else:
        print(FR.round(3).to_string(index=False))
        for dev, g in FR.groupby("device", sort=False):
            last = g.iloc[-1]
            print(f"\n{dev}: over {last.path_m/1000:.2f} km the frame translated "
                  f"{last.translation_m:.2f} m, rotated {last.yaw_deg:+.2f} deg about "
                  f"the vertical and moved {last.vertical_m:+.2f} m in height, while "
                  f"its scale changed by {last.scale_error_pct:+.2f} %.")


if __name__ == "__main__":
    main()
