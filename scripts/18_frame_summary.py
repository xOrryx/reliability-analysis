# -----------------------------------------------------------------------------
# Stage 18. Summary of the piecewise headset-controller frame fits.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Summary of the piecewise head-controller frame fits (Section 2.4.1 of the paper).

Input:  out/excursions_v2/frames.csv  (one row per hand and piece, from 12_excursions_v2.py)
Output: out/frame_summary.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
F = pd.read_csv(ROOT / "out" / "excursions_v2" / "frames.csv")

rec = ["dev", "month", "seg"]
lines = []
n_rec = F[rec].drop_duplicates().shape[0]
pieces = F[rec + ["piece"]].drop_duplicates()
per_rec = pieces.groupby(rec).size()
lines.append(f"recordings: {n_rec}")
lines.append(f"pieces: {len(pieces)}; hand fits: {len(F)}")
lines.append(f"recordings with >1 piece: {(per_rec > 1).sum()}")
lines.append(f"pieces per recording: max {per_rec.max()}")

near = np.round(F.yaw_deg / 45.0) * 45.0
dev = (F.yaw_deg - near).abs()
within = dev <= 10
lines.append(f"hand fits within 10 deg of a multiple of 45: {within.sum()} of {len(F)}")
mult = ((near + 180) % 360 - 180).astype(int)
lines.append("multiples among those: " + ", ".join(
    f"{m}: {c}" for m, c in mult[within].value_counts().sort_index().items()))
lines.append("outside 10 deg: " + "; ".join(
    f"{r.dev} {r.month} seg{r.seg} {r.hand} piece{r.piece} yaw={r.yaw_deg:.1f} resid={r.resid_m:.2f} "
    f"dur={r.t_to - r.t_from:.0f}s"
    for r in F[~within].itertuples()))
lines.append(f"deviation from multiple, within set: median {dev[within].median():.2f} deg, "
             f"max {dev[within].max():.2f} deg")

q = F.resid_m.quantile([0.25, 0.5, 0.75, 1.0])
lines.append(f"horizontal residual [m]: median {q[0.5]:.2f} (IQR {q[0.25]:.2f}-{q[0.75]:.2f}), max {q[1.0]:.2f}")

# does the relation change between pieces of one recording?
chg = []
for k, g in F.groupby(rec + ["hand"]):
    g = g.sort_values("piece")
    if len(g) > 1:
        d = np.diff(np.round(g.yaw_deg / 45.0) * 45.0)
        chg.append((k, int((d != 0).sum()), len(g) - 1))
n_changes = sum(c for _, c, _ in chg)
n_trans = sum(t for _, _, t in chg)
lines.append(f"piece transitions (per hand): {n_trans}; with a change of the 45-deg multiple: {n_changes}")

out = ROOT / "out" / "frame_summary.txt"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
