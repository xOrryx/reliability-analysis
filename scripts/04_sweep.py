# -----------------------------------------------------------------------------
# Stage 4. Systematic test of scene effects on the quality of the pose.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Stage 4: does anything about the scene change the quality of the pose?

The rule is fixed before the data are touched and is applied identically to both
descriptions of the scene.

    Every attribute is tested inside one device and one segment, so the day, the
    wearer, the time of day and the illumination are held fixed. Only windows
    with functioning tracking and a walking speed between 0.8 and 1.6 m/s enter,
    which removes the dominant covariate. Speed is controlled further: for a
    binary attribute the two sides must have median speeds within 0.15 m/s, and
    for a continuous descriptor the correlation with jitter is taken partially
    with respect to speed. The direction of the effect is then tested across the
    groups with a sign test.

An attribute with fewer than six usable groups cannot reach p < 0.05 by a sign
test at all, so it is reported as not testable rather than as a null result.
Counting it as a null would inflate the denominator and make the negative result
look stronger than the design supports.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from reliability import config as C
from reliability.stats import partial_spearman, sign_test

MM = 10.0
MIN_WINDOWS = 100
MIN_SIDE = 30
SPEED_MATCH = 0.15


def walking(W):
    w = W[(W.ui < 0.5) & (W.f_track_hmd > 0.99)
          & (W.speed >= C.WALK_MIN) & (W.speed <= C.WALK_MAX)].copy()
    w["jp"] = w.hp_pos_hmd * MM
    return w[w.jp.notna()]


# ----------------------------------------------------------------- continuous
def sweep_objective(S):
    desc = ([c for c in S.columns if c.startswith("SegFrac_")]
            + [c for c in C.PHOTOMETRIC if c in S.columns])
    w = S[(S.f_track_hmd > 0.99) & (S.speed >= C.WALK_MIN)
          & (S.speed <= C.WALK_MAX)].copy()
    w["jp"] = w.hp_pos_hmd * MM
    w = w[w.jp.notna()]
    rows = []
    for d in desc:
        rho = []
        for _, g in w.groupby(["dev", "seg"]):
            x = g[d].to_numpy(float)
            if len(g) < MIN_WINDOWS or np.nanstd(x) == 0:
                continue
            r = partial_spearman(x, g.jp.to_numpy(), g.speed.to_numpy())
            if np.isfinite(r):
                rho.append(r)
        if not rho:
            continue
        k, pos, p = sign_test(rho)
        rows.append(dict(family="objective", attribute=d, groups=k, positive=pos,
                         median_rho=float(np.median(rho)), p_sign=p,
                         testable=k >= C.MIN_GROUPS_SIGN_TEST))
    return pd.DataFrame(rows), len(w), w.groupby(["dev", "seg"]).ngroups


# ----------------------------------------------------------------- categorical
def sweep_subjective(W, A):
    attrs = [c for c in A.columns
             if c not in ("t", "dev", "month", "seg", "FAIL", "ui")]
    M = pd.merge(walking(W)[["dev", "month", "seg", "t", "jp", "speed"]],
                 A[attrs + ["dev", "month", "seg", "t"]],
                 on=["dev", "month", "seg", "t"], how="inner")
    rows = []
    for a in attrs:
        eff = []
        for _, g in M.groupby(["dev", "month", "seg"]):
            v = g[a].to_numpy(float)
            hi, lo = g[v > 0.5], g[v <= 0.5]
            if len(hi) < MIN_SIDE or len(lo) < MIN_SIDE:
                continue
            if abs(hi.speed.median() - lo.speed.median()) > SPEED_MATCH:
                continue
            eff.append(float(np.log(hi.jp.median() / lo.jp.median())))
        if not eff:
            continue
        k, pos, p = sign_test(eff)
        rows.append(dict(family="subjective", attribute=a, groups=k, positive=pos,
                         median_ratio=float(np.exp(np.median(eff))), p_sign=p,
                         testable=k >= C.MIN_GROUPS_SIGN_TEST))
    return pd.DataFrame(rows), len(M), M.groupby(["dev", "month", "seg"]).ngroups


# ----------------------------------------------------------------- vegetation
VEG = ["SegFrac_tree", "SegFrac_leaves", "SegFrac_branch", "SegFrac_bush",
       "SegFrac_grass", "SegFrac_plant_other", "SegFrac_moss", "SegFrac_flower"]


def vegetation_contrast(S):
    """Upper against lower quartile of vegetation cover, speeds matched."""
    w = S[(S.f_track_hmd > 0.99) & (S.speed >= C.WALK_MIN)
          & (S.speed <= C.WALK_MAX)].copy()
    w["jp"] = w.hp_pos_hmd * MM
    w = w[w.jp.notna()]
    w["veg"] = w[[c for c in VEG if c in w.columns]].sum(axis=1)
    rows = []
    for (dev, seg), g in w.groupby(["dev", "seg"]):
        if len(g) < MIN_WINDOWS:
            continue
        q1, q3 = g.veg.quantile([0.25, 0.75])
        lo, hi = g[g.veg <= q1], g[g.veg >= q3]
        if len(lo) < MIN_SIDE or len(hi) < MIN_SIDE:
            continue
        if abs(lo.speed.median() - hi.speed.median()) > SPEED_MATCH:
            continue
        rows.append(dict(dev=dev, seg=seg, n_low=len(lo), n_high=len(hi),
                         jitter_low=lo.jp.median(), jitter_high=hi.jp.median(),
                         ratio=hi.jp.median() / lo.jp.median(),
                         veg_contrast=hi.veg.median() - lo.veg.median()))
    T = pd.DataFrame(rows)
    k, pos, p = sign_test(np.log(T.ratio.to_numpy())) if len(T) else (0, 0, np.nan)
    return T, dict(groups=k, ratio_above_one=pos, p_sign=p,
                   median_ratio=float(T.ratio.median()) if len(T) else np.nan)


def main():
    W = pd.read_pickle(C.OUT / "windows.pkl")
    S = pd.read_pickle(C.OUT / "scene.pkl")
    A = (pd.read_pickle(C.OUT / "annotations.pkl")
         if (C.OUT / "annotations.pkl").exists() else None)

    O, n_obj, g_obj = sweep_objective(S)
    print(f"objective: {len(O)} descriptors, {n_obj} walking windows, {g_obj} groups")
    parts = [O]
    if A is not None and len(A):
        Sb, n_sub, g_sub = sweep_subjective(W, A)
        print(f"subjective: {len(Sb)} attributes, {n_sub} walking windows, {g_sub} groups")
        parts.append(Sb)
    T = pd.concat(parts, ignore_index=True)
    T.to_csv(C.OUT / "sweep.csv", index=False)

    print("\n--- systematic test of scene effects on pose quality")
    for fam, g in T.groupby("family"):
        t = g[g.testable]
        print(f"{fam:11s} tested {len(g):3d}  testable {len(t):3d}  "
              f"passed {int((t.p_sign < 0.05).sum())}  expected {0.05*len(t):.1f}")
    t = T[T.testable]
    print(f"{'TOTAL':11s} tested {len(T):3d}  testable {len(t):3d}  "
          f"passed {int((t.p_sign < 0.05).sum())}  expected {0.05*len(t):.1f}")
    if (t.p_sign < 0.05).any():
        print("\npassed:")
        print(t[t.p_sign < 0.05].to_string(index=False))

    tex = [c for c in ("Corners", "Focus_LapVar", "Entropy", "Contrast", "Brightness",
                       "SegFrac_tree", "SegFrac_bush", "SegFrac_grass",
                       "SegFrac_sky_other", "SegFrac_wall_concrete")]
    print("\neffect-size bounds, median partial rho:")
    print(t[t.attribute.isin(tex)][["attribute", "groups", "median_rho", "p_sign"]]
          .to_string(index=False))

    V, meta = vegetation_contrast(S)
    V.to_csv(C.OUT / "vegetation_contrast.csv", index=False)
    print("\n--- vegetation, upper against lower quartile")
    print(V.round(2).to_string(index=False))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in meta.items()})


if __name__ == "__main__":
    main()
