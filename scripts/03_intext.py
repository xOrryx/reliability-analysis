# -----------------------------------------------------------------------------
# Stage 3. Writes every number quoted in the running text.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Stage 3: every number that appears in the running text but not in a table.

Writes out/intext.csv, one row per quantity, with the value and the section of the
manuscript it belongs to. If a sentence in the paper carries a number, it is here.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.metrics import roc_auc_score

from reliability import config as C
from reliability.windows import load_segment, native_rate, segment_files, contiguous
from reliability.stats import precision_at_sens

R = []


def rec(section, key, value, note=""):
    R.append(dict(section=section, key=key, value=value, note=note))
    print(f"[{section}] {key} = {value}  {note}")


# ----------------------------------------------------------------- exposure
def exposure(W, S):
    w = W[W.ui < 0.5]
    rec("3", "exposure_s", round(float(w.hop_s.iloc[0]) * len(w), 0))
    rec("3", "windows_total", len(W))
    rec("3", "windows_with_scene", len(S))
    rec("2.4.2", "rate_median_hz", round(float(W[W.month == "February"].rate.median()), 1))
    rec("2.4.2", "rate_january_mqpro_hz",
        round(float(W[(W.month == "January") & (W.dev == "MQPro")].rate.median()), 1))


# ----------------------------------------------------------------- gait band
def gait(root=None):
    """Spectral peak of the vertical head position while walking, per device."""
    for dev in ("MQ3", "MQPro", "MQ2"):
        peaks = []
        for d, mon, seg, f in segment_files(root):
            if d != dev:
                continue
            a = load_segment(f)
            t = a[:, C.COL_TIME]
            fs = native_rate(t)
            y = a[:, 5]
            v = np.r_[0, np.linalg.norm(np.diff(a[:, 4:7], axis=0), axis=1)] * fs
            for i0, i1 in contiguous(v > C.WALK_MIN, int(10 * fs)):
                blk = y[i0:i1]
                fr, px = welch(blk - blk.mean(), fs=fs, nperseg=min(len(blk), int(8 * fs)))
                band = (fr > 1.2) & (fr < 2.6)
                if band.sum():
                    peaks.append(float(fr[band][np.argmax(px[band])]))
        if peaks:
            rec("2.4.2", f"gait_peak_hz_{dev}", round(float(np.median(peaks)), 2),
                f"{len(peaks)} recordings")


# ----------------------------------------------------------------- episodes
def episodes(W, E):
    rec("3.2", "episodes_total", len(E))
    rec("3.2", "episodes_time_s", round(float(E.dur_s.sum()), 0))
    rec("3.2", "episode_max_s", round(float(E.dur_s.max()), 0))
    rec("3.2", "episodes_over_50s", int((E.dur_s > 50).sum()))

    shared, shared_s = 0, 0.0
    for i, r in E.iterrows():
        o = E[(E.month == r.month) & (E.seg == r.seg) & (E.dev != r.dev)]
        ov = [min(r.end_pct, x.end_pct) - max(r.start_pct, x.start_pct)
              for _, x in o.iterrows()]
        if any(v > 0 for v in ov):
            shared += 1
            shared_s += r.dur_s
    rec("3.2", "episodes_shared", shared)
    rec("3.2", "episodes_shared_time_s", round(shared_s, 0))
    rec("3.2", "episodes_shared_time_pct", round(100 * shared_s / E.dur_s.sum(), 0))

    for mon, seg in (("February", 3), ("January", 5)):
        e = E[(E.month == mon) & (E.seg == seg)]
        lo, hi = e.start_pct.min(), e.end_pct.max()
        inter = min(e.groupby("dev").end_pct.max()) - max(e.groupby("dev").start_pct.min())
        rec("3.2", f"cooccurrence_{mon[:3]}{seg}_pct",
            round(100 * inter / (hi - lo), 0), f"{e.dev.nunique()} devices")

    w = W[W.ui < 0.5]
    fail = w.f_track_hmd <= C.FAIL_FRAC
    rec("3.2", "speed_during_loss", round(float(w[fail].speed.median()), 2),
        "artefact: the reported position stops changing")
    rec("3.2", "speed_elsewhere", round(float(w[~fail].speed.median()), 2))


# ----------------------------------------------------------------- precursor
def precursor(W):
    """Does elevated jitter precede a loss of tracking?"""
    for horizon in (5, 10, 20):
        y, s = [], []
        for _, g in W[W.ui < 0.5].groupby(["dev", "month", "seg"]):
            g = g.sort_values("t").reset_index(drop=True)
            fail = (g.f_track_hmd <= C.FAIL_FRAC).to_numpy()
            starts = g.t.to_numpy()[np.flatnonzero(fail & ~np.r_[False, fail[:-1]])]
            ok = g.hp_pos_hmd.notna().to_numpy() & ~fail
            if not len(starts) or ok.sum() < 50:
                continue
            t = g.t.to_numpy()
            lab = np.array([np.any((starts > x) & (starts <= x + horizon)) for x in t])
            y.append(lab[ok].astype(int))
            s.append(g.hp_pos_hmd.to_numpy()[ok])
        y, s = np.concatenate(y), np.concatenate(s)
        if y.sum() and (y == 0).sum():
            q = np.nanquantile(s, 0.9)
            rec("3.2", f"precursor_auc_{horizon}s", round(float(roc_auc_score(y, s)), 3))
            rec("3.2", f"precursor_top_decile_pct_{horizon}s",
                round(100 * float(y[s >= q].mean()), 2))
            rec("3.2", f"precursor_rest_pct_{horizon}s",
                round(100 * float(y[s < q].mean()), 2))


# ----------------------------------------------------------------- scene
def scene_numbers(S):
    y = S.FAIL.to_numpy().astype(bool)
    b = S.Brightness.to_numpy(float)
    rec("3.3", "base_rate_pct", round(100 * float(y.mean()), 1))
    rec("3.3", "failures_in_darkness_pct", round(100 * float((b < 20)[y].mean()), 0),
        "mean brightness below 20 of 255")
    lit = b >= 20
    sn = S.SegFrac_snow.to_numpy(float)
    rec("3.3", "lit_failure_snow_median", round(float(np.median(sn[y & lit])), 2))
    rec("3.3", "lit_normal_snow_median", round(float(np.median(sn[~y & lit])), 2))
    rec("3.3", "lit_incidence_snow_high_pct",
        round(100 * float(y[lit & (sn > 0.2)].mean()), 1))
    rec("3.3", "lit_incidence_snow_low_pct",
        round(100 * float(y[lit & (sn <= 0.2)].mean()), 1))
    rec("3.3", "lit_failures_devices",
        ",".join(sorted(S.dev[y & lit & (sn > 0.2)].unique())))

    p = np.load(C.OUT / "detector_probs.npy")[2]
    pr = precision_at_sens(y.astype(int), p, 0.80)
    rec("3.3", "precision_at_80_pct", round(100 * pr, 0),
        f"about one in {1/pr:.1f} flagged windows")

    bins = np.linspace(0, 180, 37)
    k, _ = np.histogram(b[y], bins=bins)
    n, _ = np.histogram(b, bins=bins)
    rec("3.3", "p_loss_darkest_bin", round(float(k[0] / max(n[0], 1)), 2),
        f"brightness below {bins[1]:.0f}")


# ----------------------------------------------------------------- controllers
def controllers(F, J, X, W):
    F = F.dropna(subset=["yaw_deg"])
    near = (F.yaw_deg / 45.0).round() * 45.0
    close = (F.yaw_deg - near).abs() < 10.0
    rec("2.4.1", "yaw_fits", len(F))
    rec("2.4.1", "yaw_near_multiple_of_45", int(close.sum()))
    rec("2.4.1", "yaw_multiples_observed", sorted(set(near[close].astype(int))))
    rec("2.4.1", "yaw_exceptions",
        "; ".join(f"{r.dev} {r.month[:3]}{r.seg}{r.hand} {r.yaw_deg:.0f} deg, "
                  f"fit residual {r.resid_horizontal_m:.0f} m"
                  for _, r in F[~close].iterrows()),
        "the fit is dominated by controller excursions in these recordings")
    G = F[F.resid_horizontal_m < 1.0]   # excludes the three excursion-dominated fits
    for col, key in (("resid_horizontal_m", "resid_horizontal_m"),
                     ("vertical_m", "vertical_m"),
                     ("distance_3d_m", "distance_3d_m")):
        q = G[col].quantile([0.25, 0.5, 0.75])
        rec("2.4.1", key, f"median {q[0.5]:.2f}, IQR {q[0.25]:.2f}-{q[0.75]:.2f}",
            f"over the {len(G)} fits with a residual below 1 m")

    rec("3.4", "controller_jumps", len(J))
    dtm = J.dt_s.to_numpy()
    irregular = dtm > 2 * np.median(dtm)
    rec("3.4", "jumps_at_irregular_interval_pct", round(100 * float(irregular.mean()), 1))
    rec("3.4", "implied_hand_speed_median_ms",
        round(float(np.median(J.speed_implied[~irregular])), 1))
    rec("3.4", "controller_events", len(X))
    for dev in C.DEVICES:
        x = X[X.dev == dev]
        if len(x):
            rec("3.4", f"events_{dev}",
                f"n={len(x)}, median excursion {x.excursion_m.median():.2f} m, "
                f"max {x.excursion_m.max():.1f} m, median recovery {x.recover_s.median():.1f} s")
    rec("3.4", "tracked_during_events_median_pct",
        round(float(X.tracked_during.median()), 0))

    key = W.set_index(["dev", "month", "seg"])
    ok = []
    for _, r in X.iterrows():
        g = W[(W.dev == r.dev) & (W.month == r.month) & (W.seg == r.seg)]
        i = (g.t - r.t).abs().idxmin()
        ok.append(float(g.loc[i, "f_track_hmd"]) > 0.99)
    rec("3.4", "events_with_headset_tracking_ok", int(np.sum(ok)))


def coupling(W, Rl):
    """Rank correlation between the differential controller and the headset."""
    from scipy.stats import spearmanr
    M = pd.merge(W[["dev", "month", "seg", "t", "hp_pos_hmd", "f_track_hmd", "ui"]],
                 Rl[["dev", "month", "seg", "t", "rel_left", "rel_right"]],
                 on=["dev", "month", "seg", "t"])
    M = M[(M.ui < 0.5) & (M.f_track_hmd > 0.99)]
    for dev in C.DEVICES:
        g = M[M.dev == dev].dropna(subset=["hp_pos_hmd"])
        v = pd.concat([g.rel_left, g.rel_right])
        h = pd.concat([g.hp_pos_hmd, g.hp_pos_hmd])
        m = v.notna()
        rec("3.4", f"coupling_rho_{dev}",
            round(float(spearmanr(v[m], h[m]).statistic), 2))


# ----------------------------------------------------------------- driver
def main():
    W = pd.read_pickle(C.OUT / "windows.pkl")
    S = pd.read_pickle(C.OUT / "scene.pkl")
    Rl = pd.read_pickle(C.OUT / "controllers.pkl")
    E = pd.read_csv(C.OUT / "episodes.csv")
    F = pd.read_csv(C.OUT / "frames.csv")
    J = pd.read_csv(C.OUT / "controller_jumps.csv")
    X = pd.read_csv(C.OUT / "controller_events.csv")

    exposure(W, S)
    gait()
    episodes(W, E)
    precursor(W)
    scene_numbers(S)
    controllers(F, J, X, W)
    coupling(W, Rl)

    pd.DataFrame(R).to_csv(C.OUT / "intext.csv", index=False)
    print("\nwritten", C.OUT / "intext.csv")


if __name__ == "__main__":
    main()
