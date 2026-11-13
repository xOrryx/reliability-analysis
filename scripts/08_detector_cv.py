# -----------------------------------------------------------------------------
# Stage 8: how far the scene-based detector of tracking loss can be trusted.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Stage 8: the detector under stricter evaluation.

Three questions, each answered by changing one thing at a time against the
evaluation in 02_tables.table_detector:

1. Grouping. Failures are shared between the devices worn along the same
   segment, so holding out one device's recording while the other devices'
   recordings of that segment stay in training lets the model learn the place.
   Here whole segments (all devices together) are held out as well.
2. Target. A window inside an episode is a loss the runtime already reports.
   Here the target is also the onset: windows whose centre lies 1-10 s (and
   1-5 s) before an episode begins, against windows that are neither inside an
   episode nor within 10 s before one. Windows inside episodes are left out.
3. Commensurability. Photometric values of the infrared Quest 2 stream and the
   colour streams are not on one scale, so every descriptor is also converted
   to its rank within the device (percentile over that device's February
   windows) before training.

Also reported: the number of distinct failure events in February (episodes of
different devices that overlap in time on the same segment count once), and the
probability of loss above and below the brightness levels quoted in the text.

Writes out/detector_cv/*.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from reliability import config as C, scene
from reliability.stats import spec_at_sens

OUT = C.OUT / "detector_cv"
OUT.mkdir(parents=True, exist_ok=True)
PRE = {"pre_1_10s": (1.0, 10.0), "pre_1_5s": (1.0, 5.0)}


def model():
    return HistGradientBoostingClassifier(random_state=C.SEED)


def cv(D, cols, y, groups):
    X = D[cols].to_numpy(float)
    p = np.full(len(D), np.nan)
    for g in np.unique(groups):
        m = groups == g
        tr = ~m & np.isfinite(y)
        if len(np.unique(y[tr])) < 2:
            continue
        clf = model().fit(X[tr], y[tr])
        p[m] = clf.predict_proba(X[m])[:, 1]
    return p


def summarise(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m].astype(int), p[m]
    return dict(n=int(len(y)), positives=int(y.sum()),
                auc=float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan,
                spec_at_80=spec_at_sens(y, p, 0.80), spec_at_90=spec_at_sens(y, p, 0.90))


def device_rank(D, cols):
    R = D.copy()
    for c in cols:
        R[c] = D.groupby("dev")[c].rank(pct=True)
    return R


def targets(D, E):
    """In-loss target and pre-onset targets for every window of D."""
    T = {"in_loss": D.FAIL.to_numpy(float)}
    onsets = E[E.month == "February"]
    for name, (a, b) in PRE.items():
        y = np.zeros(len(D))
        excl = D.FAIL.to_numpy() == 1
        for (dev, seg), g in D.groupby(["dev", "seg"]):
            o = onsets[(onsets.dev == dev) & (onsets.seg == seg)].t0.to_numpy()
            if not len(o):
                continue
            t = g.t.to_numpy()
            dt = o[None, :] - t[:, None]            # seconds until each onset
            pos = ((dt >= a) & (dt <= b)).any(axis=1)
            near = ((dt > 0) & (dt <= 10.0)).any(axis=1)
            y[g.index[pos]] = 1
            excl[g.index[near & ~pos]] = True
        y[excl] = np.nan
        T[name] = y
    return T


def distinct_events(E):
    F = E[E.month == "February"].sort_values("t0")
    ev = []
    for seg, g in F.groupby("seg"):
        spans = sorted(zip(g.start_pct, g.end_pct, g.dev))
        cur = None
        for s, e, d in spans:
            if cur and s <= cur[1]:
                cur = (cur[0], max(cur[1], e), cur[2] | {d})
            else:
                if cur:
                    ev.append((seg, *cur))
                cur = (s, e, {d})
        if cur:
            ev.append((seg, *cur))
    return pd.DataFrame([dict(seg=s, start_pct=a, end_pct=b, devices=",".join(sorted(d)),
                              n_devices=len(d)) for s, a, b, d in ev])


def main():
    S = pd.read_pickle(C.OUT / "scene.pkl").reset_index(drop=True)
    E = pd.read_csv(C.OUT / "episodes.csv")
    sets = scene.feature_sets(S)
    rec_groups = (S.dev + "_" + S.month + "_" + S.seg.astype(str)).to_numpy()
    seg_groups = (S.month + "_" + S.seg.astype(str)).to_numpy()
    T = targets(S, E)
    variants = {"raw": S, "device_rank": device_rank(S, sets["full"])}
    rows = []
    for tname, y in T.items():
        for fname, cols in (("brightness", sets["brightness"]), ("photometric", sets["photometric"]),
                            ("full", sets["full"])):
            for vname, D in variants.items():
                for gname, groups in (("recording", rec_groups), ("segment", seg_groups)):
                    p = cv(D, cols, y, groups)
                    rows.append(dict(target=tname, features=fname, scaling=vname, cv=gname,
                                     **summarise(y, p)))
                    print(rows[-1], flush=True)
                    if tname == "in_loss" and fname == "full":
                        b = S.Brightness.to_numpy(float)
                        thr = np.nanquantile(b, C.DARK_QUANTILE)
                        m = b < thr
                        r = summarise(y[m], p[m])
                        rows.append(dict(target="in_loss_dark_quartile", features=fname,
                                         scaling=vname, cv=gname, **r))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "detector_variants.csv", index=False)

    # transfer to an unseen device, raw and device-ranked
    tr = []
    for vname, D in variants.items():
        X, y = D[sets["full"]].to_numpy(float), S.FAIL.to_numpy()
        for dev in sorted(S.dev.unique()):
            m = (S.dev == dev).to_numpy()
            p = model().fit(X[~m], y[~m]).predict_proba(X[m])[:, 1]
            tr.append(dict(scaling=vname, dev=dev, auc=float(roc_auc_score(y[m], p))))
    pd.DataFrame(tr).to_csv(OUT / "transfer.csv", index=False)

    ev = distinct_events(E)
    ev.to_csv(OUT / "distinct_events_february.csv", index=False)

    b, f = S.Brightness.to_numpy(float), S.FAIL.to_numpy()
    rates = []
    for lo, hi in ((0, 5), (0, 15), (15, 1e9), (20, 1e9), (0, 1e9)):
        m = (b >= lo) & (b < hi)
        rates.append(dict(brightness_from=lo, brightness_to=hi, windows=int(m.sum()),
                          p_loss=float(f[m].mean())))
    pd.DataFrame(rates).to_csv(OUT / "p_loss_by_brightness.csv", index=False)
    for dev in sorted(S.dev.unique()):
        m = (S.dev == dev).to_numpy()
        rates.append(dict(dev=dev, brightness_from=15, brightness_to=1e9,
                          windows=int((m & (b >= 15)).sum()), p_loss=float(f[m & (b >= 15)].mean())))
    pd.DataFrame(rates).to_csv(OUT / "p_loss_by_brightness.csv", index=False)
    print(ev.to_string(index=False))
    print(pd.DataFrame(rates).to_string(index=False))


if __name__ == "__main__":
    main()
