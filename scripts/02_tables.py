# -----------------------------------------------------------------------------
# Stage 2. Writes every table that appears in the manuscript.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Stage 2: every table that appears in the manuscript.

Each table is written twice: a .csv with full precision for checking, and a .tex
ready to be included. Nothing is rounded before the .tex is written.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from reliability import config as C, scene, detect
from reliability.stats import block_bootstrap_median, bootstrap_median_arrays, auc

TEX = C.OUT / "tex"
TEX.mkdir(parents=True, exist_ok=True)
MM = 10.0   # window values are in centimetres; the manuscript reports millimetres


def _regimes(g):
    return {"standing": g[g.speed < C.STILL_MAX], "walking": g[g.speed > C.WALK_MIN]}


# ----------------------------------------------------------------- pose stability
def table_jitter(W, R):
    """Headset position and orientation, and controller position, February."""
    w = W[(W.month == "February") & (W.ui < 0.5) & (W.f_track_hmd > 0.99)]
    r = R[(R.month == "February") & (R.ui < 0.5)]
    rows = []
    for label, src, col, scale in (
            ("HMD position [mm]", w, "hp_pos_hmd", MM),
            ("HMD orientation [deg s]", w, "hp_rot_hmd", 1.0),
            ("Controller position [mm]", w, "ctrl", MM)):
        for cond in ("standing", "walking"):
            rec = dict(quantity=label, condition=cond)
            for dev in C.DEVICES:
                g = _regimes(src[src.dev == dev])[cond]
                if col == "ctrl":    # absolute controller jitter, both hands pooled
                    arrays = [np.r_[s.hp_pos_left.to_numpy(), s.hp_pos_right.to_numpy()]
                              for _, s in g.groupby("seg")]
                    m, lo, hi = bootstrap_median_arrays(arrays)
                    n = int(g[["hp_pos_left", "hp_pos_right"]].notna().sum().sum())
                else:
                    m, lo, hi = block_bootstrap_median(g, col, group=("seg",))
                    n = int(g[col].notna().sum())
                rec[dev] = (m * scale, lo * scale, hi * scale)
                rec[dev + "_n"] = n
            rows.append(rec)
    return pd.DataFrame(rows)


def tex_jitter(T):
    def cell(v):
        m, lo, hi = v
        return f"{m:.2f} ({lo:.2f}--{hi:.2f})"
    L = [r"\begin{table}[t]",
         r"\caption{High-frequency pose jitter of the head-mounted display and of the "
         r"controllers, February session, all three devices measured simultaneously. "
         r"Values are medians in millimetres for position and degrees per second for "
         r"orientation, with 95\,\% confidence intervals from a block bootstrap "
         r"resampled over segments. The controller row is the jitter of the controller "
         r"pose as delivered, both controllers pooled; the differential quantity with "
         r"respect to the head is reported separately in Table~\ref{tab:ctrl_rel}. "
         r"Windows during tracking loss and with the information panel open are "
         r"excluded.}\label{tab:jitter}",
         r"\begin{tabular}{llccc}", r"\toprule",
         r"Quantity & Condition & Quest~2 & Quest~3 & Quest~Pro \\", r"\midrule"]
    prev = None
    for _, r in T.iterrows():
        q = r.quantity.replace("[deg s]", r"[deg\,s$^{-1}$]")
        head = q if r.quantity != prev else ""
        if r.quantity != prev and prev is not None:
            L.append(r"\addlinespace")
        prev = r.quantity
        L.append(f"{head} & {r.condition} & " +
                 " & ".join(cell(r[d]) for d in C.DEVICES) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# ----------------------------------------------------------------- episodes
def table_episodes(W, E):
    rows = []
    for (dev, mon), g in W[W.ui < 0.5].groupby(["dev", "month"]):
        e = E[(E.dev == dev) & (E.month == mon)]
        # hop per segment: the January Quest Pro recordings change rate between
        # the two parts of the day (35.7 and 45.5 Hz), so one hop does not fit all
        expo = float(sum(float(s.hop_s.iloc[0]) * len(s) for _, s in g.groupby("seg")))
        if not len(e):
            continue
        rows.append(dict(dev=dev, month=mon, exposure_s=expo, episodes=len(e),
                         total_s=e.dur_s.sum(), median_s=e.dur_s.median(),
                         max_s=e.dur_s.max(), share_pct=100 * e.dur_s.sum() / expo))
    return pd.DataFrame(rows).sort_values(["dev", "month"])


def tex_episodes(T):
    L = [r"\begin{table}[t]",
         r"\caption{Episodes of head-tracking loss, defined as windows in which the "
         r"device reported an untracked pose for more than 80\,\% of the samples. "
         r"Episodes shorter than one second are not counted. The exposure column gives "
         r"the total recorded time outside the information panel.}\label{tab:episodes}",
         r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Device & Session & Exposure [s] & Episodes & Total [s] & Median [s] & Share [\%] \\",
         r"\midrule"]
    for _, r in T.iterrows():
        L.append(f"{C.DEVICE_LABEL[r.dev].replace(' ', '~')} & {r.month[:3]} & "
                 f"{r.exposure_s:.0f} & {r.episodes:.0f} & {r.total_s:.0f} & "
                 f"{r.median_s:.1f} & {r.share_pct:.2f} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# ----------------------------------------------------------------- descriptors
DESC_LABEL = [
    ("UnderexposedFrac", "Underexposed pixel fraction"),
    ("Brightness", "Brightness"),
    ("Entropy", "Entropy"),
    ("Contrast", "Contrast"),
    ("DynamicRangeP", "Dynamic range (P99--P1)"),
    ("Corners", "Harris corners"),
    ("SegFrac_sky_other", "Sky fraction"),
    ("SegFrac_snow", "Snow fraction"),
    ("OverexposedFrac", "Overexposed pixel fraction"),
    ("SegFrac_light", "Artificial light fraction"),
    ("Focus_LapVar", "Laplacian variance"),
]


def table_descriptors(S):
    y = S.FAIL.to_numpy()
    rows = []
    for col, label in DESC_LABEL:
        if col not in S.columns:
            continue
        x = S[col].to_numpy(float)
        rows.append(dict(descriptor=label, column=col,
                         loss=float(np.median(x[y == 1])),
                         normal=float(np.median(x[y == 0])),
                         auc=auc(y, -x)))
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def tex_descriptors(T):
    L = [r"\begin{table}[t]",
         r"\caption{Scene descriptors during tracking loss and during normal operation, "
         r"February session. Values are medians over windows. AUC is the univariate area "
         r"under the ROC curve for separating the two classes, reported so that 0.5 means "
         r"no separation.}\label{tab:descriptors}",
         r"\begin{tabular}{lrrr}", r"\toprule",
         r"Descriptor & Tracking loss & Normal & AUC \\", r"\midrule"]
    for _, r in T.iterrows():
        L.append(f"{r.descriptor} & {r.loss:.3f} & {r.normal:.3f} & {r.auc:.3f} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# ----------------------------------------------------------------- detector
def table_detector(S):
    sets = scene.feature_sets(S)
    y = S.FAIL.to_numpy()
    probs, rows = {}, []
    for name, cols in sets.items():
        p = detect.leave_one_recording_out(S, cols)
        probs[name] = p
        rows.append(dict(feature_set=name, n_features=len(cols), **detect.summarise(y, p)))
    tr = detect.transfer(S, sets["full"])
    thr, n_dark, base_dark, auc_dark = detect.dark_subset(S, probs["full"])
    np.save(C.OUT / "detector_probs.npy",
            np.vstack([probs["brightness"], probs["photometric"], probs["full"]]))
    np.save(C.OUT / "detector_y.npy", y)
    meta = dict(n=len(S), fail=int(y.sum()), base_rate=float(y.mean()),
                groups=int(len(np.unique(detect.groups(S)))),
                dark_threshold=thr, dark_n=n_dark, dark_base=base_dark, dark_auc=auc_dark)
    return pd.DataFrame(rows), tr, meta


def tex_detector(T, TR, M):
    def pct(x):
        return f"{100*x:.0f}\\,\\%"
    L = [r"\begin{table}[t]",
         r"\caption{Detection of tracking loss from scene descriptors alone, February "
         f"session, {M['n']:,} windows of which {M['fail']} ({100*M['base_rate']:.1f}\\,\\%) "
         r"are windows with tracking lost. All figures are leave-one-recording-out "
         f"cross-validated over the {M['groups']} device--segment recordings, so no window "
         r"is scored by a model that has seen its own recording. Specificity is reported "
         r"at two fixed sensitivity levels. The last row restricts the evaluation to the "
         f"darkest quartile of windows, mean brightness below {M['dark_threshold']:.1f} of "
         f"255, where the base rate of tracking loss is {100*M['dark_base']:.0f}\\,\\%."
         r"}\label{tab:detector}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Feature set & AUC & \multicolumn{2}{c}{Specificity} \\",
         r"\cmidrule(l){3-4}",
         r" & & at 80\,\% sens. & at 90\,\% sens. \\", r"\midrule"]
    names = {"brightness": "Brightness alone",
             "photometric": "Photometric (8 descriptors)",
             "full": None}
    for _, r in T.iterrows():
        nm = names[r.feature_set] or f"Photometric $+$ semantic ({r.n_features})"
        L.append(f"{nm} & {r.auc:.3f} & {pct(r.spec_at_80)} & {pct(r.spec_at_90)} " + r"\\")
    L.append(r"\addlinespace")
    for _, r in TR.iterrows():
        L.append(rf"\quad transfer to {C.DEVICE_LABEL[r.dev].replace(' ','~')} & "
                 f"{r.auc:.3f} & -- & -- " + r"\\")
    L.append(rf"\quad within dark windows only & {M['dark_auc']:.3f} & -- & -- \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# ----------------------------------------------------------------- controllers
def table_controllers(R):
    rows = []
    for mon in ("January", "February"):
        for cond in ("standing", "walking"):
            rec = dict(session=mon, condition=cond)
            for dev in C.DEVICES:
                g = _regimes(R[(R.month == mon) & (R.ui < 0.5) & (R.dev == dev)])[cond]
                if not len(g):
                    rec[dev] = (np.nan, np.nan, np.nan)
                    continue
                arrays = [np.r_[s.rel_left.to_numpy(), s.rel_right.to_numpy()]
                          for _, s in g.groupby("seg")]
                m, lo, hi = bootstrap_median_arrays(arrays)
                rec[dev] = (m * MM, lo * MM, hi * MM)
            rows.append(rec)
    return pd.DataFrame(rows)


def tex_controllers(T):
    def cell(v):
        m, lo, hi = v
        return "--" if not np.isfinite(m) else f"{m:.2f} ({lo:.2f}--{hi:.2f})"
    L = [r"\begin{table}[t]",
         r"\caption{Differential high-frequency jitter of the controllers with respect to "
         r"the head, in millimetres, after aligning the two reported coordinate frames piece by "
         r"piece between re-anchorings. Because both poses are differenced in a common "
         r"frame, body translation cancels and the residual reflects the controller "
         r"estimate rather than the walker. Both controllers are pooled. Medians with "
         r"95\,\% block-bootstrap confidence intervals over segments.}\label{tab:ctrl_rel}",
         r"\begin{tabular}{llccc}", r"\toprule",
         r"Session & Condition & Quest~2 & Quest~3 & Quest~Pro \\", r"\midrule"]
    prev = None
    for _, r in T.iterrows():
        head = r.session if r.session != prev else ""
        if r.session != prev and prev is not None:
            L.append(r"\addlinespace")
        prev = r.session
        L.append(f"{head} & {r.condition} & " +
                 " & ".join(cell(r[d]) for d in C.DEVICES) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# ----------------------------------------------------------------- replication
def table_replication(W):
    rows = []
    for dev in C.DEVICES:
        rec = dict(dev=dev)
        for mon in ("January", "February"):
            g = W[(W.dev == dev) & (W.month == mon) & (W.ui < 0.5)
                  & (W.f_track_hmd > 0.99) & (W.speed > C.WALK_MIN)]
            rec[mon + "_jitter"] = np.nanmedian(g.hp_pos_hmd) * MM if len(g) else np.nan
            rec[mon + "_speed"] = g.speed.median() if len(g) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def tex_replication(T):
    def f(x, d=2):
        return "--" if not np.isfinite(x) else f"{x:.{d}f}"
    L = [r"\begin{table}[t]",
         r"\caption{Replication across the two sessions. The headsets were exchanged "
         r"between participants between the January and the February session, so a device "
         r"effect that survives the exchange cannot be attributed to an individual walker. "
         r"Values are median high-frequency positional jitter in millimetres while walking "
         r"above 0.8\,m\,s$^{-1}$, with the median walking speed given for control. "
         r"Quest~2 was not recovered from the January session.}\label{tab:replication}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"Device & \multicolumn{2}{c}{January} & \multicolumn{2}{c}{February} \\",
         r"\cmidrule(r){2-3}\cmidrule(l){4-5}",
         r" & Jitter [mm] & Speed [m\,s$^{-1}$] & Jitter [mm] & Speed [m\,s$^{-1}$] \\",
         r"\midrule"]
    for _, r in T.iterrows():
        L.append(f"{C.DEVICE_LABEL[r.dev].replace(' ','~')} & "
                 f"{f(r.January_jitter)} & {f(r.January_speed)} & "
                 f"{f(r.February_jitter)} & {f(r.February_speed)} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


# ----------------------------------------------------------------- manual vs automatic
def table_subj_vs_obj(S, A):
    if A is None or not len(A):
        return None, None
    attrs = [c for c in A.columns
             if c not in ("t", "dev", "month", "seg", "FAIL", "ui")]
    M = pd.merge(S, A[attrs + ["t", "dev", "seg", "month"]],
                 on=["dev", "seg", "month", "t"], how="inner")
    desc = scene.feature_sets(S)["full"]
    rows = []
    for name, cols in (("Manual annotations", attrs),
                       ("Automatic descriptors", desc),
                       ("Both combined", attrs + desc)):
        p = detect.leave_one_recording_out(M, cols)
        rows.append(dict(feature_set=name, n_features=len(cols), n=len(M),
                         **detect.summarise(M.FAIL.to_numpy(), p)))
    p = detect.leave_one_recording_out(A, attrs)
    rows.append(dict(feature_set="Manual annotations, both sessions",
                     n_features=len(attrs), n=len(A),
                     **detect.summarise(A.FAIL.to_numpy(), p)))
    return pd.DataFrame(rows), dict(n_paired=len(M), fail_paired=int(M.FAIL.sum()),
                                    n_all=len(A), fail_all=int(A.FAIL.sum()))


def tex_subj_vs_obj(T, M):
    def pct(x):
        return f"{100*x:.0f}\\,\\%"
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Detection of tracking loss from manual annotations and from automatic "
         r"descriptors. All figures are leave-one-recording-out cross-validated. The first "
         r"three rows are computed on the windows for which both feature sets are "
         f"available (February session, $n={M['n_paired']:,}$, of which {M['fail_paired']} "
         r"with loss of tracking); the values for the automatic descriptors therefore "
         r"differ slightly from Table~\ref{tab:detector}, which uses the full February "
         r"set. The last row shows the coverage advantage of the manual annotations, which "
         r"exist for both sessions.}\label{tab:subj_vs_obj}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Feature set & AUC & \multicolumn{2}{c}{Specificity} \\",
         r"\cmidrule(l){3-4}",
         r" & & at 80\,\% sens. & at 90\,\% sens. \\", r"\midrule"]
    for i, (_, r) in enumerate(T.iterrows()):
        if i == 3:
            L.append(r"\addlinespace")
        L.append(f"{r.feature_set} ({r.n_features}) & {r.auc:.3f} & "
                 f"{pct(r.spec_at_80)} & {pct(r.spec_at_90)} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L).replace("both sessions (111)", "both sessions")


# ----------------------------------------------------------------- driver
def main():
    W = pd.read_pickle(C.OUT / "windows.pkl")
    R = pd.read_pickle(C.OUT / "controllers.pkl")
    S = pd.read_pickle(C.OUT / "scene.pkl")
    E = pd.read_csv(C.OUT / "episodes.csv")
    A = (pd.read_pickle(C.OUT / "annotations.pkl")
         if (C.OUT / "annotations.pkl").exists() else None)

    T = table_jitter(W, R); T.to_csv(C.OUT / "tab_jitter.csv", index=False)
    (TEX / "tab_jitter.tex").write_text(tex_jitter(T))

    T = table_episodes(W, E); T.to_csv(C.OUT / "tab_episodes.csv", index=False)
    (TEX / "tab_episodes.tex").write_text(tex_episodes(T))

    T = table_descriptors(S); T.to_csv(C.OUT / "tab_descriptors.csv", index=False)
    (TEX / "tab_descriptors.tex").write_text(tex_descriptors(T))

    T, TR, M = table_detector(S)
    T.to_csv(C.OUT / "tab_detector.csv", index=False)
    TR.to_csv(C.OUT / "tab_detector_transfer.csv", index=False)
    pd.Series(M).to_csv(C.OUT / "tab_detector_meta.csv")
    (TEX / "tab_detector.tex").write_text(tex_detector(T, TR, M))

    T = table_controllers(R); T.to_csv(C.OUT / "tab_ctrl_rel.csv", index=False)
    (TEX / "tab_ctrl_rel.tex").write_text(tex_controllers(T))

    T = table_replication(W); T.to_csv(C.OUT / "tab_replication.csv", index=False)
    (TEX / "tab_replication.tex").write_text(tex_replication(T))

    T, M2 = table_subj_vs_obj(S, A)
    if T is not None:
        T.to_csv(C.OUT / "tab_subj_vs_obj.csv", index=False)
        (TEX / "tab_subj_vs_obj.tex").write_text(tex_subj_vs_obj(T, M2))

    print("tables written to", TEX)


if __name__ == "__main__":
    main()
