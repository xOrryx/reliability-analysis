# -----------------------------------------------------------------------------
# Stage 7. Builds the Supplementary Information: ESM_1.pdf and ESM_2.csv.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Supplementary Information, generated from the same tables as the manuscript.

Springer's guidance for this journal accepts text supplements as PDF only and
states that the material is published exactly as received, so the document is
typeset here rather than exported from a word processor. Files are named
ESM_1, ESM_2, ... and each is cited in the main text as an Online Resource.
"""
import shutil
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from reliability import config as C, scene
from reliability.stats import auc

OUT = C.OUT / "supplement"
OUT.mkdir(parents=True, exist_ok=True)

LABEL = {"MQ2": "Quest~2", "MQ3": "Quest~3", "MQPro": "Quest~Pro"}
MM = 10.0


def esc(s):
    return (str(s).replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def pretty(name):
    """Descriptor column name to a readable label."""
    if name.startswith("SegFrac_"):
        return name[8:].replace("_", " ") + " fraction"
    return {"DynamicRangeP": "dynamic range (P99--P1)",
            "DynamicRange": "dynamic range (max--min)",
            "Focus_LapVar": "Laplacian variance",
            "OverexposedFrac": "overexposed fraction",
            "UnderexposedFrac": "underexposed fraction",
            "Corners": "Harris corners"}.get(name, name.lower())


# ----------------------------------------------------------------- S1
def table_descriptors_full(S):
    y = S.FAIL.to_numpy()
    cols = ([c for c in C.PHOTOMETRIC if c in S.columns] + ["DynamicRange"]
            + [c for c in S.columns if c.startswith("SegFrac_")])
    rows = []
    for c in dict.fromkeys(cols):
        x = S[c].to_numpy(float)
        if not np.isfinite(x).any():
            continue
        rows.append(dict(descriptor=pretty(c), column=c,
                         loss=float(np.nanmedian(x[y == 1])),
                         normal=float(np.nanmedian(x[y == 0])),
                         auc=auc(y, -x)))
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


# ----------------------------------------------------------------- S3
def table_recordings(W, E):
    rows = []
    for (dev, mon, seg), g in W.groupby(["dev", "month", "seg"]):
        g_ok = g[g.ui < 0.5]
        e = E[(E.dev == dev) & (E.month == mon) & (E.seg == seg)]
        st = g_ok[(g_ok.speed < C.STILL_MAX) & (g_ok.f_track_hmd > 0.99)].hp_pos_hmd
        wk = g_ok[(g_ok.speed > C.WALK_MIN) & (g_ok.f_track_hmd > 0.99)].hp_pos_hmd
        rows.append(dict(
            device=dev, session=mon, segment=seg,
            exposure_s=float(g_ok.hop_s.iloc[0]) * len(g_ok),
            windows=len(g_ok), rate_hz=float(g.rate.iloc[0]),
            standing_mm=np.nanmedian(st) * MM if st.notna().any() else np.nan,
            walking_mm=np.nanmedian(wk) * MM if wk.notna().any() else np.nan,
            episodes=len(e), lost_s=float(e.dur_s.sum()) if len(e) else 0.0))
    order = {"January": 0, "February": 1}
    return pd.DataFrame(rows).sort_values(
        ["session", "segment", "device"], key=lambda s: s.map(order).fillna(s))


# ----------------------------------------------------------------- S6
def table_excursions(X):
    rows = []
    for (dev, mon, seg), g in X.groupby(["dev", "month", "seg"]):
        rows.append(dict(device=dev, session=mon, segment=seg, events=len(g),
                         median_m=g.excursion_m.median(),
                         max_m=g.excursion_m.max(),
                         median_recovery_s=g.recover_s.median(),
                         tracked_pct=g.tracked_during.median()))
    order = {"January": 0, "February": 1}
    return pd.DataFrame(rows).sort_values(
        ["session", "device", "segment"], key=lambda s: s.map(order).fillna(s))


# ----------------------------------------------------------------- LaTeX
def tex_longtable(df, spec, header, body_row, caption, label, note="",
                  small=False):
    L = ([r"{\footnotesize\setlength{\tabcolsep}{4pt}"] if small else [])
    L += [r"\begin{longtable}{" + spec + "}",
         r"\caption{" + caption + r"}\label{" + label + r"}\\",
         r"\toprule", header + r" \\", r"\midrule", r"\endfirsthead",
         r"\multicolumn{" + str(len(spec.replace("|", ""))) +
         r"}{l}{\footnotesize\itshape continued from the previous page}\\",
         r"\toprule", header + r" \\", r"\midrule", r"\endhead",
         r"\midrule", r"\multicolumn{" + str(len(spec.replace("|", ""))) +
         r"}{r}{\footnotesize\itshape continued on the next page}\\",
         r"\endfoot", r"\bottomrule", r"\endlastfoot"]
    for _, r in df.iterrows():
        L.append(body_row(r) + r" \\")
    L.append(r"\end{longtable}")
    if small:
        L.append(r"}")
    if note:
        L.append(r"\noindent\footnotesize " + note + r"\normalsize")
    return "\n".join(L)


def build_tex(S1, S2, S3, S4, S5, S6, meta):
    def f(x, d=2):
        if not np.isfinite(x):
            return "--"
        return f"{x:.{d}f}".replace("-", "$-$")

    parts = [r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=25mm]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{amsmath}
\usepackage[T1]{fontenc}
\usepackage[hidelinks]{hyperref}
\setlength{\parskip}{0.6em}
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.05}
\begin{document}

\begin{center}
{\Large\bfseries Supplementary Information}\\[0.6em]
{\large Stability of Localization and Tracking on Meta Quest Devices:\\
Statistics, Sources of Failures and VR App for Identification}\\[0.8em]
Petr \v{S}op\'ak and Adam Chrom\'y\\
Brno University of Technology, CEITEC and Faculty of Electrical Engineering
and Communication\\
Corresponding author: \texttt{Petr.Sopak@vut.cz}
\end{center}

\vspace{1em}

This document contains the extended tables supporting the analyses in the main
manuscript: the descriptors of the scene during a loss of tracking, the
systematic test of whether the scene affects the quality of the pose, a summary
of the individual recordings, the episodes of tracking loss, the sensitivity of
the reported indicator to the cut-off frequency of the filter that defines it,
and a summary of the controller excursions. Each table is referenced from the
main text.

Every value in this document was produced by the analysis scripts that accompany
the article, from the same intermediate results from which the tables and the
figures of the main text are built.
"""]

    parts.append(r"\section*{Table S1: Scene descriptors "
                 r"during tracking loss}")
    parts.append(
        "The main manuscript reports the eleven descriptors that separate the two "
        "classes most sharply. The table below reports all "
        f"{len(S1)} of them: the median over windows in which tracking was lost, "
        "the median over the remainder, and the univariate area under the ROC "
        "curve, oriented so that $0.5$ means no separation. February session, "
        + f"{meta['n_scene']:,}".replace(",", "\\,")
        + f" windows of which {meta['n_fail']} with tracking lost.")
    parts.append(tex_longtable(
        S1, "lrrr",
        r"Descriptor & Tracking loss & Normal & AUC",
        lambda r: f"{esc(r.descriptor)} & {r.loss:.3f} & {r.normal:.3f} & {r.auc:.3f}",
        "Scene descriptors during tracking loss and during normal operation, "
        "all descriptors.", "tab:S1"))

    parts.append(r"\section*{Table S2: Systematic test of "
                 r"scene effects on pose quality}")
    parts.append(
        "Each attribute is tested inside one device and one recording, so the day, "
        "the wearer, the time of day and the illumination are held fixed. Only "
        "windows with functioning tracking and a walking speed between "
        f"{C.WALK_MIN} and {C.WALK_MAX}~m\\,s$^{{-1}}$ enter. For a continuous "
        "descriptor the association with the positional indicator is the Spearman "
        "correlation with the walking speed partialled out; for a binary "
        "annotation it is the ratio of the median indicator between the two sides, "
        "which are additionally required to have median speeds within "
        "$0.15$~m\\,s$^{-1}$ of each other. The direction is then tested across "
        "the recordings with a two-sided sign test. An attribute with fewer than "
        f"{C.MIN_GROUPS_SIGN_TEST} usable recordings cannot reach $p < 0.05$ by "
        "that test at all and is marked as not testable; counting such attributes "
        "as null results would make the negative finding look stronger than the "
        "design supports.")
    parts.append(
        f"Of the {len(S2)} attributes listed, {int(S2.testable.sum())} are "
        f"testable and {int(((S2.p_sign < 0.05) & S2.testable).sum())} reach the "
        f"$5$~\\% level, against {0.05*int(S2.testable.sum()):.1f} expected by "
        "chance.")
    parts.append(tex_longtable(
        S2, "llrrrr",
        r"Family & Attribute & Recordings & Positive & Effect & $p$",
        lambda r: (f"{esc(r.family)} & "
                   f"{esc(pretty(r.attribute) if r.family == 'objective' else r.attribute)} & "
                   f"{int(r.groups)} & "
                   f"{int(r.positive)} & "
                   f"{f(r.median_rho,3) if np.isfinite(r.median_rho) else f(r.median_ratio,2)} & "
                   f"{'--' if not np.isfinite(r.p_sign) else f'{r.p_sign:.3f}'}"
                   + ("" if r.testable else r"$^{\dagger}$")),
        "Systematic test of scene effects on the quality of the pose, every "
        "attribute.", "tab:S2",
        note=r"$^{\dagger}$~not testable: fewer than "
             f"{C.MIN_GROUPS_SIGN_TEST} usable recordings, so $p<0.05$ is "
             r"unreachable. Effect is the median partial Spearman correlation for "
             r"the objective descriptors and the median ratio of the indicator "
             r"between the two sides for the subjective annotations."))

    parts.append(r"\section*{Table S3: The recordings one by "
                 r"one}")
    parts.append(
        "Every device--segment recording with its exposure outside the information "
        "panel, the number of analysed windows, the measured logging rate, the "
        "median positional indicator in each speed regime, and the episodes of "
        "tracking loss it contains. The environmental conditions of each segment "
        "are given in the main manuscript.")
    parts.append(tex_longtable(
        S3, "llrrrrrrrr",
        r"Device & Session & Seg. & Exp. [s] & Windows & Rate [Hz] & "
        r"Standing [mm] & Walking [mm] & Episodes & Lost [s]",
        lambda r: (f"{LABEL[r.device]} & {r.session[:3]} & {int(r.segment)} & "
                   f"{r.exposure_s:.0f} & {int(r.windows)} & {r.rate_hz:.1f} & "
                   f"{f(r.standing_mm)} & {f(r.walking_mm)} & {int(r.episodes)} & "
                   f"{r.lost_s:.0f}"),
        "Per-recording summary.", "tab:S3", small=True))

    parts.append(r"\section*{Table S4: Episodes of tracking "
                 r"loss}")
    parts.append(
        "All fifteen episodes, with their position within the recording expressed "
        "as a fraction of its duration so that recordings of different length are "
        "comparable. An episode is marked as shared when it overlaps in time with "
        "an episode of another device carried along the same segment.")
    parts.append(tex_longtable(
        S4, "llrrrrl",
        r"Device & Session & Seg. & Start [\%] & End [\%] & Duration [s] & Shared",
        lambda r: (f"{LABEL[r.dev]} & {r.month[:3]} & {int(r.seg)} & "
                   f"{r.start_pct:.1f} & {r.end_pct:.1f} & {r.dur_s:.1f} & "
                   f"{'yes' if r.shared else 'no'}"),
        "Episodes of tracking loss.", "tab:S4"))

    parts.append(r"\section*{Table S5: Sensitivity to the "
                 r"high-pass cut-off}")
    parts.append(
        "The positional indicator is defined as the root mean square after a "
        "high-pass filter, and the manuscript uses a cut-off of $5$~Hz. The table "
        "shows what happens at other cut-offs. The absolute values fall as the "
        "cut-off rises, as they must, but the ordering of the devices is identical "
        "at every cut-off tested, so it is not a property of the chosen frequency. "
        "The walking-to-standing ratio does not fall as the cut-off rises either: "
        "were the increase under walking residual gait leaking through the filter, "
        "raising the cut-off above the gait band would remove it and the ratio "
        "would collapse towards unity.")
    parts.append(tex_longtable(
        S5, "rrrrrrrrrr",
        r"$f_c$ [Hz] & \multicolumn{3}{c}{Standing [mm]} & "
        r"\multicolumn{3}{c}{Walking [mm]} & \multicolumn{3}{c}{Ratio} \\"
        "\n" + r"& Q2 & Q3 & Pro & Q2 & Q3 & Pro & Q2 & Q3 & Pro",
        lambda r: (f"{r.fc_hz:.0f} & "
                   + " & ".join(f"{r[f'{d}_standing']:.2f}" for d in C.DEVICES)
                   + " & "
                   + " & ".join(f"{r[f'{d}_walking']:.2f}" for d in C.DEVICES)
                   + " & "
                   + " & ".join(f"{r[f'{d}_ratio']:.1f}" for d in C.DEVICES)),
        "Positional indicator against the high-pass cut-off frequency, February "
        "session.", "tab:S5"))

    parts.append(r"\section*{Table S6: Controller excursions}")
    parts.append(
        "Summary by recording of the events in which the reported controller "
        "position departs from its recording-specific offset to the headset by "
        f"more than {C.MIN_EXCURSION_M}~m and then returns, while the device "
        "continues to report the controller as tracked. The complete list of the "
        f"{meta['n_events']} individual events is provided as Online Resource 2.")
    parts.append(tex_longtable(
        S6, "llrrrrrr",
        r"Device & Session & Seg. & Events & Median [m] & Max [m] & "
        r"Recovery [s] & Tracked [\%]",
        lambda r: (f"{LABEL[r.device]} & {r.session[:3]} & {int(r.segment)} & "
                   f"{int(r.events)} & {r.median_m:.2f} & {r.max_m:.1f} & "
                   f"{r.median_recovery_s:.1f} & {r.tracked_pct:.0f}"),
        "Controller excursions by recording.", "tab:S6", small=True))

    parts.append(r"\end{document}")
    return "\n\n".join(parts)


def main():
    W = pd.read_pickle(C.OUT / "windows.pkl")
    S = pd.read_pickle(C.OUT / "scene.pkl")
    E = pd.read_csv(C.OUT / "episodes.csv")
    X = pd.read_csv(C.OUT / "controller_events.csv")
    SW = pd.read_csv(C.OUT / "sweep.csv")
    FC = pd.read_csv(C.OUT / "fc_sensitivity.csv")

    if "median_ratio" not in SW.columns:
        SW["median_ratio"] = np.nan
    if "median_rho" not in SW.columns:
        SW["median_rho"] = np.nan
    SW = SW.sort_values(["family", "p_sign"])

    E = E.copy()
    shared = []
    for _, r in E.iterrows():
        o = E[(E.month == r.month) & (E.seg == r.seg) & (E.dev != r.dev)]
        shared.append(any(min(r.end_pct, x.end_pct) - max(r.start_pct, x.start_pct) > 0
                          for _, x in o.iterrows()))
    E["shared"] = shared
    E = E.sort_values(["month", "seg", "dev"],
                      key=lambda s: s.map({"January": 0, "February": 1}).fillna(s))

    S1 = table_descriptors_full(S)
    S3 = table_recordings(W, E)
    S6 = table_excursions(X)
    meta = dict(n_scene=len(S), n_fail=int(S.FAIL.sum()), n_events=len(X))

    for name, df in (("S1_descriptors", S1), ("S2_sweep", SW),
                     ("S3_recordings", S3), ("S4_episodes", E),
                     ("S5_cutoff", FC), ("S6_excursions", S6)):
        df.to_csv(OUT / f"{name}.csv", index=False)

    X.sort_values(["month", "dev", "seg", "t"]).to_csv(OUT / "ESM_2.csv", index=False)

    tex = build_tex(S1, SW, S3, E, FC, S6, meta)
    (OUT / "ESM_1.tex").write_text(tex, encoding="utf-8")

    if shutil.which("pdflatex"):
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "ESM_1.tex"],
                           cwd=OUT, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        if (OUT / "ESM_1.pdf").exists():
            for ext in (".aux", ".log", ".out", ".toc"):
                (OUT / f"ESM_1{ext}").unlink(missing_ok=True)
            print("ESM_1.pdf written")
        else:
            raise SystemExit("pdflatex failed; see out/supplement/ESM_1.log")
    else:
        print("pdflatex not found; ESM_1.tex written, compile it yourself")

    print(f"S1 {len(S1)} descriptors, S2 {len(SW)} attributes "
          f"({int(SW.testable.sum())} testable), S3 {len(S3)} recordings, "
          f"S4 {len(E)} episodes, S5 {len(FC)} cut-offs, S6 {len(S6)} rows, "
          f"ESM_2 {len(X)} events")


if __name__ == "__main__":
    main()
