# -----------------------------------------------------------------------------
# Stage 5. Draws the four figures of the manuscript.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
# -----------------------------------------------------------------------------
"""Stage 5: the four figures.

They read the same intermediate tables as the tables and the in-text numbers, so
a figure cannot disagree with the value quoted next to it in the manuscript."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "grid.linewidth": 0.4, "grid.color": "0.85", "lines.linewidth": 1.2,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

DEV = ["MQ2", "MQ3", "MQPro"]
LBL = {"MQ2": "Quest 2", "MQ3": "Quest 3", "MQPro": "Quest Pro"}
# identity is carried by marker + line style; grey level is only a secondary cue,
# so the figure survives greyscale printing and colour-vision deficiency alike
STY = {"MQ2":   dict(color="0.00", marker="o", ls="-",  ms=3.6, mfc="0.00"),
       "MQ3":   dict(color="0.40", marker="s", ls="--", ms=3.4, mfc="white", mew=1.0),
       "MQPro": dict(color="0.00", marker="^", ls=":",  ms=4.0, mfc="0.70", mew=0.8)}

# ---------------------------------------------------------------- inputs
# RAW is the only primary input for figures 1 and 2: the window table is rebuilt
# from the raw pose logs by build() and cached, so those two figures have
# no hand-entered number anywhere in the chain.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reliability import config as C  # noqa: E402

OUT = C.OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
W = pd.read_pickle(C.OUT / "windows.pkl")
rng = np.random.default_rng(C.SEED)


def boot_median(vals_by_seg, n=5000):
    """Median CI resampled over whole segments; windows overlap by 50 %."""
    v = [a[np.isfinite(a)] for a in vals_by_seg]
    v = [a for a in v if len(a)]
    if len(v) < 2:
        return np.nan, np.nan
    out = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(v), len(v))
        out[i] = np.median(np.concatenate([v[p] for p in pick]))
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


# --------------------------------------------------------------- figure 1
def fig_speed():
    w = W[(W.month == "February") & (W.ui < 0.5) & (W.f_track_hmd > 0.99)].copy()
    w["jp"] = w.hp_pos_hmd * 10.0            # cm -> mm
    edges = np.array([0.0, 0.15, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.8])
    ctr = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    rows = []
    for d in DEV:
        g = w[w.dev == d]
        m, lo, hi, x = [], [], [], []
        for a, b, c in zip(edges[:-1], edges[1:], ctr):
            s = g[(g.speed >= a) & (g.speed < b)]
            if s.jp.notna().sum() < 40:
                continue
            per_seg = [s[s.seg == k].jp.to_numpy() for k in s.seg.unique()]
            l, h = boot_median(per_seg)
            x.append(c); m.append(np.nanmedian(s.jp)); lo.append(l); hi.append(h)
            rows.append(dict(dev=d, v_lo=a, v_hi=b, n=int(s.jp.notna().sum()),
                             median_mm=np.nanmedian(s.jp), ci_lo=l, ci_hi=h))
        x, m, lo, hi = map(np.array, (x, m, lo, hi))
        ax.fill_between(x, lo, hi, color=STY[d]["color"], alpha=0.12, lw=0)
        ax.plot(x, m, label=LBL[d], **STY[d])

    ax.set_yscale("log")
    ax.set_yticks([0.3, 0.5, 1, 2, 3, 5])
    ax.set_yticklabels(["0.3", "0.5", "1", "2", "3", "5"])
    ax.minorticks_off()
    ax.set_ylim(0.22, 6.5)
    ax.set_xlabel(r"head translation speed $\bar{v}$ [$\mathrm{m\,s^{-1}}$]")
    ax.set_ylabel(r"positional jitter $J_p$ [mm]")
    ax.set_xlim(0, 1.8)
    ax.grid(True, which="major", axis="y")
    ax.set_axisbelow(True)
    ax.axvspan(0, 0.15, color="0.92", zorder=0, lw=0)
    ax.text(0.075, 6.2, "standing", rotation=90, ha="center", va="top",
            fontsize=6, color="0.45")
    ax.legend(frameon=False, loc="lower right", handlelength=2.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(str(OUT / "fig_jitter_speed.pdf"))
    fig.savefig(str(OUT / "fig_jitter_speed.png"), dpi=300)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "fig_jitter_speed.csv", index=False)
    print("fig 1 done", len(rows), "bands")


# --------------------------------------------------------------- figure 2
def fig_timeline():
    E = pd.read_csv(C.OUT / "episodes.csv")
    keys = [("February", 1), ("February", 2), ("February", 3), ("February", 4),
            ("January", 3), ("January", 5), ("January", 8)]
    avail = {(m, s): [d for d in DEV
                      if ((W.month == m) & (W.seg == s) & (W.dev == d)).any()]
             for m, s in keys}

    rows = [(m, s, d) for m, s in keys for d in avail[(m, s)]]
    fig, ax = plt.subplots(figsize=(3.35, 3.4))
    ax.set_clip_on(False)
    ytick, ylab, y = [], [], 0
    sep = []
    for i, (m, s) in enumerate(keys):
        for d in avail[(m, s)]:
            ax.hlines(y, 0, 100, color="0.85", lw=3.2, zorder=1)
            e = E[(E.month == m) & (E.seg == s) & (E.dev == d)]
            for _, r in e.sort_values("start_pct").iterrows():
                x1 = max(r.end_pct, r.start_pct + 0.6)
                ax.hlines(y, r.start_pct, x1, color="0.05", lw=3.2, zorder=2)
            tot = float(e.dur_s.sum())
            ax.text(103, y, f"{tot:.0f}" if tot else "0", ha="left",
                    va="center", fontsize=6, color="0.25")
            ytick.append(y); ylab.append(LBL[d]); y -= 1
        sep.append(y + 0.5)
        y -= 0.6
    ax.set_yticks(ytick); ax.set_yticklabels(ylab)
    for yy in sep[:-1]:
        ax.axhline(yy, color="0.15", lw=0.5)
    for i, (m, s) in enumerate(keys):
        ys = [ytick[j] for j, r in enumerate(rows) if r[0] == m and r[1] == s]
        ax.text(-36, np.mean(ys), f"{m[:3]}\nseg {s}", ha="center", va="center",
                fontsize=6.5, color="0.25")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("position within segment [%]")
    ax.text(103, 1.0, "lost [s]", ha="left", va="center",
            fontsize=6, color="0.25")
    ax.set_ylim(y + 0.4, 0.8)
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[Line2D([], [], color="0.05", lw=3.2, label="tracking lost"),
                       Line2D([], [], color="0.85", lw=3.2, label="tracking held")],
              frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, handlelength=1.6)
    fig.savefig(str(OUT / "fig_failure_timeline.pdf"))
    fig.savefig(str(OUT / "fig_failure_timeline.png"), dpi=300)
    plt.close(fig)
    print("fig 2 done", len(rows), "rows")




# --------------------------------------------------------------- figure 3
def fig_roc():
    from sklearn.metrics import roc_curve, roc_auc_score
    y = np.load(C.OUT / "detector_y.npy")
    P = np.load(C.OUT / "detector_probs.npy")          # rows: brightness, photometric, full
    sets = [("Photometric $+$ semantic (66)", P[2], dict(color="0.00", ls="-",  lw=1.4)),
            ("Brightness alone",              P[0], dict(color="0.45", ls="--", lw=1.2))]
    assert len(y) == len(P[0])

    fig, ax = plt.subplots(figsize=(3.35, 3.0))
    ax.plot([0, 1], [0, 1], color="0.75", ls=":", lw=0.8, zorder=1)
    mk = {0.80: "o", 0.90: "s"}
    for name, p, st in sets:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name}, AUC {roc_auc_score(y, p):.3f}",
                zorder=3, **st)
        for sens in (0.80, 0.90):
            i = min(np.searchsorted(tpr, sens), len(tpr) - 1)
            ax.plot(fpr[i], tpr[i], marker=mk[sens], ms=4.5, mfc="white",
                    mec=st["color"], mew=1.1, ls="none", zorder=4)
            if st["ls"] == "-":
                off, ha = (7, -9), "left"
            elif fpr[i] < 0.3:
                off, ha = (-2, 20), "left"
            else:
                off, ha = (6, -9), "left"
            ax.annotate(f"spec. {(1 - fpr[i]) * 100:.0f}%", (fpr[i], tpr[i]),
                        textcoords="offset points", xytext=off, ha=ha,
                        fontsize=6, color=st["color"])
    ax.set_xlabel("1 $-$ specificity")
    ax.set_ylabel("sensitivity")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(0, 1.01, 0.25)); ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.grid(True, color="0.9", lw=0.4); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h = [Line2D([], [], marker="o", ls="none", ms=4.5, mfc="white", mec="0.2",
                label="80% sensitivity"),
         Line2D([], [], marker="s", ls="none", ms=4.5, mfc="white", mec="0.2",
                label="90% sensitivity")]
    leg1 = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02),
                     frameon=False, handlelength=1.8, fontsize=6.5,
                     borderaxespad=0.0, handletextpad=0.5)
    ax.add_artist(leg1)
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 0.20),
              frameon=False, handlelength=1.2, fontsize=6.5,
              borderaxespad=0.0, handletextpad=0.5)
    fig.savefig(str(OUT / "fig_roc_detector.pdf"))
    fig.savefig(str(OUT / "fig_roc_detector.png"), dpi=300)
    plt.close(fig)
    print("fig 3 done")


# --------------------------------------------------------------- figure 4
def fig_brightness():
    d = pd.read_pickle(C.OUT / "scene.pkl")
    b = d.Brightness.to_numpy(float)
    y = d.FAIL.to_numpy().astype(bool)
    ok = np.isfinite(b)
    b, y = b[ok], y[ok]
    bins = np.linspace(0, 180, 37)
    ctr = 0.5 * (bins[:-1] + bins[1:])

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(3.35, 3.5), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.15, 0.85],
                                                  hspace=0.12))

    a0.hist(b[~y], bins=bins, density=True, color="0.82", lw=0.5,
            edgecolor="0.6", label=f"tracking held ($n={int((~y).sum())}$)")
    a0.hist(b[y], bins=bins, density=True, histtype="step", color="0.00",
            lw=1.3, label=f"tracking lost ($n={int(y.sum())}$)")
    m0, m1 = np.median(b[~y]), np.median(b[y])
    a0.axvline(m1, color="0.00", lw=0.8, ls=(0, (1, 1.5)))
    a0.axvline(m0, color="0.45", lw=0.8, ls=(0, (1, 1.5)))
    a0.annotate(f"median {m1:.1f}", (m1, 1.0), xycoords=("data", "axes fraction"),
                xytext=(1, 3), textcoords="offset points", ha="left",
                va="bottom", fontsize=6, color="0.05", annotation_clip=False)
    a0.annotate(f"median {m0:.1f}", (m0, 1.0), xycoords=("data", "axes fraction"),
                xytext=(1, 3), textcoords="offset points", ha="left",
                va="bottom", fontsize=6, color="0.35", annotation_clip=False)
    a0.set_yscale("log"); a0.set_ylim(2e-5, 0.6)
    a0.set_ylabel("probability density")
    a0.legend(frameon=False, loc="upper right", fontsize=6.5)

    # conditional failure rate with Wilson intervals
    k, _ = np.histogram(b[y], bins=bins)
    n, _ = np.histogram(b, bins=bins)
    m = n >= 30
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    z = 1.96
    den = 1 + z**2 / n
    cen = (p + z**2 / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    a1.fill_between(ctr[m], np.clip(cen - hw, 0, 1)[m], np.clip(cen + hw, 0, 1)[m],
                    color="0.6", alpha=0.30, lw=0)
    a1.plot(ctr[m], p[m], color="0.00", lw=1.2, marker="o", ms=2.8)
    a1.axhline(y.mean(), color="0.45", lw=0.8, ls="--")
    a1.annotate(f"base rate {y.mean()*100:.1f}%", (30, y.mean()),
                ha="left", va="bottom", fontsize=6,
                textcoords="offset points", xytext=(0, 4), color="0.35")
    a1.set_ylim(0, 0.47); a1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    a1.set_ylabel("P(tracking lost)")
    a1.set_xlabel("mean scene brightness [8-bit levels]")
    a1.set_xlim(0, 180)
    for a in (a0, a1):
        a.grid(True, axis="y", color="0.92", lw=0.4); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.savefig(str(OUT / "fig_brightness_failure.pdf"))
    fig.savefig(str(OUT / "fig_brightness_failure.png"), dpi=300)
    plt.close(fig)
    print("fig 4 done  medians", round(m1, 2), round(m0, 2),
          "| P(loss) in darkest bin", round(float(p[0]), 3))


if __name__ == "__main__":
    fig_speed(); fig_timeline(); fig_roc(); fig_brightness()
