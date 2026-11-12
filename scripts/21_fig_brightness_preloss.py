"""Figure 7 with the scene BEFORE a loss in the top panel.

Top: brightness distribution of normal windows, of windows centred 1-10 s before
the onset of an episode (the scene in which tracking was lost) and, for
reference, of windows inside a loss (largely the black passthrough view).
Bottom: P(window inside a loss | brightness) with Wilson intervals, as in
05_figures.fig_brightness. Classes as in 15_descriptors_preloss.py.
Writes out/figures/fig_brightness_failure.pdf/.png (replaces the stage-5 version)
and out/figures/fig_brightness_failure.csv.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reliability import config as C

spec = importlib.util.spec_from_file_location("pre", ROOT / "scripts" / "15_descriptors_preloss.py")
pre = importlib.util.module_from_spec(spec); spec.loader.exec_module(pre)
spec = importlib.util.spec_from_file_location("figs", ROOT / "scripts" / "05_figures.py")
figs = importlib.util.module_from_spec(spec); spec.loader.exec_module(figs)   # rcParams

OUT = C.OUT / "figures"


def main():
    S = pd.read_pickle(C.OUT / "scene.pkl").reset_index(drop=True)
    E = pd.read_csv(C.OUT / "episodes.csv")
    b_all = S.Brightness.to_numpy(float)
    y_all = S.FAIL.to_numpy().astype(bool)
    ok = np.isfinite(b_all)
    Sp = S[S.ui < 0.5].reset_index(drop=True)
    Sp["cls"] = pre.classes(Sp, E)
    bn = Sp.loc[Sp.cls == "normal", "Brightness"].to_numpy(float)
    bp = Sp.loc[Sp.cls == "pre", "Brightness"].to_numpy(float)
    bl = Sp.loc[Sp.cls == "in_loss", "Brightness"].to_numpy(float)
    bn, bp, bl = (x[np.isfinite(x)] for x in (bn, bp, bl))

    bins = np.linspace(0, 180, 37)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(3.35, 3.5), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.15, 0.85], hspace=0.12))
    a0.hist(bn, bins=bins, density=True, color="0.82", lw=0.5, edgecolor="0.6",
            label=f"normal operation ($n={len(bn)}$)")
    a0.hist(bp, bins=bins, density=True, histtype="step", color="0.00", lw=1.3,
            label=f"1–10 s before a loss ($n={len(bp)}$)")
    a0.hist(bl, bins=bins, density=True, histtype="step", color="0.45", lw=0.8,
            ls=(0, (2, 1.5)), label=f"inside a loss ($n={len(bl)}$)")
    a0.set_yscale("log"); a0.set_ylim(2e-5, 0.6)
    a0.set_ylabel("probability density")
    a0.legend(frameon=False, loc="upper right", fontsize=6.5)

    b, y = b_all[ok], y_all[ok]
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
    a1.annotate(f"base rate {y.mean()*100:.1f}%", (30, y.mean()), ha="left", va="bottom",
                fontsize=6, textcoords="offset points", xytext=(0, 4), color="0.35")
    a1.set_ylim(0, 0.47); a1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    a1.set_ylabel("P(inside a loss)")
    a1.set_xlabel("mean scene brightness [8-bit levels]")
    a1.set_xlim(0, 180)
    for a in (a0, a1):
        a.grid(True, axis="y", color="0.92", lw=0.4); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.savefig(str(OUT / "fig_brightness_failure.pdf"))
    fig.savefig(str(OUT / "fig_brightness_failure.png"), dpi=300)
    plt.close(fig)
    pd.DataFrame(dict(lo=bins[:-1], hi=bins[1:], n=n, k=k, p=p)).to_csv(
        OUT / "fig_brightness_failure.csv", index=False)
    print("n normal/pre/in_loss", len(bn), len(bp), len(bl),
          "| medians", np.median(bn).round(1), np.median(bp).round(1), np.median(bl).round(1))


if __name__ == "__main__":
    main()
