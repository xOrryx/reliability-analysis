# -----------------------------------------------------------------------------
# Stage 7: Table S7 of Online Resource 1, the repeatability measurement visit by
# visit, and the rebuilt Supplementary Information PDF.
#
# Author:  Petr Šopák <Petr.Sopak@vut.cz>
#          Brno University of Technology, CEITEC and FEEC
# Part of: Stability of Localization and Tracking on Meta Quest Devices:
#          Statistics, Sources of Failures and VR App for Identification
#          (supplementary repeatability measurement)
#
# Reads   out/repeatability/deviations.csv
#         out/supplement/ESM_1.tex   the six tables built by scripts/07_supplement.py
# Writes  out/repeatability/tex/tab_S7.tex
#         out/supplement/ESM_1.tex   with Table S7 appended
#         out/supplement/ESM_1.pdf   when pdflatex is available
#
# The main text quotes only the end point of the accumulation. This table gives
# every visit, so a reader can check that the deviation grows with the distance
# walked rather than scattering, which is the claim the section rests on and is
# not visible from a single final value.
# -----------------------------------------------------------------------------
"""Table S7: the repeatability measurement, visit by visit."""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from repeatability import config as C

SI = C.OUT.parent / "supplement"

HEAD = r"""
\section*{Table S7: Repeatability measurement, visit by visit}

Every return to a marked point in the supplementary measurement of
Section~3.8, in the order in which it occurred. The path is the distance walked
since the first visit to that mark, and the deviation is the displacement of the
reported position of that mark from where it was reported on the first visit,
split into its horizontal and vertical parts. The first visit to each mark is the
reference and is therefore zero by construction. The last column records how the
position of the visit was taken: from the median over an interval in which the
wearer stood still at the mark, or, where there was no such interval, from a
fixed window around the turnaround, which additionally carries however far the
wearer walked past the mark before pivoting. The main text quotes only the final
value of each accumulation; this table shows that the deviation grows with the
distance walked rather than scattering about a fixed value.

{\footnotesize\setlength{\tabcolsep}{5pt}
\begin{longtable}{llrrrrrl}
\caption{Repeatability measurement, every visit to a marked point.}\label{tab:S7}\\
\toprule
Device & Mark & Visit & Path [m] & Horiz. [m] & Vert. [m] & Total [m] & Position from \\
\midrule
\endfirsthead
\multicolumn{8}{l}{\footnotesize\itshape continued from the previous page}\\
\toprule
Device & Mark & Visit & Path [m] & Horiz. [m] & Vert. [m] & Total [m] & Position from \\
\midrule
\endhead
\midrule
\multicolumn{8}{r}{\footnotesize\itshape continued on the next page}\\
\endfoot
\bottomrule
\endlastfoot
"""

FOOT = "\\end{longtable}\n}\n"

SOURCE_LABEL = {"stand": "stand", "turnaround": "turnaround"}


def rows(D: pd.DataFrame) -> str:
    order = {d: i for i, d in enumerate(C.DEVICES)}
    D = D.assign(_o=D.device.map(order)).sort_values(["_o", "end", "visit"])
    out, prev = [], None
    for _, r in D.iterrows():
        key = (r.device, r.end)
        if prev is not None and key != prev:
            out.append(r"\addlinespace")
        prev = key
        out.append(
            f"{C.DEVICE_LABEL.get(r.device, r.device).replace(' ', '~')} & {r.end} & "
            f"{int(r.visit)} & {r.path_since_ref_m:.0f} & {r.horizontal_m:.2f} & "
            f"{r.vertical_m:+.2f} & {r.total_m:.2f} & "
            f"{SOURCE_LABEL.get(r.source, r.source)} \\\\")
    return "\n".join(out) + "\n"


def main():
    D = pd.read_csv(C.OUT / "deviations.csv")
    table = HEAD + rows(D) + FOOT

    (C.OUT / "tex").mkdir(parents=True, exist_ok=True)
    (C.OUT / "tex" / "tab_S7.tex").write_text(table, encoding="utf-8")
    print(f"wrote {C.OUT / 'tex' / 'tab_S7.tex'}  ({len(D)} visits)")

    tex = SI / "ESM_1.tex"
    if not tex.exists():
        print(f"{tex} not found, so Online Resource 1 was not rebuilt; "
              f"run scripts/07_supplement.py first.")
        return

    body = tex.read_text(encoding="utf-8")
    if r"\label{tab:S7}" in body:
        body = body[:body.index(r"\section*{Table S7")] + "\n" + r"\end{document}"
    body = body.replace(
        "and a summary of the controller excursions. Each table is referenced from the\nmain text.",
        "a summary of the controller excursions, and the repeatability measurement of\n"
        "Section~3.8 visit by visit. Each table is referenced from the main text.")
    body = body.replace(r"\end{document}", table + "\n" + r"\end{document}")
    tex.write_text(body, encoding="utf-8")
    print(f"appended Table S7 to {tex}")

    if shutil.which("pdflatex") is None:
        print("pdflatex not on PATH, so ESM_1.pdf was not rebuilt.")
        return
    for _ in range(2):        # longtable needs two passes to settle its widths
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "ESM_1.tex"],
                       cwd=SI, check=False, stdout=subprocess.DEVNULL)
    print(f"rebuilt {SI / 'ESM_1.pdf'}")


if __name__ == "__main__":
    main()
