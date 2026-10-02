"""Score every frame we have ground truth for, write result/scores.csv.

    python run_all.py

Three independent sources of truth, none used to set anything in metrics.py:
  - two CellChek D exports, panel readout decoded from A*.dat
  - the DEKS eyebank set, Reader1, per-cell areas from A*.dat
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REPO = Path("/Users/jurimoon/PROJECTS/EC-Challenge")
AC = HERE.parent / "_actual check"
# HERE must stay first: experiments/_common also has a metrics.py
sys.path.append(str(AC))
sys.path.append(str(HERE.parent / "_DEKS-EYEBANK/_code"))
sys.path.append(str(REPO / "experiments/_common"))
sys.path.insert(0, str(HERE))
from metrics import analyse                                  # noqa: E402

UM = 0.7716049
MAG = UM ** 2 / (100 / 169)      # A*.dat areas are at the panel's mag 130


def rows_exports():
    from read_dat import read
    for lbl, fo in [("0323-23", "_dat 1-0323-23"), ("0324-23", "_dat 2-0324-23")]:
        d = read(AC / "analysis_field_bmp24" / fo)
        a = d["areas_um2"] * MAG
        yield ("exports", lbl, d["pts"], UM,
               dict(num=len(a), cd=1e6 / a.mean(), cv=a.std(ddof=0) / a.mean(),
                    hex=float((d["sides"] == 6).mean())))


def rows_deks():
    from deks_dat import read_image
    root = HERE.parent / "_DEKS-EYEBANK/Reader1"
    dat = next(p for p in root.iterdir() if p.name.startswith("DAT_"))
    for i in sorted([d.name for d in dat.iterdir() if d.is_dir()], key=int):
        for an in read_image(dat, i):
            a = an["areas_um2"]
            if a is None or len(a) < 20 or an["n_pts"] < 10:
                continue
            a = a.astype(float)
            yield ("deks", f"{i}.{an['k']}", an["pts"], UM,
                   dict(num=len(a), cd=1e6 / a.mean(), cv=a.std(ddof=0) / a.mean(),
                        hex=float((an["sides"] == 6).mean())))


def main() -> None:
    (HERE / "result").mkdir(exist_ok=True)
    out = []
    for gen in (rows_exports, rows_deks, rows_clear):
        for src, name, pts, um, gt in gen():
            r = analyse(pts, um)
            out.append(dict(source=src, frame=name, n_markers=len(pts),
                            num=r["num"], num_gt=gt["num"],
                            cd=r["cd"], cd_gt=gt["cd"],
                            cv=r["cv"], cv_gt=gt["cv"],
                            hex=r["hex"], hex_gt=gt["hex"]))
    with open(HERE / "result" / "scores.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    print(f"{'source':10}{'n':>4}{'':3}{'CD':>16}{'CV':>16}{'HEX':>16}")
    print(f"{'':17}" + "".join(f"{'bias':>8}{'MAPE':>8}" for _ in range(3)))
    for src in ("exports", "deks", "clear-ec"):
        g = [r for r in out if r["source"] == src]
        line = f"{src:10}{len(g):>4}{'':3}"
        for k in ("cd", "cv", "hex"):
            p = np.array([r[k] for r in g]); t = np.array([r[k + "_gt"] for r in g])
            e = (p - t) / t
            line += f"{100*e.mean():>+7.1f}%{100*np.abs(e).mean():>7.1f}%"
        print(line)
        n = np.array([r["num"] for r in g], float)
        tn = np.array([r["num_gt"] for r in g], float)
        if not np.isnan(tn).any():
            print(f"{'':10}Num exact {int((n == tn).sum())}/{len(g)}"
                  f"   within 2 {int((abs(n - tn) <= 2).sum())}/{len(g)}")
    print(f"\nwrote {HERE/'result'/'scores.csv'}  ({len(out)} frames)")


if __name__ == "__main__":
    main()
