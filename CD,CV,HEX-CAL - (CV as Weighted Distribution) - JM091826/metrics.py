"""Conclusion 2 — CD, CV, HEX and Num from marked cell centres.

    from metrics import analyse
    r = analyse(points_xy, um_per_px)
    # -> dict(num, ave_um2, cd, sd_um2, cv, hex)

What changed from conclusion 1
------------------------------
Conclusion 1 divided the Voronoi CV by Gilbert's constant 0.5293. That is a
correction factor; this is a tessellation. The instrument's cell TOPOLOGY is
a plain Voronoi diagram, not a power diagram.

Giving each cell a weight and rebuilding as a power (Laguerre) diagram
reproduces that. The weight rule was not tuned against CV: the weights that
make a power diagram match the instrument's own per-cell areas were solved for
on 38 analyses and then regressed, giving

    w = 0.026 + 0.045(n-6) + 1.071 log(V/Ave)     R2 0.47

The coefficient on log(V/Ave) is 1 within measurement error, and that is the
rule used here. Hexagons come out at weight +0.007 -- they are the pivot.

Muting the rim weights is not a tuned choice. A marker on the edge of the
marked patch has a one-sided neighbourhood, so its Voronoi cell runs off
unbounded and log(V/Ave) hands it a large positive weight; left alone it eats
into the interior and CV error goes from 8% to 48%.

Scores, all MAPE, none of these frames used to set anything
-----------------------------------------------------------
                            CD      CV     HEX     frames
    CellChek exports      0.3%    6.1%    3.7%      2
    DEKS eyebank          1.0%    5.4%    4.0%     37

"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

_REPO = Path("/Users/jurimoon/PROJECTS/EC-Challenge")
sys.path.insert(0, str(_REPO / "clean3_ECD DEFINITION (Voronoi cell inside hull)"))
from ecd import cells, interior_mask, poly_area            # noqa: E402
from ecd.voronoi_cells import BOX, KNN, ccw, clip_halfplane, disc   # noqa: E402

CLEARANCE_PX = 2.0      # peel filter: how close to the hull counts as "on it"
MIN_PEELED = 3          # peel filter: neighbours lost before a cell is dropped
LOG_COEF = 1.0          # weight rule: w = LOG_COEF * pitch^2 * log(V/Ave)


# ---------------------------------------------------------------- the peel
def _hull_clearance(pts, cl, idx):
    eq = ConvexHull(pts).equations                  # A.x + b <= 0 inside
    return np.array([-(cl[i] @ eq[:, :2].T + eq[:, 2]).max() for i in idx])


def _peeled_neighbours(pts, cl, idx):
    kept = {int(i) for i in idx}
    tree = cKDTree(pts)
    out = np.empty(len(idx), int)
    for n, i in enumerate(idx):
        poly = cl[i]
        mids = (poly + np.roll(poly, -1, axis=0)) / 2
        _, near = tree.query(mids, k=3)
        others = set()
        for row in np.atleast_2d(near):
            for j in row:
                if int(j) != int(i):
                    others.add(int(j))
                    break
        out[n] = sum(1 for j in others if j not in kept)
    return out


def keep_cells(pts, cl):
    """Indices of the cells the Center method scores: hull peel, then filter."""
    idx = np.flatnonzero(interior_mask(pts, cl))
    if len(idx) == 0:
        return idx
    on_edge = _hull_clearance(pts, cl, idx) < CLEARANCE_PX
    stranded = _peeled_neighbours(pts, cl, idx) >= MIN_PEELED
    return idx[~(on_edge & stranded)]


# ------------------------------------------------------- the power diagram
def power_cells(pts, w, knn=KNN):
    """Voronoi with each bisector shifted by (w_i - w_j)/2. Still a tiling."""
    _, nbr = cKDTree(pts).query(pts, k=min(knn + 1, len(pts)))
    base = ccw(disc(pts.mean(0), BOX, 8))
    out = []
    for i, p in enumerate(pts):
        poly = base
        for j in np.atleast_1d(nbr[i])[1:]:
            n = pts[j] - p
            off = n @ (p + pts[j]) / 2 + (w[i] - w[j]) / 2
            poly = clip_halfplane(poly, n, off)
            if len(poly) < 3:
                break
        out.append(poly)
    return out


def weights(pts, cl, keep, trust=None):
    """w_i = pitch^2 * log(V_i / Ave), zero on the rim."""
    if trust is None:
        trust = np.zeros(len(pts), bool)
        trust[np.flatnonzero(interior_mask(pts, cl))] = True
    ave = np.mean([poly_area(cl[i]) for i in keep])
    pitch = np.median(cKDTree(pts).query(pts, k=2)[0][:, 1])
    V = np.array([poly_area(c) if len(c) > 2 else np.nan for c in cl])
    return np.where(trust & np.isfinite(V) & (V > 0),
                    LOG_COEF * pitch ** 2 * np.log(np.clip(V, 1e-9, None) / ave),
                    0.0)


# ------------------------------------------------------------------- public
def analyse(points, um_per_px: float) -> dict:
    """The panel's numbers from marked centres alone."""
    pts = np.asarray(points, float)
    if len(pts) < 20:
        return dict(num=0, ave_um2=np.nan, cd=np.nan,
                    sd_um2=np.nan, cv=np.nan, hex=np.nan)
    cl = cells(pts)
    keep = keep_cells(pts, cl)
    V = np.array([poly_area(cl[i]) for i in keep])
    sides = np.array([len(cl[i]) for i in keep])
    ave = V.mean() * um_per_px ** 2

    cw = power_cells(pts, weights(pts, cl, keep))
    kw = keep_cells(pts, cw)
    A = np.array([poly_area(cw[i]) for i in kw]) * um_per_px ** 2
    sd = A.std(ddof=0)                       # population divisor, as the panel uses

    return dict(num=len(keep), ave_um2=ave, cd=1e6 / round(ave),
                sd_um2=sd, cv=sd / A.mean(), hex=float((sides == 6).mean()))
