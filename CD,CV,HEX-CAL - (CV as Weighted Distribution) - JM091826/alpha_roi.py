"""The polygon the markers actually cover, instead of their convex hull.

`interior_mask` peels against the CONVEX hull of every marker. Where the marked
patch is concave -- an operator working around a fold, a dark area, a lesion --
the convex hull spans the gap and admits cells whose walls were never decided by
a neighbour. This builds the concave boundary instead: Delaunay triangles whose
longest edge exceeds `k` local spacings are not tissue, so they are dropped, and
what remains is the polygon the markers really occupy.

    from alpha_roi import alpha_polygon, inside_polygon
    poly = alpha_polygon(pts, k=2.2)          # list of (x, y) rings
"""
from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import Delaunay, cKDTree


def alpha_polygon(pts: np.ndarray, k: float = 2.2):
    """Boundary rings of the alpha shape. Falls back to the convex hull."""
    pts = np.asarray(pts, float)
    if len(pts) < 4:
        return [pts]
    tri = Delaunay(pts)
    pitch = np.median(cKDTree(pts).query(pts, k=2)[0][:, 1])
    lim = k * pitch
    good = []
    for s in tri.simplices:
        a, b, c = pts[s]
        if max(np.hypot(*(a - b)), np.hypot(*(b - c)), np.hypot(*(c - a))) <= lim:
            good.append(s)
    if not good:
        from scipy.spatial import ConvexHull
        return [pts[ConvexHull(pts).vertices]]
    # edges used by exactly one kept triangle are the boundary
    cnt = {}
    for s in good:
        for e in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
            key = (min(e), max(e))
            cnt[key] = cnt.get(key, 0) + 1
    edges = [e for e, n in cnt.items() if n == 1]
    if not edges:
        from scipy.spatial import ConvexHull
        return [pts[ConvexHull(pts).vertices]]
    nxt = {}
    for a, b in edges:
        nxt.setdefault(a, []).append(b)
        nxt.setdefault(b, []).append(a)
    rings, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        ring, cur, prev = [start], start, None
        seen.add(start)
        while True:
            opts = [v for v in nxt[cur] if v != prev]
            nxt_v = next((v for v in opts if v not in seen), None)
            if nxt_v is None:
                break
            ring.append(nxt_v)
            seen.add(nxt_v)
            prev, cur = cur, nxt_v
        if len(ring) >= 3:
            rings.append(pts[ring])
    rings.sort(key=lambda r: -_area(r))
    return rings or [pts]


def _area(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def inside_polygon(cell_list, rings, eps: float = 1e-7) -> np.ndarray:
    """True where every corner of the cell lies in the largest ring."""
    path = MplPath(rings[0])
    out = np.zeros(len(cell_list), bool)
    for i, c in enumerate(cell_list):
        if len(c) < 3:
            continue
        out[i] = bool(path.contains_points(c, radius=eps).all())
    return out
