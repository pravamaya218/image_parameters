# Conclusion 2 — CD, CV, HEX and Num from marked cell centres

Input is a set of marked cell centres and a pixel scale. Output is the four
numbers the CellChek D panel prints. Validated on **64 frames** across three
independent sources of truth, none of which was used to set anything here.

```python
from metrics import analyse
r = analyse(points_xy, um_per_px)    # -> num, ave_um2, cd, sd_um2, cv, hex
```

```
python run_all.py      # score every frame, write result/scores.csv
```

|  | CD | CV | HEX | frames |
| --- | ---: | ---: | ---: | ---: |
| CellChek D exports | **0.2%** | 6.1% | 3.7% | 2 |
| DEKS eyebank, Reader1 | **1.0%** | **5.4%** | **4.0%** | 37 |

MAPE. Num is exact on 2/2 and 18/37, and within two cells on 35/37.

## The method

**1 · Tessellate.** A plain Voronoi cell for every marker.

**2 · Peel.** Keep a cell only if *all* of its corners fall inside the convex
hull of every marker. Cells that cross it are dropped whole, area included —
never clipped. Then drop a survivor that is both within 2 px of the hull *and*
has three or more wall-neighbours peeled.

That second rule is what makes the count match: on the five frames where the
instrument printed its own `Num`, the hull test alone is off by +2 and +1 on
two of them, and the filter makes all five exact. It also explains the
instrument's refusal to run the Center method on fewer than six markers —
six is the minimum that encloses one cell, five around and one in the middle.

**3 · Num, Ave, CD and HEX, from the plain Voronoi.**

```
Num = number of cells kept
Ave = mean kept-cell area, µm²
CD  = 1e6 / round(Ave)
HEX = fraction of kept cells with exactly six walls
```

**4 · Re-tessellate with weights, for CV only.**

```
pitch = median nearest-neighbour distance between markers

w_i = pitch² · log(V_i / Ave)     markers inside the hull
w_i = 0                           markers on the rim

wall between i and j:  n·x = n·(p_i+p_j)/2 + (w_i−w_j)/2,   n = p_j − p_i
```

Equivalently: each wall moves off the midpoint toward the *smaller* cell by
`δ = (pitch/2)·log(V_i/V_j)`.

**5 · SD and CV, from the weighted cells.**

```
SD = sqrt( Σ (A_i − Ave)² / n )      divide by n, not n−1
CV = SD / Ave
```

The population divisor is not a guess. It reproduces the panel on both exports
(150.2 → 150, 154.8 → 155, where ÷n−1 gives 151 and 156), and it is confirmed
independently by a pentagon test: six markers enclose one cell, and the panel
printed `SD 0` where ÷(n−1) would divide by zero.

## Why the weights

The instrument's cell **topology is a plain Voronoi**. On 0323-23 its side
histogram is 1/27/58/18/3 against our 0/28/58/18/3, HEX 54% against 54%. It
reads the same neighbour graph we do.

What differs is where the walls sit. A Voronoi puts every wall at the midpoint
between two markers regardless of which cell is really larger. Working back
from the instrument's own per-cell areas, its walls sit about **0.7 µm off the
midpoint toward the smaller cell** — one membrane's width — which is 3.4% of a
cell, identical on both exports. That is a power (Laguerre) diagram.

```
w = 0.026 + 0.045(n−6) + 1.071·log(V/Ave)        R² 0.47
```

The coefficient on `log(V/Ave)` is 1 within measurement error, and hexagons come
out at weight **+0.007** — they are the pivot, unchanged. Both facts are used
verbatim; nothing was fitted to a CV target.

Muting the rim weights is likewise not a choice. A marker on the edge of the
marked patch has a one-sided neighbourhood, so its cell runs off unbounded and
`log(V/Ave)` hands it a large positive weight. Left alone it eats into the
interior and CV error goes from 8% to 48%.

