# Path guiding: online 3D voxel grid with per-cell directional energy histograms.
# 16^3 spatial cells × 64 directional bins (8×8 equal-area octahedral) = 1 MB.
# Within-bin jitter in guide_sample gives sub-bin angular resolution.
# Data races during parallel tile rendering are benign — float non-atomicity causes
# small errors in individual bins, but aggregate energy converges with enough samples.

from std.memory import alloc
from std.math import sqrt, cos, sin, max, min
from .geometry import Point3f, Bounds3f, _is_real_ptr
from .bvh import _equal_area_square_to_sphere, _equal_area_sphere_to_square

comptime GUIDE_DIMS: Int = 16
comptime GUIDE_BINS: Int = 64    # 8 × 8 equal-area octahedral bins
comptime GUIDE_AXIS: Int = 8     # bins per axis in octahedral map
comptime GUIDE_CELLS: Int = GUIDE_DIMS * GUIDE_DIMS * GUIDE_DIMS  # 4096

# 4π in Float32 — each equal-area bin covers exactly 4π/GUIDE_BINS steradians.
comptime FOUR_PI_F: Float32 = 12.566370614359172953950

@fieldwise_init
struct GuideGrid(TrivialRegisterPassable):
    var energy: UnsafePointer[Float32, MutAnyOrigin]  # [GUIDE_CELLS × GUIDE_BINS]
    var bounds: Bounds3f

def null_guide() -> GuideGrid:
    """Sentinel GuideGrid. Guard with _is_real_ptr(g.energy) to tell an active
    guide from this disabled placeholder."""
    return GuideGrid(
        UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
        Bounds3f(Point3f(Float32(0)), Point3f(Float32(1), Float32(1), Float32(1))),
    )

def guide_create(bounds: Bounds3f) -> GuideGrid:
    var n = GUIDE_CELLS * GUIDE_BINS
    var e = alloc[Float32](n)
    for i in range(n):
        e[i] = Float32(0)
    return GuideGrid(e, bounds)

def guide_free(g: GuideGrid):
    if _is_real_ptr(g.energy):
        g.energy.free()

def guide_merge(dst: GuideGrid, src: GuideGrid):
    """Add src's energy histogram into dst's. Used to merge per-thread guide grids."""
    var n = GUIDE_CELLS * GUIDE_BINS
    for i in range(n):
        dst.energy[i] += src.energy[i]

# ── Spatial cell lookup ────────────────────────────────────────────────────────

@always_inline
def guide_pos_to_cell(g: GuideGrid, p: Point3f) -> Int:
    """Map a world-space position to a cell index. Returns -1 if outside AABB."""
    var ext = g.bounds.max - g.bounds.min
    if ext.x <= Float32(0) or ext.y <= Float32(0) or ext.z <= Float32(0):
        return -1
    var f = (p - g.bounds.min)
    var fx = f.x / ext.x; var fy = f.y / ext.y; var fz = f.z / ext.z
    if fx < Float32(0) or fx >= Float32(1) or fy < Float32(0) or fy >= Float32(1) or fz < Float32(0) or fz >= Float32(1):
        return -1
    var cx = min(Int(fx * Float32(GUIDE_DIMS)), GUIDE_DIMS - 1)
    var cy = min(Int(fy * Float32(GUIDE_DIMS)), GUIDE_DIMS - 1)
    var cz = min(Int(fz * Float32(GUIDE_DIMS)), GUIDE_DIMS - 1)
    return cz * GUIDE_DIMS * GUIDE_DIMS + cy * GUIDE_DIMS + cx

# ── Directional binning (equal-area octahedral) ───────────────────────────────
# Uses bvh.mojo's _equal_area_square_to_sphere/_equal_area_sphere_to_square —
# THE single shared implementation (see that function's docstring). This file
# used to inline its own 3rd copy of the same math, which is exactly how a
# real bug (a wrong extra branch computing the wrong angle for ~half of all
# directions) survived undetected here even after being found and fixed in
# one of the other two copies.

@always_inline
def guide_dir_to_bin(dx: Float32, dy: Float32, dz: Float32) -> Int:
    var uv = _equal_area_sphere_to_square(dx, dy, dz)
    var ui = min(Int(uv[0] * Float32(GUIDE_AXIS)), GUIDE_AXIS - 1)
    var vi = min(Int(uv[1] * Float32(GUIDE_AXIS)), GUIDE_AXIS - 1)
    return vi * GUIDE_AXIS + ui

@always_inline
def guide_bin_to_dir(bin_idx: Int) -> SIMD[DType.float32, 3]:
    var vi = bin_idx // GUIDE_AXIS
    var ui = bin_idx % GUIDE_AXIS
    var u = (Float32(ui) + Float32(0.5)) / Float32(GUIDE_AXIS)
    var v = (Float32(vi) + Float32(0.5)) / Float32(GUIDE_AXIS)
    return _equal_area_square_to_sphere(u, v)

# ── Record / query ─────────────────────────────────────────────────────────────

@always_inline
def guide_record(
    g: GuideGrid, cell_idx: Int,
    dx: Float32, dy: Float32, dz: Float32, weight: Float32,
):
    """Accumulate energy from direction (dx,dy,dz) into the cell's histogram."""
    var bin = guide_dir_to_bin(dx, dy, dz)
    g.energy[cell_idx * GUIDE_BINS + bin] += weight

@always_inline
def guide_cell_has_data(g: GuideGrid, cell_idx: Int) -> Bool:
    """Returns True when the guide has sufficient concentrated data in this cell.
    Requires total energy >= 1e-4 AND the dominant bin holds >= 5% of energy.
    A pure-noise guide (all bins equal: max_bin ~ 1/GUIDE_BINS ≈ 1.5%) fails this,
    preventing MIS from inflating BSDF weights on uninformative cells."""
    if cell_idx < 0:
        return False
    var base = cell_idx * GUIDE_BINS
    var total = Float32(0)
    var max_e = Float32(0)
    for i in range(GUIDE_BINS):
        var e = g.energy[base + i]
        total += e
        if e > max_e:
            max_e = e
    if total < Float32(1e-4):
        return False
    return max_e / total > Float32(0.15)

@always_inline
def _guide_cell_total(energy: UnsafePointer[Float32, MutAnyOrigin], cell_idx: Int) -> Float32:
    var base = cell_idx * GUIDE_BINS
    var total = Float32(0)
    for i in range(GUIDE_BINS):
        total += energy[base + i]
    return total

def guide_pdf(g: GuideGrid, cell_idx: Int, dx: Float32, dy: Float32, dz: Float32) -> Float32:
    """PDF (per steradian) for direction (dx,dy,dz) at cell_idx.
    Falls back to uniform (1/4π) when the cell has no recorded energy."""
    var total = _guide_cell_total(g.energy, cell_idx)
    if total < Float32(1e-6):
        return Float32(1.0) / FOUR_PI_F
    var bin = guide_dir_to_bin(dx, dy, dz)
    var p_bin = g.energy[cell_idx * GUIDE_BINS + bin] / total
    return max(p_bin * Float32(GUIDE_BINS) / FOUR_PI_F, Float32(1e-7))

def guide_sample(
    g: GuideGrid, cell_idx: Int, u: Float32, u2: Float32 = Float32(0.5),
) -> Tuple[Float32, Float32, Float32, Float32, Bool]:
    """Sample a direction from the guide distribution at cell_idx.
    u: bin selection. u2: within-bin jitter (uniform [0,1]).
    Returns (dx, dy, dz, pdf, ok): ok=False when cell has no energy (fall back to BSDF)."""
    var total = _guide_cell_total(g.energy, cell_idx)
    if total < Float32(1e-6):
        return (Float32(0), Float32(0), Float32(1), Float32(1.0) / FOUR_PI_F, False)
    var target = u * total
    var cum = Float32(0)
    var chosen = GUIDE_BINS - 1
    var base = cell_idx * GUIDE_BINS
    for i in range(GUIDE_BINS):
        cum += g.energy[base + i]
        if cum >= target:
            chosen = i
            break
    var p_bin = g.energy[base + chosen] / total
    var pdf = max(p_bin * Float32(GUIDE_BINS) / FOUR_PI_F, Float32(1e-7))
    # Sample uniformly within the bin's rectangle in octahedral UV space.
    # u2 drives the u-axis jitter; v-axis jitter is derived via golden-ratio offset
    # to avoid correlation between the two axes from a single random number.
    var vi = chosen // GUIDE_AXIS
    var ui = chosen % GUIDE_AXIS
    var v2 = u2 * Float32(2.6180339887)
    v2 = v2 - Float32(Int(v2))
    var u_in = (Float32(ui) + u2) / Float32(GUIDE_AXIS)
    var v_in = (Float32(vi) + v2) / Float32(GUIDE_AXIS)
    var dir = _equal_area_square_to_sphere(u_in, v_in)
    return (dir[0], dir[1], dir[2], pdf, True)
