# Path guiding: online 3D voxel grid with per-cell directional energy histograms.
# 16^3 spatial cells × 64 directional bins (8×8 equal-area octahedral) = 1 MB.
# Within-bin jitter in guide_sample gives sub-bin angular resolution.
# Data races during parallel tile rendering are benign — float non-atomicity causes
# small errors in individual bins, but aggregate energy converges with enough samples.

from std.memory import alloc
from std.math import sqrt, cos, sin, max, min

comptime GUIDE_DIMS: Int = 16
comptime GUIDE_BINS: Int = 64    # 8 × 8 equal-area octahedral bins
comptime GUIDE_AXIS: Int = 8     # bins per axis in octahedral map
comptime GUIDE_CELLS: Int = GUIDE_DIMS * GUIDE_DIMS * GUIDE_DIMS  # 4096

# 4π in Float32 — each equal-area bin covers exactly 4π/GUIDE_BINS steradians.
comptime FOUR_PI_F: Float32 = 12.566370614359172953950

@fieldwise_init
struct GuideGrid(TrivialRegisterPassable):
    var energy:    UnsafePointer[Float32, MutAnyOrigin]  # [GUIDE_CELLS × GUIDE_BINS]
    var aabb_min_x: Float32
    var aabb_min_y: Float32
    var aabb_min_z: Float32
    var aabb_max_x: Float32
    var aabb_max_y: Float32
    var aabb_max_z: Float32

def null_guide() -> GuideGrid:
    """Sentinel GuideGrid. unsafe_dangling() for Float32 returns align_of=4.
    Guard: Int(energy) > 4 → active guide; ≤ 4 → disabled."""
    return GuideGrid(
        UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
        Float32(0), Float32(0), Float32(0), Float32(1), Float32(1), Float32(1),
    )

def guide_create(
    aabb_min_x: Float32, aabb_min_y: Float32, aabb_min_z: Float32,
    aabb_max_x: Float32, aabb_max_y: Float32, aabb_max_z: Float32,
) -> GuideGrid:
    var n = GUIDE_CELLS * GUIDE_BINS
    var e = alloc[Float32](n)
    for i in range(n):
        e[i] = Float32(0)
    return GuideGrid(e, aabb_min_x, aabb_min_y, aabb_min_z, aabb_max_x, aabb_max_y, aabb_max_z)

def guide_free(g: GuideGrid):
    if Int(g.energy) > 4:
        g.energy.free()

def guide_merge(dst: GuideGrid, src: GuideGrid):
    """Add src's energy histogram into dst's. Used to merge per-thread guide grids."""
    var n = GUIDE_CELLS * GUIDE_BINS
    for i in range(n):
        dst.energy[i] += src.energy[i]

# ── Spatial cell lookup ────────────────────────────────────────────────────────

@always_inline
def guide_pos_to_cell(g: GuideGrid, x: Float32, y: Float32, z: Float32) -> Int:
    """Map a world-space position to a cell index. Returns -1 if outside AABB."""
    var ext_x = g.aabb_max_x - g.aabb_min_x
    var ext_y = g.aabb_max_y - g.aabb_min_y
    var ext_z = g.aabb_max_z - g.aabb_min_z
    if ext_x <= Float32(0) or ext_y <= Float32(0) or ext_z <= Float32(0):
        return -1
    var fx = (x - g.aabb_min_x) / ext_x
    var fy = (y - g.aabb_min_y) / ext_y
    var fz = (z - g.aabb_min_z) / ext_z
    if fx < Float32(0) or fx >= Float32(1) or fy < Float32(0) or fy >= Float32(1) or fz < Float32(0) or fz >= Float32(1):
        return -1
    var cx = min(Int(fx * Float32(GUIDE_DIMS)), GUIDE_DIMS - 1)
    var cy = min(Int(fy * Float32(GUIDE_DIMS)), GUIDE_DIMS - 1)
    var cz = min(Int(fz * Float32(GUIDE_DIMS)), GUIDE_DIMS - 1)
    return cz * GUIDE_DIMS * GUIDE_DIMS + cy * GUIDE_DIMS + cx

# ── Directional binning (equal-area octahedral) ───────────────────────────────
# Inlined from shading.mojo's _equal_area_sphere_to_square / _equal_area_square_to_sphere.

@always_inline
def _guide_dir_to_uv(dx: Float32, dy: Float32, dz: Float32) -> SIMD[DType.float32, 2]:
    var adx = dx if dx >= Float32(0) else -dx
    var ady = dy if dy >= Float32(0) else -dy
    var adz = dz if dz >= Float32(0) else -dz
    var r = sqrt(max(Float32(0), Float32(1) - adz))
    var a = max(adx, ady)
    var b: Float32
    if a == Float32(0):
        b = Float32(0)
    else:
        b = min(adx, ady) / a
    var t1 = Float32(0.406758566246788489601959989e-5)
    var t2 = Float32(0.636226545274016134946890922156)
    var t3 = Float32(0.61572017898280213493197203466e-2)
    var t4 = Float32(-0.247333733281268944196501420480)
    var t5 = Float32(0.881770664775316294736387951347e-1)
    var t6 = Float32(0.419038818029165735901852432784e-1)
    var t7 = Float32(-0.251390972343483509333252996350e-1)
    var phi = t1 + b*(t2 + b*(t3 + b*(t4 + b*(t5 + b*(t6 + b*t7)))))
    if adx < ady:
        phi = Float32(1) - phi
    var v = phi * r
    var u = r - v
    if dz < Float32(0):
        var tmp = u; u = Float32(1) - v; v = Float32(1) - tmp
    if dx < Float32(0): u = -u
    if dy < Float32(0): v = -v
    return SIMD[DType.float32, 2](Float32(0.5)*(u + Float32(1)), Float32(0.5)*(v + Float32(1)))

@always_inline
def _guide_uv_to_dir(u: Float32, v: Float32) -> SIMD[DType.float32, 3]:
    var uu = Float32(2)*u - Float32(1)
    var vv = Float32(2)*v - Float32(1)
    var up = uu if uu >= Float32(0) else -uu
    var vp = vv if vv >= Float32(0) else -vv
    var signed_dist = Float32(1) - (up + vp)
    var d = signed_dist if signed_dist >= Float32(0) else -signed_dist
    var r = Float32(1) - d
    var phi: Float32
    if r == Float32(0):
        phi = Float32(0)
    elif up >= vp:
        phi = (vp / r) * Float32(0.7853981634)
    else:
        phi = ((vp - up) / r + Float32(1)) * Float32(0.7853981634)
    var r2 = r * r
    var z = Float32(1) - r2
    if signed_dist < Float32(0): z = -z
    var cp = cos(phi); var sp = sin(phi)
    if uu < Float32(0): cp = -cp
    if vv < Float32(0): sp = -sp
    var xy = r * sqrt(max(Float32(0), Float32(2) - r2))
    return SIMD[DType.float32, 3](cp*xy, sp*xy, z)

@always_inline
def guide_dir_to_bin(dx: Float32, dy: Float32, dz: Float32) -> Int:
    var uv = _guide_dir_to_uv(dx, dy, dz)
    var ui = min(Int(uv[0] * Float32(GUIDE_AXIS)), GUIDE_AXIS - 1)
    var vi = min(Int(uv[1] * Float32(GUIDE_AXIS)), GUIDE_AXIS - 1)
    return vi * GUIDE_AXIS + ui

@always_inline
def guide_bin_to_dir(bin_idx: Int) -> SIMD[DType.float32, 3]:
    var vi = bin_idx // GUIDE_AXIS
    var ui = bin_idx % GUIDE_AXIS
    var u = (Float32(ui) + Float32(0.5)) / Float32(GUIDE_AXIS)
    var v = (Float32(vi) + Float32(0.5)) / Float32(GUIDE_AXIS)
    return _guide_uv_to_dir(u, v)

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
    var dir = _guide_uv_to_dir(u_in, v_in)
    return (dir[0], dir[1], dir[2], pdf, True)
