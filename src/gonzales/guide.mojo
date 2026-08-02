# Path guiding: adaptive SD-tree (Müller et al. 2017, "Practical Path
# Guiding for Efficient Light-Transport Simulation"). Replaces the old
# fixed 16^3 spatial grid x 64-bin directional histogram.
#
# Spatial tree: a kd-tree over world-space bounds (SNode array), splitting
# a leaf's longest axis at its midpoint once enough samples have landed in
# it during one training iteration. Directional tree: a quadtree (DNode
# array) over the SAME equal-area octahedral [0,1]^2 square the old grid
# used (bvh.mojo's _equal_area_square_to_sphere/_equal_area_sphere_to_square)
# -- one independent quadtree per spatial leaf, splitting a node when its
# energy fraction of that leaf's total exceeds a threshold. A DNode's
# `energy` is its whole subtree's total (leaves hold directly-recorded
# energy; interior nodes hold the sum of their 4 children), maintained by
# incrementing every node on the root-to-leaf path in guide_record -- this
# is what lets guide_sample do an O(depth) 4-way descent instead of an
# O(bins) linear scan.
#
# Energy accumulates CUMULATIVELY across training iterations -- it is a
# stationary Monte Carlo estimate of the (unchanging) radiance
# distribution, same reasoning as the film's own sample accumulation, so
# guide_refine never resets it. Only each spatial leaf's sample_count
# resets between iterations (see guide_refine) -- it is deliberately a
# per-iteration statistic driving the NEXT refine's spatial-split decision,
# not a cumulative one.
#
# Both trees are STRUCTURALLY FROZEN during a rendering pass -- only leaf
# (and ancestor) energy floats and spatial sample_count are mutated by
# concurrent tile threads. Data races there are benign (same tolerance the
# old flat grid documented: float non-atomicity causes small errors that
# converge out with enough samples) PROVIDED the tree shape itself never
# changes concurrently with reads/writes. guide_refine (which does change
# shape) must only ever run single-threaded, strictly between passes.

from std.memory import alloc
from std.math import sqrt, cos, sin, max, min
from .geometry import Point3f, Bounds3f, _is_real_ptr
from .bvh import _equal_area_square_to_sphere, _equal_area_sphere_to_square

# 4π in Float32 — total solid angle of the unit sphere, which the equal-area
# octahedral map spreads uniformly over the [0,1]^2 square: a quadtree leaf
# at depth d covers exactly FOUR_PI_F / 4^d steradians.
comptime FOUR_PI_F: Float32 = 12.566370614359172953950

comptime SPATIAL_MAX_DEPTH: Int32 = 16          # cap on kd-tree depth
comptime DIR_MAX_DEPTH: Int32 = 5               # cap on per-leaf quadtree depth (≤ 4^5 = 1024 leaves)
comptime SPATIAL_SPLIT_SAMPLES: Int32 = 12000   # Müller's default spatial-subdivision sample threshold
comptime DIR_SPLIT_FRACTION: Float32 = 0.01     # Müller's default directional-subdivision energy fraction (ρ)

@fieldwise_init
struct SNode(TrivialRegisterPassable):
    """kd-tree node. Leaf iff child0 < 0 (child0/child1 always both -1 or
    both >= 0 together)."""
    var split_axis:   Int32    # 0=x, 1=y, 2=z; -1 on a leaf
    var split_pos:    Float32  # world-space split coordinate along split_axis (interior only)
    var depth:        Int32
    var child0:       Int32    # "< split_pos" child (interior only)
    var child1:       Int32    # ">= split_pos" child (interior only)
    var dtree_root:   Int32    # index into the shared dnodes array (leaf only)
    var sample_count: Int32    # samples routed here THIS iteration (leaf only); refine input, reset by guide_refine

@fieldwise_init
struct DNode(TrivialRegisterPassable):
    """Quadtree node over the octahedral [0,1)^2 square. Leaf iff
    child0 < 0. Quadrant order: 0=(u<.5,v<.5) 1=(u>=.5,v<.5)
    2=(u<.5,v>=.5) 3=(u>=.5,v>=.5)."""
    var energy: Float32   # subtree total: own recorded energy (leaf) or sum of children (interior)
    var depth:  Int32
    var child0: Int32
    var child1: Int32
    var child2: Int32
    var child3: Int32

@fieldwise_init
struct GuideGrid(TrivialRegisterPassable):
    """Handle into externally-owned, flat node arrays. Cheap to copy by
    value (matches every other pointer+metadata handle in this codebase);
    see file header for the structural-freeze invariant that makes that
    safe during a render pass."""
    var snodes:   UnsafePointer[SNode, MutAnyOrigin]
    var n_snodes: Int32
    var dnodes:   UnsafePointer[DNode, MutAnyOrigin]
    var n_dnodes: Int32
    var bounds:   Bounds3f

def null_guide() -> GuideGrid:
    """Sentinel GuideGrid. Guard with guide_is_active(g) to tell an active
    guide from this disabled placeholder."""
    return GuideGrid(
        snodes=UnsafePointer[SNode, MutAnyOrigin].unsafe_dangling(),
        n_snodes=Int32(0),
        dnodes=UnsafePointer[DNode, MutAnyOrigin].unsafe_dangling(),
        n_dnodes=Int32(0),
        bounds=Bounds3f(Point3f(Float32(0)), Point3f(Float32(1), Float32(1), Float32(1))),
    )

@always_inline
def guide_is_active(g: GuideGrid) -> Bool:
    """True for a real (non-null_guide) GuideGrid."""
    return _is_real_ptr(g.snodes)

def guide_create(bounds: Bounds3f) -> GuideGrid:
    """A fresh SD-tree: one spatial leaf covering `bounds`, whose
    directional quadtree is a single leaf covering the whole sphere.
    Zero energy/sample_count everywhere -- ready to record into directly."""
    var snodes = alloc[SNode](1)
    var dnodes = alloc[DNode](1)
    snodes[0] = SNode(split_axis=Int32(-1), split_pos=Float32(0), depth=Int32(0),
                       child0=Int32(-1), child1=Int32(-1), dtree_root=Int32(0), sample_count=Int32(0))
    dnodes[0] = DNode(energy=Float32(0), depth=Int32(0),
                       child0=Int32(-1), child1=Int32(-1), child2=Int32(-1), child3=Int32(-1))
    return GuideGrid(snodes=snodes, n_snodes=Int32(1), dnodes=dnodes, n_dnodes=Int32(1), bounds=bounds)

def guide_free(g: GuideGrid):
    if _is_real_ptr(g.snodes):
        g.snodes.free()
    if _is_real_ptr(g.dnodes):
        g.dnodes.free()

def guide_clone_empty(g: GuideGrid) -> GuideGrid:
    """Copy g's current tree STRUCTURE (spatial splits, directional splits,
    depths) into a new, independently-owned GuideGrid with every
    energy/sample_count field zeroed -- a fresh accumulator for one
    parallel shard's or one training iteration's own samples, meant to be
    guide_merge'd into the real, cumulative tree afterward. Never use the
    result as guide_read directly -- it has no distribution to sample from
    yet."""
    var snodes = alloc[SNode](Int(g.n_snodes))
    var dnodes = alloc[DNode](Int(g.n_dnodes))
    for i in range(Int(g.n_snodes)):
        var n = g.snodes[i]
        snodes[i] = SNode(split_axis=n.split_axis, split_pos=n.split_pos, depth=n.depth,
                           child0=n.child0, child1=n.child1, dtree_root=n.dtree_root, sample_count=Int32(0))
    for i in range(Int(g.n_dnodes)):
        var n = g.dnodes[i]
        dnodes[i] = DNode(energy=Float32(0), depth=n.depth,
                           child0=n.child0, child1=n.child1, child2=n.child2, child3=n.child3)
    return GuideGrid(snodes=snodes, n_snodes=g.n_snodes, dnodes=dnodes, n_dnodes=g.n_dnodes, bounds=g.bounds)

def guide_merge(dst: GuideGrid, src: GuideGrid):
    """Add src's energy/sample_count into dst's, elementwise. Requires src
    and dst share the SAME tree structure (guaranteed: both descend from
    guide_clone_empty of the same tree, or dst is that tree itself
    absorbing an already-merged shard of matching shape)."""
    for i in range(Int(dst.n_dnodes)):
        dst.dnodes[i].energy += src.dnodes[i].energy
    for i in range(Int(dst.n_snodes)):
        if dst.snodes[i].child0 < Int32(0):
            dst.snodes[i].sample_count += src.snodes[i].sample_count

# ── Spatial leaf lookup ─────────────────────────────────────────────────────

@always_inline
def guide_pos_to_cell(g: GuideGrid, p: Point3f) -> Int:
    """Map a world-space position to a spatial leaf index (an SNode array
    index). Returns -1 if outside the guide's AABB. Half-open convention:
    [min, max) on every axis."""
    var ext = g.bounds.max - g.bounds.min
    if ext.x <= Float32(0) or ext.y <= Float32(0) or ext.z <= Float32(0):
        return -1
    if p.x < g.bounds.min.x or p.x >= g.bounds.max.x:
        return -1
    if p.y < g.bounds.min.y or p.y >= g.bounds.max.y:
        return -1
    if p.z < g.bounds.min.z or p.z >= g.bounds.max.z:
        return -1
    var idx = 0
    while g.snodes[idx].child0 >= Int32(0):
        var n = g.snodes[idx]
        var coord: Float32
        if n.split_axis == Int32(0):
            coord = p.x
        elif n.split_axis == Int32(1):
            coord = p.y
        else:
            coord = p.z
        idx = Int(n.child0) if coord < n.split_pos else Int(n.child1)
    return idx

# ── Directional quadtree descent (equal-area octahedral) ────────────────────
# Uses bvh.mojo's _equal_area_square_to_sphere/_equal_area_sphere_to_square —
# THE single shared implementation (see that function's docstring; guide.mojo
# used to inline its own 3rd copy of the same math, which is exactly how a
# real bug survived undetected here even after being fixed elsewhere).

@always_inline
def _dtree_leaf_for_uv(g: GuideGrid, droot: Int32, u_in: Float32, v_in: Float32) -> Int32:
    """Descend from droot to the quadtree leaf containing UV point
    (u_in, v_in) in [0,1)^2. Returns the leaf's node index."""
    var u = u_in
    var v = v_in
    var idx = droot
    while g.dnodes[Int(idx)].child0 >= Int32(0):
        var n = g.dnodes[Int(idx)]
        if u < Float32(0.5):
            if v < Float32(0.5):
                idx = n.child0; u = u * Float32(2); v = v * Float32(2)
            else:
                idx = n.child2; u = u * Float32(2); v = (v - Float32(0.5)) * Float32(2)
        else:
            if v < Float32(0.5):
                idx = n.child1; u = (u - Float32(0.5)) * Float32(2); v = v * Float32(2)
            else:
                idx = n.child3; u = (u - Float32(0.5)) * Float32(2); v = (v - Float32(0.5)) * Float32(2)
    return idx

@always_inline
def _four_pow(depth: Int32) -> Float32:
    var r = Float32(1.0)
    for _ in range(Int(depth)):
        r *= Float32(4.0)
    return r

# ── Record / query ─────────────────────────────────────────────────────────

@always_inline
def guide_record(
    g: GuideGrid, cell_idx: Int,
    dx: Float32, dy: Float32, dz: Float32, weight: Float32,
):
    """Accumulate energy from direction (dx,dy,dz) into cell_idx's (a
    spatial leaf's) directional quadtree -- incrementing the found leaf AND
    every ancestor up to that quadtree's root, so an interior node's energy
    always equals its subtree's total (required by guide_sample's descent).
    Also bumps the spatial leaf's per-iteration sample_count."""
    if cell_idx < 0:
        return
    g.snodes[cell_idx].sample_count += Int32(1)
    var droot = g.snodes[cell_idx].dtree_root
    var uv = _equal_area_sphere_to_square(dx, dy, dz)
    var u = uv[0]
    var v = uv[1]
    var idx = droot
    g.dnodes[Int(idx)].energy += weight
    while g.dnodes[Int(idx)].child0 >= Int32(0):
        var n = g.dnodes[Int(idx)]
        if u < Float32(0.5):
            if v < Float32(0.5):
                idx = n.child0; u = u * Float32(2); v = v * Float32(2)
            else:
                idx = n.child2; u = u * Float32(2); v = (v - Float32(0.5)) * Float32(2)
        else:
            if v < Float32(0.5):
                idx = n.child1; u = (u - Float32(0.5)) * Float32(2); v = v * Float32(2)
            else:
                idx = n.child3; u = (u - Float32(0.5)) * Float32(2); v = (v - Float32(0.5)) * Float32(2)
        g.dnodes[Int(idx)].energy += weight

@always_inline
def guide_cell_has_data(g: GuideGrid, cell_idx: Int) -> Bool:
    """Returns True when the guide has sufficient concentrated data in this
    leaf. Requires total energy >= 1e-4 AND the directional quadtree having
    subdivided beyond its root -- a uniform/uninformative distribution
    never earns a split (see DIR_SPLIT_FRACTION), so "did it split" is
    itself the concentration test; no separate scan needed like the old
    fixed-bin grid's max-bin-fraction check."""
    if cell_idx < 0:
        return False
    var droot = g.snodes[cell_idx].dtree_root
    var total = g.dnodes[Int(droot)].energy
    if total < Float32(1e-4):
        return False
    return g.dnodes[Int(droot)].child0 >= Int32(0)

def guide_pdf(g: GuideGrid, cell_idx: Int, dx: Float32, dy: Float32, dz: Float32) -> Float32:
    """PDF (per steradian) for direction (dx,dy,dz) at cell_idx.
    Falls back to uniform (1/4π) when the leaf has no recorded energy."""
    if cell_idx < 0:
        return Float32(1.0) / FOUR_PI_F
    var droot = g.snodes[cell_idx].dtree_root
    var total = g.dnodes[Int(droot)].energy
    if total < Float32(1e-6):
        return Float32(1.0) / FOUR_PI_F
    var uv = _equal_area_sphere_to_square(dx, dy, dz)
    var leaf_idx = _dtree_leaf_for_uv(g, droot, uv[0], uv[1])
    var leaf = g.dnodes[Int(leaf_idx)]
    var p_bin = leaf.energy / total
    var solid_angle = FOUR_PI_F / _four_pow(leaf.depth)
    return max(p_bin / solid_angle, Float32(1e-7))

def guide_sample(
    g: GuideGrid, cell_idx: Int, u: Float32, u2: Float32 = Float32(0.5),
) -> Tuple[Float32, Float32, Float32, Float32, Bool]:
    """Sample a direction from the guide distribution at cell_idx.
    u: leaf selection (4-way CDF descent). u2: within-leaf jitter, both UV
    axes, decorrelated via a golden-ratio offset from the same number
    (matches the old grid's within-bin jitter trick).
    Returns (dx, dy, dz, pdf, ok): ok=False when the leaf has no energy
    (fall back to BSDF)."""
    if cell_idx < 0:
        return (Float32(0), Float32(0), Float32(1), Float32(1.0) / FOUR_PI_F, False)
    var droot = g.snodes[cell_idx].dtree_root
    var total = g.dnodes[Int(droot)].energy
    if total < Float32(1e-6):
        return (Float32(0), Float32(0), Float32(1), Float32(1.0) / FOUR_PI_F, False)
    var target = u * total
    var idx = droot
    var u0 = Float32(0.0)
    var v0 = Float32(0.0)
    var size = Float32(1.0)
    while g.dnodes[Int(idx)].child0 >= Int32(0):
        var n = g.dnodes[Int(idx)]
        var e0 = g.dnodes[Int(n.child0)].energy
        var e1 = g.dnodes[Int(n.child1)].energy
        var e2 = g.dnodes[Int(n.child2)].energy
        var half = size * Float32(0.5)
        if target < e0:
            idx = n.child0
        elif target < e0 + e1:
            target -= e0
            idx = n.child1; u0 += half
        elif target < e0 + e1 + e2:
            target -= (e0 + e1)
            idx = n.child2; v0 += half
        else:
            # Unconditional remainder -- also the safe fallback if
            # concurrent, non-atomic guide_record writes (see file header)
            # made e0+e1+e2+e3 drift slightly below `target`.
            target -= (e0 + e1 + e2)
            idx = n.child3; u0 += half; v0 += half
        size = half
    var leaf = g.dnodes[Int(idx)]
    var p_bin = leaf.energy / total
    var solid_angle = FOUR_PI_F / _four_pow(leaf.depth)
    var pdf = max(p_bin / solid_angle, Float32(1e-7))
    var v2 = u2 * Float32(2.6180339887)
    v2 = v2 - Float32(Int(v2))
    var u_in = u0 + u2 * size
    var v_in = v0 + v2 * size
    var dir = _equal_area_square_to_sphere(u_in, v_in)
    return (dir[0], dir[1], dir[2], pdf, True)

# ── Refinement (between training iterations, single-threaded) ──────────────

def _count_dtree_growth(dnodes: UnsafePointer[DNode, MutAnyOrigin], node_idx: Int32, root_total: Float32, depth: Int32) -> Int:
    var n = dnodes[Int(node_idx)]
    if n.child0 >= Int32(0):
        return (_count_dtree_growth(dnodes, n.child0, root_total, depth + Int32(1))
              + _count_dtree_growth(dnodes, n.child1, root_total, depth + Int32(1))
              + _count_dtree_growth(dnodes, n.child2, root_total, depth + Int32(1))
              + _count_dtree_growth(dnodes, n.child3, root_total, depth + Int32(1)))
    if depth >= DIR_MAX_DEPTH or root_total < Float32(1e-6):
        return 0
    if n.energy / root_total > DIR_SPLIT_FRACTION:
        return 4
    return 0

def _grow_dtree(dnodes: UnsafePointer[DNode, MutAnyOrigin], node_idx: Int32, root_total: Float32, depth: Int32, mut next_free: Int32):
    """Split leaves whose energy fraction of root_total exceeds
    DIR_SPLIT_FRACTION into 4 fresh children (each seeded with energy/4, an
    inherited estimate -- refined for real once the next iteration's
    samples land in them). One level of growth per call; a leaf that's
    still too concentrated after that will qualify again on the NEXT
    refine, once it has re-accumulated enough energy at the new depth --
    deliberately simpler than growing multiple levels in a single pass."""
    var n = dnodes[Int(node_idx)]
    if n.child0 >= Int32(0):
        _grow_dtree(dnodes, n.child0, root_total, depth + Int32(1), next_free)
        _grow_dtree(dnodes, n.child1, root_total, depth + Int32(1), next_free)
        _grow_dtree(dnodes, n.child2, root_total, depth + Int32(1), next_free)
        _grow_dtree(dnodes, n.child3, root_total, depth + Int32(1), next_free)
        return
    if depth >= DIR_MAX_DEPTH or root_total < Float32(1e-6):
        return
    if n.energy / root_total <= DIR_SPLIT_FRACTION:
        return
    var child_e = n.energy / Float32(4.0)
    var c0 = next_free
    var c1 = next_free + Int32(1)
    var c2 = next_free + Int32(2)
    var c3 = next_free + Int32(3)
    var cd = depth + Int32(1)
    dnodes[Int(c0)] = DNode(energy=child_e, depth=cd, child0=Int32(-1), child1=Int32(-1), child2=Int32(-1), child3=Int32(-1))
    dnodes[Int(c1)] = DNode(energy=child_e, depth=cd, child0=Int32(-1), child1=Int32(-1), child2=Int32(-1), child3=Int32(-1))
    dnodes[Int(c2)] = DNode(energy=child_e, depth=cd, child0=Int32(-1), child1=Int32(-1), child2=Int32(-1), child3=Int32(-1))
    dnodes[Int(c3)] = DNode(energy=child_e, depth=cd, child0=Int32(-1), child1=Int32(-1), child2=Int32(-1), child3=Int32(-1))
    dnodes[Int(node_idx)].child0 = c0
    dnodes[Int(node_idx)].child1 = c1
    dnodes[Int(node_idx)].child2 = c2
    dnodes[Int(node_idx)].child3 = c3
    next_free += Int32(4)

@fieldwise_init
struct _SplitCandidate(TrivialRegisterPassable):
    var leaf_idx: Int32
    var lo: Point3f
    var hi: Point3f

def _longest_axis(lo: Point3f, hi: Point3f) -> Int32:
    var ex = hi.x - lo.x
    var ey = hi.y - lo.y
    var ez = hi.z - lo.z
    if ex >= ey and ex >= ez:
        return Int32(0)
    if ey >= ez:
        return Int32(1)
    return Int32(2)

def _axis_mid(lo: Point3f, hi: Point3f, axis: Int32) -> Float32:
    if axis == Int32(0):
        return (lo.x + hi.x) * Float32(0.5)
    if axis == Int32(1):
        return (lo.y + hi.y) * Float32(0.5)
    return (lo.z + hi.z) * Float32(0.5)

def _collect_spatial_splits(snodes: UnsafePointer[SNode, MutAnyOrigin], idx: Int32, lo: Point3f, hi: Point3f, mut out: List[_SplitCandidate]):
    var n = snodes[Int(idx)]
    if n.child0 >= Int32(0):
        if n.split_axis == Int32(0):
            _collect_spatial_splits(snodes, n.child0, lo, Point3f(n.split_pos, hi.y, hi.z), out)
            _collect_spatial_splits(snodes, n.child1, Point3f(n.split_pos, lo.y, lo.z), hi, out)
        elif n.split_axis == Int32(1):
            _collect_spatial_splits(snodes, n.child0, lo, Point3f(hi.x, n.split_pos, hi.z), out)
            _collect_spatial_splits(snodes, n.child1, Point3f(lo.x, n.split_pos, lo.z), hi, out)
        else:
            _collect_spatial_splits(snodes, n.child0, lo, Point3f(hi.x, hi.y, n.split_pos), out)
            _collect_spatial_splits(snodes, n.child1, Point3f(lo.x, lo.y, n.split_pos), hi, out)
        return
    if n.depth < SPATIAL_MAX_DEPTH and n.sample_count > SPATIAL_SPLIT_SAMPLES:
        out.append(_SplitCandidate(leaf_idx=idx, lo=lo, hi=hi))

def guide_refine(g: GuideGrid) -> GuideGrid:
    """Grow the SD-tree between training iterations, then reset each
    spatial leaf's per-iteration sample_count (directional energy is NOT
    reset -- see file header). Frees g's old arrays; returns a new
    GuideGrid the caller must guide_free later. Single-threaded, between
    passes only -- see file header for why concurrent calls would corrupt
    the tree.

    New spatial leaves start with a fresh (single-node, zero-energy)
    directional quadtree rather than a deep copy of the parent's learned
    shape -- a deliberate simplification vs. the full Müller warm-start:
    unbiased either way, just slower for that specific child to re-learn
    its directional distribution over the following iterations."""
    # ── Phase 1: grow directional quadtrees for existing spatial leaves ──
    var extra_d = 0
    for i in range(Int(g.n_snodes)):
        if g.snodes[i].child0 < Int32(0):
            var droot = g.snodes[i].dtree_root
            extra_d += _count_dtree_growth(g.dnodes, droot, g.dnodes[Int(droot)].energy, Int32(0))
    var n_dnodes1 = Int(g.n_dnodes) + extra_d
    var dnodes1 = alloc[DNode](n_dnodes1)
    for i in range(Int(g.n_dnodes)):
        dnodes1[i] = g.dnodes[i]
    var next_free_d = g.n_dnodes
    for i in range(Int(g.n_snodes)):
        if g.snodes[i].child0 < Int32(0):
            var droot = g.snodes[i].dtree_root
            _grow_dtree(dnodes1, droot, dnodes1[Int(droot)].energy, Int32(0), next_free_d)
    g.dnodes.free()

    # ── Phase 2: split spatial leaves with enough samples this iteration ──
    var to_split = List[_SplitCandidate]()
    _collect_spatial_splits(g.snodes, Int32(0), g.bounds.min, g.bounds.max, to_split)
    var n_new_leaves = 2 * len(to_split)
    var n_snodes2 = Int(g.n_snodes) + n_new_leaves
    var n_dnodes2 = Int(next_free_d) + n_new_leaves
    var snodes2 = alloc[SNode](n_snodes2)
    var dnodes2 = alloc[DNode](n_dnodes2)
    for i in range(Int(g.n_snodes)):
        snodes2[i] = g.snodes[i]
    for i in range(Int(next_free_d)):
        dnodes2[i] = dnodes1[i]
    dnodes1.free()
    g.snodes.free()

    var next_s = g.n_snodes
    var next_d = next_free_d
    for k in range(len(to_split)):
        var cand = to_split[k]
        var axis = _longest_axis(cand.lo, cand.hi)
        var pos = _axis_mid(cand.lo, cand.hi, axis)
        var leaf = snodes2[Int(cand.leaf_idx)]
        var c0 = next_s
        var c1 = next_s + Int32(1)
        var d0 = next_d
        var d1 = next_d + Int32(1)
        dnodes2[Int(d0)] = DNode(energy=Float32(0), depth=Int32(0), child0=Int32(-1), child1=Int32(-1), child2=Int32(-1), child3=Int32(-1))
        dnodes2[Int(d1)] = DNode(energy=Float32(0), depth=Int32(0), child0=Int32(-1), child1=Int32(-1), child2=Int32(-1), child3=Int32(-1))
        snodes2[Int(c0)] = SNode(split_axis=Int32(-1), split_pos=Float32(0), depth=leaf.depth + Int32(1),
                                  child0=Int32(-1), child1=Int32(-1), dtree_root=d0, sample_count=Int32(0))
        snodes2[Int(c1)] = SNode(split_axis=Int32(-1), split_pos=Float32(0), depth=leaf.depth + Int32(1),
                                  child0=Int32(-1), child1=Int32(-1), dtree_root=d1, sample_count=Int32(0))
        snodes2[Int(cand.leaf_idx)].split_axis = axis
        snodes2[Int(cand.leaf_idx)].split_pos = pos
        snodes2[Int(cand.leaf_idx)].child0 = c0
        snodes2[Int(cand.leaf_idx)].child1 = c1
        next_s += Int32(2)
        next_d += Int32(2)

    # ── Phase 3: reset the per-iteration spatial statistic only ──────────
    for i in range(Int(next_s)):
        if snodes2[i].child0 < Int32(0):
            snodes2[i].sample_count = Int32(0)

    return GuideGrid(snodes=snodes2, n_snodes=next_s, dnodes=dnodes2, n_dnodes=next_d, bounds=g.bounds)
