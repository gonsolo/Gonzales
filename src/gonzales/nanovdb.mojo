"""NanoVDB sparse volume grids: loading (via the C bridge) and index-space
sampling (in Mojo, so CPU and GPU share one accessor).

STATUS: loading and index-space sampling (`nvdb_sample_index`) both work,
verified against NanoVDB's own C++ accessor by `Tools/nvdb_diff.mojo`
(`make nvdb_diff`) on all three real scene assets (bunny_cloud, fire,
wdas_cloud_quarter -- 200K+ probes each, zero mismatches). NOT yet wired
into the renderer: no parser branch (pbrt_parser.mojo's `is_hom`/`is_grid`
need a sibling `is_nanovdb` for `"string type" "nanovdb"`), no per-leaf
majorant extraction for delta/ratio tracking, no GPU upload of the blob.

WHY THE ACCESSOR MUST BE MOJO, NOT C++
The bridge could obviously expose `getValue` (it does -- `nvdb_get_value`,
but only as the test oracle). Using that as the real sampler would strand
volumes on the CPU: a GPU kernel cannot call back into a host C++ library.
PNanoVDB.h is NanoVDB's own C99/HLSL/GLSL implementation, written as flat
offset arithmetic over the blob precisely so it can run in a shader -- so a
Mojo transcription of it runs unchanged on both backends, exactly like
`traverse_bvh2_core` does. Writing two samplers is the mistake this
codebase already paid for with the VCM backends (see the light/camera step
sharing in bdpt.mojo); do not repeat it here.

LAYOUT MAP (from /usr/include/nanovdb/PNanoVDB.h, NanoVDB 32.9.0)
The blob is a single contiguous allocation, addressed by byte offset:

    Grid   at 0,              PNANOVDB_GRID_SIZE = 672 bytes
    Tree   at 672,            PNANOVDB_TREE_SIZE = 64 bytes
    Root   at 672 + tree.nodeOffset[3]
    then Upper (5 bits, 32^3), Lower (4 bits, 16^3), Leaf (3 bits, 8^3) nodes.

  Useful Grid field offsets (bytes from grid base):
    MAGIC 0, CHECKSUM 8, VERSION 16, FLAGS 20, GRID_INDEX 24, GRID_COUNT 28,
    GRID_SIZE 32, GRID_NAME 40, MAP 296, WORLD_BBOX 560, VOXEL_SIZE 608,
    GRID_CLASS 632, GRID_TYPE 636, BLIND_METADATA_OFFSET 640,
    BLIND_METADATA_COUNT 648.
  GRID_TYPE_FLOAT = 1 (the only type wired up; the bridge refuses others).

  The traversal itself is the standard NanoVDB 3-level descent: root hash
  table (linear-scanned; tile counts are small) -> upper -> lower -> leaf,
  each level testing a child mask bit to decide "descend" vs "this is a
  constant tile", the leaf holding a dense 512-value array indexed by an
  8x8x8 local coordinate. Ported from `pnanovdb_root_get_value_address_
  and_level` and its upper/lower/leaf counterparts in PNanoVDB.h -- see
  `nvdb_sample_index` below for the actual traversal and its own comments
  for a genuine compiler-bug workaround (`_nvdb_coord_to_key`) found while
  porting this; read that comment before touching the widening arithmetic
  anywhere in this file.

VERSION NOTE
The scene assets are written by NanoVDB 32.3.3 while these headers are
32.9.0. Same MAJOR version means same ABI and file format by NanoVDB's own
versioning rule, and the 32.9 C++ accessor demonstrably reads these blobs
correctly, so the 32.9 offsets above apply. If that ever stops being true
the differential harness catches it immediately rather than silently
returning wrong densities.
"""
from std.ffi import external_call

# ── Loading (C bridge, host only) ────────────────────────────────────────

def nvdb_load(path: UnsafePointer[UInt8, MutExternalOrigin], n: Int32) -> UnsafePointer[UInt8, MutExternalOrigin]:
    """Opens a .nvdb and returns an opaque handle, or a null-ish pointer on
    failure. The handle owns the DECOMPRESSED grid blob (the on-disk one is
    codec-compressed -- every scene asset here is ZIP)."""
    return external_call["nvdb_load", UnsafePointer[UInt8, MutExternalOrigin]](path, n)

def nvdb_data(handle: UnsafePointer[UInt8, MutExternalOrigin]) -> UnsafePointer[UInt8, MutExternalOrigin]:
    """The flat grid blob. memcpy-able straight to the GPU."""
    return external_call["nvdb_data", UnsafePointer[UInt8, MutExternalOrigin]](handle)

def nvdb_size(handle: UnsafePointer[UInt8, MutExternalOrigin]) -> Int64:
    return external_call["nvdb_size", Int64](handle)

def nvdb_free(handle: UnsafePointer[UInt8, MutExternalOrigin]):
    external_call["nvdb_free", NoneType](handle)

# ── Test oracle (C bridge, host only -- NOT for rendering) ───────────────

def nvdb_get_value_ref(handle: UnsafePointer[UInt8, MutExternalOrigin], i: Int32, j: Int32, k: Int32) -> Float32:
    """NanoVDB's own C++ accessor. Exists ONLY so `nvdb_sample_index` can be
    diffed against it -- offset/bitmask errors in a tree traversal are
    silent (wrong densities, not crashes). Never call this from rendering
    code: it cannot run on the GPU."""
    return external_call["nvdb_get_value", Float32](handle, i, j, k)

def nvdb_active_count(handle: UnsafePointer[UInt8, MutExternalOrigin]) -> Int64:
    """Count of LEAF-RESIDENT active voxels -- deliberately not the grid's
    activeVoxelCount(), which also counts voxels covered by active upper
    tiles that `nvdb_active_coord` cannot enumerate."""
    return external_call["nvdb_active_count", Int64](handle)

def nvdb_active_coord(handle: UnsafePointer[UInt8, MutExternalOrigin], n: Int64,
                      out3: UnsafePointer[Int32, MutExternalOrigin]):
    """Coordinate of the n-th leaf-resident active voxel. Deterministic for
    a given blob. Needed because uniform random sampling of the index bbox
    mostly lands in empty space on a sparse grid -- a stub that always
    returns background would pass such a test vacuously."""
    external_call["nvdb_active_coord", NoneType](handle, n, out3)

# ── The accessor (Mojo, CPU+GPU) ─────────────────────────────────────────

comptime NVDB_GRID_SIZE = 672
comptime NVDB_TREE_SIZE = 64
comptime NVDB_GRID_TYPE_FLOAT = 1

# Tree header field offset (bytes from tree base).
comptime NVDB_TREE_OFF_NODE_OFFSET_ROOT = 24

# Root header field offsets (bytes from root base) -- Float grid row of
# pnanovdb_grid_type_constants: root_off_background=28, root_size=64,
# root_tile_off_value=20, root_tile_size=32.
comptime NVDB_ROOT_OFF_TABLE_SIZE = 24
comptime NVDB_ROOT_OFF_BACKGROUND = 28
comptime NVDB_ROOT_SIZE = 64
comptime NVDB_ROOT_TILE_OFF_KEY = 0
comptime NVDB_ROOT_TILE_OFF_CHILD = 8
comptime NVDB_ROOT_TILE_OFF_VALUE = 20
comptime NVDB_ROOT_TILE_SIZE = 32

# Upper node (5-bit, 32^3 children) -- Float row: upper_off_table=8256,
# table_stride=8 (a child byte-offset (int64) and a constant-tile value
# (float) share this same 8-byte table slot, discriminated by the child
# mask bit -- see this module's docstring).
comptime NVDB_UPPER_OFF_CHILD_MASK = 4128
comptime NVDB_UPPER_OFF_TABLE = 8256
comptime NVDB_TABLE_STRIDE = 8  # same for upper and lower, Float grid

# Lower node (4-bit, 16^3 children) -- Float row: lower_off_table=1088.
comptime NVDB_LOWER_OFF_CHILD_MASK = 544
comptime NVDB_LOWER_OFF_TABLE = 1088

# Leaf node (3-bit, 8^3 voxels) -- Float row: leaf_off_table=96, dense
# 4-byte-per-voxel value array (value_stride_bits=32).
comptime NVDB_LEAF_OFF_VALUE_MASK = 16
comptime NVDB_LEAF_OFF_TABLE = 96

@always_inline
def _nvdb_u32(blob: UnsafePointer[UInt8, MutExternalOrigin], off: Int) -> UInt32:
    return (blob + off).bitcast[UInt32]()[0]

@always_inline
def _nvdb_u64(blob: UnsafePointer[UInt8, MutExternalOrigin], off: Int) -> UInt64:
    return (blob + off).bitcast[UInt64]()[0]

@always_inline
def _nvdb_i64(blob: UnsafePointer[UInt8, MutExternalOrigin], off: Int) -> Int64:
    return (blob + off).bitcast[Int64]()[0]

@always_inline
def _nvdb_f32(blob: UnsafePointer[UInt8, MutExternalOrigin], off: Int) -> Float32:
    return (blob + off).bitcast[Float32]()[0]

@always_inline
def _nvdb_coord_to_key(i: Int32, j: Int32, k: Int32) -> UInt64:
    """pnanovdb_coord_to_key, PNANOVDB_NATIVE_64 form: reinterpret each
    coordinate's bits as unsigned (two's complement -> a large positive
    number for negatives, exactly like C's cast) and keep its top 20 bits
    -- the tile this voxel's root region hashes to.

    TOOLCHAIN QUIRK -- unconfirmed, not filed upstream, read before
    "simplifying" this function. `UInt64(UInt32(x)) >> 12` for a negative
    Int32 x prints the SIGN-extended 64-bit value of x (as if the UInt32
    step never happened), even though `UInt32(x)` alone, and even
    `UInt64(UInt32(x))` alone with NO following shift, both print
    correctly in isolation. A first attempt at fixing this by re-masking
    the result (`(UInt64(UInt32(x)) >> 12) & UInt64(0xFFFFFFFF)`) worked
    in an isolated repro but the mask was then silently eliminated once
    this function was actually inlined into its real caller, so the wrong
    result survived the first fix. What DOES survive inlining: do the
    `>> 12` shift entirely within UInt32 (verified correct on its own: the
    very first thing checked while porting this accessor), and only THEN
    widen the already-small (<=20-bit) UInt32 result to UInt64 as its own
    statement, never fused with the shift. Confirmed via `make nvdb_diff`
    on all three real scene assets, cross-checked byte-for-byte against
    the real C++ NanoVDB accessor -- not just the isolated repro, which
    already fooled one earlier attempt at this same fix.

    Deliberately NOT labeled a "confirmed Mojo compiler bug": this
    project's one actual filed report of a similar-looking symptom
    (reference_mojo_compiler_bug_6759, a TrivialRegisterPassable struct
    allegedly corrupted across a call boundary) was retracted by its own
    reporter after the founding instance turned out to trace to an
    uncommitted, never-pinned local edit, and a faithful reconstruction at
    the exact historical commit and toolchain reproduced CLEANLY, 10/10
    runs, with 250 more clean trials across two rounds of synthetic
    isolation. That history means "looks like the same kind of thing" is
    not corroboration and should not be cited as such. This finding
    stands only on what was directly, repeatedly observed today against
    the real build and the real reference accessor -- it does not borrow
    that other report's credibility, because that report no longer has
    any to lend. If revisited: build a real minimal reproducer FIRST
    (something #6759's history shows is not guaranteed to be possible
    even when the underlying symptom is real and repeatable in the full
    build) before concluding this is Modular's bug rather than something
    in this code or this specific toolchain install. Do not restructure
    this function into a single fused expression without re-running
    `make nvdb_diff` on all three real assets first."""
    var iu32: UInt32 = UInt32(i) >> 12
    var ju32: UInt32 = UInt32(j) >> 12
    var ku32: UInt32 = UInt32(k) >> 12
    var iu: UInt64 = UInt64(iu32)
    var ju: UInt64 = UInt64(ju32)
    var ku: UInt64 = UInt64(ku32)
    return ku | (ju << 21) | (iu << 42)

@always_inline
def _nvdb_upper_offset(i: Int32, j: Int32, k: Int32) -> Int:
    return Int((((i & 4095) >> 7) << 10) + (((j & 4095) >> 7) << 5) + ((k & 4095) >> 7))

@always_inline
def _nvdb_lower_offset(i: Int32, j: Int32, k: Int32) -> Int:
    return Int((((i & 127) >> 3) << 8) + (((j & 127) >> 3) << 4) + ((k & 127) >> 3))

@always_inline
def _nvdb_leaf_offset(i: Int32, j: Int32, k: Int32) -> Int:
    return Int(((i & 7) << 6) + ((j & 7) << 3) + (k & 7))

@always_inline
def _nvdb_mask_is_on(blob: UnsafePointer[UInt8, MutExternalOrigin], mask_base: Int, bit: Int) -> Bool:
    var word = _nvdb_u32(blob, mask_base + 4 * (bit >> 5))
    return (word & (UInt32(1) << UInt32(bit & 31))) != UInt32(0)

def nvdb_sample_index(
    blob: UnsafePointer[UInt8, MutExternalOrigin],
    i: Int32, j: Int32, k: Int32,
) -> Float32:
    """Value at index-space voxel (i,j,k), or the background value outside
    any active region. Pure offset arithmetic over `blob` -- no allocation,
    no host calls -- so this same function serves CPU and GPU.

    Transcribed from pnanovdb_root_get_value_address_and_level +
    pnanovdb_upper/lower/leaf_get_value_address_and_level (PNanoVDB.h): a
    root hash-table lookup (linear scan -- root tile counts are small,
    tens to low hundreds, not worth a real hash) followed by up to three
    levels of "is there a child here, or is this whole region one constant
    tile value" descent. See this module's LAYOUT MAP for the byte offsets.
    Proven against NanoVDB's own C++ accessor by `Tools/nvdb_diff.mojo`
    (`make nvdb_diff`)."""
    var tree_base = NVDB_GRID_SIZE
    var root_base = tree_base + Int(_nvdb_u64(blob, tree_base + NVDB_TREE_OFF_NODE_OFFSET_ROOT))

    var table_size = Int(_nvdb_u32(blob, root_base + NVDB_ROOT_OFF_TABLE_SIZE))
    var key = _nvdb_coord_to_key(i, j, k)

    var tile_base = root_base + NVDB_ROOT_SIZE
    var found = False
    for _ in range(table_size):
        if _nvdb_u64(blob, tile_base + NVDB_ROOT_TILE_OFF_KEY) == key:
            found = True
            break
        tile_base += NVDB_ROOT_TILE_SIZE

    if not found:
        return _nvdb_f32(blob, root_base + NVDB_ROOT_OFF_BACKGROUND)

    var root_child = _nvdb_i64(blob, tile_base + NVDB_ROOT_TILE_OFF_CHILD)
    if root_child == Int64(0):
        return _nvdb_f32(blob, tile_base + NVDB_ROOT_TILE_OFF_VALUE)

    # Upper node: root_child is a byte offset relative to root_base (not
    # tile_base) -- see pnanovdb_root_get_child.
    var upper_base = root_base + Int(root_child)
    var n_upper = _nvdb_upper_offset(i, j, k)
    if _nvdb_mask_is_on(blob, upper_base + NVDB_UPPER_OFF_CHILD_MASK, n_upper):
        var upper_slot = upper_base + NVDB_UPPER_OFF_TABLE + NVDB_TABLE_STRIDE * n_upper
        var lower_base = upper_base + Int(_nvdb_i64(blob, upper_slot))
        var n_lower = _nvdb_lower_offset(i, j, k)
        if _nvdb_mask_is_on(blob, lower_base + NVDB_LOWER_OFF_CHILD_MASK, n_lower):
            var lower_slot = lower_base + NVDB_LOWER_OFF_TABLE + NVDB_TABLE_STRIDE * n_lower
            var leaf_base = lower_base + Int(_nvdb_i64(blob, lower_slot))
            var n_leaf = _nvdb_leaf_offset(i, j, k)
            return _nvdb_f32(blob, leaf_base + NVDB_LEAF_OFF_TABLE + 4 * n_leaf)
        else:
            # Constant lower-level tile: the value shares the table slot
            # the child offset would otherwise occupy.
            return _nvdb_f32(blob, lower_base + NVDB_LOWER_OFF_TABLE + NVDB_TABLE_STRIDE * n_lower)
    else:
        return _nvdb_f32(blob, upper_base + NVDB_UPPER_OFF_TABLE + NVDB_TABLE_STRIDE * n_upper)
