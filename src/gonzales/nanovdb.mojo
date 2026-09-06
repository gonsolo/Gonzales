"""NanoVDB sparse volume grids: loading (via the C bridge) and index-space
sampling (in Mojo, so CPU and GPU share one accessor).

STATUS: loading works; `nvdb_sample_index` below is a STUB awaiting the
PNanoVDB accessor port. `Tools/nvdb_diff.mojo` is the differential harness
that proves the port correct -- run it, it is red until this is done.

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
  table -> upper -> lower -> leaf, each level testing a child mask bit to
  decide "descend" vs "this is a constant tile", and the leaf holding a
  512-bit value mask plus a dense 512-value array. Port
  `pnanovdb_readaccessor_get_value_float` / `pnanovdb_root_get_value` and
  friends; they are already branch-free offset arithmetic.

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

def nvdb_sample_index(
    blob: UnsafePointer[UInt8, MutExternalOrigin],
    i: Int32, j: Int32, k: Int32,
) -> Float32:
    """Value at index-space voxel (i,j,k), or the background value outside
    any active region. Pure offset arithmetic over `blob` -- no allocation,
    no host calls -- so this same function serves CPU and GPU.

    TODO(port): transcribe pnanovdb_readaccessor_get_value_float from
    /usr/include/nanovdb/PNanoVDB.h. See this module's LAYOUT MAP. Prove it
    with `Tools/nvdb_diff.mojo`, which compares against NanoVDB's own
    accessor over the real scene assets."""
    return Float32(0)
