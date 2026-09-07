"""Differential harness for the NanoVDB accessor port.

Runs gonzales's Mojo `nvdb_sample_index` and NanoVDB's own C++ accessor
over the SAME blob in the SAME process and compares them voxel by voxel.
No reference files, so nothing can drift out of date.

    make nvdb_diff
    build/nvdb_diff Scenes/pbrt-v4-scenes/bunny-cloud/bunny_cloud.nvdb

Red until `nvdb_sample_index` is implemented; that is the point.

Sampling is deliberately mixed, because the obvious test is vacuous: a
sparse grid is mostly empty, so uniformly random coordinates in the index
bbox nearly all return background, and a stub that always returns 0 would
"agree" with the oracle on ~99% of them. So most probes walk ACTIVE voxels
(via the bridge's deterministic leaf enumeration), and a minority probe
empty space and far-outside-bbox on purpose -- background must be right
too, and it is the one case a stub gets free.
"""
from std.sys import argv
from std.memory import alloc
from gonzales.nanovdb import (
    nvdb_load, nvdb_data, nvdb_size, nvdb_free,
    nvdb_get_value_ref, nvdb_active_count, nvdb_active_coord, nvdb_majorant_at,
    nvdb_sample_index,
)

def main() raises:
    var av = argv()
    if len(av) < 2:
        print("usage: nvdb_diff <file.nvdb> [n_active_probes]")
        return
    var path = String(av[1])
    var n_probe = 20000
    if len(av) > 2:
        n_probe = Int(String(av[2]))

    # NUL-terminated copy for the C bridge -- same idiom pbrt_parser uses
    # when handing texture paths to the OIIO bridge.
    var slen = path.byte_length()
    var cpath = alloc[UInt8](slen + 1)
    for ci in range(slen):
        cpath[ci] = path.unsafe_ptr()[ci]
    cpath[slen] = UInt8(0)
    var h = nvdb_load(cpath, Int32(0))
    if Int(h) == 0:
        print("FAIL: could not load", path)
        return
    var blob = nvdb_data(h)
    var n_active = nvdb_active_count(h)
    print("loaded", path)
    print("  blob =", nvdb_size(h), "bytes,  leaf-resident active voxels =", n_active)
    if n_active == 0:
        print("FAIL: grid has no active voxels to compare against")
        nvdb_free(h)
        return

    var coord = alloc[Int32](3)
    var mismatches = 0
    var checked = 0
    var nonzero_ref = 0
    var first_bad_shown = 0

    # 1) active voxels -- the probes a stub cannot fake
    var stride = n_active // Int64(n_probe)
    if stride < Int64(1): stride = Int64(1)
    var idx = Int64(0)
    while idx < n_active:
        nvdb_active_coord(h, idx, coord)
        var expect = nvdb_get_value_ref(h, coord[0], coord[1], coord[2])
        var got = nvdb_sample_index(blob, coord[0], coord[1], coord[2])
        checked += 1
        if expect != Float32(0): nonzero_ref += 1
        if expect != got:
            mismatches += 1
            if first_bad_shown < 5:
                print("    MISMATCH at (", coord[0], coord[1], coord[2], ") ref=", expect, " got=", got)
                first_bad_shown += 1
        idx += stride

    # 2) empty space inside the bbox, and far outside it -- background must
    #    also be right, though a stub gets these for free.
    var lcg = UInt64(0x9E3779B97F4A7C15)
    for _ in range(2000):
        lcg = lcg * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var x = Int32(Int((lcg >> 16) % UInt64(1200)) - 600)
        var y = Int32(Int((lcg >> 28) % UInt64(1200)) - 600)
        var z = Int32(Int((lcg >> 40) % UInt64(1200)) - 600)
        var expect = nvdb_get_value_ref(h, x, y, z)
        var got = nvdb_sample_index(blob, x, y, z)
        checked += 1
        if expect != Float32(0): nonzero_ref += 1
        if expect != got:
            mismatches += 1
            if first_bad_shown < 5:
                print("    MISMATCH at (", x, y, z, ") ref=", expect, " got=", got)
                first_bad_shown += 1

    # ── Majorant validity ────────────────────────────────────────────────
    # A LOCAL majorant that is ever SMALLER than a real density inside the
    # region it claims to bound silently biases delta tracking (it makes the
    # medium look thinner, with no crash and no obvious artefact), so it is
    # checked here rather than trusted. For each probe: the reported bound
    # must be >= the true max over the whole aligned box it covers.
    var maj_checked = 0
    var maj_bad = 0
    var maj_shown = 0
    var maj_total_ratio = Float64(0)
    for probe in range(min(n_active, Int64(400))):
        var ci = Int64(probe) * (n_active // Int64(400) + Int64(1))
        if ci >= n_active: break
        nvdb_active_coord(h, ci, coord)
        var bx = coord[0]; var by = coord[1]; var bz = coord[2]
        var mr = nvdb_majorant_at(blob, bx, by, bz)
        var bound = mr[0]
        var dim = Int32(mr[1])
        # aligned box base (arithmetic shift keeps negatives correct)
        var shift = Int32(0)
        var d = dim
        while d > Int32(1):
            d = d >> Int32(1); shift += Int32(1)
        var b0 = (bx >> shift) << shift
        var b1 = (by >> shift) << shift
        var b2 = (bz >> shift) << shift
        # true max over the box, subsampled for boxes too big to scan whole
        var step = Int32(1)
        if dim > Int32(16): step = dim // Int32(16)
        var truemax = Float32(0)
        for zz in range(0, Int(dim), Int(step)):
            for yy in range(0, Int(dim), Int(step)):
                for xx in range(0, Int(dim), Int(step)):
                    var v = nvdb_sample_index(blob, b0 + Int32(xx), b1 + Int32(yy), b2 + Int32(zz))
                    if v > truemax: truemax = v
        maj_checked += 1
        if truemax > bound * Float32(1.0001):
            maj_bad += 1
            if maj_shown < 5:
                print("    MAJORANT TOO SMALL at (", bx, by, bz, ") bound=", bound,
                      " true max in", dim, "^3 box =", truemax)
                maj_shown += 1
        if bound > Float32(0):
            maj_total_ratio += Float64(truemax / bound)

    coord.free()
    cpath.free()
    nvdb_free(h)

    print("  majorant: checked", maj_checked, "regions,", maj_bad, "invalid;",
          "mean tightness (true max / bound) =", maj_total_ratio / Float64(max(maj_checked, 1)))
    print("  checked", checked, "voxels,", nonzero_ref, "of them non-background")
    if nonzero_ref == 0:
        print("FAIL: every probe was background -- the comparison proves nothing")
        return
    if mismatches == 0:
        print("PASS: Mojo accessor matches NanoVDB exactly on all", checked, "probes")
    else:
        print("FAIL:", mismatches, "mismatches of", checked)
