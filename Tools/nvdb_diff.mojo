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
    nvdb_get_value_ref, nvdb_active_count, nvdb_active_coord,
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

    coord.free()
    cpath.free()
    nvdb_free(h)

    print("  checked", checked, "voxels,", nonzero_ref, "of them non-background")
    if nonzero_ref == 0:
        print("FAIL: every probe was background -- the comparison proves nothing")
        return
    if mismatches == 0:
        print("PASS: Mojo accessor matches NanoVDB exactly on all", checked, "probes")
    else:
        print("FAIL:", mismatches, "mismatches of", checked)
