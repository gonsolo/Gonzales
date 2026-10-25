"""Builds a bottom-level acceleration structure in the format the RT cores walk (docs/rtcore/ACCEL_FORMAT.md), with no
driver involved. Input: triangles as nine floats each. Output: the structure's bytes (the root node at 0x380, the nodes,
then the triangle blocks) and the map from the hit's primitive id to the input triangle."""
from max.algorithm import parallelize
from std.atomic import Atomic
from std.math import floor, ceil
from std.memory import bitcast
from std.memory.alloc import unsafe_alloc
from std.os import getenv
from std.sys import get_defined_int
from std.sys.info import num_performance_cores

comptime RT_NODE = 128
comptime RT_ROOT = 0x380
comptime RT_LEAF_TRIS = 12
comptime RT_BLOCK_TRIS = get_defined_int["RT_BLOCK_TRIS", 8]()      # a block holds up to 16 distinct vertices; ids of 5..8 triangles use 3-bit offsets in an aligned group of 8
comptime RT_SLOT_TRIS = 4
comptime RT_STREAM_BITS = (0x68 - 12) * 8

comptime PF = Pointer[Float32, MutUntrackedOrigin]
comptime PD = Pointer[Float64, MutUntrackedOrigin]
comptime PI = Pointer[Int32, MutUntrackedOrigin]
comptime PB = Pointer[UInt8, MutUntrackedOrigin]


struct RtBlas(Movable):
    var bytes: PB
    var size: Int
    var gid_to_tri: PI          # primitive id (4 * block + slot) -> input triangle, -1 where unused
    var n_gid: Int
    var n_nodes: Int
    var n_blocks: Int
    var box: PD                 # object-space bounds: min corner, then max corner (six doubles)
    var gid_to_box: PI          # AABB candidate id (leaf payload + slot) -> input box
    var n_box: Int

    def __init__(out self, bytes: PB, size: Int, gid_to_tri: PI, n_gid: Int, n_nodes: Int, n_blocks: Int, box: PD,
                 gid_to_box: PI = PI.unsafe_dangling(), n_box: Int = 0):
        self.gid_to_box = gid_to_box
        self.n_box = n_box
        self.bytes = bytes
        self.size = size
        self.gid_to_tri = gid_to_tri
        self.n_gid = n_gid
        self.n_nodes = n_nodes
        self.n_blocks = n_blocks
        self.box = box


@always_inline
def _fbits(x: Float32) -> UInt32:
    return bitcast[DType.uint32, 1](x)


@always_inline
def _bit_length(v: UInt32) -> Int:
    var n = 0
    var x = v
    while x != UInt32(0):
        n += 1
        x = x >> UInt32(1)
    return n


@no_inline
def _put_bits(buf: PB, bit: Int, value: UInt64, nbits: Int):
    """Set bits [bit, bit + nbits) of buf (bit k = byte k / 8, bit k % 8) to value (they are zero before)."""
    for b in range(nbits):
        if ((value >> UInt64(b)) & UInt64(1)) != UInt64(0):
            var p = bit + b
            buf[unsafe_offset=p >> 3] = buf[unsafe_offset=p >> 3] | UInt8(1 << (p & 7))


@no_inline
def _put_u32(buf: PB, off: Int, v: UInt32):
    buf[unsafe_offset=off] = UInt8(v & UInt32(0xff))
    buf[unsafe_offset=off + 1] = UInt8((v >> UInt32(8)) & UInt32(0xff))
    buf[unsafe_offset=off + 2] = UInt8((v >> UInt32(16)) & UInt32(0xff))
    buf[unsafe_offset=off + 3] = UInt8((v >> UInt32(24)) & UInt32(0xff))


@no_inline
def _block_try_impl(tv: PF, tri: PI, n: Int, blk: PB, prim_base: Int, write: Bool, vx: PF, vy: PF, vz: PF, idx: Pointer[Int, MutUntrackedOrigin], e: Pointer[Int, MutUntrackedOrigin]) -> Bool:
    """Encode triangles tri[0..n) (indices into tv, nine floats each) as one 128-byte block. False when they do not
    fit (more than 16 distinct vertices, a first triangle with a repeated vertex, or the stream exceeds its space)."""
    var nv = 0
    for t in range(n):
        for k in range(3):
            var x = tv[unsafe_offset=Int(tri[unsafe_offset=t]) * 9 + k * 3]
            var y = tv[unsafe_offset=Int(tri[unsafe_offset=t]) * 9 + k * 3 + 1]
            var z = tv[unsafe_offset=Int(tri[unsafe_offset=t]) * 9 + k * 3 + 2]
            var found = -1
            for q in range(nv):
                if vx[unsafe_offset=q] == x and vy[unsafe_offset=q] == y and vz[unsafe_offset=q] == z:
                    found = q
            if found < 0:
                if nv >= 16:
                    return False
                vx[unsafe_offset=nv] = x; vy[unsafe_offset=nv] = y; vz[unsafe_offset=nv] = z
                found = nv
                nv += 1
            idx[unsafe_offset=t * 3 + k] = found
    if idx[unsafe_offset=0] != 0 or idx[unsafe_offset=1] != 1 or idx[unsafe_offset=2] != 2:
        return False
    var m = 0
    for q in range(nv):
        for a in range(3):
            var c = vx[unsafe_offset=q]
            if a == 1: c = vy[unsafe_offset=q]
            if a == 2: c = vz[unsafe_offset=q]
            var man = _fbits(c) & UInt32(0x7fffff)
            if man != UInt32(0):
                var tz = 0
                while ((man >> UInt32(tz)) & UInt32(1)) == UInt32(0):
                    tz += 1
                m = max(m, 23 - tz)
    for a in range(3):
        var c0 = vx[unsafe_offset=0]
        if a == 1: c0 = vy[unsafe_offset=0]
        if a == 2: c0 = vz[unsafe_offset=0]
        var r0 = _fbits(c0) >> UInt32(23)
        var span = UInt32(0)
        for q in range(1, nv):
            var c = vx[unsafe_offset=q]
            if a == 1: c = vy[unsafe_offset=q]
            if a == 2: c = vz[unsafe_offset=q]
            span = span | ((_fbits(c) >> UInt32(23)) ^ r0)
        e[unsafe_offset=a] = max(1, _bit_length(span))
    var total = (nv - 1) * (3 * m + e[unsafe_offset=0] + e[unsafe_offset=1] + e[unsafe_offset=2])
    if total > RT_STREAM_BITS:
        return False
    if not write:
        return True
    for i in range(128): blk[unsafe_offset=i] = UInt8(0)
    _put_u32(blk, 0, _fbits(vx[unsafe_offset=0])); _put_u32(blk, 4, _fbits(vy[unsafe_offset=0])); _put_u32(blk, 8, _fbits(vz[unsafe_offset=0]))
    var pos = 96
    for q in range(1, nv):
        for a in range(3):
            var c = vx[unsafe_offset=q]
            if a == 1: c = vy[unsafe_offset=q]
            if a == 2: c = vz[unsafe_offset=q]
            var f = _fbits(c)
            var man = UInt32(0)
            if m > 0:
                man = (f & UInt32(0x7fffff)) >> UInt32(23 - m)
            var ex = (f >> UInt32(23)) & ((UInt32(1) << UInt32(e[unsafe_offset=a])) - UInt32(1))
            var fld = UInt64(man | (ex << UInt32(m)))
            _put_bits(blk, pos, fld, m + e[unsafe_offset=a])
            pos += m + e[unsafe_offset=a]
    var s = 126 - 14 * (n - 1)
    var ow = 2 if n <= 4 else 3         # id offset width: ids are prim_base + j inside an aligned group of 4 or 8
    for j in range(1, n):
        var rec = UInt64(idx[unsafe_offset=j * 3] | (idx[unsafe_offset=j * 3 + 1] << 4) | (idx[unsafe_offset=j * 3 + 2] << 8)) << UInt64(2)
        _put_bits(blk, 0x68 * 8 + 126 - 14 * j, rec, 14)
        _put_bits(blk, 0x68 * 8 + s - ow * j, UInt64(j & ((1 << ow) - 1)), ow)
    _put_u32(blk, 0x78, UInt32(0x80000000) | UInt32(prim_base))
    # descriptor bit 15: 2-bit id offsets; bits 16..19 = 1: 3-bit offsets (blocks of 5..8 triangles)
    var desc = (UInt32(55 - m) << UInt32(24)) | (UInt32(n - 1) << UInt32(20)) | (UInt32(0x8000) if n <= 4 else UInt32(0x10000))
    desc = desc | (UInt32(m + e[unsafe_offset=2] - 1) << UInt32(10)) | (UInt32(m + e[unsafe_offset=1] - 1) << UInt32(5)) | UInt32(m + e[unsafe_offset=0] - 1)
    _put_u32(blk, 0x7c, desc)
    return True


@no_inline
def _block_try(tv: PF, tri: PI, n: Int, blk: PB, prim_base: Int, write: Bool) -> Bool:
    var vx = unsafe_alloc[Float32](16); var vy = unsafe_alloc[Float32](16); var vz = unsafe_alloc[Float32](16)
    var idx = unsafe_alloc[Int](3 * RT_BLOCK_TRIS); var e = unsafe_alloc[Int](3)
    var ok = _block_try_impl(tv, tri, n, blk, prim_base, write, vx, vy, vz, idx, e)
    vx.unsafe_free(); vy.unsafe_free(); vz.unsafe_free(); idx.unsafe_free(); e.unsafe_free()
    return ok


@no_inline
def _select(order: PI, tc: PD, lo: Int, hi: Int, kth: Int, ax: Int):
    """Rearrange order[lo..hi) so that position kth holds the element of that rank by centroid[ax] (ties by index)."""
    var l = lo
    var r = hi - 1
    while l < r:
        var pv = Int(order[unsafe_offset=(l + r) >> 1])
        var pk = tc[unsafe_offset=pv * 3 + ax]
        var i = l
        var j = r
        while i <= j:
            while True:
                var a = Int(order[unsafe_offset=i])
                var ka = tc[unsafe_offset=a * 3 + ax]
                if ka < pk or (ka == pk and a < pv):
                    i += 1
                else:
                    break
            while True:
                var b = Int(order[unsafe_offset=j])
                var kb = tc[unsafe_offset=b * 3 + ax]
                if kb > pk or (kb == pk and b > pv):
                    j -= 1
                else:
                    break
            if i <= j:
                var t = order[unsafe_offset=i]
                order[unsafe_offset=i] = order[unsafe_offset=j]
                order[unsafe_offset=j] = t
                i += 1
                j -= 1
        if kth <= j:
            r = j
        elif kth >= i:
            l = i
        else:
            return


comptime SAH_BINS = 48


@no_inline
def _split_sah(order: PI, tlo: PD, thi: PD, tc: PD, start: Int, count: Int, granule: Int = 1) -> Int:
    """Partition order[start..start + count) by the best binned surface-area split over the three centroid axes;
    returns the size of the left part (the median split when no bin split separates anything). With a granule the left
    part is moved to a multiple of it, so that the leaves below it come out full."""
    var bc = unsafe_alloc[Int32](SAH_BINS)
    var bb = unsafe_alloc[Float64](SAH_BINS * 6)
    var cl = unsafe_alloc[Float64](3); var ch = unsafe_alloc[Float64](3)
    for a in range(3):
        cl[unsafe_offset=a] = Float64(1e300); ch[unsafe_offset=a] = Float64(-1e300)
    for i in range(start, start + count):
        var k = Int(order[unsafe_offset=i])
        for a in range(3):
            cl[unsafe_offset=a] = min(cl[unsafe_offset=a], tc[unsafe_offset=k * 3 + a])
            ch[unsafe_offset=a] = max(ch[unsafe_offset=a], tc[unsafe_offset=k * 3 + a])
    var best_cost = Float64(1e300)
    var best_ax = -1
    var best_bin = 0
    for ax in range(3):
        var ext = ch[unsafe_offset=ax] - cl[unsafe_offset=ax]
        if ext <= 0.0:
            continue
        for b in range(SAH_BINS):
            bc[unsafe_offset=b] = Int32(0)
            for q in range(3):
                bb[unsafe_offset=b * 6 + q] = Float64(1e300); bb[unsafe_offset=b * 6 + 3 + q] = Float64(-1e300)
        var scale = Float64(SAH_BINS) / ext
        for i in range(start, start + count):
            var k = Int(order[unsafe_offset=i])
            var b = min(SAH_BINS - 1, Int((tc[unsafe_offset=k * 3 + ax] - cl[unsafe_offset=ax]) * scale))
            bc[unsafe_offset=b] = bc[unsafe_offset=b] + Int32(1)
            for q in range(3):
                bb[unsafe_offset=b * 6 + q] = min(bb[unsafe_offset=b * 6 + q], tlo[unsafe_offset=k * 3 + q])
                bb[unsafe_offset=b * 6 + 3 + q] = max(bb[unsafe_offset=b * 6 + 3 + q], thi[unsafe_offset=k * 3 + q])
        # right-to-left areas
        var rarea = unsafe_alloc[Float64](SAH_BINS)
        var rx0 = Float64(1e300); var ry0 = Float64(1e300); var rz0 = Float64(1e300)
        var rx1 = Float64(-1e300); var ry1 = Float64(-1e300); var rz1 = Float64(-1e300)
        for back in range(SAH_BINS):
            var b = SAH_BINS - 1 - back
            if bc[unsafe_offset=b] > Int32(0):
                rx0 = min(rx0, bb[unsafe_offset=b * 6]); ry0 = min(ry0, bb[unsafe_offset=b * 6 + 1]); rz0 = min(rz0, bb[unsafe_offset=b * 6 + 2])
                rx1 = max(rx1, bb[unsafe_offset=b * 6 + 3]); ry1 = max(ry1, bb[unsafe_offset=b * 6 + 4]); rz1 = max(rz1, bb[unsafe_offset=b * 6 + 5])
            var dx = rx1 - rx0; var dy = ry1 - ry0; var dz = rz1 - rz0
            rarea[unsafe_offset=b] = Float64(0) if dx < 0.0 else 2.0 * (dx * dy + dy * dz + dz * dx)
        var lx0 = Float64(1e300); var ly0 = Float64(1e300); var lz0 = Float64(1e300)
        var lx1 = Float64(-1e300); var ly1 = Float64(-1e300); var lz1 = Float64(-1e300)
        var ln = 0
        for b in range(SAH_BINS - 1):
            if bc[unsafe_offset=b] > Int32(0):
                lx0 = min(lx0, bb[unsafe_offset=b * 6]); ly0 = min(ly0, bb[unsafe_offset=b * 6 + 1]); lz0 = min(lz0, bb[unsafe_offset=b * 6 + 2])
                lx1 = max(lx1, bb[unsafe_offset=b * 6 + 3]); ly1 = max(ly1, bb[unsafe_offset=b * 6 + 4]); lz1 = max(lz1, bb[unsafe_offset=b * 6 + 5])
            ln += Int(bc[unsafe_offset=b])
            if ln == 0 or ln == count:
                continue
            var dx = lx1 - lx0; var dy = ly1 - ly0; var dz = lz1 - lz0
            var la = 2.0 * (dx * dy + dy * dz + dz * dx)
            var cost = la * Float64(ln) + rarea[unsafe_offset=b + 1] * Float64(count - ln)
            if cost < best_cost:
                best_cost = cost; best_ax = ax; best_bin = b
        rarea.unsafe_free()
    var left = 0
    var used_ax = best_ax
    if best_ax >= 0:
        var ext = ch[unsafe_offset=best_ax] - cl[unsafe_offset=best_ax]
        var scale = Float64(SAH_BINS) / ext
        var i = start
        var j = start + count - 1
        while i <= j:
            var k = Int(order[unsafe_offset=i])
            var b = min(SAH_BINS - 1, Int((tc[unsafe_offset=k * 3 + best_ax] - cl[unsafe_offset=best_ax]) * scale))
            if b <= best_bin:
                i += 1
            else:
                var t = order[unsafe_offset=i]
                order[unsafe_offset=i] = order[unsafe_offset=j]
                order[unsafe_offset=j] = t
                j -= 1
        left = i - start
    if left <= 0 or left >= count:
        left = count >> 1
        var ax = 0
        var bestext = ch[unsafe_offset=0] - cl[unsafe_offset=0]
        if ch[unsafe_offset=1] - cl[unsafe_offset=1] > bestext:
            ax = 1; bestext = ch[unsafe_offset=1] - cl[unsafe_offset=1]
        if ch[unsafe_offset=2] - cl[unsafe_offset=2] > bestext:
            ax = 2
        used_ax = ax
        _select(order, tc, start, start + count, start + left, ax)
    if granule > 1 and count > granule and used_ax >= 0:
        var q = (left + granule // 2) // granule * granule
        q = max(granule, min(q, (count - 1) // granule * granule))
        if q != left:
            _select(order, tc, start, start + count, start + q, used_ax)
            left = q
    bc.unsafe_free(); bb.unsafe_free(); cl.unsafe_free(); ch.unsafe_free()
    return left



@no_inline
def _range_box(order: PI, tlo: PD, thi: PD, start: Int, count: Int, box: PD):
    """box[0..3) = min corner, box[3..6) = max corner of the triangles order[start..start + count)."""
    var lx = Float64(1e300); var ly = Float64(1e300); var lz = Float64(1e300)
    var hx = Float64(-1e300); var hy = Float64(-1e300); var hz = Float64(-1e300)
    for i in range(start, start + count):
        var k = Int(order[unsafe_offset=i])
        lx = min(lx, tlo[unsafe_offset=k * 3]); ly = min(ly, tlo[unsafe_offset=k * 3 + 1]); lz = min(lz, tlo[unsafe_offset=k * 3 + 2])
        hx = max(hx, thi[unsafe_offset=k * 3]); hy = max(hy, thi[unsafe_offset=k * 3 + 1]); hz = max(hz, thi[unsafe_offset=k * 3 + 2])
    box[unsafe_offset=0] = lx; box[unsafe_offset=1] = ly; box[unsafe_offset=2] = lz; box[unsafe_offset=3] = hx; box[unsafe_offset=4] = hy; box[unsafe_offset=5] = hz


@no_inline
def _leaf_blocks(tvv: PF, u_first: PI, u_n: PI, order: PI, start: Int, count: Int, tri4: PI, scratch: PB, slots: PI) -> Int:
    """Blocks the units order[start..start + count) need when placed together: 1 if they encode as one block (at most
    four triangles), 0 if they cannot be a leaf (too many triangles), else the count of a greedy split."""
    var nt = 0
    for i in range(count):
        var u = Int(order[unsafe_offset=start + i])
        for q in range(Int(u_n[unsafe_offset=u])):
            if nt >= RT_BLOCK_TRIS:
                return 0
            tri4[unsafe_offset=nt] = Int32(Int(u_first[unsafe_offset=u]) + q); nt += 1
    if _block_try(tvv, tri4, nt, scratch, 0, False):
        slots[unsafe_offset=0] = Int32((nt + RT_SLOT_TRIS - 1) // RT_SLOT_TRIS)
        return 1
    if count > 1:
        return 0
    # a single unit that does not encode as one block: split it by triangle (a single triangle always encodes)
    slots[unsafe_offset=0] = Int32(nt)
    return nt


@no_inline
def _build_binary(order: PI, tlo: PD, thi: PD, tc: PD, tvv: PF, u_first: PI, u_n: PI, m: Int, bn_s: PI, bn_c: PI, bn_l: PI, bn_r: PI, bn_nb: PI, bn_box: PD, sah: Bool) -> Int:
    """Binary tree over the units (a unit is one triangle or a quad's two); a leaf is the units that become one block.
    bn_nb gets the blocks of a leaf. Returns the node count."""
    var stack = unsafe_alloc[Int32](128 + 64)
    var nb = 1
    bn_s[unsafe_offset=0] = Int32(0); bn_c[unsafe_offset=0] = Int32(m)
    var sp: Int
    stack[unsafe_offset=0] = Int32(0); sp = 1
    var cb = unsafe_alloc[Float64](6)
    var tri4 = unsafe_alloc[Int32](RT_BLOCK_TRIS)
    var scratch = unsafe_alloc[UInt8](128)
    var lslots = unsafe_alloc[Int32](1)
    while sp > 0:
        sp -= 1
        var ni = Int(stack[unsafe_offset=sp])
        var start = Int(bn_s[unsafe_offset=ni]); var count = Int(bn_c[unsafe_offset=ni])
        _range_box(order, tlo, thi, start, count, cb)
        for q in range(6): bn_box[unsafe_offset=ni * 6 + q] = cb[unsafe_offset=q]
        if count <= RT_BLOCK_TRIS:
            var nbk = _leaf_blocks(tvv, u_first, u_n, order, start, count, tri4, scratch, lslots)
            if nbk > 0:
                bn_l[unsafe_offset=ni] = Int32(-1); bn_r[unsafe_offset=ni] = Int32(-1)
                bn_nb[unsafe_offset=ni] = lslots[unsafe_offset=0]
                continue
        # axis of the largest centroid extent
        var cl0 = Float64(1e300); var cl1 = Float64(1e300); var cl2 = Float64(1e300)
        var ch0 = Float64(-1e300); var ch1 = Float64(-1e300); var ch2 = Float64(-1e300)
        for i in range(start, start + count):
            var k = Int(order[unsafe_offset=i])
            cl0 = min(cl0, tc[unsafe_offset=k * 3]); ch0 = max(ch0, tc[unsafe_offset=k * 3])
            cl1 = min(cl1, tc[unsafe_offset=k * 3 + 1]); ch1 = max(ch1, tc[unsafe_offset=k * 3 + 1])
            cl2 = min(cl2, tc[unsafe_offset=k * 3 + 2]); ch2 = max(ch2, tc[unsafe_offset=k * 3 + 2])
        var ax = 0
        var best = ch0 - cl0
        if ch1 - cl1 > best:
            ax = 1; best = ch1 - cl1
        if ch2 - cl2 > best:
            ax = 2
        var half = count >> 1
        if sah:
            half = _split_sah(order, tlo, thi, tc, start, count)
        else:
            _select(order, tc, start, start + count, start + half, ax)
        var li = nb; var ri = nb + 1
        nb += 2
        bn_l[unsafe_offset=ni] = Int32(li); bn_r[unsafe_offset=ni] = Int32(ri)
        bn_s[unsafe_offset=li] = Int32(start); bn_c[unsafe_offset=li] = Int32(half)
        bn_s[unsafe_offset=ri] = Int32(start + half); bn_c[unsafe_offset=ri] = Int32(count - half)
        stack[unsafe_offset=sp] = Int32(ri); stack[unsafe_offset=sp + 1] = Int32(li)
        sp += 2
    stack.unsafe_free()
    cb.unsafe_free()
    tri4.unsafe_free()
    scratch.unsafe_free()
    lslots.unsafe_free()
    return nb


comptime BOX_SUBTREE = 65536       # boxes a subtree task covers at most
comptime BOX_MAX_TASKS = 4096


@no_inline
def _box_subtree(order: PI, tlo: PD, thi: PD, tc: PD, root: Int, nb0: Int, bn_s: PI, bn_c: PI, bn_l: PI, bn_r: PI, bn_nb: PI, bn_box: PD) -> Int:
    """Binary tree over the boxes of node `root` (bn_s, bn_c set), its other nodes from nb0 on; a leaf holds at most
    RT_LEAF_TRIS boxes. Returns the next free node."""
    var stack = unsafe_alloc[Int32](128 + 64)
    var nb = nb0
    stack[unsafe_offset=0] = Int32(root)
    var sp = 1
    var cb = unsafe_alloc[Float64](6)
    while sp > 0:
        sp -= 1
        var ni = Int(stack[unsafe_offset=sp])
        var start = Int(bn_s[unsafe_offset=ni]); var count = Int(bn_c[unsafe_offset=ni])
        _range_box(order, tlo, thi, start, count, cb)
        for q in range(6): bn_box[unsafe_offset=ni * 6 + q] = cb[unsafe_offset=q]
        if count <= RT_LEAF_TRIS:
            bn_l[unsafe_offset=ni] = Int32(-1); bn_r[unsafe_offset=ni] = Int32(-1); bn_nb[unsafe_offset=ni] = Int32(count)
            continue
        var half = _split_sah(order, tlo, thi, tc, start, count)
        var li = nb; var ri = nb + 1
        nb += 2
        bn_l[unsafe_offset=ni] = Int32(li); bn_r[unsafe_offset=ni] = Int32(ri)
        bn_s[unsafe_offset=li] = Int32(start); bn_c[unsafe_offset=li] = Int32(half)
        bn_s[unsafe_offset=ri] = Int32(start + half); bn_c[unsafe_offset=ri] = Int32(count - half)
        stack[unsafe_offset=sp] = Int32(ri); stack[unsafe_offset=sp + 1] = Int32(li)
        sp += 2
    stack.unsafe_free(); cb.unsafe_free()
    return nb


@no_inline
def _build_box_binary(order: PI, tlo: PD, thi: PD, tc: PD, m: Int, first: Int, bn_s: PI, bn_c: PI, bn_l: PI, bn_r: PI, bn_nb: PI, bn_box: PD) -> Int:
    """Binary tree over boxes from node `first` on, on all cores: the top split level by level (a level's nodes in
    parallel), then subtrees of at most BOX_SUBTREE boxes, each in its own node range (twice its boxes). A child's index is
    above its parent's; unused nodes have no children and no boxes. Returns the end of the node range."""
    bn_s[unsafe_offset=first] = Int32(0); bn_c[unsafe_offset=first] = Int32(m)
    var level = unsafe_alloc[Int32](BOX_MAX_TASKS); var next_level = unsafe_alloc[Int32](BOX_MAX_TASKS)
    var half = unsafe_alloc[Int](BOX_MAX_TASKS)
    var subs = unsafe_alloc[Int32](BOX_MAX_TASKS)
    var n_subs = 0
    level[unsafe_offset=0] = Int32(first)
    var n_level = 1
    var nb = first + 1
    while n_level > 0:
        def split_one(k: Int) {imm}:
            var ni = Int(level[unsafe_offset=k])
            var start = Int(bn_s[unsafe_offset=ni]); var count = Int(bn_c[unsafe_offset=ni])
            half[unsafe_offset=k] = -1
            if count > BOX_SUBTREE:
                _range_box(order, tlo, thi, start, count, bn_box.unsafe_offset(ni * 6))
                half[unsafe_offset=k] = _split_sah(order, tlo, thi, tc, start, count)

        parallelize(split_one, n_level)
        var n_next = 0
        for k in range(n_level):
            var ni = Int(level[unsafe_offset=k])
            if half[unsafe_offset=k] < 0 or n_subs + n_next + 2 * (n_level - k) > BOX_MAX_TASKS:
                subs[unsafe_offset=n_subs] = Int32(ni); n_subs += 1
                continue
            var start = Int(bn_s[unsafe_offset=ni]); var count = Int(bn_c[unsafe_offset=ni])
            var h = half[unsafe_offset=k]
            bn_l[unsafe_offset=ni] = Int32(nb); bn_r[unsafe_offset=ni] = Int32(nb + 1); bn_nb[unsafe_offset=ni] = Int32(0)
            bn_s[unsafe_offset=nb] = Int32(start); bn_c[unsafe_offset=nb] = Int32(h)
            bn_s[unsafe_offset=nb + 1] = Int32(start + h); bn_c[unsafe_offset=nb + 1] = Int32(count - h)
            next_level[unsafe_offset=n_next] = Int32(nb); next_level[unsafe_offset=n_next + 1] = Int32(nb + 1)
            nb += 2; n_next += 2
        var tmp = level; level = next_level; next_level = tmp
        n_level = n_next
    var base = unsafe_alloc[Int](n_subs + 1)
    base[unsafe_offset=0] = nb
    for j in range(n_subs):
        base[unsafe_offset=j + 1] = base[unsafe_offset=j] + 2 * Int(bn_c[unsafe_offset=Int(subs[unsafe_offset=j])])
    var ends = unsafe_alloc[Int](max(n_subs, 1))
    var next_sub = unsafe_alloc[Int32](1)
    next_sub[unsafe_offset=0] = Int32(0)

    def build_worker(_w: Int) {imm}:
        while True:
            var j = Int(Atomic.fetch_add(next_sub, Int32(1)))
            if j >= n_subs:
                break
            ends[unsafe_offset=j] = _box_subtree(order, tlo, thi, tc, Int(subs[unsafe_offset=j]), base[unsafe_offset=j], bn_s, bn_c, bn_l, bn_r, bn_nb, bn_box)
            for i in range(ends[unsafe_offset=j], base[unsafe_offset=j + 1]):
                bn_l[unsafe_offset=i] = Int32(-1); bn_r[unsafe_offset=i] = Int32(-1); bn_nb[unsafe_offset=i] = Int32(0)

    parallelize(build_worker, min(num_performance_cores(), max(n_subs, 1)))
    var end = base[unsafe_offset=n_subs]
    level.unsafe_free(); next_level.unsafe_free(); half.unsafe_free(); subs.unsafe_free(); base.unsafe_free(); ends.unsafe_free(); next_sub.unsafe_free()
    return end


@no_inline
def _hw_children(root: Int, bn_nb: PI, bn_l: PI, bn_r: PI, bn_box: PD, kids: PI) -> Int:
    """Expand a binary node into up to 12 hardware children (binary node ids in kids); returns how many."""
    kids[unsafe_offset=0] = Int32(root)
    var nk = 1
    while nk < 12:
        var pick = -1
        var pa = Float64(-1.0)
        for i in range(nk):
            var kb = Int(kids[unsafe_offset=i])
            var c = Int(bn_nb[unsafe_offset=kb])
            var dx = bn_box[unsafe_offset=kb * 6 + 3] - bn_box[unsafe_offset=kb * 6]
            var dy = bn_box[unsafe_offset=kb * 6 + 4] - bn_box[unsafe_offset=kb * 6 + 1]
            var dz = bn_box[unsafe_offset=kb * 6 + 5] - bn_box[unsafe_offset=kb * 6 + 2]
            var area = dx * dy + dy * dz + dz * dx
            if c > RT_LEAF_TRIS and area > pa:
                pick = i; pa = area
        if pick < 0:
            break
        var bi = Int(kids[unsafe_offset=pick])
        # keep the left child in place, put the right one after it
        for t in range(nk - pick - 1):
            var j = nk - t
            kids[unsafe_offset=j] = kids[unsafe_offset=j - 1]
        kids[unsafe_offset=pick] = bn_l[unsafe_offset=bi]
        kids[unsafe_offset=pick + 1] = bn_r[unsafe_offset=bi]
        nk += 1
    return nk


@no_inline
def _quant_frame(lox: Float64, loy: Float64, loz: Float64, hix: Float64, hiy: Float64, hiz: Float64, e: PI, org: PF):
    """Exponents and origin of a node frame covering the box (the origin is rounded down to a float)."""
    for a in range(3):
        var lo = lox
        var hi = hix
        if a == 1:
            lo = loy; hi = hiy
        if a == 2:
            lo = loz; hi = hiz
        var ext = hi - lo
        var ee = 0
        if ext > 0.0:
            var bits = bitcast[DType.uint64, 1](ext)
            ee = Int((bits >> UInt64(52)) & UInt64(0x7ff)) - 1022
            ee = max(1, min(254, ee + 127))
        e[unsafe_offset=a] = Int32(ee)
        var o = Float32(lo)
        if Float64(o) > lo:
            var ob = _fbits(o)
            if o > Float32(0):
                ob = ob - UInt32(1)
            elif o < Float32(0):
                ob = ob + UInt32(1)
            else:
                ob = UInt32(0x80000001)
            o = bitcast[DType.float32, 1](ob)
        org[unsafe_offset=a] = o


@no_inline
def _put_slot(ob: PB, off: Int, k: Int, e: PI, org: PF, lox: Float64, loy: Float64, loz: Float64, hix: Float64, hiy: Float64, hiz: Float64, code: Int):
    var o = off + 0x20 + 8 * k
    for a in range(3):
        var lo = lox
        var hi = hix
        if a == 1:
            lo = loy; hi = hiy
        if a == 2:
            lo = loz; hi = hiz
        var sc = Float64(1.0)
        if e[unsafe_offset=a] != Int32(0):
            sc = Float64(bitcast[DType.float32, 1](UInt32(e[unsafe_offset=a]) << UInt32(23)))
        var l = Int(floor((lo - Float64(org[unsafe_offset=a])) / sc * 255.0 - 1e-6))
        var h = Int(ceil((hi - Float64(org[unsafe_offset=a])) / sc * 255.0 + 1e-6))
        l = max(0, min(255, l)); h = max(0, min(255, h))
        ob[unsafe_offset=o + a * 2] = UInt8(l)
        ob[unsafe_offset=o + a * 2 + 1] = UInt8(h)
    ob[unsafe_offset=o + 6] = UInt8(0xff)
    ob[unsafe_offset=o + 7] = UInt8(code)


@no_inline
def _put_empty_slot(ob: PB, off: Int, k: Int):
    var o = off + 0x20 + 8 * k
    for q in range(8): ob[unsafe_offset=o + q] = UInt8(0)
    ob[unsafe_offset=o + 4] = UInt8(0xff)


@no_inline
def rt_patch_ids(bytes: PB, n_nodes: Int, n_blocks: Int, id_base: Int):
    """Shift the primitive ids of a BLAS built with id_base 0 (so BLASes can be built concurrently and numbered after)."""
    var blk_base = RT_ROOT + n_nodes * RT_NODE
    for b in range(n_blocks):
        var o = blk_base + b * RT_NODE + 0x78
        var v = UInt32(bytes[unsafe_offset=o]) | (UInt32(bytes[unsafe_offset=o + 1]) << 8) | (UInt32(bytes[unsafe_offset=o + 2]) << 16) | (UInt32(bytes[unsafe_offset=o + 3]) << 24)
        _put_u32(bytes, o, v + UInt32(id_base))


def rt_build_blas(tv: PF, n: Int, tight: Bool = True, sah: Bool = True, quads: Bool = False, id_base: Int = 0, pair_ok: PB = PB.unsafe_dangling(),
                  bx: PF = PF.unsafe_dangling(), n_bx: Int = 0) -> RtBlas:
    """Build a BLAS over n triangles (tv: nine floats each). Triangles with a repeated vertex are left out. With quads,
    triangles 2q and 2q + 1 with pair_ok[2q] set are a quad that stays in one block when possible. The n_bx boxes (bx: min
    corner, max corner) become AABB leaves under the same root: the trace stops at them with a candidate id (gid_to_box)."""
    # valid triangles
    var vid = unsafe_alloc[Int32](max(n, 1))
    var m = 0
    for t in range(n):
        var a = tv.unsafe_offset(t * 9)
        var d01 = a[unsafe_offset=0] == a[unsafe_offset=3] and a[unsafe_offset=1] == a[unsafe_offset=4] and a[unsafe_offset=2] == a[unsafe_offset=5]
        var d02 = a[unsafe_offset=0] == a[unsafe_offset=6] and a[unsafe_offset=1] == a[unsafe_offset=7] and a[unsafe_offset=2] == a[unsafe_offset=8]
        var d12 = a[unsafe_offset=3] == a[unsafe_offset=6] and a[unsafe_offset=4] == a[unsafe_offset=7] and a[unsafe_offset=5] == a[unsafe_offset=8]
        if not (d01 or d02 or d12):
            vid[unsafe_offset=m] = Int32(t); m += 1
    var tvv = unsafe_alloc[Float32](9 * max(m, 1))
    for k in range(m):
        var s = tv.unsafe_offset(Int(vid[unsafe_offset=k]) * 9)
        for q in range(9): tvv[unsafe_offset=k * 9 + q] = s[unsafe_offset=q]
    # units: a valid pair 2q, 2q + 1 (consecutive valid triangles) or a single triangle
    var u_first = unsafe_alloc[Int32](max(m, 1)); var u_n = unsafe_alloc[Int32](max(m, 1))
    var nu = 0
    var k0 = 0
    while k0 < m:
        var pair = False
        if quads and k0 + 1 < m:
            var t0 = Int(vid[unsafe_offset=k0])
            pair = Int(vid[unsafe_offset=k0 + 1]) == t0 + 1 and pair_ok[unsafe_offset=t0] != UInt8(0)
        u_first[unsafe_offset=nu] = Int32(k0); u_n[unsafe_offset=nu] = Int32(2 if pair else 1)
        nu += 1
        k0 += 2 if pair else 1
    var tlo = unsafe_alloc[Float64](3 * max(nu, 1)); var thi = unsafe_alloc[Float64](3 * max(nu, 1)); var tc = unsafe_alloc[Float64](3 * max(nu, 1))
    var order = unsafe_alloc[Int32](max(nu, 1))
    for u in range(nu):
        for a in range(3):
            var lo = Float64(1e300); var hi = Float64(-1e300)
            for q in range(Int(u_n[unsafe_offset=u])):
                var tk = Int(u_first[unsafe_offset=u]) + q
                for c in range(3):
                    var x = Float64(tvv[unsafe_offset=tk * 9 + c * 3 + a])
                    lo = min(lo, x); hi = max(hi, x)
            tlo[unsafe_offset=u * 3 + a] = lo; thi[unsafe_offset=u * 3 + a] = hi
            tc[unsafe_offset=u * 3 + a] = (lo + hi) * 0.5
        order[unsafe_offset=u] = Int32(u)
    var nbmax = 2 * nu + 2 * n_bx + 2 + BOX_MAX_TASKS
    var bn_s = unsafe_alloc[Int32](nbmax); var bn_c = unsafe_alloc[Int32](nbmax)
    var bn_l = unsafe_alloc[Int32](nbmax); var bn_r = unsafe_alloc[Int32](nbmax)
    var bn_nb = unsafe_alloc[Int32](nbmax)
    var bn_box = unsafe_alloc[Float64](6 * nbmax)
    var nb = 0
    if nu > 0:
        nb = _build_binary(order, tlo, thi, tc, tvv, u_first, u_n, nu, bn_s, bn_c, bn_l, bn_r, bn_nb, bn_box, sah)
    for back in range(nb):
        var i = nb - 1 - back
        if Int(bn_l[unsafe_offset=i]) >= 0:
            bn_nb[unsafe_offset=i] = bn_nb[unsafe_offset=Int(bn_l[unsafe_offset=i])] + bn_nb[unsafe_offset=Int(bn_r[unsafe_offset=i])]
    # boxes: their own binary tree after the triangles' nodes (box nodes are those >= box0), joined under a new root
    var box0 = nb
    var b_order = unsafe_alloc[Int32](max(n_bx, 1))
    var b_lo = unsafe_alloc[Float64](3 * max(n_bx, 1)); var b_hi = unsafe_alloc[Float64](3 * max(n_bx, 1)); var b_c = unsafe_alloc[Float64](3 * max(n_bx, 1))
    var broot = 0
    if n_bx > 0:
        for j in range(n_bx):
            for a in range(3):
                b_lo[unsafe_offset=j * 3 + a] = Float64(bx[unsafe_offset=j * 6 + a]); b_hi[unsafe_offset=j * 3 + a] = Float64(bx[unsafe_offset=j * 6 + 3 + a])
                b_c[unsafe_offset=j * 3 + a] = (b_lo[unsafe_offset=j * 3 + a] + b_hi[unsafe_offset=j * 3 + a]) * 0.5
            b_order[unsafe_offset=j] = Int32(j)
        var bend = _build_box_binary(b_order, b_lo, b_hi, b_c, n_bx, box0, bn_s, bn_c, bn_l, bn_r, bn_nb, bn_box)
        for back in range(bend - box0):
            var i = bend - 1 - back
            if Int(bn_l[unsafe_offset=i]) >= 0:
                bn_nb[unsafe_offset=i] = bn_nb[unsafe_offset=Int(bn_l[unsafe_offset=i])] + bn_nb[unsafe_offset=Int(bn_r[unsafe_offset=i])]
        broot = box0
        nb = bend
        if nu > 0:
            # a mixed node is never a leaf: its slot count is set above a leaf's
            broot = nb; nb += 1
            bn_l[unsafe_offset=broot] = Int32(0); bn_r[unsafe_offset=broot] = Int32(box0)
            bn_nb[unsafe_offset=broot] = Int32(max(Int(bn_nb[unsafe_offset=0]) + Int(bn_nb[unsafe_offset=box0]), RT_LEAF_TRIS + 1))
            for q in range(3):
                bn_box[unsafe_offset=broot * 6 + q] = min(bn_box[unsafe_offset=q], bn_box[unsafe_offset=box0 * 6 + q])
                bn_box[unsafe_offset=broot * 6 + 3 + q] = max(bn_box[unsafe_offset=3 + q], bn_box[unsafe_offset=box0 * 6 + 3 + q])
    # hardware nodes in breadth-first order: a node with at most 12 blocks is a leaf node, else it has up to 12 children
    var hn_bn = unsafe_alloc[Int32](max(nb, 1)); var hn_par = unsafe_alloc[Int32](max(nb, 1)); var hn_sib = unsafe_alloc[Int32](max(nb, 1))
    var hn_first = unsafe_alloc[Int32](max(nb, 1)); var hn_nk = unsafe_alloc[Int32](max(nb, 1))
    var hn_blk0 = unsafe_alloc[Int32](max(nb, 1)); var hn_nblk = unsafe_alloc[Int32](max(nb, 1))
    var kids = unsafe_alloc[Int32](16)
    var nn = 0
    if nu > 0 or n_bx > 0:
        hn_bn[unsafe_offset=0] = Int32(broot); hn_par[unsafe_offset=0] = Int32(-1); hn_sib[unsafe_offset=0] = Int32(0); nn = 1
    var hi = 0
    while hi < nn:
        var bi = Int(hn_bn[unsafe_offset=hi])
        hn_nk[unsafe_offset=hi] = Int32(0); hn_first[unsafe_offset=hi] = Int32(0)
        if Int(bn_nb[unsafe_offset=bi]) > RT_LEAF_TRIS:
            var nk = _hw_children(bi, bn_nb, bn_l, bn_r, bn_box, kids)
            hn_nk[unsafe_offset=hi] = Int32(nk); hn_first[unsafe_offset=hi] = Int32(nn)
            for k in range(nk):
                hn_bn[unsafe_offset=nn] = kids[unsafe_offset=k]; hn_par[unsafe_offset=nn] = Int32(hi); hn_sib[unsafe_offset=nn] = Int32(k)
                nn += 1
        hi += 1
    # the blocks of every leaf node: for each binary leaf below it (left to right) one block, or a greedy split
    var blk_tri = unsafe_alloc[Int32](RT_BLOCK_TRIS * max(m, 1)); var blk_cnt = unsafe_alloc[Int32](max(m, 1)); var blk_box = unsafe_alloc[Float64](6 * max(m, 1))
    var nblk = 0
    var dstack = unsafe_alloc[Int32](192)
    var tri4 = unsafe_alloc[Int32](RT_BLOCK_TRIS)
    var scratch = unsafe_alloc[UInt8](128)
    for h in range(nn):
        hn_blk0[unsafe_offset=h] = Int32(nblk); hn_nblk[unsafe_offset=h] = Int32(0)
        if Int(hn_nk[unsafe_offset=h]) != 0 or Int(hn_bn[unsafe_offset=h]) >= box0:
            continue
        var sp = 1
        dstack[unsafe_offset=0] = hn_bn[unsafe_offset=h]
        while sp > 0:
            sp -= 1
            var bi = Int(dstack[unsafe_offset=sp])
            if Int(bn_l[unsafe_offset=bi]) >= 0:
                dstack[unsafe_offset=sp] = bn_r[unsafe_offset=bi]; dstack[unsafe_offset=sp + 1] = bn_l[unsafe_offset=bi]
                sp += 2
                continue
            var nt = 0
            for i in range(Int(bn_c[unsafe_offset=bi])):
                var u = Int(order[unsafe_offset=Int(bn_s[unsafe_offset=bi]) + i])
                for q in range(Int(u_n[unsafe_offset=u])):
                    tri4[unsafe_offset=nt] = Int32(Int(u_first[unsafe_offset=u]) + q); nt += 1
            var p = 0
            while p < nt:
                var take = nt - p
                if take > 1 and Int(bn_nb[unsafe_offset=bi]) >= nt:
                    take = 1                      # a unit that did not encode as one block: one triangle a block
                for q in range(take): blk_tri[unsafe_offset=nblk * RT_BLOCK_TRIS + q] = tri4[unsafe_offset=p + q]
                blk_cnt[unsafe_offset=nblk] = Int32(take)
                for q in range(6): blk_box[unsafe_offset=nblk * 6 + q] = bn_box[unsafe_offset=bi * 6 + q]
                nblk += 1; hn_nblk[unsafe_offset=h] = hn_nblk[unsafe_offset=h] + Int32(1)
                p += take
    # bytes
    var size = RT_ROOT + (nn + nblk) * RT_NODE
    var out = unsafe_alloc[UInt8](size)
    for i in range(size): out[unsafe_offset=i] = UInt8(0)
    _put_u32(out, 0, UInt32(0x00100003)); _put_u32(out, 4, UInt32(8)); _put_u32(out, 0x20, UInt32(RT_ROOT))
    var gid = unsafe_alloc[Int32](RT_BLOCK_TRIS * max(nblk, 1))
    for i in range(RT_BLOCK_TRIS * max(nblk, 1)): gid[unsafe_offset=i] = Int32(-1)
    var blk_base = RT_ROOT + nn * RT_NODE
    var next_id = 0
    for b in range(nblk):
        var cnt = Int(blk_cnt[unsafe_offset=b])
        var align = 4 if cnt <= 4 else 8
        next_id = (next_id + align - 1) // align * align
        for q in range(cnt):
            tri4[unsafe_offset=q] = blk_tri[unsafe_offset=b * RT_BLOCK_TRIS + q]
            gid[unsafe_offset=next_id + q] = vid[unsafe_offset=Int(tri4[unsafe_offset=q])]
        var blkp = out.unsafe_offset(blk_base + b * RT_NODE)
        _ = _block_try(tvv, tri4, cnt, blkp.unsafe_origin_cast[MutUntrackedOrigin](), id_base + next_id, True)
        next_id += cnt
    next_id = (next_id + 7) // 8 * 8     # the next BLAS's ids start aligned
    var e = unsafe_alloc[Int32](3); var org = unsafe_alloc[Float32](3)
    var sbox = unsafe_alloc[Float64](6 * 12)       # slot boxes of the node being written
    var scode = unsafe_alloc[Int32](12)
    var gid_box = unsafe_alloc[Int32](max(n_bx, 1))
    var n_gbox = 0
    var bstack = unsafe_alloc[Int32](192)
    for h in range(nn):
        var off = RT_ROOT + h * RT_NODE
        var ns: Int
        var w10 = 0
        var w1c = 1
        if Int(hn_nk[unsafe_offset=h]) != 0:
            ns = Int(hn_nk[unsafe_offset=h])
            for k in range(ns):
                var kb = Int(hn_bn[unsafe_offset=Int(hn_first[unsafe_offset=h]) + k])
                for q in range(6): sbox[unsafe_offset=k * 6 + q] = bn_box[unsafe_offset=kb * 6 + q]
                scode[unsafe_offset=k] = Int32(0)
            w1c = Int(hn_first[unsafe_offset=h]) - h
            out[unsafe_offset=off] = UInt8(0x40)
        elif Int(hn_bn[unsafe_offset=h]) >= box0:
            # AABB leaf: one slot a box, ids payload + slot
            ns = 0
            w10 = n_gbox
            var sp = 1
            bstack[unsafe_offset=0] = hn_bn[unsafe_offset=h]
            while sp > 0:
                sp -= 1
                var bi = Int(bstack[unsafe_offset=sp])
                if Int(bn_l[unsafe_offset=bi]) >= 0:
                    bstack[unsafe_offset=sp] = bn_r[unsafe_offset=bi]; bstack[unsafe_offset=sp + 1] = bn_l[unsafe_offset=bi]
                    sp += 2
                    continue
                for i in range(Int(bn_c[unsafe_offset=bi])):
                    var j = Int(b_order[unsafe_offset=Int(bn_s[unsafe_offset=bi]) + i])
                    for q in range(3):
                        sbox[unsafe_offset=ns * 6 + q] = b_lo[unsafe_offset=j * 3 + q]; sbox[unsafe_offset=ns * 6 + 3 + q] = b_hi[unsafe_offset=j * 3 + q]
                    scode[unsafe_offset=ns] = Int32(0x41)
                    gid_box[unsafe_offset=n_gbox] = Int32(j); n_gbox += 1; ns += 1
            out[unsafe_offset=off] = UInt8(0x00)
        else:
            ns = 0
            for g in range(Int(hn_nblk[unsafe_offset=h])):
                var b = Int(hn_blk0[unsafe_offset=h]) + g
                var cnt = Int(blk_cnt[unsafe_offset=b])
                var nsl = (cnt + RT_SLOT_TRIS - 1) // RT_SLOT_TRIS
                for sl in range(nsl):
                    var lo = Float64(1e300); var lo1 = Float64(1e300); var lo2 = Float64(1e300)
                    var hx = Float64(-1e300); var hy = Float64(-1e300); var hz = Float64(-1e300)
                    for q in range(sl * RT_SLOT_TRIS, min(cnt, (sl + 1) * RT_SLOT_TRIS)):
                        var tk = Int(blk_tri[unsafe_offset=b * RT_BLOCK_TRIS + q])
                        for c in range(3):
                            var x = Float64(tvv[unsafe_offset=tk * 9 + c * 3]); var y = Float64(tvv[unsafe_offset=tk * 9 + c * 3 + 1]); var z = Float64(tvv[unsafe_offset=tk * 9 + c * 3 + 2])
                            lo = min(lo, x); lo1 = min(lo1, y); lo2 = min(lo2, z)
                            hx = max(hx, x); hy = max(hy, y); hz = max(hz, z)
                    sbox[unsafe_offset=ns * 6] = lo; sbox[unsafe_offset=ns * 6 + 1] = lo1; sbox[unsafe_offset=ns * 6 + 2] = lo2
                    sbox[unsafe_offset=ns * 6 + 3] = hx; sbox[unsafe_offset=ns * 6 + 4] = hy; sbox[unsafe_offset=ns * 6 + 5] = hz
                    if sl == nsl - 1:
                        scode[unsafe_offset=ns] = Int32(1)
                    else:
                        scode[unsafe_offset=ns] = Int32((sl * RT_SLOT_TRIS + RT_SLOT_TRIS) << 3)
                    ns += 1
            w10 = blk_base + Int(hn_blk0[unsafe_offset=h]) * RT_NODE - off
            out[unsafe_offset=off] = UInt8(0x60)
        # frame over the slot boxes
        var lx = Float64(1e300); var ly = Float64(1e300); var lz = Float64(1e300)
        var ux = Float64(-1e300); var uy = Float64(-1e300); var uz = Float64(-1e300)
        for k in range(ns):
            lx = min(lx, sbox[unsafe_offset=k * 6]); ly = min(ly, sbox[unsafe_offset=k * 6 + 1]); lz = min(lz, sbox[unsafe_offset=k * 6 + 2])
            ux = max(ux, sbox[unsafe_offset=k * 6 + 3]); uy = max(uy, sbox[unsafe_offset=k * 6 + 4]); uz = max(uz, sbox[unsafe_offset=k * 6 + 5])
        _quant_frame(lx, ly, lz, ux, uy, uz, e, org)
        out[unsafe_offset=off + 1] = UInt8(e[unsafe_offset=0]); out[unsafe_offset=off + 2] = UInt8(e[unsafe_offset=1]); out[unsafe_offset=off + 3] = UInt8(e[unsafe_offset=2])
        for a in range(3): _put_u32(out, off + 4 + 4 * a, _fbits(org[unsafe_offset=a]))
        _put_u32(out, off + 0x10, UInt32(w10))
        if h == 0:
            _put_u32(out, off + 0x14, UInt32(0xf0000000)); _put_u32(out, off + 0x18, UInt32(0))
        else:
            _put_u32(out, off + 0x14, UInt32(Int(hn_sib[unsafe_offset=h])) << UInt32(28))
            _put_u32(out, off + 0x18, UInt32((0x100000000 - (h - Int(hn_par[unsafe_offset=h]))) & 0xffffffff))
        _put_u32(out, off + 0x1c, UInt32(w1c))
        for k in range(12):
            if k < ns:
                _put_slot(out, off, k, e, org, sbox[unsafe_offset=k * 6], sbox[unsafe_offset=k * 6 + 1], sbox[unsafe_offset=k * 6 + 2],
                          sbox[unsafe_offset=k * 6 + 3], sbox[unsafe_offset=k * 6 + 4], sbox[unsafe_offset=k * 6 + 5], Int(scode[unsafe_offset=k]))
                if Int(scode[unsafe_offset=k]) == 0x41:
                    out[unsafe_offset=off + 0x20 + 8 * k + 6] = UInt8(0)
            else:
                _put_empty_slot(out, off, k)
    var bn_box_root = unsafe_alloc[Float64](6)
    for q in range(6): bn_box_root[unsafe_offset=q] = bn_box[unsafe_offset=broot * 6 + q] if nb > 0 else Float64(0)
    # free the temporaries
    vid.unsafe_free(); tvv.unsafe_free(); u_first.unsafe_free(); u_n.unsafe_free()
    tlo.unsafe_free(); thi.unsafe_free(); tc.unsafe_free(); order.unsafe_free()
    bn_s.unsafe_free(); bn_c.unsafe_free(); bn_l.unsafe_free(); bn_r.unsafe_free(); bn_box.unsafe_free(); bn_nb.unsafe_free()
    hn_bn.unsafe_free(); hn_par.unsafe_free(); hn_sib.unsafe_free(); hn_first.unsafe_free(); hn_nk.unsafe_free()
    hn_blk0.unsafe_free(); hn_nblk.unsafe_free(); kids.unsafe_free(); blk_tri.unsafe_free(); blk_cnt.unsafe_free(); blk_box.unsafe_free()
    dstack.unsafe_free(); tri4.unsafe_free(); scratch.unsafe_free(); e.unsafe_free(); org.unsafe_free(); sbox.unsafe_free(); scode.unsafe_free()
    bstack.unsafe_free(); b_order.unsafe_free(); b_lo.unsafe_free(); b_hi.unsafe_free(); b_c.unsafe_free()
    var box = bn_box_root
    return RtBlas(out, size, gid, next_id, nn, nblk, box, gid_box, n_gbox)


struct RtTlas(Movable):
    var bytes: PB
    var size: Int
    var rec_off: Int            # byte offset of the first instance record; record j is 64 bytes, its pointer field (offset 8,
                                # six bytes) holds the index of the BLAS until the loader writes the address
    var n_records: Int
    var inst_of_record: PI      # record -> instance index

    def __init__(out self, bytes: PB, size: Int, rec_off: Int, n_records: Int, inst_of_record: PI):
        self.bytes = bytes
        self.size = size
        self.rec_off = rec_off
        self.n_records = n_records
        self.inst_of_record = inst_of_record


@no_inline
def _build_binary_items(order: PI, tlo: PD, thi: PD, tc: PD, m: Int, bn_s: PI, bn_c: PI, bn_l: PI, bn_r: PI, bn_box: PD, n_fixed: Int) -> Int:
    """Binary tree whose leaves are single items (instances). The first n_fixed items keep their places: the root separates
    them from the rest and they are split at the middle of their range, so item i stays at position i."""
    var leaf_granule = 1 if "loose" in getenv("GONZALES_RT_ABLATE") else RT_LEAF_TRIS      # ablation: leaves not filled
    var stack = unsafe_alloc[Int32](256)
    var nb = 1
    bn_s[unsafe_offset=0] = Int32(0); bn_c[unsafe_offset=0] = Int32(m)
    stack[unsafe_offset=0] = Int32(0)
    var sp = 1
    var cb = unsafe_alloc[Float64](6)
    while sp > 0:
        sp -= 1
        var ni = Int(stack[unsafe_offset=sp])
        var start = Int(bn_s[unsafe_offset=ni]); var count = Int(bn_c[unsafe_offset=ni])
        _range_box(order, tlo, thi, start, count, cb)
        for q in range(6): bn_box[unsafe_offset=ni * 6 + q] = cb[unsafe_offset=q]
        if count <= 1:
            bn_l[unsafe_offset=ni] = Int32(-1); bn_r[unsafe_offset=ni] = Int32(-1)
            continue
        var half: Int
        if n_fixed > 0 and start == 0 and count > n_fixed:
            half = n_fixed
        elif start + count <= n_fixed:
            half = count // 2
        else:
            half = _split_sah(order, tlo, thi, tc, start, count, leaf_granule)
        var li = nb; var ri = nb + 1
        nb += 2
        bn_l[unsafe_offset=ni] = Int32(li); bn_r[unsafe_offset=ni] = Int32(ri)
        bn_s[unsafe_offset=li] = Int32(start); bn_c[unsafe_offset=li] = Int32(half)
        bn_s[unsafe_offset=ri] = Int32(start + half); bn_c[unsafe_offset=ri] = Int32(count - half)
        stack[unsafe_offset=sp] = Int32(ri); stack[unsafe_offset=sp + 1] = Int32(li)
        sp += 2
    stack.unsafe_free()
    cb.unsafe_free()
    return nb


@no_inline
def rt_build_tlas(w2o: PF, binst: PI, n: Int, blas_box: PD, n_blas: Int, n_fixed: Int = 0, renumber: Bool = False) -> RtTlas:
    """Top-level structure over n instances. w2o: the world-to-object matrix of each as twelve floats, the top three rows
    of a column-major matrix; binst: its BLAS; blas_box: six doubles per BLAS (RtBlas.box). Records are laid out in tree order
    (RtTlas.inst_of_record); with renumber a record reports its own index as the instance, so the caller reorders its instances
    to match, the first n_fixed staying where they are."""
    var tlo = unsafe_alloc[Float64](3 * max(n, 1)); var thi = unsafe_alloc[Float64](3 * max(n, 1)); var tc = unsafe_alloc[Float64](3 * max(n, 1))
    var order = unsafe_alloc[Int32](max(n, 1))
    for i in range(n):
        var a = w2o.unsafe_offset(i * 12)
        # world-to-object y = A x + t, A with columns c0 c1 c2 (a[0..9]), t = a[9..12]; object-to-world: x = A^-1 (y - t)
        var m00 = Float64(a[unsafe_offset=0]); var m10 = Float64(a[unsafe_offset=1]); var m20 = Float64(a[unsafe_offset=2])
        var m01 = Float64(a[unsafe_offset=3]); var m11 = Float64(a[unsafe_offset=4]); var m21 = Float64(a[unsafe_offset=5])
        var m02 = Float64(a[unsafe_offset=6]); var m12 = Float64(a[unsafe_offset=7]); var m22 = Float64(a[unsafe_offset=8])
        var t0 = Float64(a[unsafe_offset=9]); var t1 = Float64(a[unsafe_offset=10]); var t2 = Float64(a[unsafe_offset=11])
        var det = m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20)
        var id = Float64(1.0)
        if det != 0.0:
            id = 1.0 / det
        var i00 = (m11 * m22 - m12 * m21) * id; var i01 = (m02 * m21 - m01 * m22) * id; var i02 = (m01 * m12 - m02 * m11) * id
        var i10 = (m12 * m20 - m10 * m22) * id; var i11 = (m00 * m22 - m02 * m20) * id; var i12 = (m02 * m10 - m00 * m12) * id
        var i20 = (m10 * m21 - m11 * m20) * id; var i21 = (m01 * m20 - m00 * m21) * id; var i22 = (m00 * m11 - m01 * m10) * id
        var bi = Int(binst[unsafe_offset=i])
        var lx = Float64(1e300); var ly = Float64(1e300); var lz = Float64(1e300)
        var hx = Float64(-1e300); var hy = Float64(-1e300); var hz = Float64(-1e300)
        for c in range(8):
            var ox = blas_box[unsafe_offset=bi * 6 + (3 if (c & 1) != 0 else 0)]
            var oy = blas_box[unsafe_offset=bi * 6 + (4 if (c & 2) != 0 else 1)]
            var oz = blas_box[unsafe_offset=bi * 6 + (5 if (c & 4) != 0 else 2)]
            # world = M^-1 (obj... ) : object coordinates o = A w + t  =>  w = A^-1 (o - t)
            var dx = ox - t0; var dy = oy - t1; var dz = oz - t2
            var wx = i00 * dx + i01 * dy + i02 * dz
            var wy = i10 * dx + i11 * dy + i12 * dz
            var wz = i20 * dx + i21 * dy + i22 * dz
            lx = min(lx, wx); ly = min(ly, wy); lz = min(lz, wz)
            hx = max(hx, wx); hy = max(hy, wy); hz = max(hz, wz)
        tlo[unsafe_offset=i * 3] = lx; tlo[unsafe_offset=i * 3 + 1] = ly; tlo[unsafe_offset=i * 3 + 2] = lz
        thi[unsafe_offset=i * 3] = hx; thi[unsafe_offset=i * 3 + 1] = hy; thi[unsafe_offset=i * 3 + 2] = hz
        tc[unsafe_offset=i * 3] = (lx + hx) * 0.5; tc[unsafe_offset=i * 3 + 1] = (ly + hy) * 0.5; tc[unsafe_offset=i * 3 + 2] = (lz + hz) * 0.5
        order[unsafe_offset=i] = Int32(i)
    var nbmax = 2 * n + 1
    var bn_s = unsafe_alloc[Int32](nbmax); var bn_c = unsafe_alloc[Int32](nbmax)
    var bn_l = unsafe_alloc[Int32](nbmax); var bn_r = unsafe_alloc[Int32](nbmax)
    var bn_nb = unsafe_alloc[Int32](nbmax)
    var bn_box = unsafe_alloc[Float64](6 * nbmax)
    var nb = 0
    if n > 0:
        nb = _build_binary_items(order, tlo, thi, tc, n, bn_s, bn_c, bn_l, bn_r, bn_box, n_fixed)
    for back in range(nb):
        var i = nb - 1 - back
        if Int(bn_l[unsafe_offset=i]) < 0:
            bn_nb[unsafe_offset=i] = Int32(1)
        else:
            bn_nb[unsafe_offset=i] = bn_nb[unsafe_offset=Int(bn_l[unsafe_offset=i])] + bn_nb[unsafe_offset=Int(bn_r[unsafe_offset=i])]
    var hn_bn = unsafe_alloc[Int32](max(nb, 1)); var hn_par = unsafe_alloc[Int32](max(nb, 1)); var hn_sib = unsafe_alloc[Int32](max(nb, 1))
    var hn_first = unsafe_alloc[Int32](max(nb, 1)); var hn_nk = unsafe_alloc[Int32](max(nb, 1)); var hn_rec0 = unsafe_alloc[Int32](max(nb, 1))
    var kids = unsafe_alloc[Int32](16)
    var nn = 0
    if n > 0:
        hn_bn[unsafe_offset=0] = Int32(0); hn_par[unsafe_offset=0] = Int32(-1); hn_sib[unsafe_offset=0] = Int32(0); nn = 1
    var hi = 0
    while hi < nn:
        var bi = Int(hn_bn[unsafe_offset=hi])
        hn_nk[unsafe_offset=hi] = Int32(0); hn_first[unsafe_offset=hi] = Int32(0)
        if Int(bn_nb[unsafe_offset=bi]) > RT_LEAF_TRIS:
            var nk = _hw_children(bi, bn_nb, bn_l, bn_r, bn_box, kids)
            hn_nk[unsafe_offset=hi] = Int32(nk); hn_first[unsafe_offset=hi] = Int32(nn)
            for k in range(nk):
                hn_bn[unsafe_offset=nn] = kids[unsafe_offset=k]; hn_par[unsafe_offset=nn] = Int32(hi); hn_sib[unsafe_offset=nn] = Int32(k)
                nn += 1
        hi += 1
    # records: one per tree position, so the instances of a leaf node are consecutive
    var rec_inst = unsafe_alloc[Int32](max(n, 1))
    var nrec = n
    for r in range(n):
        rec_inst[unsafe_offset=r] = order[unsafe_offset=r]
    for h in range(nn):
        hn_rec0[unsafe_offset=h] = bn_s[unsafe_offset=Int(hn_bn[unsafe_offset=h])]
    var rec_off = RT_ROOT
    var nodes_base = RT_ROOT + ((nrec * 64 + RT_NODE - 1) // RT_NODE) * RT_NODE      # nodes must be 128-byte aligned
    var size = nodes_base + nn * RT_NODE
    var out = unsafe_alloc[UInt8](size)
    for i in range(size): out[unsafe_offset=i] = UInt8(0)
    _put_u32(out, 0, UInt32(0x00080003)); _put_u32(out, 4, UInt32(8)); _put_u32(out, 0x20, UInt32(nodes_base))
    for r in range(nrec):
        var inst = Int(rec_inst[unsafe_offset=r])
        var ro = rec_off + r * 64
        _put_u32(out, ro, UInt32((r if renumber else inst) + 1))
        var bi = UInt64(binst[unsafe_offset=inst])
        _put_u32(out, ro + 4, UInt32(binst[unsafe_offset=inst]))
        for q in range(6): out[unsafe_offset=ro + 8 + q] = UInt8((bi >> UInt64(8 * q)) & UInt64(0xff))
        out[unsafe_offset=ro + 15] = UInt8(0x14)
        var a = w2o.unsafe_offset(inst * 12)
        # row-major 3 x 4 from the column-major top three rows: row i = (c0[i], c1[i], c2[i], c3[i])
        for row in range(3):
            for col in range(4):
                _put_u32(out, ro + 16 + (row * 4 + col) * 4, _fbits(a[unsafe_offset=col * 3 + row]))
    var e = unsafe_alloc[Int32](3); var org = unsafe_alloc[Float32](3)
    var sbox = unsafe_alloc[Float64](6 * 12)
    var scode = unsafe_alloc[Int32](12)
    for h in range(nn):
        var off = nodes_base + h * RT_NODE
        var ns: Int
        var w10 = 0
        var w1c = 1
        if Int(hn_nk[unsafe_offset=h]) != 0:
            ns = Int(hn_nk[unsafe_offset=h])
            for k in range(ns):
                var kb = Int(hn_bn[unsafe_offset=Int(hn_first[unsafe_offset=h]) + k])
                for q in range(6): sbox[unsafe_offset=k * 6 + q] = bn_box[unsafe_offset=kb * 6 + q]
                scode[unsafe_offset=k] = Int32(0)
            w1c = Int(hn_first[unsafe_offset=h]) - h
            out[unsafe_offset=off] = UInt8(0x80)
        else:
            ns = Int(bn_nb[unsafe_offset=Int(hn_bn[unsafe_offset=h])])
            for g in range(ns):
                var inst = Int(rec_inst[unsafe_offset=Int(hn_rec0[unsafe_offset=h]) + g])
                for q in range(3):
                    sbox[unsafe_offset=g * 6 + q] = tlo[unsafe_offset=inst * 3 + q]
                    sbox[unsafe_offset=g * 6 + 3 + q] = thi[unsafe_offset=inst * 3 + q]
                scode[unsafe_offset=g] = Int32(0x41)
            w10 = rec_off + Int(hn_rec0[unsafe_offset=h]) * 64 - off
            out[unsafe_offset=off] = UInt8(0xa0)
        var lx = Float64(1e300); var ly = Float64(1e300); var lz = Float64(1e300)
        var ux = Float64(-1e300); var uy = Float64(-1e300); var uz = Float64(-1e300)
        for k in range(ns):
            lx = min(lx, sbox[unsafe_offset=k * 6]); ly = min(ly, sbox[unsafe_offset=k * 6 + 1]); lz = min(lz, sbox[unsafe_offset=k * 6 + 2])
            ux = max(ux, sbox[unsafe_offset=k * 6 + 3]); uy = max(uy, sbox[unsafe_offset=k * 6 + 4]); uz = max(uz, sbox[unsafe_offset=k * 6 + 5])
        _quant_frame(lx, ly, lz, ux, uy, uz, e, org)
        out[unsafe_offset=off + 1] = UInt8(e[unsafe_offset=0]); out[unsafe_offset=off + 2] = UInt8(e[unsafe_offset=1]); out[unsafe_offset=off + 3] = UInt8(e[unsafe_offset=2])
        for a in range(3): _put_u32(out, off + 4 + 4 * a, _fbits(org[unsafe_offset=a]))
        _put_u32(out, off + 0x10, UInt32(w10 & 0xffffffff))
        if h == 0:
            _put_u32(out, off + 0x14, UInt32(0xf001ffff)); _put_u32(out, off + 0x18, UInt32(0))
        else:
            _put_u32(out, off + 0x14, (UInt32(Int(hn_sib[unsafe_offset=h])) << UInt32(28)) | UInt32(0x0001ffff))
            _put_u32(out, off + 0x18, UInt32((0x100000000 - (h - Int(hn_par[unsafe_offset=h]))) & 0xffffffff))
        _put_u32(out, off + 0x1c, UInt32(w1c))
        for k in range(12):
            if k < ns:
                _put_slot(out, off, k, e, org, sbox[unsafe_offset=k * 6], sbox[unsafe_offset=k * 6 + 1], sbox[unsafe_offset=k * 6 + 2],
                          sbox[unsafe_offset=k * 6 + 3], sbox[unsafe_offset=k * 6 + 4], sbox[unsafe_offset=k * 6 + 5], Int(scode[unsafe_offset=k]))
                if Int(scode[unsafe_offset=k]) == 0x41:
                    out[unsafe_offset=off + 0x20 + 8 * k + 6] = UInt8(0)
            else:
                _put_empty_slot(out, off, k)
    tlo.unsafe_free(); thi.unsafe_free(); tc.unsafe_free(); order.unsafe_free()
    bn_s.unsafe_free(); bn_c.unsafe_free(); bn_l.unsafe_free(); bn_r.unsafe_free(); bn_box.unsafe_free(); bn_nb.unsafe_free()
    hn_bn.unsafe_free(); hn_par.unsafe_free(); hn_sib.unsafe_free(); hn_first.unsafe_free(); hn_nk.unsafe_free(); hn_rec0.unsafe_free()
    kids.unsafe_free(); e.unsafe_free(); org.unsafe_free(); sbox.unsafe_free(); scode.unsafe_free()
    return RtTlas(out, size, rec_off, nrec, rec_inst)
