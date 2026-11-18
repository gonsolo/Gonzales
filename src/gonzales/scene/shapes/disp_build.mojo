# Displaced Catmull-Clark meshes from a displacement Ptex (its cage in the meta data): patches, height chains and
# nodes from src/oiio/disp.cc, sub-square levels per patch, the welded corner grid (displaced_patches.mojo).

from std.ffi import external_call
from std.math import max, min, sqrt
from std.memory.alloc import unsafe_alloc
from max.algorithm import parallelize
from gonzales.core.geometry import Vec3f, dot
from gonzales.scene.shapes.patches import BezierPatch
from gonzales.scene.shapes.displaced_patches import DispSurface, Levels, LEVELS, make_levels, patch_box, corner_off, \
    corner_step, corner_grid, weld_corner_grid


@fieldwise_init
struct DispOptions(TrivialRegisterPassable):
    """The camera (eye, pixel angle; pixel 0: none) and the level choices: height levels kept down to `lod` pixels
    (0: all), sub-squares about `tps` texels a side but no smaller than `pps` pixels (0: off), at least level `lmin`;
    `grid` >= 0 fixes the level."""
    var eye: Vec3f
    var pixel: Float32
    var lod: Float32
    var pps: Float32
    var tps: Float32
    var lmin: Int
    var grid: Int
    var weld: Bool


@fieldwise_init
struct DispMesh(Copyable, Movable):
    """One displaced mesh's host data, patches sorted by sub-square level."""
    var np: Int
    var nf: Int
    var npyr: Int
    var nchain: Int
    var ncg: Int
    var texels: Int
    var patches: Pointer[BezierPatch, MutUntrackedOrigin]
    var sq: Pointer[Float32, MutUntrackedOrigin]
    var face: Pointer[Int32, MutUntrackedOrigin]
    var hdr: Pointer[Int32, MutUntrackedOrigin]
    var pyr: Pointer[UInt8, MutUntrackedOrigin]
    var chain_off: Pointer[Int32, MutUntrackedOrigin]
    var chain: Pointer[UInt8, MutUntrackedOrigin]
    var fm: Pointer[Float32, MutUntrackedOrigin]
    var greg: Pointer[UInt8, MutUntrackedOrigin]
    var corners: Pointer[Float32, MutUntrackedOrigin]
    var lv: Levels
    var lo: Vec3f
    var hi: Vec3f

    def surface(self, scale: Float32, cam: Vec3f, pixel: Float32, leaf_mode: Int32) -> DispSurface:
        return DispSurface(self.patches, self.sq, self.face, self.hdr, self.pyr, scale, cam, pixel, self.chain_off, self.chain,
                           self.fm, self.greg, self.lv, leaf_mode, self.corners)

    def prims(self) -> Int:
        return Int(self.lv.prim[LEVELS])


def _cstr(s: String) -> Pointer[UInt8, MutUntrackedOrigin]:
    var c = unsafe_alloc[UInt8](s.byte_length() + 1)
    for i in range(s.byte_length()):
        c[unsafe_offset=i] = s.unsafe_ptr()[unsafe_offset=i]
    c[unsafe_offset=s.byte_length()] = UInt8(0)
    return c


def disp_bounds(ptx: String) raises -> Tuple[Vec3f, Vec3f]:
    """The cage's box, from the meta data only."""
    var c = _cstr(ptx); var bb = unsafe_alloc[Float32](6)
    var ok = external_call["disp_bounds", Int32](c, bb)
    var r = (Vec3f(bb[unsafe_offset=0], bb[unsafe_offset=1], bb[unsafe_offset=2]), Vec3f(bb[unsafe_offset=3], bb[unsafe_offset=4], bb[unsafe_offset=5]))
    c.unsafe_free(); bb.unsafe_free()
    if ok == 0:
        raise Error("cannot read " + ptx)
    return r


def disp_levels(np: Int, patches: Pointer[BezierPatch, MutUntrackedOrigin], sq: Pointer[Float32, MutUntrackedOrigin],
                face: Pointer[Int32, MutUntrackedOrigin], hdr: Pointer[Int32, MutUntrackedOrigin], fdrop: Pointer[UInt8, MutUntrackedOrigin],
                o: DispOptions) -> List[Int]:
    """Each patch's sub-square level, from its full Ptex resolution (dropped height levels keep the grid)."""
    var lev = List[Int](length=np, fill=0)
    for p in range(np):
        var l = 0
        if o.grid >= 0:
            l = min(o.grid, LEVELS - 1)
        else:
            var f = Int(face[unsafe_offset=p])
            var side = sq[unsafe_offset=3 * p + 2] * Float32(1 << (max(Int(hdr[unsafe_offset=3 * f + 1]), Int(hdr[unsafe_offset=3 * f + 2])) + Int(fdrop[unsafe_offset=f])))
            while l < LEVELS - 1 and Float32(1 << (l + 1)) * o.tps <= side * Float32(1.41421356):
                l += 1
            if o.pps > Float32(0) and o.pixel > Float32(0):   # the patch's control box in pixels at its nearest point to the eye
                var b = patches[unsafe_offset=p]
                var lo = Vec3f(b.x.reduce_min(), b.y.reduce_min(), b.z.reduce_min()); var hi = Vec3f(b.x.reduce_max(), b.y.reduce_max(), b.z.reduce_max())
                var dx = max(max(lo.x - o.eye.x, o.eye.x - hi.x), Float32(0)); var dy = max(max(lo.y - o.eye.y, o.eye.y - hi.y), Float32(0))
                var dz = max(max(lo.z - o.eye.z, o.eye.z - hi.z), Float32(0)); var e = hi - lo; var ext = sqrt(dot(e, e))
                var px = ext / (o.pixel * max(sqrt(dx * dx + dy * dy + dz * dz), Float32(1e-3) * ext))
                while l > 0 and px / Float32(1 << l) < o.pps:
                    l -= 1
            l = max(l, min(o.lmin, LEVELS - 1))
        lev[p] = l
    return lev^


def load_disp_mesh(ptx: String, o: DispOptions) raises -> DispMesh:
    """The displaced mesh of `ptx` for options `o`."""
    var c = _cstr(ptx)
    var counts = unsafe_alloc[Int64](4)
    var view = unsafe_alloc[Float32](5)
    view[unsafe_offset=0] = o.eye.x; view[unsafe_offset=1] = o.eye.y; view[unsafe_offset=2] = o.eye.z
    view[unsafe_offset=3] = o.pixel; view[unsafe_offset=4] = o.lod if o.pixel > Float32(0) else Float32(0)
    var hd = external_call["disp_open", Int](c, Int32(2), counts, view)
    c.unsafe_free(); view.unsafe_free()
    if hd == 0:
        raise Error("cannot read " + ptx)
    var np = Int(counts[0]); var nf = Int(counts[1]); var npyr = Int(counts[2]); var nchain = Int(counts[3])
    counts.unsafe_free()
    var cpf = unsafe_alloc[Float32](48 * np)
    var sq = unsafe_alloc[Float32](3 * np); var face = unsafe_alloc[Int32](np)
    var hdr = unsafe_alloc[Int32](3 * nf); var pyr = unsafe_alloc[UInt8](max(npyr, 1)); var cbox = unsafe_alloc[Float32](6)
    var chain_off = unsafe_alloc[Int32](nf); var chain = unsafe_alloc[UInt8](max(nchain, 1))
    external_call["disp_take_chains", NoneType](hd, chain_off, chain)
    var fm = unsafe_alloc[Float32](12 * np); var greg = unsafe_alloc[UInt8](np)
    external_call["disp_take_gregory", NoneType](hd, fm, greg)
    var fdrop = unsafe_alloc[UInt8](nf)
    external_call["disp_take_drop", NoneType](hd, fdrop)
    external_call["disp_take", NoneType](hd, cpf, sq, face, hdr, pyr, cbox)
    var patches = unsafe_alloc[BezierPatch](np)
    for p in range(np):
        var x = SIMD[DType.float32, 16](0); var y = SIMD[DType.float32, 16](0); var z = SIMD[DType.float32, 16](0)
        for k in range(16):
            x[k] = cpf[unsafe_offset=48 * p + 3 * k]; y[k] = cpf[unsafe_offset=48 * p + 3 * k + 1]; z[k] = cpf[unsafe_offset=48 * p + 3 * k + 2]
        patches[unsafe_offset=p] = BezierPatch(x, y, z)
    cpf.unsafe_free()
    var plev = disp_levels(np, patches, sq, face, hdr, fdrop, o)
    fdrop.unsafe_free()
    # Patches sorted by level: a primitive's patch follows from its index (prim_patch).
    var p2 = unsafe_alloc[BezierPatch](np); var sq2 = unsafe_alloc[Float32](3 * np); var face2 = unsafe_alloc[Int32](np)
    var fm2 = unsafe_alloc[Float32](12 * np); var greg2 = unsafe_alloc[UInt8](np); var slev = List[Int](capacity=np)
    var i = 0
    for l in range(LEVELS):
        for p in range(np):
            if plev[p] != l:
                continue
            p2[unsafe_offset=i] = patches[unsafe_offset=p]; face2[unsafe_offset=i] = face[unsafe_offset=p]; greg2[unsafe_offset=i] = greg[unsafe_offset=p]
            for k in range(3):
                sq2[unsafe_offset=3 * i + k] = sq[unsafe_offset=3 * p + k]
            for k in range(12):
                fm2[unsafe_offset=12 * i + k] = fm[unsafe_offset=12 * p + k]
            slev.append(l); i += 1
    patches.unsafe_free(); sq.unsafe_free(); face.unsafe_free(); fm.unsafe_free(); greg.unsafe_free()
    var lv = make_levels(slev)
    var texels = 0
    for f in range(nf):
        texels += (1 << Int(hdr[unsafe_offset=3 * f + 1])) * (1 << Int(hdr[unsafe_offset=3 * f + 2]))
    var ncg = Int(lv.corner[LEVELS])
    var cg = unsafe_alloc[Float32](max(ncg, 1))
    var m = DispMesh(np, nf, npyr, nchain, ncg, texels, p2, sq2, face2, hdr, pyr, chain_off, chain, fm2, greg2, cg, lv,
                     Vec3f(cbox[unsafe_offset=0], cbox[unsafe_offset=1], cbox[unsafe_offset=2]), Vec3f(cbox[unsafe_offset=3], cbox[unsafe_offset=4], cbox[unsafe_offset=5]))
    cbox.unsafe_free()
    var d = m.surface(Float32(0), Vec3f(0, 0, 0), Float32(0), Int32(5))
    var plo = List[Vec3f](capacity=np); var pext = Float32(0)
    for p in range(np):
        var b = patch_box(d, p); plo.append(b[0])
        pext = max(pext, max(b[1].x - b[0].x, max(b[1].y - b[0].y, b[1].z - b[0].z)))
    var cst = corner_step(pext)
    def fill(p: Int) {imm}:
        corner_grid(d, p, plo[p], cst, cg.unsafe_offset(corner_off(lv, p)))
    parallelize(fill, np)
    if o.weld:
        weld_corner_grid(np, lv, cg)
    return m^
