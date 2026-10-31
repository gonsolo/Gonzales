"""Texture lookup and the surface maps applied at a hit."""
from std.atomic import Atomic
from gonzales.scene.ptex_cache import PTEX_QUEUE_CAP
from std.math import sqrt, floor, log2, ceil, abs
from std.ffi import external_call
from std.memory.alloc import unsafe_alloc
from gonzales.core.geometry import RGB, Vec3f, dot, cross, Frame, _is_real_ptr
from gonzales.scene.materials import Material
from gonzales.scene.materials import GpuTexture
from gonzales.scene.primitives import Intersection, TriangleMesh, Instance, mesh_uv, mesh_has_uvs, mesh_vi, mesh_face
from gonzales.scene.footprint import CameraFootprint, UVFootprint, TriWorld, tri_world, hit_uv_footprint


@always_inline
def _srgb_to_linear(c: Float32) -> Float32:
    if c <= Float32(0.04045):
        return c / Float32(12.92)
    else:
        return Float32(((c + Float32(0.055)) / Float32(1.055)) ** Float32(2.4))


# Linear RGB of texel `i`, counted in texels from the start of tex.data. uint8
# textures decode each channel through tex.lut, so every blend happens in
# linear space; single-channel textures replicate into all three channels.
@always_inline
def _texel(tex: GpuTexture, i: Int) -> RGB:
    if Int(tex.format) == GpuTexture.FORMAT_U8:
        if Int(tex.channels) == 1:
            var l = tex.lut[unsafe_offset=Int(tex.data[unsafe_offset=i])]
            return RGB(l, l, l)
        var j = i * 3
        return RGB(tex.lut[unsafe_offset=Int(tex.data[unsafe_offset=j])], tex.lut[unsafe_offset=Int(tex.data[unsafe_offset=j + 1])], tex.lut[unsafe_offset=Int(tex.data[unsafe_offset=j + 2])])
    var f = tex.data.unsafe_bitcast[Float32]()
    var k = i * 3
    return RGB(f[unsafe_offset=k], f[unsafe_offset=k + 1], f[unsafe_offset=k + 2])


# Bilinear sample of ONE mip level: `off` = texel offset of the level in
# tex.data, (lw, lh) = that level's dimensions. Pixel centres at +0.5, wrap.
@always_inline
def _sample_level(tex: GpuTexture, off: Int, lw: Int, lh: Int, u: Float32, v: Float32) -> RGB:
    var s = u - Float32(Int(u))
    if s < Float32(0.0): s += Float32(1.0)
    var t = v - Float32(Int(v))
    if t < Float32(0.0): t += Float32(1.0)
    var fx = s * Float32(lw) - Float32(0.5)
    var fy = t * Float32(lh) - Float32(0.5)
    var x0 = Int(floor(fx)); var y0 = Int(floor(fy))
    var wx = fx - Float32(x0); var wy = fy - Float32(y0)
    # Wrap without integer division: x0 is in [-1, lw-1] here; % only for out-of-range inputs.
    var x0w = x0
    if x0w < 0 or x0w >= lw: x0w = ((x0 % lw) + lw) % lw
    var y0w = y0
    if y0w < 0 or y0w >= lh: y0w = ((y0 % lh) + lh) % lh
    var x1w = x0w + 1
    if x1w == lw: x1w = 0
    var y1w = y0w + 1
    if y1w == lh: y1w = 0
    var i00 = off + y0w * lw + x0w
    var i10 = off + y0w * lw + x1w
    var i01 = off + y1w * lw + x0w
    var i11 = off + y1w * lw + x1w
    var w00 = (Float32(1.0) - wx) * (Float32(1.0) - wy)
    var w10 = wx * (Float32(1.0) - wy)
    var w01 = (Float32(1.0) - wx) * wy
    var w11 = wx * wy
    return _texel(tex, i00) * w00 + _texel(tex, i10) * w10 + _texel(tex, i01) * w01 + _texel(tex, i11) * w11


# Trilinear mip sample. lod 0 = base level (full res); higher = coarser.
# With a 1-level texture (no pyramid) this is plain bilinear on the base.
@always_inline
def _sample_tex(tex: GpuTexture, u: Float32, v: Float32, lod: Float32 = Float32(0.0)) -> RGB:
    var nl = Int(tex.n_levels)
    if nl <= 1:
        return _sample_level(tex, 0, Int(tex.width), Int(tex.height), u, v)
    var clamped = lod
    if clamped < Float32(0.0): clamped = Float32(0.0)
    var maxl = Float32(nl - 1)
    if clamped > maxl: clamped = maxl
    var l0 = Int(floor(clamped))
    var f = clamped - Float32(l0)
    # Walk to level l0, tracking its texel offset and dims.
    var off = 0; var w = Int(tex.width); var h = Int(tex.height)
    for _k in range(l0):
        off += w * h
        w = max(1, w // 2); h = max(1, h // 2)
    var c0 = _sample_level(tex, off, w, h, u, v)
    if f <= Float32(0.0) or l0 >= nl - 1:
        return c0
    var off1 = off + w * h
    var w1 = max(1, w // 2); var h1 = max(1, h // 2)
    var c1 = _sample_level(tex, off1, w1, h1, u, v)
    return c0 + (c1 - c0) * f


# pbrt's MIPMap level for its default "bilinear" filter: floor(nLevels - 1 +
# log2(width)), one level, no blend. `width` is UVFootprint.width (uv units).
@always_inline
def _footprint_lod(tex: GpuTexture, width: Float32) -> Float32:
    var texels = width * Float32(max(Int(tex.width), Int(tex.height)))
    if texels > Float32(1.0):
        return floor(log2(texels))
    return Float32(0.0)


# ── Packed Ptex (GpuTexture.FORMAT_PTEX; layout in oiio.cc) ─────────────────
# Bilinear inside the face, blended between the two mip levels the footprint straddles. Each level has a
# one-texel border holding the neighbouring faces' texels, so the lookup filters across face edges.
# (u, v) are the face's own 0..1.
@always_inline
def _ptex_texel(tex: GpuTexture, texels: Pointer[UInt8, MutUntrackedOrigin], i: Int) -> RGB:
    var j = i * 3
    return RGB(tex.lut[unsafe_offset=Int(texels[unsafe_offset=j])], tex.lut[unsafe_offset=Int(texels[unsafe_offset=j + 1])],
               tex.lut[unsafe_offset=Int(texels[unsafe_offset=j + 2])])


@always_inline
def _ptex_level(tex: GpuTexture, texels: Pointer[UInt8, MutUntrackedOrigin], off: Int, lw: Int, lh: Int, u: Float32, v: Float32) -> RGB:
    var fx = min(max(u, Float32(0.0)), Float32(1.0)) * Float32(lw) + Float32(0.5)   # in bordered texel coordinates
    var fy = min(max(v, Float32(0.0)), Float32(1.0)) * Float32(lh) + Float32(0.5)
    var x0 = min(Int(floor(fx)), lw); var y0 = min(Int(floor(fy)), lh)
    var wx = fx - Float32(x0); var wy = fy - Float32(y0)
    var row = off + y0 * (lw + 2) + x0
    var c00 = _ptex_texel(tex, texels, row)
    var c10 = _ptex_texel(tex, texels, row + 1)
    var c01 = _ptex_texel(tex, texels, row + lw + 2)
    var c11 = _ptex_texel(tex, texels, row + lw + 3)
    return (c00 * (Float32(1.0) - wx) + c10 * wx) * (Float32(1.0) - wy) + (c01 * (Float32(1.0) - wx) + c11 * wx) * wy


@always_inline
def _ptex_need(fp_width: Float32) -> Int:
    """log2 of the face resolution a lookup with this footprint reads."""
    if fp_width > Float32(0.0):
        return min(max(Int(ceil(-log2(fp_width))), 0), 12)
    return 12


@always_inline
def _ptex_tri(inter: Intersection) -> Int:
    if inter.primId.type == 0:
        return Int(inter.primId.id2) // 3
    return Int(inter.primId.id2 & 0xFFFFFFFF)


@always_inline
def _ptex_ask(tex: GpuTexture, entry: Pointer[UInt8, MutUntrackedOrigin], face: Int, need: Int):
    """Raise the face's wanted byte to `need`. With the viewer's queue on (pool byte 1), the thread that
    raised it also queues the face; a full queue takes the raise back so the face asks again next frame."""
    var w = entry.unsafe_offset(4).unsafe_bitcast[UInt32]()
    var old = w[unsafe_offset=0]
    var prev: UInt32
    while True:
        prev = (old >> 16) & 0xFF
        if prev >= UInt32(254) or prev >= UInt32(need):
            return
        if Atomic[UInt32].compare_exchange(w, old, (old & 0xFF00FFFF) | (UInt32(need) << 16)):
            break
    if tex.pool[unsafe_offset=1] == UInt8(0) or tex.height < Int32(0):
        return
    var q = tex.pool.unsafe_offset(4).unsafe_bitcast[UInt32]()
    var i = Int(Atomic.fetch_add(q, UInt32(1)))
    if i < PTEX_QUEUE_CAP:
        q[unsafe_offset=1 + 2 * i] = (UInt32(tex.height) << 4) | UInt32(need)
        q[unsafe_offset=2 + 2 * i] = UInt32(face)
        return
    var cur = w[unsafe_offset=0]
    while ((cur >> 16) & 0xFF) == UInt32(need):
        if Atomic[UInt32].compare_exchange(w, cur, (cur & 0xFF00FFFF) | (prev << 16)):
            break


@always_inline
def _sample_ptex(tex: GpuTexture, face_in: Int, u: Float32, v: Float32, fp_width: Float32) -> RGB:
    var n_faces = Int(tex.width)
    var face = face_in if (face_in >= 0 and face_in < n_faces) else 0
    var entry = tex.data.unsafe_offset(face * 8)
    var off = Int(entry.unsafe_bitcast[UInt32]()[unsafe_offset=0])
    var ul = Int(entry[unsafe_offset=4]); var vl = Int(entry[unsafe_offset=5])
    var texels = tex.data.unsafe_offset(n_faces * 8)
    var flags = Int(entry[unsafe_offset=7])
    if (flags & 1) != 0:
        texels = tex.pool
        if (flags & 2) == 0:
            entry[unsafe_offset=7] = UInt8(3)
    var want = Int(entry[unsafe_offset=6])
    if want != 255:
        # Ask the host for the resolution this footprint needs (ptex_cache.mojo).
        var need = _ptex_need(fp_width)
        if need > max(ul, vl) and need > want:
            _ptex_ask(tex, entry, face, need)
    var n_lev = max(ul, vl) + 1
    var lod = Float32(0.0)
    var fp_texels = fp_width * Float32(1 << max(ul, vl))
    if fp_texels > Float32(1.0):
        lod = min(log2(fp_texels), Float32(n_lev - 1))
    var l0 = Int(floor(lod))
    var f = lod - Float32(l0)
    for _k in range(l0):
        off += ((1 << ul) + 2) * ((1 << vl) + 2)
        ul = max(ul - 1, 0); vl = max(vl - 1, 0)
    var c0 = _ptex_level(tex, texels, off, 1 << ul, 1 << vl, u, v)
    if f <= Float32(0.0) or l0 >= n_lev - 1:
        return c0
    var off1 = off + ((1 << ul) + 2) * ((1 << vl) + 2)
    var c1 = _ptex_level(tex, texels, off1, 1 << max(ul - 1, 0), 1 << max(vl - 1, 0), u, v)
    return c0 + (c1 - c0) * f


# Unified 2D-texture fetch — the single use_gpu seam for texture sampling.
# Both backends read the in-memory GpuTexture table (device buffers on GPU, host
# pyramids on CPU); a CPU caller without a table falls back to OIIO by filename.
# (u, v) are the interpolated, NOT-yet-V-flipped coords; this applies pbrt's
# V-flip (1 - v). raw=True skips the sRGB decode in the OIIO fallback only: the
# table's textures are decoded (or not) once at load.
# Sets `found` False when there is no texture/data (caller uses its fallback).
@always_inline
def sample_texture[use_gpu: Bool](
    tex_idx: Int,
    u: Float32, v: Float32,
    raw: Bool,
    fp_width: Float32,   # UVFootprint.width (pbrt MIPMap filter width, uv units); <=0 => LOD 0
    tex_filenames: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
    textures: Pointer[GpuTexture, MutUntrackedOrigin],
    n_textures: Int,
    mut found: Bool,
    ptex_face: Int = 0,   # Ptex textures only: the hit triangle's face id; (u, v) are then the face's own
) -> RGB:
    found = False
    if tex_idx < 0:
        return RGB(Float32(0.0))
    var su = u
    var tv = Float32(1.0) - v  # pbrt V-flip: V=0 at top
    if tex_idx < n_textures:
        var tex = textures[unsafe_offset=tex_idx]
        if Int(tex.width) > 0:
            found = True
            if Int(tex.format) == GpuTexture.FORMAT_PTEX:
                # A float Ptex is the mean of its channels (pbrt's FloatPtexTexture); no V-flip.
                var pc = _sample_ptex(tex, ptex_face, u, v, fp_width)
                return RGB((pc.r + pc.g + pc.b) * Float32(1.0 / 3.0))
            return _sample_tex(tex, su, tv, _footprint_lod(tex, fp_width))
    comptime if not use_gpu:
        if n_textures == 0 and Int(tex_filenames) > 1:
            var filename = tex_filenames[unsafe_offset=tex_idx]
            if Int(filename) > 1:
                su = su - Float32(Int(su))
                if su < Float32(0.0): su += Float32(1.0)
                tv = tv - Float32(Int(tv))
                if tv < Float32(0.0): tv += Float32(1.0)
                var tr = unsafe_alloc[Float32](3)
                tr[unsafe_offset=0] = Float32(0.0); tr[unsafe_offset=1] = Float32(0.0); tr[unsafe_offset=2] = Float32(0.0)
                _ = external_call["texture", Bool,
                    Pointer[UInt8, MutUntrackedOrigin], Float32, Float32,
                    Pointer[Float32, MutUntrackedOrigin]](filename, su, tv, tr)
                var rr = tr[unsafe_offset=0]; var gg = tr[unsafe_offset=1]; var bb = tr[unsafe_offset=2]
                tr.unsafe_free()
                found = True
                if raw:
                    return RGB(rr, gg, bb)
                return RGB(_srgb_to_linear(rr), _srgb_to_linear(gg), _srgb_to_linear(bb))
    return RGB(Float32(0.0))


@always_inline
def _tex_lookup[use_gpu: Bool](
    mat: Material,
    inter: Intersection,
    v0: Int, v1: Int, v2: Int,
    mesh: TriangleMesh,
    tex_filenames: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
    textures: Pointer[GpuTexture, MutUntrackedOrigin],
    n_textures: Int,
    fp_width: Float32 = Float32(0.0),   # UVFootprint.width; 0 = LOD 0
) -> RGB:
    var ti = Int(mat.tex_idx)
    if ti == -2:
        # Procedural checkerboard (pbrt "checkerboard" texture class): evaluated
        # analytically per-shading-point from Material's embedded checker_*
        # fields, identically on CPU and GPU — no image data or texture-table
        # index involved. See material_builder.mojo's "reflectance" handler.
        if mesh_has_uvs(mesh):
            var w0 = Float32(1.0) - inter.u - inter.v
            var su = w0*mesh_uv(mesh, v0)[0]   + inter.u*mesh_uv(mesh, v1)[0]   + inter.v*mesh_uv(mesh, v2)[0]
            var tv = w0*mesh_uv(mesh, v0)[1] + inter.u*mesh_uv(mesh, v1)[1] + inter.v*mesh_uv(mesh, v2)[1]
            # No V-flip here (unlike the image-texture path): pbrt evaluates its
            # checkerboard directly on the raw surface (u, v), with no notion of
            # image row order. Flipping v would toggle the sum-parity and swap
            # tex1/tex2 across the whole grid.
            var iu = Int(floor(su * mat.checker_uscale))
            var iv = Int(floor(tv * mat.checker_vscale))
            if (iu + iv) % 2 == 0:
                return mat.checker_tex1
            return mat.checker_tex2
        return mat.checker_tex1
    if ti >= 0 and ti < n_textures and Int(textures[unsafe_offset=ti].format) == GpuTexture.FORMAT_PTEX:
        # Ptex: the triangle's face id is in mesh.faceIndices; the mesh uv is the face's own 0..1 (pbrt passes it through).
        var ptx = textures[unsafe_offset=ti]
        var tri = _ptex_tri(inter)
        var pu = inter.u + inter.v   # pbrt's default triangle uv: (0,0), (1,0), (1,1)
        var pv = inter.v
        if (mesh.uv_corners != Int32(0) or _is_real_ptr(mesh.uvs)):
            var pw0 = Float32(1.0) - inter.u - inter.v
            pu = pw0*mesh_uv(mesh, v0)[0]   + inter.u*mesh_uv(mesh, v1)[0]   + inter.v*mesh_uv(mesh, v2)[0]
            pv = pw0*mesh_uv(mesh, v0)[1] + inter.u*mesh_uv(mesh, v1)[1] + inter.v*mesh_uv(mesh, v2)[1]
        var pt = _sample_ptex(ptx, Int(mesh_face(mesh, tri)), pu, pv, fp_width)
        return RGB(mat.tex_bias.r + mat.tex_scale.r * pt.r, mat.tex_bias.g + mat.tex_scale.g * pt.g, mat.tex_bias.b + mat.tex_scale.b * pt.b)
    var has_uvs = mesh_has_uvs(mesh)
    if ti >= 0 and ti < n_textures:
        var tex = textures[unsafe_offset=ti]
        if Int(tex.width) > 0:
            var w0 = Float32(1.0) - inter.u - inter.v
            var su = inter.u + inter.v   # pbrt's default triangle uv: (0,0), (1,0), (1,1)
            var tv = inter.v
            if has_uvs:
                su = w0*mesh_uv(mesh, v0)[0]   + inter.u*mesh_uv(mesh, v1)[0]   + inter.v*mesh_uv(mesh, v2)[0]
                tv = w0*mesh_uv(mesh, v0)[1] + inter.u*mesh_uv(mesh, v1)[1] + inter.v*mesh_uv(mesh, v2)[1]
            tv = Float32(1.0) - tv  # PBRT V-flip: V=0 at top
            # bias + scale*texel is pbrt's "scale"/"mix" texture graph
            # folded into the lookup (scale=1, bias=0 when absent) --
            # see material_builder.mojo's _resolve_affine_rgb.
            var t = _sample_tex(tex, su, tv, _footprint_lod(tex, fp_width))
            return RGB(mat.tex_bias.r + mat.tex_scale.r * t.r,
                       mat.tex_bias.g + mat.tex_scale.g * t.g,
                       mat.tex_bias.b + mat.tex_scale.b * t.b)
    comptime if not use_gpu:
        if ti >= 0 and n_textures == 0 and _is_real_ptr(tex_filenames):
            var filename = tex_filenames[unsafe_offset=ti]
            if Int(filename) > 1 and has_uvs:
                var w0 = Float32(1.0) - inter.u - inter.v
                var su = w0*mesh_uv(mesh, v0)[0]   + inter.u*mesh_uv(mesh, v1)[0]   + inter.v*mesh_uv(mesh, v2)[0]
                var tv = w0*mesh_uv(mesh, v0)[1] + inter.u*mesh_uv(mesh, v1)[1] + inter.v*mesh_uv(mesh, v2)[1]
                tv = Float32(1.0) - tv  # PBRT V-flip: V=0 at top
                su = su - Float32(Int(su))
                if su < Float32(0.0): su += Float32(1.0)
                tv = tv - Float32(Int(tv))
                if tv < Float32(0.0): tv += Float32(1.0)
                var tr = unsafe_alloc[Float32](3)
                tr[unsafe_offset=0] = Float32(0.0); tr[unsafe_offset=1] = Float32(0.0); tr[unsafe_offset=2] = Float32(0.0)
                _ = external_call["texture", Bool,
                    Pointer[UInt8, MutUntrackedOrigin], Float32, Float32,
                    Pointer[Float32, MutUntrackedOrigin]](filename, su, tv, tr)
                var result = RGB(_srgb_to_linear(tr[unsafe_offset=0]), _srgb_to_linear(tr[unsafe_offset=1]), _srgb_to_linear(tr[unsafe_offset=2]))
                tr.unsafe_free()
                return RGB(mat.tex_bias.r + mat.tex_scale.r * result.r,
                           mat.tex_bias.g + mat.tex_scale.g * result.g,
                           mat.tex_bias.b + mat.tex_scale.b * result.b)
    return mat.albedo


# ── Triangle primitive helper ─────────────────────────────────────────────────
# Decodes the primId encoding into (mesh, v0, v1, v2). Returns ok=False for
# non-triangle hits (caller should deactivate path and return).
@always_inline
def _get_tri_verts(
    inter: Intersection,
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
) -> Tuple[TriangleMesh, Int, Int, Int, Bool]:
    var mi: Int
    var bv: Int
    if inter.primId.type == 0:
        mi = Int(inter.primId.id1)
        bv = Int(inter.primId.id2)
    elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
        mi = Int(inter.primId.id2 >> 32)
        bv = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
    else:
        return (meshes[unsafe_offset=0], 0, 0, 0, False)
    var m = meshes[unsafe_offset=mi]
    return (m, Int(mesh_vi(m, bv)), Int(mesh_vi(m, bv+1)), Int(mesh_vi(m, bv+2)), True)


# Apply tangent-space normal map: samples the texture, decodes [-1,1] normal,
# rotates it into world space via UV-gradient tangent frame.
# Returns geom_normal unchanged when normal_tex_idx < 0 or no UVs.
@always_inline
def _decode_tangent_normal(ns: RGB, tangent: Vec3f, bitangent: Vec3f, geom_normal: Vec3f) -> Vec3f:
    """A normal-map texel, sampled raw ([0,1], no sRGB), decoded to a
    tangent-space normal ([-1,1]) and rotated into world space -- the
    identical tail both normal-map call sites in this file had (one for a
    mesh's own UVs, one for an analytic sphere's Mitsuba-convention UVs);
    falls back to `geom_normal` on a degenerate rotation, same as both did."""
    var nx = ns.r * Float32(2.0) - Float32(1.0)
    var ny = ns.g * Float32(2.0) - Float32(1.0)
    var nz = ns.b * Float32(2.0) - Float32(1.0)
    var world_n = tangent * nx + bitangent * ny + geom_normal * nz
    var wn_len = dot(world_n, world_n)
    if wn_len > Float32(0.0):
        return world_n * (Float32(1.0) / sqrt(wn_len))
    return geom_normal


@always_inline
def _apply_normal_map[use_gpu: Bool](
    mat: Material,
    v0: Int, v1: Int, v2: Int,
    mesh: TriangleMesh,
    inter: Intersection,
    geom_normal: Vec3f,
    p0: Vec3f,
    p1: Vec3f,
    p2: Vec3f,
    tex_filenames: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
    textures: Pointer[GpuTexture, MutUntrackedOrigin],
    n_textures: Int,
    pixel_uv: Float32 = Float32(0.0),
) -> Vec3f:
    if mat.normal_tex_idx < Int32(0) or not mesh_has_uvs(mesh):
        return geom_normal
    # Compute barycentric UV coordinates
    var dp1 = p1 - p0; var dp2 = p2 - p0
    var u0f = mesh_uv(mesh, v0)[0]; var v0f = mesh_uv(mesh, v0)[1]
    var u1f = mesh_uv(mesh, v1)[0]; var v1f = mesh_uv(mesh, v1)[1]
    var u2f = mesh_uv(mesh, v2)[0]; var v2f = mesh_uv(mesh, v2)[1]
    var du1 = u1f - u0f; var dv1 = v1f - v0f
    var du2 = u2f - u0f; var dv2 = v2f - v0f
    var det = du1 * dv2 - du2 * dv1
    if det == Float32(0.0):
        return geom_normal
    var inv_det = Float32(1.0) / det
    # Tangent = dP/du; bitangent = cross(N, tangent). This matches pbrt's shading
    # frame Frame::FromXZ(dpdu, n) (whose Y axis is cross(n, dpdu)) — NOT dP/dv,
    # which has the opposite handedness here and mirrors the relief in a grazing
    # view (the seam this once showed was the view-ray faceforward bug, fixed
    # separately).
    var tangent = (dp1 * dv2 - dp2 * dv1) * inv_det
    var tlen = dot(tangent, tangent)
    if tlen <= Float32(0.0): return geom_normal
    tangent = tangent * (Float32(1.0) / sqrt(tlen))
    # pbrt's frame is FromXZ(dpdu, n): the bitangent is cross(n, dpdu) (matches
    # materials.h NormalMap), NOT dP/dv.
    var bitangent = cross(geom_normal, tangent)
    var blen = dot(bitangent, bitangent)
    if blen <= Float32(0.0): return geom_normal
    bitangent = bitangent * (Float32(1.0) / sqrt(blen))
    # Interpolate UV at the hit point using the barycentrics (inter.u, inter.v).
    # (A centroid approximation here makes the normal map constant per triangle,
    #  so bumps vanish on low-poly meshes like a 2-triangle floor.)
    var bw0 = Float32(1.0) - inter.u - inter.v
    var uv_u = bw0 * u0f + inter.u * u1f + inter.v * u2f
    var uv_v = bw0 * v0f + inter.u * v1f + inter.v * v2f  # unflipped; sample_texture applies the V-flip
    # Sample the normal map raw (no sRGB) and decode [0,1] -> [-1,1].
    var found = False
    var ns = sample_texture[use_gpu](Int(mat.normal_tex_idx), uv_u, uv_v, True, pixel_uv, tex_filenames, textures, n_textures, found)
    if found:
        return _decode_tangent_normal(ns, tangent, bitangent, geom_normal)
    return geom_normal


@always_inline
def _apply_bump_map[use_gpu: Bool](
    mat: Material,
    v0: Int, v1: Int, v2: Int,
    mesh: TriangleMesh,
    inter: Intersection,
    geom_normal: Vec3f,
    tri: TriWorld,
    tex_filenames: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
    textures: Pointer[GpuTexture, MutUntrackedOrigin],
    n_textures: Int,
    fp: UVFootprint = UVFootprint.none(),
) -> Vec3f:
    """PBRT-style bump map: perturb dpdu/dpdv by the finite-difference height
    gradient of `mat.bump_tex_idx` (scaled by `mat.bump_scale`), then rebuild
    the normal as cross(dpdu', dpdv'). Distinct from `_apply_normal_map` (a
    tangent-space normal image) -- this is a scalar HEIGHT texture, so unlike
    that function it needs the RAW (non-normalized) dpdu/dpdv, since the
    height gradient's magnitude must compose against their actual
    world-per-parametric-unit scale, not a unit direction.

    The finite-difference steps are pbrt's du/dv from the hit's footprint
    (footprint.mojo) -- never a fixed UV constant. A first attempt at this used a fixed epsilon
    and made barcelona-pavilion's water WORSE (1.682x -> 1.804x pbrt): a
    fixed step over a heavily-tiled UV layout produces enormous gradients
    wherever texels are small in world space. Matches pbrt's own adaptive
    `du = .5*(|dudx|+|dudy|)`, falling back to a small constant only when no
    footprint is available (first ray, or non-UV-mapped geometry) -- see
    project_barcelona_pavilion_mnee memory for the full history."""
    if mat.bump_tex_idx < Int32(0) or not mesh_has_uvs(mesh):
        return geom_normal
    var p0 = tri.p0; var p1 = tri.p1; var p2 = tri.p2
    var dp1 = p1 - p0; var dp2 = p2 - p0
    var u0f = mesh_uv(mesh, v0)[0]; var v0f = mesh_uv(mesh, v0)[1]
    var u1f = mesh_uv(mesh, v1)[0]; var v1f = mesh_uv(mesh, v1)[1]
    var u2f = mesh_uv(mesh, v2)[0]; var v2f = mesh_uv(mesh, v2)[1]
    var du1 = u1f - u0f; var dv1 = v1f - v0f
    var du2 = u2f - u0f; var dv2 = v2f - v0f
    var det = du1 * dv2 - du2 * dv1
    if det == Float32(0.0):
        return geom_normal
    var inv_det = Float32(1.0) / det
    # Raw (unnormalized) dp/du, dp/dv -- same barycentric solve as
    # _apply_normal_map's tangent, but kept at real world-per-parametric-unit
    # magnitude instead of normalizing immediately.
    var dpdu = (dp1 * dv2 - dp2 * dv1) * inv_det
    var dpdv = (dp2 * du1 - dp1 * du2) * inv_det

    var bw0 = Float32(1.0) - inter.u - inter.v
    var uv_u = bw0 * u0f + inter.u * u1f + inter.v * u2f
    var uv_v = bw0 * v0f + inter.u * v1f + inter.v * v2f  # unflipped; sample_texture applies its own V-flip consistently to every sample below

    var hu_step = fp.du if fp.du > Float32(0.0) else Float32(0.0005)
    var hv_step = fp.dv if fp.dv > Float32(0.0) else Float32(0.0005)

    # raw=False: unlike a normal map (tangent-space vectors, never gamma
    # data), a height texture sourced from an 8-bit PNG is sRGB-encoded by
    # convention -- pbrt-v4 itself defaults every PNG-sourced FloatImageTexture
    # to sRGB decoding (textures.cpp: `HasExtension(filename, "png") ? "sRGB"
    # : "linear"`), confirmed empirically here: an sRGB-decoded ramp produces
    # a nonlinear (gamma ~2.2) height profile, and using raw=True (no decode)
    # against a scene authored assuming pbrt's default made this feature's
    # own controlled validation (a monotonic ramp vs a real pbrt render)
    # diverge from pbrt's falloff shape. GPU already gets this for free --
    # gpu.mojo's upload only marks normal_tex_idx textures raw, so
    # bump_tex_idx defaults to the sRGB-decoded (non-raw) upload path.
    var found = False
    var bump_face = 0
    if Int(mat.bump_tex_idx) < n_textures and Int(textures[unsafe_offset=Int(mat.bump_tex_idx)].format) == GpuTexture.FORMAT_PTEX:
        var btri = Int(inter.primId.id2 & 0xFFFFFFFF)
        if inter.primId.type == 0:
            btri = Int(inter.primId.id2) // 3
        bump_face = Int(mesh_face(mesh, btri))
    var h0 = sample_texture[use_gpu](Int(mat.bump_tex_idx), uv_u, uv_v, False, fp.width, tex_filenames, textures, n_textures, found, bump_face).r
    if not found:
        return geom_normal
    var hu = sample_texture[use_gpu](Int(mat.bump_tex_idx), uv_u + hu_step, uv_v, False, fp.width, tex_filenames, textures, n_textures, found, bump_face).r
    var hv = sample_texture[use_gpu](Int(mat.bump_tex_idx), uv_u, uv_v + hv_step, False, fp.width, tex_filenames, textures, n_textures, found, bump_face).r
    var dhdu = (hu - h0) * (Float32(1.0) / hu_step) * mat.bump_scale
    var dhdv = (hv - h0) * (Float32(1.0) / hv_step) * mat.bump_scale

    # pbrt's BumpMap perturbs the SHADING frame, not the triangle's own
    # derivatives. Without vertex normals the two are the same (dpdu, dpdv,
    # the geometric normal). With them, pbrt's Triangle sets ss = dpdu made
    # perpendicular to ns and ts = cross(ns, ss) -- so |ts| = |ss|, NOT |dpdv|,
    # which changes the v-slope wherever the uv mapping is anisotropic -- and
    # carries the vertex normals' dn/du, dn/dv as `displace * dndu`.
    var ss = dpdu
    var ts = dpdv
    var dndu = Vec3f(Float32(0.0))
    var dndv = Vec3f(Float32(0.0))
    if tri.has_n:
        var ns = geom_normal
        ts = cross(ns, ss)
        if dot(ts, ts) > Float32(0.0):
            ss = cross(ts, ns)
        else:
            var fr = Frame.from_z(ns)
            ss = Vec3f(fr.x.x, fr.x.y, fr.x.z)
            ts = Vec3f(fr.y.x, fr.y.y, fr.y.z)
        var dn1 = tri.n0 - tri.n2
        var dn2 = tri.n1 - tri.n2
        var du02 = u0f - u2f; var dv02 = v0f - v2f
        var du12 = u1f - u2f; var dv12 = v1f - v2f
        var uv_det = du02 * dv12 - dv02 * du12
        if abs(uv_det) >= Float32(1e-32):
            var inv_uv = Float32(1.0) / uv_det
            dndu = (dn1 * dv12 - dn2 * dv02) * inv_uv
            dndv = (dn2 * du02 - dn1 * du12) * inv_uv
            # The vertex normals' orientation, not ns's: ns may have been
            # turned toward the ray, and dn/du must turn with it.
            var ni = tri.n0 * bw0 + tri.n1 * inter.u + tri.n2 * inter.v
            if dot(ni, ns) < Float32(0.0):
                dndu = -dndu
                dndv = -dndv
    var displace = h0 * mat.bump_scale
    var new_dpdu = ss + geom_normal * dhdu + dndu * displace
    var new_dpdv = ts + geom_normal * dhdv + dndv * displace
    var world_n = cross(new_dpdu, new_dpdv)
    var wn_len2 = dot(world_n, world_n)
    if wn_len2 <= Float32(0.0):
        return geom_normal
    world_n = world_n * (Float32(1.0) / sqrt(wn_len2))
    if dot(world_n, geom_normal) < Float32(0.0):
        world_n = -world_n
    return world_n


# Apply normal + bump maps to an already-interpolated shading normal.
#
# WHY THIS EXISTS: _apply_normal_map/_apply_bump_map used to be called from
# exactly one site, inside _build_geom_context_full -- which only
# shade_diffuse calls. Every other material builds its geometry inline (see
# the comment above), so coateddiffuse, diffusetransmission, conductor,
# measured, coated_conductor and dielectric silently ignored their
# "texture displacement"/"normalmap" parameters entirely. Across the pbrt-v4
# corpus that is 284 coateddiffuse and 146 diffusetransmission materials
# dropped against 73 diffuse ones honoured -- i.e. ~86% of them. Confirmed on
# barcelona-pavilion's water (a `dielectric` with "texture displacement"):
# high-frequency structure in the water was identical with the displacement
# present and stripped (ratio 0.999, matching a no-displacement control
# region), i.e. the map had no effect whatsoever.
#
# `orient_to` is the reference direction the perturbed normal is kept on the
# same side of. Pass the FACE-FORWARDED geometric normal for ordinary
# reflective/transmissive shading (matching _build_geom_context_full), but for
# dielectric/thin_dielectric pass the RAW WINDING normal instead: those decide
# entering-vs-exiting from `dot(ray_dir, n) < 0`, so flipping the perturbed
# normal toward the ray would make that test tautological and resurrect the
# 1/eta^4 transmission loss documented in shade_dielectric.
@always_inline
def _apply_surface_maps[use_gpu: Bool](
    mat: Material,
    v0: Int, v1: Int, v2: Int,
    mesh: TriangleMesh,
    inter: Intersection,
    shading_normal: Vec3f,
    orient_to: Vec3f,
    tri: TriWorld,       # WORLD-space corners: the maps build tangents from them
    fp: UVFootprint,
    tex_filenames: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
    textures: Pointer[GpuTexture, MutUntrackedOrigin],
    n_textures: Int,
) -> Vec3f:
    if mat.normal_tex_idx < Int32(0) and mat.bump_tex_idx < Int32(0):
        return shading_normal
    var p0 = tri.p0; var p1 = tri.p1; var p2 = tri.p2
    # Normal map at LOD 0, like pbrt's NormalMap() (a plain bilerp of the
    # full-resolution image, no mip chain). A box-filtered mip of encoded
    # normals averages toward the flat (0,0,1) texel, so at a distant or
    # grazing hit the footprint LOD silently erased the map -- on Bistro's
    # pavement pbrt's normal map darkens the ground ~10% and gonzales's
    # did nothing. The bump map keeps the footprint: pbrt's BumpMap()
    # evaluates a mipmapped FloatImageTexture.
    # pbrt applies a normal map OR a displacement, never both: the normal map
    # wins (SurfaceInteraction::GetBSDF).
    var n: Vec3f
    if mat.normal_tex_idx >= Int32(0):
        n = _apply_normal_map[use_gpu](mat, v0, v1, v2, mesh, inter, shading_normal, p0, p1, p2,
            tex_filenames, textures, n_textures, Float32(0.0))
    else:
        n = _apply_bump_map[use_gpu](mat, v0, v1, v2, mesh, inter, shading_normal, tri,
            tex_filenames, textures, n_textures, fp)
    if dot(n, orient_to) < Float32(0.0):
        n = -n
    return n


# Resolve the hit's triangle and apply its normal/bump maps, in one call.
#
# Exists because every caller of _apply_surface_maps outside this file has to
# do the same three steps -- _get_tri_verts, test `ok`, call through -- and
# getting that wrong is silent: a missed site renders a plausible image with
# the map simply absent, which is how SPPM's photon pass and ALL of bdpt_*.mojo
# went without bump/normal maps while the path tracer had them since 2026-09.
# One helper, so a new integrator branch is one line and the `ok` test cannot
# be forgotten. Returns `shading_normal` unchanged for a non-triangle hit (an
# analytic sphere: _get_tri_verts reports type 4 as not-ok) and for a material
# with neither map.
@always_inline
def apply_surface_maps_at_hit[use_gpu: Bool](
    mat: Material,
    inter: Intersection,
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    instances: Pointer[Instance, MutUntrackedOrigin],
    shading_normal: Vec3f,
    orient_to: Vec3f,
    hit: Vec3f,
    ray_dir: Vec3f,
    cone_w: Float32,   # >= 0: specular-chain cone width here; < 0: camera approximation
    cam: CameraFootprint,
    tex_filenames: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
    textures: Pointer[GpuTexture, MutUntrackedOrigin],
    n_textures: Int,
) -> Vec3f:
    if mat.normal_tex_idx < Int32(0) and mat.bump_tex_idx < Int32(0):
        return shading_normal
    var (mesh, v0, v1, v2, ok) = _get_tri_verts(inter, meshes)
    if not ok:
        return shading_normal
    var tri = tri_world(mesh, v0, v1, v2, inter.primId.instanceIdx, instances)
    var fp = hit_uv_footprint(cam, tri, mesh, v0, v1, v2, hit, ray_dir, cone_w)
    return _apply_surface_maps[use_gpu](mat, v0, v1, v2, mesh, inter,
        shading_normal, orient_to, tri, fp,
        tex_filenames, textures, n_textures)


# The same footprint, for an albedo/texture lookup at a hit that resolves its
# own maps elsewhere (0 = LOD 0 for a non-triangle hit).
@always_inline
def uv_footprint_at_hit(
    inter: Intersection,
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    instances: Pointer[Instance, MutUntrackedOrigin],
    hit: Vec3f,
    ray_dir: Vec3f,
    cone_w: Float32,
    cam: CameraFootprint,
) -> UVFootprint:
    var (mesh, v0, v1, v2, ok) = _get_tri_verts(inter, meshes)
    if not ok:
        return UVFootprint.none()
    return hit_uv_footprint(cam, tri_world(mesh, v0, v1, v2, inter.primId.instanceIdx, instances),
        mesh, v0, v1, v2, hit, ray_dir, cone_w)
