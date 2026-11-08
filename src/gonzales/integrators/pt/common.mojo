"""What the path tracer's material shaders share: hit geometry, spectra, shadow rays, Russian roulette."""
from std.sys import size_of
from std.math import sqrt, abs
from gonzales.core.geometry import offset_point, RGB, Point3f, Vec3f, dot, face_toward, cross, Frame, INV_PI, _is_real_ptr, _atan2f
from gonzales.scene.appearance.materials import Material
from gonzales.transport.render_state import PathState, ShadowTask, SHADOW_SLOTS
from gonzales.scene.appearance.materials import GpuTexture
from gonzales.scene.shapes.primitives import Intersection, TriangleMesh, Sphere, Instance, mesh_uv, mesh_has_uvs, mesh_face
from gonzales.core.geometry import Ray
from gonzales.scene.shapes.primitives import mesh_pos
from gonzales.scene.appearance.lights import AreaLight
from gonzales.transport.shadow_media import shadow_transmittance
from gonzales.scene.shapes.curves import Curve, curve_at, curve_piece_endpoints, curve_hit_normal
from gonzales.transport.bxdf import BxDFSample, GeomContext, SobolSamples8, BxDFFlags
from gonzales.core.rng import PCG32
from gonzales.scene.appearance.footprint import CameraFootprint, UVFootprint, TriWorld, tri_world, hit_uv_footprint
from gonzales.scene.accel.bvh import any_hit_dispatch, curve_offset_eps
from gonzales.core.sampling import sobol_sample_n, sobol_dim_seed
from gonzales.core.transform import transform_normal, Mat4
from gonzales.transport.guide import GuideGrid, guide_pos_to_cell, guide_record, null_guide, guide_is_active
from gonzales.core.spectrum import spec_refl_unbounded, SpectralSample, SampledWavelengths, rgb_to_spectral_sample, rgb_illuminant_to_spectral_sample
from gonzales.transport.texturing import _apply_surface_maps, _decode_tangent_normal, _get_tri_verts, _ptex_ask, _ptex_need, _ptex_tri, _tex_lookup, sample_texture
from gonzales.transport.shade_context import GIPendingX1, ShadeContext
from gonzales.transport.glossy_sms import GSMS_DEBUG
from gonzales.transport.manifold import _emitter_face_normal, _interp_vertex_normal


@always_inline
def gi_pending_x1_init() -> GIPendingX1:
    return GIPendingX1(
        active=Int8(0),
        hit_point=Vec3f(Float32(0)),
        normal=Vec3f(Float32(0)),
        alb=RGB(Float32(0)),
        throughput=SpectralSample(Float32(0)),
    )


@always_inline
def area_light_hit_cos(
    inter: Intersection,
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    instances: Pointer[Instance, MutUntrackedOrigin],
    ray_dir: Vec3f,
) -> Float32:
    """Cosine between an area-light TRIANGLE's emitting face and the ray that
    hit it (primId.type 3): > 0 on the emitting front, <= 0 on the back. The
    emitting face is _emitter_face_normal's -- the winding normal flipped to
    agree with the interpolated declared normal, pbrt's rule. Unit-length, so
    it also serves as the area -> solid-angle Jacobian's cosine."""
    var (m, v0, v1, v2, ok) = _get_tri_verts(inter, meshes)
    if not ok:
        return Float32(0)
    var p0 = mesh_pos(m, v0)
    var p1 = mesh_pos(m, v1)
    var p2 = mesh_pos(m, v2)
    var en = _emitter_face_normal(m, v0, v1, v2, inter.u, inter.v, p0, p1, p2,
                                  inter.primId.instanceIdx, instances)
    var l2 = dot(en, en)
    if l2 <= Float32(0.0):
        return Float32(0)
    var c = -dot(en, ray_dir) / sqrt(l2)
    return abs(c) if m.emit_two_sided != Int32(0) else c


@always_inline
def curve_light_hit(
    inter: Intersection,
    curves: Pointer[Curve, MutUntrackedOrigin],
    area_lights: Pointer[AreaLight, MutUntrackedOrigin],
    n_area_lights: Int,
    ray_dir: Vec3f,
) -> Tuple[Int, Float32]:
    """(al_idx, cos_l) for a ray hitting an emissive CURVE (primId.type 5).

    al_idx is the curve's AreaLight slot (kind 1), -1 if it has none. Curve
    lights are rare -- a handful of emissive strands at most -- so a linear
    scan beats threading a reverse index through PrimId. cos_l is taken
    against the outward radial normal at the hit, reconstructed from (u, v)
    exactly as shade_hair does. The EMISSION is not returned: it lives in the
    curve's own material slot (mat.emission), which the caller already has."""
    var curve_idx = Int(inter.primId.id1)
    var al_idx = -1
    for li in range(n_area_lights):
        var cand = area_lights[unsafe_offset=li]
        if cand.kind == Int8(1) and Int(cand.meshIdx) == curve_idx:
            al_idx = li
            break
    var curve = curve_at(curves, curve_idx)
    var h = max(Float32(-0.99), min(Float32(0.99), inter.u))
    var piece = min(Int(curve.n_pieces) - 1, max(0, Int(inter.v * Float32(curve.n_pieces))))
    var (q0, q1, _, _) = curve_piece_endpoints(curve, piece)
    var seg_axis = q1 - q0
    var seg_len = sqrt(dot(seg_axis, seg_axis))
    var tangent = Vec3f(Float32(1.0), Float32(0.0), Float32(0.0))
    if seg_len > Float32(1e-8):
        tangent = seg_axis * (Float32(1.0) / seg_len)
    var geo_normal = curve_hit_normal(tangent, -ray_dir, h)
    return (al_idx, -dot(geo_normal, ray_dir))


@always_inline
def _shading_normal(
    mesh: TriangleMesh,
    v0: Int, v1: Int, v2: Int,
    bu: Float32, bv: Float32,
    geo_normal: Vec3f,
    instance_idx: Int32 = Int32(-1),
    instances: Pointer[Instance, MutUntrackedOrigin] = Pointer[Instance, MutUntrackedOrigin].unsafe_dangling(),
) -> Vec3f:
    """Interpolate per-vertex shading normals with barycentric (bu, bv).
    Falls back to the geometric normal if the mesh has no shading normals.
    The result is aligned to the same hemisphere as geo_normal, which the
    caller has already (a) oriented against the incoming ray and (b), if this
    hit came from an instanced BLAS (instance_idx >= 0), transformed to world
    space — mesh.normals are per-vertex OBJECT-space data in that case, so
    the interpolated result is transformed the same way before comparison."""
    var sn = _interp_vertex_normal(mesh, v0, v1, v2, bu, bv, instance_idx, instances)
    if dot(sn, sn) == Float32(0.0):
        return geo_normal
    if dot(sn, geo_normal) < Float32(0.0):
        sn = -sn
    return sn


@always_inline
def _ptex_missed(textures: Pointer[GpuTexture, MutUntrackedOrigin], n_textures: Int, ti: Int,
                 inter: Intersection, mesh: TriangleMesh, fp_width: Float32) -> Bool:
    """Whether texture `ti` is a paged Ptex whose face at this hit is coarser than the lookup needs and can
    still be loaded (wanted byte 255 = nothing finer exists, 254 = no room). Texel 0 of the pool switches
    the test off, for the final passes that take what is there."""
    if ti < 0 or ti >= n_textures or Int(textures[unsafe_offset=ti].format) != GpuTexture.FORMAT_PTEX:
        return False
    var tex = textures[unsafe_offset=ti]
    var n_faces = Int(tex.width)
    var face = Int(mesh_face(mesh, _ptex_tri(inter)))
    if face < 0 or face >= n_faces:
        face = 0
    var entry = tex.data.unsafe_offset(face * 8)
    var want = Int(entry[unsafe_offset=6])
    var need = _ptex_need(fp_width)
    if want >= 254 or need <= max(Int(entry[unsafe_offset=4]), Int(entry[unsafe_offset=5])):
        return False
    if need > want:
        _ptex_ask(tex, entry, face, need)   # a withheld sample always has its request placed
    return tex.pool[unsafe_offset=0] != UInt8(0)


# ── Shading frame helper ──────────────────────────────────────────────────────
# Computes geom normal from triangle cross product, normalizes, faceforwards it
# toward the incoming ray, and extracts ray_dir/ray_org from path state.
# Returns (geom_normal_ff, ray_dir, ray_org).
@always_inline
def _geom_normal_and_ray(
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    p0: Vec3f,
    p1: Vec3f,
    p2: Vec3f,
    instance_idx: Int32 = Int32(-1),
    instances: Pointer[Instance, MutUntrackedOrigin] = Pointer[Instance, MutUntrackedOrigin].unsafe_dangling(),
) -> Tuple[Vec3f, Vec3f, Vec3f]:
    """p0/p1/p2 come straight from `mesh.points` — object-space if this hit
    came from an instanced BLAS (instance_idx >= 0), so the resulting normal
    is transformed to world space before face-forwarding against the ray
    (which is always world-space). The hit point (`ro`, the ray origin the
    caller combines with tHit) needs no such fixup — tHit is preserved 1:1
    between object/world parameterizations, see bvh.mojo's
    _transform_ray_to_instance_space."""
    var gn = cross(p1 - p0, p2 - p0)
    if instance_idx >= Int32(0):
        gn = transform_normal(Mat4(instances[unsafe_offset=Int(instance_idx)].world_to_obj()), gn)
    var nlen = dot(gn, gn)
    if nlen > Float32(0.0):
        gn = gn * (Float32(1.0) / sqrt(nlen))
    var rd = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    if dot(gn, rd) > Float32(0.0):
        gn = -gn
    var ro = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    return (gn, rd, ro)


# Analytic-sphere counterpart of _geom_normal_and_ray: exact outward normal
# at the hit point ((hit - center)/radius), face-forwarded toward the
# incoming ray. Spheres are never instanced (see Instance's docs), so
# unlike the triangle path there is no object/world transform to apply.
@always_inline
def _sphere_geom_normal_and_ray(
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    inter: Intersection,
    spheres: Pointer[Sphere, MutUntrackedOrigin],
) -> Tuple[Vec3f, Vec3f, Vec3f]:
    var sph = spheres[unsafe_offset=Int(inter.primId.id1)]
    var rd = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    var ro = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    var hit = ro + rd * inter.tHit
    var c = Vec3f(sph.center.x, sph.center.y, sph.center.z)
    var gn = hit - c
    var nlen = dot(gn, gn)
    if nlen > Float32(0.0):
        gn = gn * (Float32(1.0) / sqrt(nlen))
    if dot(gn, rd) > Float32(0.0):
        gn = -gn
    return (gn, rd, ro)


@always_inline
def _hit_curve(inter: Intersection, curves: Pointer[Curve, MutUntrackedOrigin],
               instances: Pointer[Instance, MutUntrackedOrigin]) -> Curve:
    """The curve a hit is on, in world space. A curve of an object template is stored in the template's
    space; its control points go through the hit instance's transform (a cubic's control points transform
    like the curve) and its widths through that transform's mean scale."""
    var curve = curve_at(curves, Int(inter.primId.id1))
    if inter.primId.instanceIdx >= Int32(0) and _is_real_ptr(instances):
        var o2w = instances[unsafe_offset=Int(inter.primId.instanceIdx)].obj_to_world()
        var m = Mat4(o2w)
        curve.cp0 = m.transform_point(curve.cp0); curve.cp1 = m.transform_point(curve.cp1)
        curve.cp2 = m.transform_point(curve.cp2); curve.cp3 = m.transform_point(curve.cp3)
        var scale = (sqrt(o2w[0] * o2w[0] + o2w[1] * o2w[1] + o2w[2] * o2w[2])
                     + sqrt(o2w[4] * o2w[4] + o2w[5] * o2w[5] + o2w[6] * o2w[6])
                     + sqrt(o2w[8] * o2w[8] + o2w[9] * o2w[9] + o2w[10] * o2w[10])) / Float32(3)
        curve.width0 *= scale; curve.width1 *= scale
    return curve


@always_inline
def _curve_geom_normal_and_ray(
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    inter: Intersection,
    curves: Pointer[Curve, MutUntrackedOrigin],
    instances: Pointer[Instance, MutUntrackedOrigin] = Pointer[Instance, MutUntrackedOrigin].unsafe_dangling(),
) -> Tuple[Vec3f, Vec3f, Vec3f]:
    var rd = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    var ro = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    var curve = _hit_curve(inter, curves, instances)
    var h = max(Float32(-0.99), min(Float32(0.99), inter.u))
    var piece = min(Int(curve.n_pieces) - 1, max(0, Int(inter.v * Float32(curve.n_pieces))))
    var (q0, q1, _, _) = curve_piece_endpoints(curve, piece)
    var seg_axis = q1 - q0
    var seg_len = sqrt(dot(seg_axis, seg_axis))
    var tangent = Vec3f(Float32(1.0), Float32(0.0), Float32(0.0))
    if seg_len > Float32(1e-8):
        tangent = seg_axis * (Float32(1.0) / seg_len)
    var gn: Vec3f
    var rperp = rd - tangent * dot(rd, tangent)
    var rl = dot(rperp, rperp)
    if curve.flat != Int32(0) and rl > Float32(1e-12):
        # A flat curve is a ribbon facing the ray: one normal across the width.
        gn = rperp * (Float32(-1.0) / sqrt(rl))
    else:
        # Radial normal from the hit position itself: (u, v) alone lose the sign of the second radial component.
        var hp = ro + rd * inter.tHit - q0
        gn = hp - tangent * dot(hp, tangent)
        var gl = dot(gn, gn)
        if gl > Float32(1e-20):
            gn = gn * (Float32(1.0) / sqrt(gl))
        else:
            gn = curve_hit_normal(tangent, -rd, h)
        if dot(gn, rd) > Float32(0.0):
            gn = -gn
    return (gn, rd, ro)


@always_inline
def _curve_extra_offset(inter: Intersection, curves: Pointer[Curve, MutUntrackedOrigin],
                        instances: Pointer[Instance, MutUntrackedOrigin] = Pointer[Instance, MutUntrackedOrigin].unsafe_dangling()) -> Float32:
    """Radius-scaled origin offset for a ray leaving a curve hit (same rule as shade_hair); 0 for other primitives."""
    if inter.primId.type != Int8(5) or not _is_real_ptr(curves):
        return Float32(0.0)
    var curve = _hit_curve(inter, curves, instances)
    return curve_offset_eps(Float32(0.5) * (curve.width0 + (curve.width1 - curve.width0) * inter.v))


# ── Unified per-hit geometry (triangle OR analytic sphere) ────────────────────
# Every material shader used to hand-roll its own "if primId.type==4: sphere
# branch, else: _get_tri_verts + cross product" — the same handful of lines
# copy-pasted into 8 different functions. This is the ONE place that decides
# how to turn an Intersection into (geo_normal, ray_dir, ray_org), for
# either primitive type; callers needing UV-dependent extras (textures,
# normal maps, anisotropy tangents, shading-normal interpolation — all of
# which have no sphere analogue, since Sphere has no UV parameterization)
# still branch on `is_sphere` themselves, but only for THAT material-specific
# logic, not for re-deriving the normal/ray every time.
# `mesh`/v0/v1/v2 are only meaningful when is_sphere is False; when ok is
# False the caller must deactivate the path without reading anything else.
@always_inline
def _hit_geom(
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    inter: Intersection,
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    spheres: Pointer[Sphere, MutUntrackedOrigin],
    instance_idx: Int32 = Int32(-1),
    instances: Pointer[Instance, MutUntrackedOrigin] = Pointer[Instance, MutUntrackedOrigin].unsafe_dangling(),
    curves: Pointer[Curve, MutUntrackedOrigin] = Pointer[Curve, MutUntrackedOrigin].unsafe_dangling(),
) -> Tuple[Bool, Bool, Vec3f, Vec3f, Vec3f, TriangleMesh, Int, Int, Int]:
    """Returns (ok, is_sphere, geo_normal, ray_dir, ray_org, mesh, v0, v1, v2). A curve hit (non-hair material) counts as is_sphere: no mesh, no UVs."""
    if inter.primId.type == Int8(5) and _is_real_ptr(curves):
        var crv = _curve_geom_normal_and_ray(path_ptr, inter, curves, instances)
        return (True, True, crv[0], crv[1], crv[2], meshes[unsafe_offset=0], 0, 0, 0)
    if inter.primId.type == Int8(4):
        var sph_r = _sphere_geom_normal_and_ray(path_ptr, inter, spheres)
        return (True, True, sph_r[0], sph_r[1], sph_r[2], meshes[unsafe_offset=0], 0, 0, 0)
    var (mesh, v0, v1, v2, ok) = _get_tri_verts(inter, meshes)
    if not ok:
        var z = Vec3f(Float32(0.0), Float32(0.0), Float32(0.0))
        return (False, False, z, z, z, mesh, 0, 0, 0)
    var p0 = mesh_pos(mesh, v0)
    var p1 = mesh_pos(mesh, v1)
    var p2 = mesh_pos(mesh, v2)
    var gnr = _geom_normal_and_ray(path_ptr, p0, p1, p2, instance_idx, instances)
    return (True, False, gnr[0], gnr[1], gnr[2], mesh, v0, v1, v2)


@always_inline
def _to_spec_refl(ctx: ShadeContext, c: RGB, wl: SampledWavelengths) -> SpectralSample:
    """RGB REFLECTANCE -> spectral, at the material boundary. Path transport is
    spectral; a material's albedo is authored as RGB, so this is exactly where
    the conversion belongs -- once, on the way in, rather than converting a
    finished RGB answer back and forth per evaluation."""
    return rgb_to_spectral_sample(ctx.spectral.coeffs, ctx.spectral.res,
        ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z,
        ctx.spectral.d65, c.r, c.g, c.b, wl)


@always_inline
def _to_spec_weight(ctx: ShadeContext, c: RGB, wl: SampledWavelengths) -> SpectralSample:
    """A BxDF SAMPLING WEIGHT (f*cos/pdf) -> spectral -- the THIRD converter
    at this boundary, and the one the conductor was missing.

    A weight is not a reflectance even though both arrive as an RGB triple
    off a material: a reflectance is bounded by 1 and _to_spec_refl CLAMPS
    to that, because the sigmoid table's domain is a bounded reflectance. A
    weight is a throughput multiplier and is routinely larger -- Kulla-Conty's
    compensation lobe reaches ~1.7 wherever its cosine branch is sampled
    toward grazing, and all of that excess was being deleted on the way into
    the spectral path. The conductor furnace read 0.81 instead of 1.0 from
    this clamp alone, with the compensation itself already correct.

    spec_refl_unbounded is that converter and it already existed; the bug was
    never a missing capability, only the wrong one of three being called."""
    return spec_refl_unbounded(ctx.spectral.coeffs, ctx.spectral.res,
        ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z,
        ctx.spectral.d65, c.r, c.g, c.b, wl)


@always_inline
@always_inline
def _hit_emission[use_gpu: Bool](mat: Material, inter: Intersection, ctx: ShadeContext) -> RGB:
    """A material's emitted radiance (Material.emit_tex_idx) at a mesh hit; no light samples it, so it counts in full."""
    var (mesh, v0, v1, v2, ok) = _get_tri_verts(inter, ctx.meshes)
    if not ok or not mesh_has_uvs(mesh):
        return RGB(Float32(0.0))
    var w0 = Float32(1.0) - inter.u - inter.v
    var su = w0 * mesh_uv(mesh, v0)[0] + inter.u * mesh_uv(mesh, v1)[0] + inter.v * mesh_uv(mesh, v2)[0]
    var sv = w0 * mesh_uv(mesh, v0)[1] + inter.u * mesh_uv(mesh, v1)[1] + inter.v * mesh_uv(mesh, v2)[1]
    var found = False
    return sample_texture[use_gpu](Int(mat.emit_tex_idx), su, sv, True, Float32(0.0), ctx.tex_filenames, ctx.textures,
                                   ctx.n_textures, found)


@always_inline
def _clamped(ctx: ShadeContext, c: SpectralSample, bounce: Int) -> SpectralSample:
    """Cycles' sample clamp (film_clamp_light): a light contribution whose channels sum past the limit is scaled down to
    it. `bounce` is Cycles' index: 0 for light reaching the camera through the first scattering vertex (direct),
    above for indirect; emission found by a path counts one less than the path's bounces. A limit of 0 is none."""
    comptime if GSMS_DEBUG != 0:
        return SpectralSample(Float32(0))
    var limit = ctx.clamp[1] if bounce > 0 else ctx.clamp[0]
    if limit <= Float32(0):
        return c
    var s = (abs(c.v0) + abs(c.v1) + abs(c.v2) + abs(c.v3)) * Float32(0.75)   # four wavelengths for Cycles' three channels
    return c * (limit / s) if s > limit else c


def _to_spec_illum(ctx: ShadeContext, c: RGB, wl: SampledWavelengths) -> SpectralSample:
    """RGB EMISSION -> spectral, at the light boundary. Uses the ILLUMINANT
    upsampling (unbounded, energy-normalised for light spectra), which is a
    different curve from the reflectance one above -- mixing them up is how
    the homogeneous medium ended up multiplying throughput by a
    "transmittance" greater than 1."""
    return rgb_illuminant_to_spectral_sample(ctx.spectral.coeffs, ctx.spectral.res,
        ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z,
        ctx.spectral.d65, c.r, c.g, c.b, wl)


@always_inline
def _to_spec_diffuse_lit(ctx: ShadeContext, alb: RGB, le: RGB, wl: SampledWavelengths) -> SpectralSample:
    """Diffuse albedo spectrum times emission spectrum. The two are upsampled separately: their RGB product is not
    a valid input to either converter (it matches only for grey albedo). The 1/pi stays outside: a table fit
    of albedo/pi is a different, more saturated curve than the fit of albedo."""
    return _to_spec_refl(ctx, alb, wl) * _to_spec_illum(ctx, le, wl) * INV_PI


@always_inline
def _emitter_invisible(ctx: ShadeContext, al: AreaLight) -> Bool:
    """An alpha-0 emitter is cut out of every ray, so BSDF sampling never finds it and light sampling takes the whole weight."""
    if al.kind != Int8(0):
        return False
    return ctx.meshes[unsafe_offset=Int(al.meshIdx)].alpha_w == Int32(0) and ctx.meshes[unsafe_offset=Int(al.meshIdx)].alpha_const <= Float32(0.0)


@always_inline
def _shadow_inline(ctx: ShadeContext, guide_write: GuideGrid) -> Bool:
    """True when _shadow_contribute would trace its ray right here as a plain any-hit test (no deferral, media or guiding)."""
    return ctx.n_mediums == 0 and not _is_real_ptr(ctx.shadow_tasks) and not guide_is_active(guide_write)


@always_inline
def _shadow_blocked(ctx: ShadeContext, origin: Vec3f, dir: Vec3f, tmax: Float32) -> Bool:
    """The plain any-hit shadow test of _shadow_contribute."""
    var shadow_ray = Ray(Point3f(origin[0], origin[1], origin[2]), Vec3f(dir[0], dir[1], dir[2]))
    return any_hit_dispatch(ctx.bvh4, ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, shadow_ray, tmax,
                             ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
                             ctx.lights.spheres, ctx.lights.sphere_count,
                             materials=ctx.materials)


@always_inline
def _shadow_contribute[enqueue_shadow: Bool, defer_only: Bool = False](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    ctx: ShadeContext,
    origin: Vec3f,
    dir: Vec3f,
    tmax: Float32,
    contrib_in: SpectralSample,
    guide_write: GuideGrid = null_guide(),
):
    var contrib = _clamped(ctx, contrib_in, Int(path_ptr[].bounce))
    # Deferral needs a real task buffer (only handed out for --rt-hardware) and no guiding; it takes the first free of the
    # path's SHADOW_SLOTS slots, and a candidate that finds none is traced inline below.
    comptime if defer_only:
        # The host guarantees a free slot (slots >= candidates), no media and no guiding; sw_resolve_shadow_gpu traces the ray.
        var path_index = (Int(path_ptr) - Int(ctx.paths_base)) // size_of[PathState]()
        var base = path_index * SHADOW_SLOTS
        comptime for s in range(SHADOW_SLOTS):
            if ctx.shadow_tasks[unsafe_offset=base + s].active == Int32(0):
                ctx.shadow_tasks[unsafe_offset=base + s] = ShadowTask(
                    Point3f(origin[0], origin[1], origin[2]),
                    Vec3f(dir[0], dir[1], dir[2]),
                    tmax, contrib, Int32(1), Int32(0))
                return
        return
    comptime if enqueue_shadow:
        if _is_real_ptr(ctx.shadow_tasks) and not guide_is_active(guide_write) and ctx.n_mediums == 0:
            var path_index = ctx.path_idx
            if _is_real_ptr(ctx.paths_base):
                path_index = (Int(path_ptr) - Int(ctx.paths_base)) // size_of[PathState]()
            var base = path_index * SHADOW_SLOTS
            comptime for s in range(SHADOW_SLOTS):
                if ctx.shadow_tasks[unsafe_offset=base + s].active == Int32(0):
                    ctx.shadow_tasks[unsafe_offset=base + s] = ShadowTask(
                        Point3f(origin[0], origin[1], origin[2]),
                        Vec3f(dir[0], dir[1], dir[2]),
                        tmax, contrib, Int32(1), Int32(0))
                    return
    var shadow_ray = Ray(Point3f(origin[0], origin[1], origin[2]), Vec3f(dir[0], dir[1], dir[2]))
    var tr = SpectralSample(Float32(1.0))
    var blocked: Bool
    if ctx.n_mediums > 0:
        # Salt the stream with the direction so the path's candidates in one bounce do not share ratio-tracking draws.
        var salt = UInt64(Int(abs(dir[0]) * Float32(1e7))) * UInt64(0x9E3779B97F4A7C15) + UInt64(Int(abs(dir[2]) * Float32(1e7)))
        var pcg = PCG32(path_ptr[].pcgState ^ salt, path_ptr[].pcgInc + UInt64(1))
        tr = shadow_transmittance(origin, dir, tmax, path_ptr[].current_medium_idx, ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves,
                                  ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances, ctx.lights.spheres, ctx.lights.sphere_count,
                                  ctx.materials, ctx.mediums, ctx.medium_ifaces, ctx.grids, ctx.nvdb_grids,
                                  ctx.spectral, path_ptr[].wavelengths, pcg)
        blocked = tr.is_black()
    else:
        blocked = any_hit_dispatch(ctx.bvh4, ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, shadow_ray, tmax,
                                    ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
                                    ctx.lights.spheres, ctx.lights.sphere_count,
                                    materials=ctx.materials)
    if not blocked:
        path_ptr[].estimate += _clamped(ctx, contrib * tr, Int(path_ptr[].bounce))
        # Record in the guide at the PARENT surface (one bounce back):
        # "scatter direction ray.dir from parent_cell leads to illumination W here."
        # This teaches indirect-illumination guiding — path_ptr[].ray.origin is where
        # the path came from and ray.direction is the scatter direction that arrived here.
        if guide_is_active(guide_write) and path_ptr[].bounce > 0:
            var parent_cell = guide_pos_to_cell(guide_write, path_ptr[].ray.origin)
            if parent_cell >= 0:
                var w = contrib.luma()
                # Normalize by current throughput to record incoming radiance at
                # the parent surface, independent of path history (Li, not T*Li).
                var t = path_ptr[].throughput
                var t_lum = t.luma()
                if t_lum > Float32(1e-7):
                    w = w / t_lum
                if w > Float32(1e-7):
                    var sd = path_ptr[].ray.direction
                    guide_record(guide_write, parent_cell, sd.x, sd.y, sd.z, w)


comptime RR_THROUGHPUT_CLAMP: Float32 = 32.0


# Russian roulette after the first bounce, then save PCG state -- the same
# 8-line epilogue every shade_* path ends with. u_rr must be drawn by the
# caller (not inside here) since some callers already drew it earlier in
# the same bounce for other purposes and must not draw a second one.
comptime BOUNCE_DIFFUSE = 0


comptime BOUNCE_GLOSSY = 1


comptime BOUNCE_TRANSMIT = 2


comptime BOUNCE_NONE = -1   # not a bounce Cycles counts (a subsurface boundary)


@always_inline
def _count_bounce(path_ptr: Pointer[PathState, MutUntrackedOrigin], cycles: SIMD[DType.int32, 4], kind: Int):
    """Cycles' diffuse, glossy and transmission bounce limits: past one, the path ends at its next scattering."""
    if cycles[3] == 0 or kind < 0:
        return
    var n = path_ptr[].kind_bounces[kind] + 1
    path_ptr[].kind_bounces[kind] = n
    if Int32(n) > cycles[kind]:
        path_ptr[].at_cap = Int8(1)


@always_inline
def _apply_russian_roulette(
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    cycles: SIMD[DType.int32, 4],
    pcg: PCG32,
    u_rr: Float32,
    kind: Int,
):
    _count_bounce(path_ptr, cycles, kind)
    if path_ptr[].bounce > 1 and cycles[3] == 0:   # Cycles' sessions roulette at the next surface (shade_nee_core)
        # rr_lum, not raw throughput luma, decides termination -- eta_scale
        # (1.0 for any path that hasn't transmitted through a dielectric)
        # undoes the temporary eta^2 compression bxdf_sample_dielectric
        # applies mid-transit through a higher-index medium, exactly like
        # pbrt-v4's `rrBeta = beta * etaScale` (PathIntegrator::Li). Without
        # this, RR reads that compression as real attenuation and kills
        # paths that were about to be restored to full brightness on exit --
        # see PathState.eta_scale's docstring.
        var rr_lum = path_ptr[].throughput.luma() * path_ptr[].eta_scale
        var q = Float32(1.0) - (rr_lum if rr_lum < Float32(0.95) else Float32(0.95))
        if u_rr < q:
            path_ptr[].active = 0
        else:
            path_ptr[].throughput *= Float32(1.0) / (Float32(1.0) - q)
            var new_lum = path_ptr[].throughput.luma()
            if new_lum > RR_THROUGHPUT_CLAMP:
                path_ptr[].throughput *= RR_THROUGHPUT_CLAMP / new_lum
    path_ptr[].pcgState = pcg.state


# ── Shared epilogue for all delta/glossy-delta BSDFs (conductor, coated_conductor,
# dielectric, thin_dielectric): apply the sample, update throughput/bounce
# bookkeeping, run Russian roulette, and save PCG state. Each material's shade_*
# wrapper only differs in how it builds (bs, hit_point) — this is everything that
# happens once those are known. bs.is_valid is always 1 for the two dielectric
# variants, so the early-return there is a harmless no-op for them.
@always_inline
@always_inline
def spawn_origin(hit_point: Vec3f, geo_n: Vec3f, dir: Vec3f) -> Vec3f:
    """Origin for a ray leaving a surface along `dir`, pbrt's SpawnRay rule:
    offset to whichever side of the GEOMETRIC surface `dir` points to.

    `hit_point` is the usual front-side point (surface + geo_n * 1e-4). A bump
    or normal map tilts the shading normal, so a direction sampled around it
    can point below the geometric surface. Launched from the front-side point,
    such a ray re-hit the same surface from above and re-scattered upward,
    keeping light that pbrt's ray -- spawned below, into the geometry -- loses:
    barcelona-pavilion-day's bump-mapped deck and water read 4-10% bright in
    every indirect bounce (exact once the bump maps were removed)."""
    if dot(dir, geo_n) < Float32(0.0):
        return hit_point - geo_n * Float32(0.0002)
    return hit_point


def _finish_delta_bounce(
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    cycles: SIMD[DType.int32, 4],
    mut pcg: PCG32,
    bs: BxDFSample,
    f_spec: SpectralSample,
    hit_point: Vec3f,
    default_albedo: RGB,
    charge_depth: Bool = True,
    geo_n: Vec3f = Vec3f(0),   # REFLECTIVE callers pass it (see spawn_origin); dielectrics already chose a side
):
    if bs.is_valid == Int8(0):
        path_ptr[].active = 0
        path_ptr[].pcgState = pcg.state
        return

    var org = spawn_origin(hit_point, geo_n, bs.wi) if dot(geo_n, geo_n) > Float32(0.0) else hit_point
    path_ptr[].ray = Ray(Point3f(org[0], org[1], org[2]), Vec3f(bs.wi[0], bs.wi[1], bs.wi[2]))
    if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
        path_ptr[].albedo = default_albedo
    path_ptr[].throughput *= f_spec
    path_ptr[].specularBounce = Int8(1)
    path_ptr[].lastBsdfPdf = Float32(0.0)
    path_ptr[].ns_spec = min(path_ptr[].ns_spec + Int8(1), Int8(100))
    # `charge_depth` is False only at the boundary of a subsurface interior
    # (Material.sss_boundary). Entry, exit and total internal reflection
    # there are all parts of ONE BSSRDF scattering event, so charging them to
    # maxdepth kills TIR-trapped light before it can escape -- the interior
    # walk steps are already exempt for exactly this reason (Medium.is_sss).
    # Termination still comes from absorption, Russian roulette and the
    # render loop's round budget.
    if charge_depth:
        path_ptr[].bounce += 1

    var u_rr = pcg.next_float()
    var kind = BOUNCE_GLOSSY if Int(bs.flags) & Int(BxDFFlags.reflect) != 0 else BOUNCE_TRANSMIT
    _apply_russian_roulette(path_ptr, cycles, pcg, u_rr, kind if charge_depth else BOUNCE_NONE)


@always_inline
def _apply_normal_map_sphere[use_gpu: Bool](
    mat: Material,
    center: Vec3f,
    hit_point: Vec3f,
    geom_normal: Vec3f,
    tex_filenames: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
    textures: Pointer[GpuTexture, MutUntrackedOrigin],
    n_textures: Int,
) -> Vec3f:
    """Analytic-sphere counterpart to `_apply_normal_map` -- no mesh/UVs
    exist for a `Sphere`, so u/v and the tangent frame are derived from
    the standard spherical parameterization instead of barycentrics.
    Matches Mitsuba's OWN `Sphere::compute_surface_interaction` convention
    exactly (`src/shapes/sphere.cpp`: `theta = unit_angle_z(local)` --
    polar angle from +Z, not +Y -- `phi = atan2(local.y, local.x)`,
    `dp_du = (-local.y, local.x, 0) * 2*pi`, `dp_dv = (local.z*cos_phi,
    local.z*sin_phi, -rd) * pi` where `rd = hypot(local.x, local.y)`) --
    NOT the pbrt/y-up convention this originally used, which put the
    swirl pattern from a normal map at the wrong place on the sphere
    (right UV topology, wrong pole axis, so the whole pattern sat rotated
    90 degrees relative to a real Mitsuba render of the same scene). Only
    correct when the sphere's own object-to-world transform is identity
    (true for `sphere_sms.xml` and any other `<shape type="sphere">` with
    no `<transform>` block) -- `Sphere` stores no orientation, so a
    rotated sphere's local Z axis can't be recovered here; out of scope,
    same "no non-uniform scale" limitation `_mit_process_sphere` already
    documents.
    Degenerates to a zero tangent at the poles (`rd == 0`) like any sphere
    UV parameterization -- falls back to an arbitrary tangent frame there
    via `Frame.from_z`, matching how `_build_geom_context_full`'s own
    sphere branch already builds a frame with no meaningful tangent
    direction to prefer."""
    if mat.normal_tex_idx < Int32(0):
        return geom_normal
    var local = hit_point - center
    var rd2 = local[0]*local[0] + local[1]*local[1]
    var rd = sqrt(rd2)
    var theta = _atan2f(rd, local[2])
    var phi = _atan2f(local[1], local[0])
    if phi < Float32(0.0):
        phi += Float32(2.0) * Float32(3.14159265358979)
    var u = phi / (Float32(2.0) * Float32(3.14159265358979))
    var v = theta / Float32(3.14159265358979)
    var dp_du = Vec3f(-local[1], local[0], Float32(0.0)) * (Float32(2.0) * Float32(3.14159265358979))
    var tangent: Vec3f
    var du_len2 = dot(dp_du, dp_du)
    if du_len2 > Float32(1e-12):
        tangent = dp_du * (Float32(1.0) / sqrt(du_len2))
    else:
        var frame = Frame.from_z(Vec3f(geom_normal[0], geom_normal[1], geom_normal[2]))
        tangent = Vec3f(frame.x.x, frame.x.y, frame.x.z)
    var bitangent = cross(geom_normal, tangent)
    var b_len2 = dot(bitangent, bitangent)
    if b_len2 <= Float32(1e-12):
        return geom_normal
    bitangent = bitangent * (Float32(1.0) / sqrt(b_len2))
    var found = False
    # `sample_texture` applies pbrt's V-flip (tv = 1 - v) because a pbrt mesh
    # puts V=0 at the TOP of the image. This sphere's (u, v) are built in
    # MITSUBA's convention instead (see this function's own docstring), so
    # that flip is wrong here and has to be pre-cancelled -- otherwise the
    # normal map is read vertically MIRRORED, which tilts the shading normal
    # by ~11 degrees everywhere and bends every refracted ray.
    #
    # Verified against the reference renderer directly, at a matched hit
    # point (its own `normalmap.cpp::frame()` instrumented to dump
    # p/uv/n_local): our (u, v) already reproduce its uv to 5 decimals, and
    # sampling the map at v matches its shading normal to 0.00 degrees while
    # sampling at 1 - v is off by 11.53 degrees. Only the analytic-sphere
    # path is affected -- a pbrt trianglemesh normal map goes through
    # `_apply_normal_map`, which genuinely wants pbrt's flip.
    var ns = sample_texture[use_gpu](Int(mat.normal_tex_idx), u, Float32(1.0) - v, True, Float32(0.0), tex_filenames, textures, n_textures, found)
    if not found:
        return geom_normal
    return _decode_tangent_normal(ns, tangent, bitangent, geom_normal)


# The path tracer's footprint at this hit: the specular-chain cone while
# PathState.cone_len >= 0 (camera differentials), pbrt's camera approximation
# once a non-specular scatter has cleared it to -1.
@always_inline
def _pt_hit_footprint(
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    mat: Material,
    cam: CameraFootprint,
    instances: Pointer[Instance, MutUntrackedOrigin],
    mesh: TriangleMesh, v0: Int, v1: Int, v2: Int,
    inter: Intersection, ray_org: Vec3f, ray_dir: Vec3f,
    textures: Pointer[GpuTexture, MutUntrackedOrigin], n_textures: Int,
) -> Tuple[TriWorld, UVFootprint]:
    var tri = tri_world(mesh, v0, v1, v2, inter.primId.instanceIdx, instances)
    # The footprint only ever feeds a texture or map lookup. A material with
    # none of them (the common case) skips the whole camera-approximation
    # computation, a tenth of the diffuse kernel's instructions.
    if (mat.tex_idx < Int32(0) and mat.normal_tex_idx < Int32(0)
            and mat.bump_tex_idx < Int32(0) and mat.rough_tex_idx < Int32(0)):
        return (tri, UVFootprint.none())
    var cl = path_ptr[].cone_len
    var cone_w = cam.cone_spread * cl if cl >= Float32(0.0) else Float32(-1.0)
    var hit = ray_org + ray_dir * inter.tHit
    var fp = hit_uv_footprint(cam, tri, mesh, v0, v1, v2, hit, ray_dir, cone_w)
    # Ptex demand paging: a face this hit needs is not loaded yet, so the sample is rendered again later.
    if _ptex_missed(textures, n_textures, Int(mat.tex_idx), inter, mesh, fp.width) or \
       _ptex_missed(textures, n_textures, Int(mat.bump_tex_idx), inter, mesh, fp.width):
        path_ptr[].ptex_missed = Int8(1)
    return (tri, fp)


# Full context for NEE materials (diffuse, diffuse_transmit, coated_diffuse).
# Computes the hit footprint, applies normal/bump maps, looks up albedo texture.
# hit_point is offset along the geometric normal.
# The shading normal comes back turned toward wo (face_toward).
@always_inline
def _build_geom_context_full[use_gpu: Bool](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    inter: Intersection,
    mat: Material,
    ctx: ShadeContext,
) -> Tuple[GeomContext, Bool]:
    var (ok, is_sphere, geo_normal, ray_dir, ray_org, mesh, v0, v1, v2) = _hit_geom(
        path_ptr, inter, ctx.meshes, ctx.lights.spheres, inter.primId.instanceIdx, ctx.instances, ctx.curves)
    if not ok:
        var z = Vec3f(Float32(0.0), Float32(0.0), Float32(0.0))
        return (GeomContext(z, z, z, z, z, z, RGB(Float32(0.0)), Float32(0.0)), False)

    if is_sphere:
        # Analytic sphere: exact normal, no UVs/texture/normal-map to resolve
        # -- alb falls back to the material's flat albedo (see project_mitsuba_parser
        # memory: no sphere UV parameterization exists in this codebase yet).
        var hit_point = offset_point(ray_org + ray_dir * inter.tHit, geo_normal)
        var wo = Vec3f(-ray_dir[0], -ray_dir[1], -ray_dir[2])
        var frame = Frame.from_z(Vec3f(geo_normal[0], geo_normal[1], geo_normal[2]))
        var tangent = Vec3f(frame.x.x, frame.x.y, frame.x.z)
        var bitangent = Vec3f(frame.y.x, frame.y.y, frame.y.z)
        return (GeomContext(geo_normal, geo_normal, hit_point, wo, tangent, bitangent, mat.albedo, Float32(0.0)), True)

    var ng_ff = geo_normal
    var (tri, fp) = _pt_hit_footprint(path_ptr, mat, ctx.cam_fp, ctx.instances, mesh, v0, v1, v2, inter, ray_org, ray_dir, ctx.textures, ctx.n_textures)

    var normal = _shading_normal(mesh, v0, v1, v2, inter.u, inter.v, geo_normal, inter.primId.instanceIdx, ctx.instances)
    normal = _apply_surface_maps[use_gpu](mat, v0, v1, v2, mesh, inter, normal, ng_ff,
        tri, fp, ctx.tex_filenames, ctx.textures, ctx.n_textures)
    var wo = Vec3f(-ray_dir[0], -ray_dir[1], -ray_dir[2])
    normal = face_toward(normal, wo)

    # Offset along the GEOMETRIC normal (pbrt's OffsetRayOrigin): the shading
    # normal just turned toward wo can point into the surface.
    var hit_point = offset_point(ray_org + ray_dir * inter.tHit, ng_ff)
    var alb = _tex_lookup[use_gpu](mat, inter, v0, v1, v2, mesh, ctx.tex_filenames, ctx.textures, ctx.n_textures, fp.width)
    var frame = Frame.from_z(Vec3f(normal[0], normal[1], normal[2]))
    var tangent   = Vec3f(frame.x.x, frame.x.y, frame.x.z)
    var bitangent = Vec3f(frame.y.x, frame.y.y, frame.y.z)
    return (GeomContext(normal, ng_ff, hit_point, wo, tangent, bitangent, alb, fp.width), True)


# Extract 8 consecutive Sobol dimensions for one scattering event and advance
# the path's sampler_dim counter.
@always_inline
def _draw_sobol_8(
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
) -> SobolSamples8:
    var _sidx = Int(path_ptr[].sobol_idx)
    var _sdim = Int(path_ptr[].sampler_dim)
    var seeds = SIMD[DType.uint32, 8](0)
    comptime for k in range(8):
        seeds[k] = sobol_dim_seed(path_ptr[].sobol_seed, _sdim + k)
    var u = sobol_sample_n[8](_sidx, _sdim, seeds, sobol_matrices)
    var u_light = u[0]
    var u_bary1 = u[1]
    var u_bary2 = u[2]
    var u_env1  = u[3]
    var u_env2  = u[4]
    var u_scat1 = u[5]
    var u_scat2 = u[6]
    var u_rr    = u[7]
    path_ptr[].sampler_dim += Int32(8)
    return SobolSamples8(u_light, u_bary1, u_bary2, u_env1, u_env2, u_scat1, u_scat2, u_rr)

# ── Hair BSDF helpers ────────────────────────────────────────────────────────
# Live in bvh.mojo, not here — sppm.mojo/bdpt_*.mojo need them too (for
# connectible-vertex/gather/NEE evaluation of a stored hair vertex), and
# shading.mojo already imports FROM sppm.mojo, so sppm.mojo can't import back
# from shading.mojo. See bvh.mojo's _hair_precompute/_hair_eval_lobes/
# _hair_sample_dir/HairLobeConstants.
