"""The path tracer's shading step: dispatch to a material, on CPU and GPU."""
from std.collections import Array
from std.math import sqrt, floor, abs
from std.ffi import external_call
from std.memory.alloc import unsafe_alloc
from gonzales.core.geometry import offset_point, RGB, Point3f, Vec3f, dot, Frame, fast_sincos, PI, TWO_PI, INV_FOUR_PI, _is_real_ptr
from gonzales.transport.render_state import PDF_DROP_DIRECT
from gonzales.scene.appearance.materials import Material, MatKind, MeasuredBRDF, resolve_mix
from gonzales.transport.render_state import PathState, ShadowTask, SuppressTask
from gonzales.scene.appearance.materials import GpuTexture, NormalSlopeMap
from gonzales.scene.shapes.primitives import Intersection, PrimId, TriangleMesh, Sphere, Instance
from gonzales.core.geometry import Ray
from gonzales.scene.shapes.primitives import mesh_pos
from gonzales.scene.appearance.lights import AreaLight, DistantLight, PointLight, InfiniteLight, LightSampler, il_rgb, light_sampler_pdf_at
from gonzales.scene.appearance.media import Medium, MediumInterface, Grid, NvdbGrid
from gonzales.scene.shapes.curves import Curve
from gonzales.core.rng import PCG32
from gonzales.scene.appearance.footprint import CameraFootprint
from gonzales.scene.accel.bvh import BVH2Node, BVH4, traverse_dispatch, _equal_area_sphere_to_square
from gonzales.core.sampling import power_heuristic
from gonzales.core.transform import Mat4
from gonzales.transport.guide import GuideGrid, null_guide
from gonzales.core.spectrum import SpectralHandle, null_spectral_handle, SpectralSample, rgb_to_spectral_sample, rgb_illuminant_to_spectral_sample
from gonzales.transport.restir_di import ReservoirIO, reservoir_io_null
from gonzales.transport.restir_gi import GIReservoirIO, gi_reservoir_io_null
from gonzales.integrators.pt.restir_sms import SMSReservoirIO, sms_reservoir_io_null
from gonzales.transport.texturing import _get_tri_verts
from gonzales.transport.shade_context import GIPendingX1, LightContext, ShadeContext
from gonzales.transport.manifold import _emitter_face_normal
from gonzales.integrators.pt.common import area_light_hit_cos, curve_light_hit, _geom_normal_and_ray, _hit_emission, _clamped, _to_spec_illum
from gonzales.integrators.pt.nee import _mnee_bsdf_weight, _glossy_sms_bsdf_weight
from gonzales.transport.glossy_sms import glossy_sms_live, GSMS_DEBUG, GLOSSY_SMS_MIN_PDF
from gonzales.transport.manifold import _mnee_first_probe
from gonzales.integrators.pt.materials import shade_diffuse_transmission, _shade_sss_exit, shade_coated_diffuse, shade_dielectric, shade_thin_dielectric, shade_conductor, shade_measured, shade_hair, shade_diffuse, shade_interface


@always_inline
def shade_core(
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    materials: Pointer[Material, MutUntrackedOrigin],
    spectral: SpectralHandle,
    tid: Int,
):
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].active == 0:
        return

    var inter = intersections[unsafe_offset=tid]
    if inter.hit == 0:
        path_ptr[].active = 0
        return

    var mat_idx = Int(inter.primId.materialIndex)
    var mat = materials[unsafe_offset=mat_idx]

    if mat.type == MatKind.area_light and GSMS_DEBUG == 0:
        path_ptr[].estimate += path_ptr[].throughput * rgb_illuminant_to_spectral_sample(
            spectral.coeffs, spectral.res, spectral.cie_x, spectral.cie_y,
            spectral.cie_z, spectral.d65,
            mat.emission.r, mat.emission.g, mat.emission.b, path_ptr[].wavelengths)
        path_ptr[].active = 0
        return

    if mat.type == MatKind.diffuse:
        # Construct Normal
        var (mesh, v0, v1, v2, ok) = _get_tri_verts(inter, meshes)
        if not ok:
            path_ptr[].active = 0
            return

        var p0 = mesh_pos(mesh, v0)
        var p1 = mesh_pos(mesh, v1)
        var p2 = mesh_pos(mesh, v2)

        var (normal, ray_dir, _) = _geom_normal_and_ray(path_ptr, p0, p1, p2)

        # Cosine weighted hemisphere sampling
        var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
        var u1 = pcg.next_float()
        var u2 = pcg.next_float()
        path_ptr[].pcgState = pcg.state

        var r = sqrt(u1)
        var theta = 2.0 * PI * u2
        var (stheta, ctheta) = fast_sincos(theta)
        var x = r * ctheta
        var y = r * stheta
        var z2 = 1.0 - u1
        var z = sqrt(z2 if z2 > 0.0 else Float32(0.0))

        # Build tangent basis
        var frame = Frame.from_z(Vec3f(normal[0], normal[1], normal[2]))
        var tangent = Vec3f(frame.x.x, frame.x.y, frame.x.z)
        var bitangent = Vec3f(frame.y.x, frame.y.y, frame.y.z)

        var dir = tangent * x + bitangent * y + normal * z
        var dlen = dot(dir, dir)
        if dlen > 0:
            dir = dir * (1.0 / sqrt(dlen))

        # Update Ray
        var org = offset_point(Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z) + ray_dir * inter.tHit, normal)
        path_ptr[].ray = Ray(Point3f(org[0], org[1], org[2]), Vec3f(dir[0], dir[1], dir[2]))

        # Update Throughput (albedo)
        path_ptr[].throughput *= rgb_to_spectral_sample(
            spectral.coeffs, spectral.res, spectral.cie_x, spectral.cie_y,
            spectral.cie_z, spectral.d65,
            mat.albedo.r, mat.albedo.g, mat.albedo.b, path_ptr[].wavelengths)
    else:
        # Unknown material type — deactivate to prevent infinite loops
        path_ptr[].active = 0


# Mix material: randomly select one of two sub-materials using amount as probability.
# Sub-material indices are packed into mat.tex_idx: low 16 bits = idx1, high 16 bits = idx2.
# mat.roughU = blend amount (probability of picking mat2).
# Not @always_inline: shade_mix ↔ _shade_dispatch would form an always_inline recursion.
def shade_mix[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    inter: Intersection,
    ctx: ShadeContext,
    mat: Material,
):
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var sub_mat = mat
    var sub_idx = Int(inter.primId.materialIndex)
    resolve_mix(ctx.materials, pcg, sub_mat, sub_idx)
    path_ptr[].pcgState = pcg.state
    var sub_inter = inter
    sub_inter.primId.materialIndex = Int64(sub_idx)
    _shade_dispatch[use_gpu, enqueue_shadow](sub_mat, path_ptr, sub_inter, ctx)


@always_inline
def _shade_dispatch[use_gpu: Bool, enqueue_shadow: Bool](
    mat: Material,
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    inter: Intersection,
    ctx: ShadeContext,
    guide_write: GuideGrid = null_guide(),
    restir_io: ReservoirIO = reservoir_io_null(),
    pixel_idx: Int = -1,
    sms_io: SMSReservoirIO = sms_reservoir_io_null(),
):
    if mat.type == MatKind.diffuse:
        shade_diffuse[use_gpu, enqueue_shadow](path_ptr, inter, ctx, mat, guide_write, restir_io, pixel_idx, sms_io)
    # Delta BSDFs (dielectric variants) need only triangle geometry — no NEE,
    # textures, or Sobol. Passing ctx.meshes directly keeps GPU kernel
    # argument counts minimal.
    elif mat.type == MatKind.conductor or mat.type == MatKind.coated_conductor:
        shade_conductor[use_gpu, enqueue_shadow](path_ptr, inter, ctx, mat)
    elif mat.type == MatKind.dielectric:
        shade_dielectric[False, enqueue_shadow](path_ptr, inter, ctx, mat)
    elif mat.type == MatKind.coated_diffuse:
        shade_coated_diffuse[use_gpu, enqueue_shadow](path_ptr, inter, ctx, mat)
    elif mat.type == MatKind.diffuse_transmit:
        shade_diffuse_transmission[use_gpu, enqueue_shadow](path_ptr, inter, ctx)
    elif mat.type == MatKind.mix:
        shade_mix[use_gpu, enqueue_shadow](path_ptr, inter, ctx, mat)
    elif mat.type == MatKind.thin_dielectric:
        shade_thin_dielectric(path_ptr, inter, ctx.meshes, mat, ctx.lights.spheres, ctx.cycles)
    elif mat.type == MatKind.interface:
        shade_interface(path_ptr, inter)
    elif mat.type == MatKind.hair:
        comptime if not use_gpu:
            shade_hair[use_gpu, enqueue_shadow](path_ptr, inter, ctx, mat)
    elif mat.type == MatKind.measured:
        shade_measured[use_gpu, enqueue_shadow](path_ptr, inter, ctx, mat)
    else:
        path_ptr[].active = 0


@always_inline
def _wrap_octahedral_texel(x: Int, y: Int, iw: Int, ih: Int) -> Tuple[Int, Int]:
    """One INTEGER texel coordinate of an equal-area (octahedral) environment
    map, mirrored back into [0,iw)x[0,ih) when a bilinear filter's neighbour
    sample falls just outside the image -- pbrt's RemapPixelCoords
    (util/image.h, WrapMode::OctahedralSphere), not a plain clamp.

    The equal-area square isn't a flat, seamlessly-tiling image: its four
    edges each connect to ANOTHER part of the same edge, reflected through
    its midpoint (Clarberg's mapping folds the sphere's octants onto the
    square along those edges). A bilinear tap that clamps at an edge instead
    of mirroring re-reads the LAST valid texel as an extra sample, silently
    blending in that edge's own color a second time instead of the correct
    across-the-seam neighbour -- invisible on a flat/constant environment
    (every tap reads the same value either way), but a systematic energy/
    colour error near every edge and the four corners of a real texture.
    Confirmed against pbrt-v4's own logic (RemapPixelCoords doesn't fall
    through to `Clamp`; MIS/NEE's CDF-based sample already lands on a valid
    in-range texel by construction and needs no equivalent here)."""
    var xr = x
    var yr = y
    if xr < 0:
        xr = -xr
        yr = ih - 1 - yr
    elif xr >= iw:
        xr = 2 * iw - 1 - xr
        yr = ih - 1 - yr
    if yr < 0:
        xr = iw - 1 - xr
        yr = -yr
    elif yr >= ih:
        xr = iw - 1 - xr
        yr = 2 * ih - 1 - yr
    return (max(0, min(iw - 1, xr)), max(0, min(ih - 1, yr)))


# Unified NEE core — comptime-specialized for CPU (use_gpu=False) and GPU (use_gpu=True).
# Texture lookup samples the in-memory GpuTexture table on both backends (OIIO only as the CPU's no-table fallback).
@always_inline
def shade_nee_core[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    inter: Intersection,
    ctx: ShadeContext,
    guide_write: GuideGrid = null_guide(),
    restir_io: ReservoirIO = reservoir_io_null(),
    pixel_idx: Int = -1,
    sms_io: SMSReservoirIO = sms_reservoir_io_null(),
):
    # ── Miss handler: ray escaped — add infinite light and deactivate ──────────
    if inter.hit == 0:
        # Volume scatter sets hit=0 to skip surface shading but is not a true miss.
        if path_ptr[].volume_scattered == Int8(1):
            path_ptr[].volume_scattered = Int8(0)
            return
        var ray_dir = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
        var ray_org = Point3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
        var miss_albedo = RGB(Float32(0.0))
        for inf_i in range(ctx.lights.infinite_count):
            var ilight = ctx.lights.infinite_lights[unsafe_offset=inf_i]
            if not ilight.portal_admits(ray_org, ray_dir):
                continue
            # Transform world-space ray direction into light's local frame
            var local_dir = Mat4.load(ilight.world_to_light) * ray_dir
            var env_rgb: RGB
            if ilight.tex_idx >= Int32(0) and _is_real_ptr(ilight.pixels_ptr) and ilight.cdf_w > Int32(0):
                # Bilinear lookup in GPU/CPU-resident pixels (cdf_w × cdf_h, 3 floats/pixel)
                var iw = Int(ilight.cdf_w); var ih = Int(ilight.cdf_h)
                var ea_uv = _equal_area_sphere_to_square(local_dir[0], local_dir[1], local_dir[2])
                var u = ea_uv[0]; var v = ea_uv[1]
                var fx = u * Float32(iw) - Float32(0.5)
                var fy = v * Float32(ih) - Float32(0.5)
                # Un-clamped: may be -1 or iw/ih right at an edge, which is
                # exactly what _wrap_octahedral_texel below needs to tell a
                # genuine off-image tap from an in-range one. The bilinear
                # WEIGHT stays tied to this raw floor, not to wherever the
                # corner's mirrored texel ends up.
                var x0r = Int(floor(fx))
                var y0r = Int(floor(fy))
                var wx = fx - Float32(x0r); var wy = fy - Float32(y0r)
                @always_inline
                def texel(x: Int, y: Int) {imm} -> RGB:
                    return il_rgb(ilight, y*iw+x)
                var (cx00, cy00) = _wrap_octahedral_texel(x0r,     y0r,     iw, ih)
                var (cx10, cy10) = _wrap_octahedral_texel(x0r + 1, y0r,     iw, ih)
                var (cx01, cy01) = _wrap_octahedral_texel(x0r,     y0r + 1, iw, ih)
                var (cx11, cy11) = _wrap_octahedral_texel(x0r + 1, y0r + 1, iw, ih)
                var c00 = texel(cx00, cy00); var c10 = texel(cx10, cy10)
                var c01 = texel(cx01, cy01); var c11 = texel(cx11, cy11)
                env_rgb = (c00*(Float32(1)-wx) + c10*wx)*(Float32(1)-wy) + (c01*(Float32(1)-wx) + c11*wx)*wy
                env_rgb = env_rgb * ilight.scale
            else:
                comptime if not use_gpu:
                    if ilight.tex_idx >= Int32(0):
                        var fname = ctx.tex_filenames[unsafe_offset=Int(ilight.tex_idx)]
                        var ea_uv2 = _equal_area_sphere_to_square(local_dir[0], local_dir[1], local_dir[2])
                        var u = ea_uv2[0]; var v = ea_uv2[1]
                        var tr = unsafe_alloc[Float32](3)
                        tr[unsafe_offset=0] = Float32(0.0); tr[unsafe_offset=1] = Float32(0.0); tr[unsafe_offset=2] = Float32(0.0)
                        _ = external_call["texture", Bool,
                            Pointer[UInt8, MutUntrackedOrigin], Float32, Float32,
                            Pointer[Float32, MutUntrackedOrigin]](fname, u, v, tr)
                        env_rgb = RGB(tr[unsafe_offset=0], tr[unsafe_offset=1], tr[unsafe_offset=2]) * ilight.scale
                        tr.unsafe_free()
                    else:
                        env_rgb = ilight.scale
                else:
                    env_rgb = ilight.scale
            miss_albedo += env_rgb
            var mis_weight = Float32(1.0)
            # A scatter whose direct-light term NEE has already reported (the
            # layered coat exit; see PDF_DROP_DIRECT) must contribute only its
            # indirect bounces here. The 1.0 default above is for the CAMERA
            # ray, whose lastBsdfPdf is also 0 and which must take the full
            # background -- hence a negative sentinel rather than a zero.
            if path_ptr[].lastBsdfPdf == PDF_DROP_DIRECT:
                mis_weight = Float32(0.0)
            # env_rgb_contrib defaults to the bilinear-filtered env_rgb (smooth
            # sky appearance for direct/specular escapes, where mis_weight stays
            # 1.0 and no pdf is involved). For MIS-weighted indirect escapes
            # (bounce>0, non-specular) with a CDF-textured light, it's
            # overridden below to the SAME nearest-texel value the pdf is
            # computed from — bilinear-blending the radiance while reading the
            # pdf from one sharp texel is a radiance/pdf registration mismatch
            # that biases (and adds fireflies to) this MIS term specifically.
            var env_rgb_contrib = env_rgb
            # `lastBsdfPdf > 0`, NOT `bounce > 0`. Both mean "a sampled
            # scattering event already happened, and its pdf is the MIS
            # partner for this escape" -- and they agree for every ordinary
            # path (a non-delta bounce sets both; a delta one zeroes the pdf
            # and is excluded by specularBounce anyway). They diverge for a
            # SUBSURFACE interior, whose walk is deliberately not charged to
            # `bounce` (Material.sss_boundary / Medium.is_sss): `bounce`
            # stays 0 for hundreds of real phase-sampled scatters, so every
            # ray that escaped the skin and hit the sky took FULL weight
            # instead of its MIS weight -- double-counting against the volume
            # NEE at that same vertex, and handing each escape that happened
            # to land on the sun the raw sun radiance. head.pbrt: 48 pixels
            # held 95% of the frame's energy, max 52747 against a pbrt
            # reference whose whole-frame mean is 0.282, and the mean did not
            # converge (64spp 4.40, 256spp 3.99, 1024spp 11.79). Same class of
            # bug as _dielectric_bounce's old `bounce == 0` force-entering
            # test, which this file already documents.
            if path_ptr[].specularBounce == Int8(0) and path_ptr[].lastBsdfPdf > Float32(0.0):
                var pdf_bsdf = path_ptr[].lastBsdfPdf
                # Untextured env: the MIS partner is whatever pdf the last
                # scattering material's OWN env NEE sampler assigns to this
                # direction, carried on the path because there are two such
                # samplers and this handler cannot tell which one ran. See
                # PathState.lastEnvNeePdf for the full story -- hardcoding
                # `pdf_bsdf` here (exact for diffuse, wrong for everyone
                # else) pinned every other material's escape at MIS 0.5.
                var pdf_light = path_ptr[].lastEnvNeePdf
                # Use CDF-based pdf when available (env-map importance sampling)
                if ilight.cdf_w > Int32(0) and ilight.cdf_h > Int32(0):
                        var iw = Int(ilight.cdf_w); var ih = Int(ilight.cdf_h)
                        var ea_uv3 = _equal_area_sphere_to_square(local_dir[0], local_dir[1], local_dir[2])
                        var u = ea_uv3[0]; var v = ea_uv3[1]
                        var px = Int(min(Float32(iw - 1), max(Float32(0.0), u * Float32(iw))))
                        var py = Int(min(Float32(ih - 1), max(Float32(0.0), v * Float32(ih))))
                        var marginal_base = ih + 1
                        var row_cdf_base = marginal_base + py * (iw + 1)
                        # pdf of this texel in the CDF (equal-area: uniform solid angle)
                        var dp_row = ilight.cdf_ptr[unsafe_offset=py + 1] - ilight.cdf_ptr[unsafe_offset=py]
                        var dp_col_base = row_cdf_base + px
                        var dp_col = ilight.cdf_ptr[unsafe_offset=dp_col_base + 1] - ilight.cdf_ptr[unsafe_offset=dp_col_base]
                        if dp_row > Float32(0.0):
                            pdf_light = dp_row * dp_col * Float32(iw) * Float32(ih) * INV_FOUR_PI
                        if _is_real_ptr(ilight.pixels_ptr):
                            env_rgb_contrib = il_rgb(ilight, py * iw + px) * ilight.scale
                mis_weight = power_heuristic(pdf_bsdf, pdf_light)
            path_ptr[].estimate += _clamped(ctx, path_ptr[].throughput * _to_spec_illum(ctx, env_rgb_contrib, path_ptr[].wavelengths) * mis_weight, Int(path_ptr[].bounce) - 1)
        if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
            path_ptr[].albedo = miss_albedo
        path_ptr[].active = 0
        return

    var mat = ctx.materials[unsafe_offset=Int(inter.primId.materialIndex)]

    # ── Analytical sphere hit: collect emission and terminate, but ONLY for
    # actual area-light spheres. A non-emissive sphere (e.g. a medium-bounding
    # volume with Material "interface", or a regular conductor/diffuse
    # sphere) falls through to the normal material dispatch below, same as
    # any other primitive type — it used to unconditionally terminate here
    # regardless of isAreaLight, silently killing every ray that hit any
    # non-light sphere (e.g. smoke-plume's medium-bounding sphere never got
    # to run its "interface" material's pass-through logic).
    if inter.primId.type == Int8(4) and ctx.lights.sphere_count > 0:
        var sph_idx = Int(inter.primId.id1)
        var sph = ctx.lights.spheres[unsafe_offset=sph_idx]
        if sph.isAreaLight == Int8(1):
            if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
                path_ptr[].estimate += _clamped(ctx, path_ptr[].throughput * _to_spec_illum(ctx, sph.emission, path_ptr[].wavelengths), Int(path_ptr[].bounce) - 1)
            else:
                # MIS: BSDF pdf vs sphere solid-angle pdf from previous shading point
                var pdf_bsdf = path_ptr[].lastBsdfPdf
                var ray_org = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
                var to_c_x = sph.center.x - ray_org[0]
                var to_c_y = sph.center.y - ray_org[1]
                var to_c_z = sph.center.z - ray_org[2]
                var dc_sq = to_c_x*to_c_x + to_c_y*to_c_y + to_c_z*to_c_z
                var sin2_max = sph.radius * sph.radius / dc_sq
                if sin2_max < Float32(1.0) and pdf_bsdf > Float32(0.0):
                    var cos_max = sqrt(Float32(1.0) - sin2_max)
                    var solid_angle = TWO_PI * (Float32(1.0) - cos_max)
                    # No 1/count factor -- see _sample_sphere_light_nee,
                    # whose pdf this must match exactly for MIS to be right.
                    var pdf_light = Float32(1.0) / solid_angle
                    var w = power_heuristic(pdf_bsdf, pdf_light)
                    path_ptr[].estimate += _clamped(ctx, path_ptr[].throughput * _to_spec_illum(ctx, sph.emission, path_ptr[].wavelengths) * w, Int(path_ptr[].bounce) - 1)
            path_ptr[].active = 0
            return

    if inter.primId.type == Int8(3):
        # Area light triangle hit — use emission from AreaLight directly so
        # NamedMaterial area lights (mat.type == 1) also emit correctly.
        var al_idx = Int(inter.primId.id1)
        var al = ctx.lights.area_lights[unsafe_offset=al_idx]
        var emission = al.emission
        var ns_n = path_ptr[].last_ns_n
        var mnee_w_b = Float32(1.0)   # this path's share against MNEE through the same glass chain
        var hp = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z) \
                 + Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z) * inter.tHit
        var shared_with = 0   # 1: MNEE from a diffuse vertex may sample this path, 2: glossy SMS may
        if path_ptr[].specularBounce == Int8(1) and dot(ns_n, ns_n) > Float32(0.0) and al.kind == Int8(0):   # marked only where MNEE ran
            # This arrived through a specular chain from a vertex that
            # delegates such paths to MNEE/SMS. Drop it exactly when MNEE
            # owns the point reached, by re-asking MNEE's own questions for
            # it: that vertex's NEE reaches the point (straight-line cosines
            # positive at both ends -- _nee_area_lights' gate) and the first
            # thing on the straight segment is glass (the probe's). Anything
            # else -- a lamp face turned away from the straight line, glass
            # only off to the side -- MNEE never samples, and BSDF sampling
            # is the only strategy that does, so it is kept. Curve lights
            # (kind 1) never take the MNEE branch at all.
            var seg = hp - path_ptr[].last_ns_p
            var seg_len = sqrt(dot(seg, seg))
            if seg_len > Float32(0.0001):
                var seg_dir = seg * (Float32(1.0) / seg_len)
                if dot(ns_n, seg_dir) > Float32(0.0) \
                   and area_light_hit_cos(inter, ctx.meshes, ctx.instances, seg_dir) > Float32(0.0):
                    if _mnee_first_probe(ctx, path_ptr[].last_ns_p, seg_dir, seg_len)[0]:
                        shared_with = 1
        elif path_ptr[].gl_pdf >= GLOSSY_SMS_MIN_PDF and path_ptr[].specularBounce == Int8(1) and path_ptr[].ns_spec >= Int8(1) \
             and al.kind == Int8(0) and glossy_sms_live(ctx):
            shared_with = 2
        if shared_with != 0:
            comptime if use_gpu:
                # The solve is far too big for this kernel: park the hit for resolve_suppress_gpu.
                if _is_real_ptr(ctx.suppress_tasks):
                    var edir_d = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
                    var park = SpectralSample(Float32(0.0))
                    if area_light_hit_cos(inter, ctx.meshes, ctx.instances, edir_d) > Float32(0.0):
                        park = path_ptr[].throughput * _to_spec_illum(ctx, emission, path_ptr[].wavelengths)
                    ctx.suppress_tasks[unsafe_offset=ctx.path_idx] = SuppressTask(park, Int32(shared_with), Int32(0), Int32(0), Int32(0))
                if mat.type == MatKind.lamp:   # the ray goes on (Cycles' shade_light.h)
                    shade_interface(path_ptr, inter)
                    return
                path_ptr[].active = 0
                return
            else:
                if shared_with == 1:
                    mnee_w_b = _mnee_bsdf_weight(ctx, path_ptr, inter, hp)
                else:
                    mnee_w_b = _glossy_sms_bsdf_weight(ctx, path_ptr, inter, hp)
        # A pbrt area light emits from its FRONT face only
        # (DiffuseAreaLight::L: `if (!twoSided && Dot(n, w) < 0) return 0`).
        # The MIS branch below has always enforced that via its own
        # `cos_l_hit > 0`, but this camera-ray/specular branch did not, so a
        # ray landing on the BACK of an emitter was credited full radiance.
        # Invisible for a flat quad light (you cannot see its back without
        # going behind it) and glaring for a CLOSED emissive shell, where the
        # far side of the shell is visible through every gap: the lantern in
        # barcelona-pavilion rendered as a solid bright blob at 2.07x pbrt's
        # emitted radiance, where pbrt shows dark structure between its bars.
        # "bool twosided" is honoured: area_light_hit_cos returns |cos| for a two-sided mesh.
        var e_front = True
        if inter.primId.type == Int8(3):
            var edir = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
            e_front = area_light_hit_cos(inter, ctx.meshes, ctx.instances, edir) > Float32(0.0)
        if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
            comptime if GSMS_DEBUG >= 2:
                if e_front and shared_with == 2:
                    path_ptr[].estimate += path_ptr[].throughput * _to_spec_illum(ctx, emission, path_ptr[].wavelengths) * (mnee_w_b if GSMS_DEBUG == 2 else Float32(1))
            if e_front and mnee_w_b > Float32(0.0):
                path_ptr[].estimate += _clamped(ctx, path_ptr[].throughput * _to_spec_illum(ctx, emission, path_ptr[].wavelengths) * mnee_w_b, Int(path_ptr[].bounce) - 1)
        else:
            var pdf_bsdf = path_ptr[].lastBsdfPdf
            if pdf_bsdf > Float32(0.0):
                var (lmesh, lv0, lv1, lv2, _) = _get_tri_verts(inter, ctx.meshes)
                var lp0 = mesh_pos(lmesh, lv0)
                var lp1 = mesh_pos(lmesh, lv1)
                var lp2 = mesh_pos(lmesh, lv2)
                # Same supplied-normal orientation as the NEE sampler above --
                # this branch MIS-weights against that sampler's pdf, so the
                # two must agree on which way the emitter faces.
                var lnorm = _emitter_face_normal(lmesh, lv0, lv1, lv2, inter.u, inter.v,
                                                 lp0, lp1, lp2,
                                                 inter.primId.instanceIdx, ctx.instances)
                var lnlen = dot(lnorm, lnorm)
                if lnlen > Float32(0.0):
                    lnorm = lnorm * (Float32(1.0) / sqrt(lnlen))
                var ray_dir = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
                var cos_l = -dot(lnorm, ray_dir)
                if lmesh.emit_two_sided != Int32(0):
                    cos_l = abs(cos_l)
                # Distance from the vertex whose sample generated this
                # direction, NOT from the current ray origin -- those differ by
                # every null interface crossed in between (PathState.mis_null_dist).
                var dist  = inter.tHit + path_ptr[].mis_null_dist
                var dist2 = dist * dist
                if cos_l > Float32(0.0) and al.total_area > Float32(0.0):
                    var ls = ctx.lights.light_sampler
                    var al_sel_pdf = ls.cdf[unsafe_offset=al_idx + 1] - ls.cdf[unsafe_offset=al_idx]
                    if dot(path_ptr[].last_ns_n, path_ptr[].last_ns_n) > Float32(0.0):
                        al_sel_pdf = light_sampler_pdf_at(ls, Int32(al_idx), path_ptr[].last_ns_p)
                    var pdf_light = dist2 * max(al_sel_pdf, Float32(1e-6)) / (cos_l * al.total_area)
                    var w = power_heuristic(pdf_bsdf, pdf_light)
                    path_ptr[].estimate += _clamped(ctx, path_ptr[].throughput * _to_spec_illum(ctx, emission, path_ptr[].wavelengths) * w, Int(path_ptr[].bounce) - 1)
        if mat.type == MatKind.lamp:   # the ray goes on (Cycles' shade_light.h)
            shade_interface(path_ptr, inter)
            return
        path_ptr[].active = 0
        return

    if inter.primId.type == Int8(5) and mat.type == MatKind.area_light:
        # Emissive curve hit directly by the camera/bounce ray. Curves are
        # now explicitly NEE-sampled too (AreaLight.kind==1, see al_list
        # in finalize_scene) — symmetric with the type==3 triangle case
        # above, this MIS-weights against the curve's own selection pdf
        # instead of always crediting full emission (which would double-
        # count against the NEE strategy now that one exists).
        var emission = mat.emission
        if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
            path_ptr[].estimate += _clamped(ctx, path_ptr[].throughput * _to_spec_illum(ctx, emission, path_ptr[].wavelengths), Int(path_ptr[].bounce) - 1)
        else:
            var pdf_bsdf = path_ptr[].lastBsdfPdf
            if pdf_bsdf > Float32(0.0):
                var ray_dir = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
                var (al_idx, cos_l) = curve_light_hit(inter, ctx.curves, ctx.lights.area_lights,
                                                      Int(ctx.lights.area_light_count), ray_dir)
                if al_idx >= 0:
                    var al = ctx.lights.area_lights[unsafe_offset=al_idx]
                    if al.total_area > Float32(0.0):
                        var dist2 = inter.tHit * inter.tHit
                        if cos_l > Float32(0.0):
                            var ls = ctx.lights.light_sampler
                            var al_sel_pdf = ls.cdf[unsafe_offset=al_idx + 1] - ls.cdf[unsafe_offset=al_idx]
                            if dot(path_ptr[].last_ns_n, path_ptr[].last_ns_n) > Float32(0.0):
                                al_sel_pdf = light_sampler_pdf_at(ls, Int32(al_idx), path_ptr[].last_ns_p)
                            var pdf_light = dist2 * max(al_sel_pdf, Float32(1e-6)) / (cos_l * al.total_area)
                            var w = power_heuristic(pdf_bsdf, pdf_light)
                            path_ptr[].estimate += _clamped(ctx, path_ptr[].throughput * _to_spec_illum(ctx, emission, path_ptr[].wavelengths) * w, Int(path_ptr[].bounce) - 1)
        path_ptr[].active = 0
        return

    if mat.emit_tex_idx >= Int32(0) and inter.primId.type == Int8(0):
        path_ptr[].estimate += _clamped(ctx, path_ptr[].throughput * _to_spec_illum(ctx, _hit_emission[use_gpu](mat, inter, ctx), path_ptr[].wavelengths), Int(path_ptr[].bounce) - 1)

    # A path that has used its full maxdepth budget dies HERE, at the next
    # real scattering event -- after any emission this segment landed on has
    # been collected above, and before this vertex would do NEE or sample a
    # new direction. That is exactly pbrt's ordering (integrators.cpp:
    # emission, then `if (depth++ >= maxDepth) return L;`, then SampleLd).
    # A null interface is NOT a scattering event, so a capped path is still
    # allowed to cross it and keep looking for an emitter beyond.
    # It sits here, ahead of BOTH handoffs, because it used to live only in
    # the CPU's _shade_dispatch: the GPU's per-material kernels never saw it,
    # so once the GPU round budget grew by the terminal-segment grace round
    # every GPU path tracer render ran one extra shaded bounce (cornell-box at
    # maxdepth 1 read pbrt's maxdepth-2 value).
    if path_ptr[].at_cap != Int8(0) and mat.type != MatKind.interface:
        path_ptr[].active = 0
        return

    # Cycles' Russian roulette (integrator_intersect_terminate): once the ray has hit a surface, so what the ray found
    # (sky, lights, emission above) is never weighted by it; the survival odds are the sqrt of the largest channel.
    if ctx.cycles[3] != 0 and mat.type != MatKind.interface and path_ptr[].bounce > 1:
        var p = min(sqrt(path_ptr[].throughput.max_component() * path_ptr[].eta_scale), Float32(1))
        if p < Float32(1):
            var pcg_rr = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
            var u = pcg_rr.next_float()
            path_ptr[].pcgState = pcg_rr.state
            if u >= p:
                path_ptr[].active = 0
                return
            path_ptr[].throughput *= Float32(1) / p

    # Any material other than a null interface is a REAL scattering event, so
    # the null-interface distance accumulated on the way here has served its
    # purpose and must not leak into the next segment's MIS. Reset here rather
    # than at each of the dozen sites that write lastBsdfPdf: one place, and it
    # cannot be forgotten when a new material is added. It must sit ahead of
    # BOTH handoffs below (same trap as at_cap above): in _shade_dispatch it
    # never ran on the GPU, so a leaf hit after a diffuse floor kept the
    # floor's cos/pi as its escape partner and the GPU path tracer lost 11%
    # of the sky light transmitted through diffusetransmission foliage.
    if mat.type != MatKind.interface:
        path_ptr[].mis_null_dist = Float32(0.0)
        if path_ptr[].specularBounce == Int8(0) and path_ptr[].lastBsdfPdf > Float32(0.0):
            path_ptr[].min_ray_pdf = min(path_ptr[].min_ray_pdf, path_ptr[].lastBsdfPdf)
        # Same reasoning, same one place: default the escape's env-NEE MIS
        # partner to the material-agnostic sampler (_sample_infinite_light_nee,
        # uniform sphere) that every material but `diffuse` uses, and let
        # shade_diffuse override it once it knows its scattered direction.
        # Defaulting HERE rather than at each scatter site means a material
        # added later inherits the right pdf instead of silently keeping the
        # previous bounce's -- and means forgetting it cannot reproduce the
        # flat-0.5 escape weight documented on PathState.lastEnvNeePdf.
        path_ptr[].lastEnvNeePdf = INV_FOUR_PI
        # A specular chain now starts HERE, not at an earlier MNEE vertex:
        # forget that vertex unless this one re-marks itself (only the
        # diffuse family's _nee_area_lights does). Glass continues a chain.
        if not mat.is_specular_glass():
            path_ptr[].last_ns_n = Vec3f(Float32(0.0))
            if not mat.sss_walk_interior():   # glossy SMS chains also reflect off skin
                path_ptr[].gl_pdf = Float32(0.0)

    if path_ptr[].sss.exiting != Int32(0):
        comptime if use_gpu:
            path_ptr[].pending_mat = MatKind.diffuse   # shade_diffuse_gpu shades the exit
        else:
            _shade_sss_exit[enqueue_shadow](path_ptr, inter, ctx)
        return
    comptime if use_gpu:
        # GPU: mark material for its dedicated per-material kernel
        path_ptr[].pending_mat = mat.type
    else:
        _shade_dispatch[False, enqueue_shadow](mat.glossy_filtered(path_ptr[].min_ray_pdf, ctx.filter_glossy), path_ptr, inter, ctx, guide_write, restir_io, pixel_idx, sms_io)


@always_inline
def cpu_shade_context(
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    curves: Pointer[Curve, MutUntrackedOrigin],
    materials: Pointer[Material, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight, MutUntrackedOrigin],
    areaLightCount: Int,
    tex_filenames: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
    distantLights: Pointer[DistantLight, MutUntrackedOrigin],
    distantLightCount: Int,
    pointLights: Pointer[PointLight, MutUntrackedOrigin],
    pointLightCount: Int,
    infiniteLights: Pointer[InfiniteLight, MutUntrackedOrigin],
    infiniteLightCount: Int,
    spheres: Pointer[Sphere, MutUntrackedOrigin],
    sphereCount: Int,
    light_sampler: LightSampler,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    guide: GuideGrid,
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin] = Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
    blasPrimIdsArr: Pointer[Pointer[PrimId, MutUntrackedOrigin], MutUntrackedOrigin] = Pointer[Pointer[PrimId, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
    instances: Pointer[Instance, MutUntrackedOrigin] = Pointer[Instance, MutUntrackedOrigin].unsafe_dangling(),
    guide_write: GuideGrid = null_guide(),
    spectral: SpectralHandle = null_spectral_handle(),
    measured_brdfs: Pointer[MeasuredBRDF, MutUntrackedOrigin] = Pointer[MeasuredBRDF, MutUntrackedOrigin].unsafe_dangling(),
    use_restir: Bool = False,
    gi_pending: Pointer[GIPendingX1, MutUntrackedOrigin] = Pointer[GIPendingX1, MutUntrackedOrigin].unsafe_dangling(),
    gi_io: GIReservoirIO = gi_reservoir_io_null(),
    nmaps: Pointer[NormalSlopeMap, MutUntrackedOrigin] = Pointer[NormalSlopeMap, MutUntrackedOrigin].unsafe_dangling(),
    textures: Pointer[GpuTexture, MutUntrackedOrigin] = Pointer[GpuTexture, MutUntrackedOrigin].unsafe_dangling(),
    n_textures: Int = 0,
    has_glass: Bool = True,
    mnee_stats: Pointer[Int32, MutUntrackedOrigin] = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
    bvh4: BVH4 = BVH4.none(),
    filter_glossy: Float32 = Float32(0),
    clamp: SIMD[DType.float32, 2] = SIMD[DType.float32, 2](0),
    cycles: SIMD[DType.int32, 4] = SIMD[DType.int32, 4](0),
) -> ShadeContext:
    """The per-render-invariant ShadeContext of the CPU wavefront; build once, set path_idx per path."""
    return ShadeContext(
        paths_base=Pointer[PathState, MutUntrackedOrigin].unsafe_dangling(), has_glass=has_glass, glossy_sms=has_glass, mnee_stats=mnee_stats, bvh4=bvh4, suppress_tasks=Pointer[SuppressTask, MutUntrackedOrigin].unsafe_dangling(),
        n_mediums=0, mediums=Pointer[Medium, MutUntrackedOrigin].unsafe_dangling(), medium_ifaces=Pointer[MediumInterface, MutUntrackedOrigin].unsafe_dangling(),
        grids=Pointer[Grid, MutUntrackedOrigin].unsafe_dangling(), nvdb_grids=Pointer[NvdbGrid, MutUntrackedOrigin].unsafe_dangling(),
                path_idx=0, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=tex_filenames,
        textures=textures, n_textures=n_textures,
        nmaps=nmaps,
        shadow_tasks=Pointer[ShadowTask, MutUntrackedOrigin].unsafe_dangling(),
        cam_fp=CameraFootprint.none(), sobol_matrices=sobol_matrices, guide=guide, guide_write=guide_write, use_restir=use_restir,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=spectral, measured_brdfs=measured_brdfs,
        gi_pending=gi_pending, gi_io=gi_io,
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=distantLightCount,
            point_lights=pointLights, point_count=pointLightCount,
            infinite_lights=infiniteLights, infinite_count=infiniteLightCount,
            spheres=spheres, sphere_count=sphereCount, light_sampler=light_sampler), filter_glossy=filter_glossy, clamp=clamp, cycles=cycles)


@always_inline
def shade_core_cpu_nee(
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    curves: Pointer[Curve, MutUntrackedOrigin],
    materials: Pointer[Material, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight, MutUntrackedOrigin],
    areaLightCount: Int,
    tex_filenames: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
    tid: Int,
    distantLights: Pointer[DistantLight, MutUntrackedOrigin],
    distantLightCount: Int,
    pointLights: Pointer[PointLight, MutUntrackedOrigin],
    pointLightCount: Int,
    infiniteLights: Pointer[InfiniteLight, MutUntrackedOrigin],
    infiniteLightCount: Int,
    spheres: Pointer[Sphere, MutUntrackedOrigin],
    sphereCount: Int,
    light_sampler: LightSampler,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    guide: GuideGrid,
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin] = Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
    blasPrimIdsArr: Pointer[Pointer[PrimId, MutUntrackedOrigin], MutUntrackedOrigin] = Pointer[Pointer[PrimId, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
    instances: Pointer[Instance, MutUntrackedOrigin] = Pointer[Instance, MutUntrackedOrigin].unsafe_dangling(),
    guide_write: GuideGrid = null_guide(),
    spectral: SpectralHandle = null_spectral_handle(),
    measured_brdfs: Pointer[MeasuredBRDF, MutUntrackedOrigin] = Pointer[MeasuredBRDF, MutUntrackedOrigin].unsafe_dangling(),
    use_restir: Bool = False,
    restir_io: ReservoirIO = reservoir_io_null(),
    pixel_idx: Int = -1,
    gi_pending: Pointer[GIPendingX1, MutUntrackedOrigin] = Pointer[GIPendingX1, MutUntrackedOrigin].unsafe_dangling(),
    gi_io: GIReservoirIO = gi_reservoir_io_null(),
    sms_io: SMSReservoirIO = sms_reservoir_io_null(),
    nmaps: Pointer[NormalSlopeMap, MutUntrackedOrigin] = Pointer[NormalSlopeMap, MutUntrackedOrigin].unsafe_dangling(),
    textures: Pointer[GpuTexture, MutUntrackedOrigin] = Pointer[GpuTexture, MutUntrackedOrigin].unsafe_dangling(),
    n_textures: Int = 0,
    has_glass: Bool = True,
    mnee_stats: Pointer[Int32, MutUntrackedOrigin] = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
    bvh4: BVH4 = BVH4.none(),
):
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].active == 0:
        return
    var inter = intersections[unsafe_offset=tid]
    var ctx = cpu_shade_context(bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials, areaLights=areaLights, areaLightCount=areaLightCount, tex_filenames=tex_filenames, distantLights=distantLights, distantLightCount=distantLightCount, pointLights=pointLights, pointLightCount=pointLightCount, infiniteLights=infiniteLights, infiniteLightCount=infiniteLightCount, spheres=spheres, sphereCount=sphereCount, light_sampler=light_sampler, sobol_matrices=sobol_matrices, guide=guide, blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances, guide_write=guide_write, spectral=spectral, measured_brdfs=measured_brdfs, use_restir=use_restir, gi_pending=gi_pending, gi_io=gi_io, nmaps=nmaps, textures=textures, n_textures=n_textures, has_glass=has_glass, mnee_stats=mnee_stats, bvh4=bvh4)
    ctx.path_idx = tid
    shade_nee_core[False, False](path_ptr, inter, ctx, guide_write, restir_io, pixel_idx, sms_io)


@always_inline
def shade_core_cpu_nee_ctx(
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    tid: Int,
    mut ctx: ShadeContext,
    guide_write: GuideGrid,
    restir_io: ReservoirIO,
    pixel_idx: Int,
    sms_io: SMSReservoirIO,
):
    """shade_core_cpu_nee with the ShadeContext built once by the caller (cpu_shade_context)."""
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].active == 0:
        return
    ctx.path_idx = tid
    shade_nee_core[False, False](path_ptr, intersections[unsafe_offset=tid], ctx, guide_write, restir_io, pixel_idx, sms_io)
