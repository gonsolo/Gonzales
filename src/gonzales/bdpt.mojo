# Bidirectional Path Tracing (CPU, single-threaded)
# Supports homogeneous participating media and specular chains (glass).
# Strategies: t >= 1, s >= 1 only (no lens sampling for s=0).
# MIS: balance heuristic over all valid connection strategies.

from std.sys import has_accelerator
from std.sys.info import size_of
from std.gpu import block_idx, thread_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, cos, sin, floor, log, exp, max, abs, ceildiv
from std.memory import alloc
from std.atomic import Atomic
from .geometry import (
    RGB, SampledSpectrum, Point3f, Vec3f, vec3f, point3f, Ray_C, Intersection_C, Frame,
    TriangleMesh_C, Material_C, MatKind, AreaLight_C, Medium_C, MediumInterface_C,
    Sphere_C, Curve_C, PrimId_C, Instance_C, LightSampler_C,
    DistantLight_C, PointLight_C, InfiniteLight_C, Grid_C,
    dot, cross, fr_dielectric, sphere_outward_normal, PI, INV_FOUR_PI, INV_PI,
)
from .bvh import BVH2Node, SceneDescriptor2_C, traverse_bvh2_core, any_hit_bvh2_core, test_spheres
from .rng import PCG32
from .pbrt_parser import ParsedScene_Mojo
from .postprocess import write_image
from .sppm import _geom_normal, _dielectric_bounce, _sppm_update_medium, _cosine_hemisphere_sample, sample_homogeneous_free_flight, sample_area_light_uniform
from .bxdf import GeomContext, bxdf_sample_conductor, bxdf_is_delta, ggx_D, ggx_G2
from .gpu import GpuSceneHandle
from .gpu_sppm import sppm_reset_i32_gpu

comptime _BDPT_MAX_DEPTH = 40  # max surface/medium interactions per subpath (incl.
                                # non-stored delta/dielectric bounces — glass-of-water's
                                # nested water/ice/glass interfaces need ~30 crossings
                                # just to reach a real (diffuse) vertex)
comptime _BDPT_MAX_VERTS = 10  # max non-delta vertices stored per light subpath (caps
                                # each light path's contribution to the shared cache below)

# ── Light Vertex Cache (LVC-BPT, Davidovic et al. 2014) ──────────────────────
# Instead of pairing one light subpath with one camera subpath (the old,
# CPU-only design this replaced), one *shared* cache of light-subpath vertices
# is built once per spp sample — from many independently-traced light paths,
# not tied to any one pixel — and every camera-subpath vertex connects to a
# few uniformly-random vertices from that shared pool. This is what makes the
# algorithm GPU-friendly (the light-tracing and camera-tracing+connect phases
# are each embarrassingly parallel across independent threads, with no
# per-pixel light-subpath pairing to serialize on) while remaining exactly the
# same algorithm on CPU — see `_bdpt_trace_light_path`/
# `_bdpt_trace_camera_and_connect` below, both `comptime[use_gpu: Bool]`
# parameterized so CPU and GPU share one implementation.
comptime _BDPT_K_CONNECTIONS = 2  # random light-vertex connections per non-delta camera vertex

# ── Vertex types ──────────────────────────────────────────────────────────────

@fieldwise_init
struct BDPTVertex(TrivialRegisterPassable):
    """A vertex on a camera or light subpath."""
    var pos:    Point3f  # world position
    var normal: Vec3f    # geometric normal (0 for volume)
    var beta: RGB  # throughput to here
    var alb:  RGB  # BSDF albedo (F0 for conductor)
    var pdf_fwd: Float32  # area PDF forward (from previous vertex)
    var pdf_bwd: Float32  # unused by the equal-weight MIS scheme — repurposed to hold
                           # the isotropic GGX alpha for mat_kind=1 (conductor) vertices
    var is_surface: Int32  # 1 = surface hit, 0 = volume scatter
    var is_delta:   Int32  # 1 = specular (mirror conductor / dielectric) — cannot be connected
    var is_light:   Int32  # 1 = this is a light-source vertex (s=0 in BDPT notation)
    var med_idx:    Int32  # medium index AFTER this vertex (-1 = vacuum)
    var mat_kind:   Int32  # 0 = Lambertian (diffuse/volume), 1 = rough conductor (GGX)
    # Direction back toward this vertex's own predecessor on its subpath
    # (-incoming ray direction). Only populated for mat_kind=1 (GGX needs both
    # directions around the half-vector); Lambertian/volume don't need it.
    var wo: Vec3f

@always_inline
def _null_vertex() -> BDPTVertex:
    return BDPTVertex(
        pos=Point3f(Float32(0)),
        normal=Vec3f(Float32(0), Float32(1), Float32(0)),
        beta=RGB(Float32(0)),
        alb=RGB(Float32(0)),
        pdf_fwd=Float32(0), pdf_bwd=Float32(0),
        is_surface=Int32(0), is_delta=Int32(0), is_light=Int32(0),
        med_idx=Int32(-1), mat_kind=Int32(0),
        wo=Vec3f(Float32(0)),
    )

# ── Geometry helpers ──────────────────────────────────────────────────────────

@always_inline
def _bdpt_medium_update(
    ray_dir: SIMD[DType.float32, 3],
    inter:   Intersection_C,
    mat:     Material_C,
    sd:      SceneDescriptor2_C,
    hit: Point3f,
) -> Int32:
    """Determine new current_medium_idx after crossing a surface."""
    if mat.medium_interface_idx < Int32(0) or sd.mediumIfaceCount == Int64(0):
        return Int32(-1)
    var iface = sd.mediumInterfaces[Int(mat.medium_interface_idx)]
    var n: Vec3f
    if inter.primId.type == Int8(4):
        # Analytic sphere: outward normal = normalize(hit - center)
        var si = Int(inter.primId.id1)
        var sph = sd.spheres[si]
        n = sphere_outward_normal(hit, sph.center)
    else:
        var mi: Int; var bv: Int
        if inter.primId.type == 0:
            mi = Int(inter.primId.id1); bv = Int(inter.primId.id2)
        else:
            mi = Int(inter.primId.id2 >> 32); bv = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
        var m  = sd.meshes[mi]
        var v0 = Int(m.vertexIndices[bv]); var v1 = Int(m.vertexIndices[bv+1]); var v2 = Int(m.vertexIndices[bv+2])
        var p0 = Point3f(m.points[v0*4], m.points[v0*4+1], m.points[v0*4+2])
        var p1 = Point3f(m.points[v1*4], m.points[v1*4+1], m.points[v1*4+2])
        var p2 = Point3f(m.points[v2*4], m.points[v2*4+1], m.points[v2*4+2])
        var e1 = p1 - p0; var e2 = p2 - p0
        n = Vec3f(e1.y*e2.z - e1.z*e2.y, e1.z*e2.x - e1.x*e2.z, e1.x*e2.y - e1.y*e2.x)
    var md = ray_dir[0]*n.x + ray_dir[1]*n.y + ray_dir[2]*n.z
    return iface.outside_medium_idx if md > Float32(0) else iface.inside_medium_idx

# ── Visibility with transmittance ─────────────────────────────────────────────

def _visible_transmittance(
    a: Point3f, b: Point3f,
    med_idx: Int32,
    sd:      SceneDescriptor2_C,
    scratch: UnsafePointer[Intersection_C, MutAnyOrigin],
) -> SIMD[DType.float32, 3]:
    """Returns transmittance along segment AB, or (0,0,0) if occluded.
    Glass (dielectric) surfaces are passed through with Fresnel transmittance.
    `scratch` is one caller-owned Intersection_C slot (no internal alloc/free)
    so this is safe to call from a GPU kernel thread — every existing GPU
    kernel in this codebase takes pre-allocated, thread-indexed scratch
    instead of allocating per-thread (see sppm_gen_vp_gpu's inter_scratch)."""
    var d = b - a
    var dist_total = d.length()
    if dist_total < Float32(1e-5):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var inv = Float32(1) / dist_total
    var dir = SIMD[DType.float32, 3](d.x*inv, d.y*inv, d.z*inv)

    var Tr = RGB(Float32(1))
    var org = a + Vec3f(dir[0], dir[1], dir[2]) * Float32(0.0002)
    var remaining = dist_total - Float32(0.0002)
    var cur_med = med_idx

    var inter_mem = scratch
    for _ in range(8):
        if remaining < Float32(1e-4): break
        var ray = Ray_C(org, Vec3f(dir[0], dir[1], dir[2]))
        inter_mem[0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, remaining * Float32(0.9995), inter_mem,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        # test_spheres (analytic spheres, e.g. the caustic sphere) aren't part of
        # the BVH — traverse_bvh2_core only tests triangles/curves — so they need
        # a separate pass. Seed a sentinel tHit=remaining*0.9995 when the BVH found
        # nothing, so test_spheres's own internal tMax (it only bounds itself by
        # result[0].tHit when result[0].hit is already set) respects the shadow
        # ray's segment length instead of defaulting to unbounded (1e38).
        var had_bvh_hit = inter_mem[0].hit != Int8(0)
        if not had_bvh_hit:
            inter_mem[0].hit = Int8(1)
            inter_mem[0].tHit = remaining * Float32(0.9995)
        test_spheres(sd.spheres, Int(sd.sphereCount), ray, inter_mem)
        if not had_bvh_hit and inter_mem[0].primId.type != Int8(4):
            inter_mem[0].hit = Int8(0)
        if inter_mem[0].hit == Int8(0):
            # Nothing between here and destination: apply remaining Beer-Lambert
            if Int(cur_med) >= 0:
                var med = sd.mediums[Int(cur_med)]
                var sigma_t = med.sigma_a + med.sigma_s
                Tr *= RGB(exp(-sigma_t.r*remaining), exp(-sigma_t.g*remaining), exp(-sigma_t.b*remaining))
            break

        var inter = inter_mem[0]
        var t_hit = inter.tHit
        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[mat_idx]
        var hit = org + Vec3f(dir[0], dir[1], dir[2]) * t_hit

        # Beer-Lambert through medium segment up to hit
        if Int(cur_med) >= 0:
            var med = sd.mediums[Int(cur_med)]
            var sigma_t = med.sigma_a + med.sigma_s
            Tr *= RGB(exp(-sigma_t.r*t_hit), exp(-sigma_t.g*t_hit), exp(-sigma_t.b*t_hit))

        if mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            # Pass through glass with Fresnel transmittance
            var gn: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si = Int(inter.primId.id1)
                var sph = sd.spheres[si]
                gn = sphere_outward_normal(hit, sph.center).to_simd()
            else:
                gn = _geom_normal(inter, sd.meshes, sd.instances)
            var facing = dot(dir, gn) < Float32(0)
            var n_for_cos = gn if facing else gn*Float32(-1)
            var cos_i = -dot(dir, n_for_cos)
            if cos_i < Float32(0): cos_i = -cos_i
            var ior = mat.albedo.r
            var fr = fr_dielectric(cos_i, Float32(1)/ior if facing else ior)
            var T = Float32(1) - fr
            Tr *= T
            if Tr.r < Float32(1e-7) and Tr.g < Float32(1e-7) and Tr.b < Float32(1e-7):
                return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
            # Update medium after crossing glass surface
            if mat.medium_interface_idx >= Int32(0) and sd.mediumIfaceCount > Int64(0):
                var iface = sd.mediumInterfaces[Int(mat.medium_interface_idx)]
                var md = dir[0]*gn[0]+dir[1]*gn[1]+dir[2]*gn[2]
                cur_med = iface.outside_medium_idx if md > Float32(0) else iface.inside_medium_idx
            org = hit + Vec3f(dir[0], dir[1], dir[2]) * Float32(0.0002)
            remaining = remaining - t_hit - Float32(0.0002)

        elif mat.type == MatKind.interface:
            # Pure medium boundary: update medium, continue
            if mat.medium_interface_idx >= Int32(0) and sd.mediumIfaceCount > Int64(0):
                var iface = sd.mediumInterfaces[Int(mat.medium_interface_idx)]
                var igna = _geom_normal(inter, sd.meshes, sd.instances)
                var md = dir[0]*igna[0]+dir[1]*igna[1]+dir[2]*igna[2]
                cur_med = iface.outside_medium_idx if md > Float32(0) else iface.inside_medium_idx
            org = hit + Vec3f(dir[0], dir[1], dir[2]) * Float32(0.0002)
            remaining = remaining - t_hit - Float32(0.0002)

        else:
            # Opaque surface blocks the segment
            return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))

    return SIMD[DType.float32, 3](Tr.r, Tr.g, Tr.b)

# ── Cosine-area PDF conversion ────────────────────────────────────────────────

@always_inline
def _pdf_solid_to_area(pdf_solid: Float32, cos_theta: Float32, dist2: Float32) -> Float32:
    """Convert solid-angle PDF to area PDF: p_A = p_ω * |cosθ| / r²."""
    if dist2 < Float32(1e-8): return Float32(0)
    return pdf_solid * (cos_theta if cos_theta > Float32(0) else -cos_theta) / dist2

# ── Store a vertex in the shared Light Vertex Cache ──────────────────────────

@always_inline
def _bdpt_store_lvc_vertex[use_gpu: Bool](
    v: BDPTVertex,
    lvc: UnsafePointer[BDPTVertex, MutAnyOrigin],
    lvc_cap: Int,
    lvc_counter: UnsafePointer[Int32, MutAnyOrigin],
):
    """Reserve a slot in the shared cache and store `v`, dropping the write
    if the cache is already full — same cap-and-drop convention as
    sppm_emit_photons_gpu's photon buffer (gpu_sppm.mojo). The ONLY line
    that differs between CPU and GPU: concurrent GPU threads need an atomic
    fetch-add to reserve a slot; the CPU driver traces light paths
    sequentially (bdpt_render's `for lp_idx in range(...)` loop), so a plain
    increment is correct there and the comptime branch below means the
    non-taken side is never even compiled into that specialization."""
    var slot: Int
    comptime if use_gpu:
        slot = Int(Atomic.fetch_add(lvc_counter, Int32(1)))
    else:
        slot = Int(lvc_counter[0])
        lvc_counter[0] += 1
    if slot < lvc_cap:
        lvc[slot] = v

# ── LVC connection scale factor ──────────────────────────────────────────────

@always_inline
def _bdpt_lvc_connection_scale(lvc_count: Int, n_light_paths: Int) -> Float32:
    """The factor _bdpt_trace_camera_and_connect must scale its k random
    cache connections by: avg_light_path_len / _BDPT_K_CONNECTIONS, where
    avg_light_path_len = lvc_count / n_light_paths is the average number of
    non-delta vertices each traced light path contributed to the cache.
    NOT lvc_count/k (that reconstructs "sum over the ENTIRE cache", wildly
    over-counting since the cache holds n_light_paths independent light
    paths' worth of energy) and NOT a bare 1/k (that estimates the MEAN
    over a light path's D distinct depth-strategies, when the correct
    target is their SUM — light-source-direct, one-bounce-indirect,
    two-bounce-indirect, ... are different additive terms of the rendering
    equation, not interchangeable samples of the same term). See
    `_bdpt_trace_camera_and_connect`'s docstring for the full derivation."""
    if n_light_paths <= 0:
        return Float32(0)
    var avg_light_path_len = Float32(lvc_count) / Float32(n_light_paths)
    return avg_light_path_len / Float32(_BDPT_K_CONNECTIONS)

# ── Connect one camera vertex to k random cache vertices ─────────────────────

@always_inline
def _bdpt_connect_to_cache(
    cv: BDPTVertex,
    sd: SceneDescriptor2_C,
    has_med: Bool,
    scratch: UnsafePointer[Intersection_C, MutAnyOrigin],
    lvc: UnsafePointer[BDPTVertex, MutAnyOrigin],
    lvc_count: Int,
    scale: Float32,
    mut pcg: PCG32,
) -> RGB:
    """_BDPT_K_CONNECTIONS uniformly-random connections from cv into the
    shared cache, scaled by the caller-supplied `scale` (see
    `_bdpt_trace_camera_and_connect`'s docstring for the
    avg_light_path_len/k derivation — each individual `_connect(cv, lv)` is
    already a complete, unbiased sample of ONE depth-strategy's
    contribution; averaging k such samples and rescaling by the average
    number of depth-strategies per light path reconstructs their sum). This
    REPLACES the old exhaustive same-total-path-length strategy count (see
    bdpt_render's docstring for why that scheme doesn't compose with a
    cache shared across all pixels)."""
    var sum = SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    for _ in range(_BDPT_K_CONNECTIONS):
        var ridx = Int(pcg.next_uint() % UInt32(lvc_count))
        var lv = lvc[ridx]
        sum += _connect(cv, lv, sd, has_med, scratch)
    return RGB(sum[0], sum[1], sum[2]) * scale

# ── Trace one camera subpath, connecting to the shared cache inline ─────────

def _bdpt_trace_camera_and_connect[use_gpu: Bool](
    r2c:     UnsafePointer[Float32, MutAnyOrigin],
    c2w:     UnsafePointer[Float32, MutAnyOrigin],
    px:      Int, py:      Int,
    sd:      SceneDescriptor2_C,
    mut pcg: PCG32,
    has_med: Bool,
    scratch: UnsafePointer[Intersection_C, MutAnyOrigin],
    lvc:     UnsafePointer[BDPTVertex, MutAnyOrigin],
    lvc_count: Int,
    scale: Float32,
    start_med_idx: Int32 = Int32(-1),
) -> RGB:
    """Trace one camera subpath from pixel (px,py). At each non-delta vertex,
    connect inline/synchronously to the shared Light Vertex Cache via
    `_bdpt_connect_to_cache` — mirrors how every live GPU shading kernel in
    this codebase already does its shadow ray (any_hit test, straight into
    the thread's own accumulator; gpu.mojo's queued ShadowTask_C mechanism
    is dead code, never used by the live render loop). Returns this camera
    path's total contribution for one spp sample. `use_gpu` is unused inside
    this function's own body today (no CPU/GPU divergence needed here beyond
    what `_bdpt_connect_to_cache`'s `_connect`/`_visible_transmittance` calls
    already share) — kept as a parameter so its signature matches
    `_bdpt_trace_light_path`'s and the two thin kernels wrapping this stay
    symmetric in Phase (b).

    `scale` (from `_bdpt_lvc_connection_scale` — see its docstring for the
    full derivation of why it's avg_light_path_len/k, not lvc_count/k or a
    bare 1/k) corrects for the cache mixing multiple depth-strategies
    together: a light path of length D contributes D DIFFERENT
    depth-strategies (direct light-source connection, one-bounce-indirect,
    two-bounce-indirect, ...) that must be SUMMED, not averaged — each is a
    distinct additive term of the rendering equation, not D interchangeable
    samples of the same term."""
    var fX = Float32(px) + Float32(0.5)
    var fY = Float32(py) + Float32(0.5)
    var cx = r2c[0]*fX + r2c[4]*fY + r2c[12]
    var cy = r2c[1]*fX + r2c[5]*fY + r2c[13]
    var cz = r2c[2]*fX + r2c[6]*fY + r2c[14]
    var cw = r2c[3]*fX + r2c[7]*fY + r2c[15]
    if cw != Float32(0.0) and cw != Float32(1.0):
        cx /= cw; cy /= cw; cz /= cw
    var cl = sqrt(cx*cx + cy*cy + cz*cz)
    if cl > Float32(0.0): cx /= cl; cy /= cl; cz /= cl
    var rd = Vec3f(
        c2w[0]*cx + c2w[4]*cy + c2w[8]*cz,
        c2w[1]*cx + c2w[5]*cy + c2w[9]*cz,
        c2w[2]*cx + c2w[6]*cy + c2w[10]*cz,
    )
    var dl = rd.length()
    if dl > Float32(0.0): rd = rd / dl
    var ro = Point3f(c2w[12], c2w[13], c2w[14])

    var n_verts = 0
    var n_bounces = 0  # total surface hits including glass (for _dielectric_bounce entering logic)
    var beta = RGB(Float32(1))
    var cur_med_idx = start_med_idx
    var total = RGB(Float32(0))

    for _ in range(_BDPT_MAX_DEPTH):
        if n_verts >= _BDPT_MAX_VERTS: break
        var ray = Ray_C(ro, rd)
        scratch[0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), scratch,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        test_spheres(sd.spheres, Int(sd.sphereCount), ray, scratch)
        if scratch[0].hit == Int8(0): break

        var inter = scratch[0]
        var t_hit = inter.tHit
        var ray_dir = rd.to_simd()

        # Volume free-flight
        if has_med and Int(cur_med_idx) >= 0:
            var med = sd.mediums[Int(cur_med_idx)]
            var ff = sample_homogeneous_free_flight(med, t_hit, pcg)
            if ff.collided:
                # Volume scatter vertex
                var sp = ro + rd*ff.t_free
                # BDPT vertex beta: Tr/pdf_free × phase/pdf_phase = 1/sig_t × sig_s = alb_s
                # (exp(-sig_t×t) cancels between Tr numerator and pdf denominator)
                var v = _null_vertex()
                v.pos = sp
                v.beta = beta * ff.albedo
                v.alb = ff.albedo
                v.is_surface = Int32(0); v.is_delta = Int32(0)
                v.pdf_fwd = ff.sig_t * exp(-ff.sig_t * ff.t_free)
                v.med_idx = cur_med_idx
                n_verts += 1
                if lvc_count > 0:
                    total += _bdpt_connect_to_cache(v, sd, has_med, scratch, lvc, lvc_count, scale, pcg)
                # Continuation beta = prev × alb_s (same as stored vertex beta)
                beta *= ff.albedo
                var u1 = pcg.next_float(); var u2 = pcg.next_float()
                var cosT = Float32(2)*u1 - Float32(1)
                var sinT = sqrt(max(Float32(0), Float32(1)-cosT*cosT))
                var phi  = Float32(2)*PI*u2
                rd = Vec3f(sinT*cos(phi), sinT*sin(phi), cosT)
                ro = sp + rd*Float32(0.0002)
                continue
            else:
                # Beer-Lambert through full segment to surface
                beta *= ff.transmittance

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[mat_idx]
        var hit = ro + rd*t_hit

        if mat.type == MatKind.area_light:
            break  # camera hits light directly → handled by unidirectional contribution

        elif mat.type == MatKind.diffuse or mat.type == MatKind.coated_diffuse or mat.type == MatKind.diffuse_transmit:
            var gn = _geom_normal(inter, sd.meshes, sd.instances)
            if dot(gn, ray_dir) > Float32(0): gn = gn * Float32(-1)
            var v = _null_vertex()
            v.pos = hit
            v.normal = vec3f(gn)
            v.beta = beta
            v.alb = mat.albedo
            v.is_surface = Int32(1); v.is_delta = Int32(0)
            v.pdf_fwd = Float32(1)  # unused by the uniform-subsample estimator
            v.med_idx = cur_med_idx
            n_verts += 1
            if lvc_count > 0:
                total += _bdpt_connect_to_cache(v, sd, has_med, scratch, lvc, lvc_count, scale, pcg)
            # Cosine-weighted scatter direction
            var u1 = pcg.next_float(); var u2 = pcg.next_float()
            rd = vec3f(_cosine_hemisphere_sample(gn, u1, u2))
            ro = hit + rd*Float32(0.0002)
            # Update beta: f/pdf for Lambertian = (alb/π) / (cosθ/π) = alb
            beta *= mat.albedo

        elif mat.type == MatKind.conductor:
            var gn_c: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si_c = Int(inter.primId.id1)
                var sph_c = sd.spheres[si_c]
                gn_c = sphere_outward_normal(hit, sph_c.center).to_simd()
            else:
                gn_c = _geom_normal(inter, sd.meshes, sd.instances)
            if dot(gn_c, ray_dir) > Float32(0): gn_c = gn_c * Float32(-1)
            var wo_c = (-rd).to_simd()
            var frm_c = Frame.from_z(Vec3f(gn_c[0], gn_c[1], gn_c[2]))
            var gc_c = GeomContext(
                normal=gn_c, geo_normal=gn_c, hit_point=hit.to_simd(), wo=wo_c,
                tangent=SIMD[DType.float32, 3](frm_c.x.x, frm_c.x.y, frm_c.x.z),
                bitangent=SIMD[DType.float32, 3](frm_c.y.x, frm_c.y.y, frm_c.y.z),
                alb=mat.albedo, pixel_uv=Float32(0),
            )
            var uc1 = pcg.next_float(); var uc2 = pcg.next_float()
            var bs_c = bxdf_sample_conductor(gc_c, mat, uc1, uc2)
            if bs_c.is_valid == Int8(0):
                break
            if not bxdf_is_delta(bs_c.flags):
                var v = _null_vertex()
                v.pos = hit
                v.normal = vec3f(gn_c)
                v.beta = beta
                v.alb = mat.albedo
                v.is_surface = Int32(1); v.is_delta = Int32(0); v.mat_kind = Int32(1)
                v.pdf_bwd = max(mat.roughU, mat.roughV)
                v.wo = vec3f(wo_c)
                v.pdf_fwd = Float32(1)
                v.med_idx = cur_med_idx
                n_verts += 1
                if lvc_count > 0:
                    total += _bdpt_connect_to_cache(v, sd, has_med, scratch, lvc, lvc_count, scale, pcg)
            beta *= bs_c.f
            rd = vec3f(bs_c.wi)
            ro = hit + rd*Float32(0.0002)

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            var gn: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si = Int(inter.primId.id1)
                var sph = sd.spheres[si]
                gn = sphere_outward_normal(hit, sph.center).to_simd()
            else:
                gn = _geom_normal(inter, sd.meshes, sd.instances)
            var (new_dir, new_org) = _dielectric_bounce(ray_dir, hit.to_simd(), gn, mat.albedo.r, n_bounces, pcg)
            n_bounces += 1
            # Specular vertex: no BSDF record needed, just track throughput
            if has_med:
                var new_idx = _bdpt_medium_update(ray_dir, inter, mat, sd, hit)
                if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
            rd = vec3f(new_dir)
            ro = point3f(new_org)

        elif mat.type == MatKind.interface:
            if has_med:
                var new_idx = _bdpt_medium_update(ray_dir, inter, mat, sd, hit)
                if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
            ro = hit + rd*Float32(0.0002)

        else:
            break

    return total

# ── Trace one light subpath, storing its vertices into the shared cache ─────

def _bdpt_trace_light_path[use_gpu: Bool](
    sd:      SceneDescriptor2_C,
    mut pcg: PCG32,
    has_med: Bool,
    default_emit_med: Int32,
    scratch: UnsafePointer[Intersection_C, MutAnyOrigin],
    lvc:      UnsafePointer[BDPTVertex, MutAnyOrigin],
    lvc_cap:  Int,
    lvc_counter: UnsafePointer[Int32, MutAnyOrigin],
):
    """Emit a photon from a random light and trace a light subpath, storing
    every non-delta vertex (including the light-source point itself, the
    s=1/NEE-equivalent strategy) into the shared Light Vertex Cache via
    `_bdpt_store_lvc_vertex` instead of a private per-path array — see the
    module's LVC-BPT docstring above. `_BDPT_MAX_VERTS` still caps how many
    vertices any ONE light path contributes (unchanged from before), it's
    just that they now land in a cache shared across the whole frame."""
    var n_lights = Int(sd.areaLightCount)
    if n_lights == 0:
        return

    # Pick a light uniformly + a random triangle + barycentric point on it.
    var light_sample = sample_area_light_uniform(sd.areaLights, sd.meshes, n_lights, pcg, sd.curves)
    var al = light_sample.light
    var lp = light_sample.point
    var ln = light_sample.normal

    # Cosine-weighted emission direction
    var du1 = pcg.next_float(); var du2 = pcg.next_float()
    var pdir = _cosine_hemisphere_sample(ln, du1, du2)
    var ly = sqrt(max(Float32(0), Float32(1)-du1))  # cos_theta_emitted, reused below

    # Light vertex (the emission point, s=1 strategy connects here directly).
    # beta = 1/p_A = area × n_lights (area sampling PDF correction only).
    # alb  = Le (emission) — used as f_lgt in _connect for the light vertex.
    # G already handles the cos_l factor, so f_lgt = Le (no extra cos multiply).
    var area_weight = al.total_area * Float32(n_lights)
    var lv0_vert = _null_vertex()
    lv0_vert.pos = point3f(lp)
    lv0_vert.normal = vec3f(ln)
    lv0_vert.beta = RGB(area_weight, area_weight, area_weight)
    lv0_vert.alb = al.emission
    lv0_vert.is_surface = Int32(1); lv0_vert.is_light = Int32(1)
    lv0_vert.pdf_fwd = Float32(1) / area_weight
    lv0_vert.med_idx = default_emit_med
    _bdpt_store_lvc_vertex[use_gpu](lv0_vert, lvc, lvc_cap, lvc_counter)

    # For traced vertices: beta = Le / (p_A × p_ω) where p_ω = cos_θ/π for cosine-weighted emission.
    # cos_θ_emitted = ly (the cosine of the sampled direction against the light normal).
    # β = Le × area × n_lights × π / cos_θ_emitted
    var cos_theta_emit = max(ly, Float32(0.01))
    var flux = al.emission * (area_weight * PI / cos_theta_emit)
    var ro = point3f(lp) + vec3f(ln)*Float32(0.0001)
    var rd = vec3f(pdir)
    var cur_med_idx = default_emit_med
    var n_verts = 1  # vertex 0 is the light point itself
    var n_lbounces = 1  # counts all surface hits (1 = after initial light vertex)

    for _ in range(_BDPT_MAX_DEPTH):
        if n_verts >= _BDPT_MAX_VERTS: break
        var ray = Ray_C(ro, rd)
        scratch[0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), scratch,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        test_spheres(sd.spheres, Int(sd.sphereCount), ray, scratch)
        if scratch[0].hit == Int8(0): break

        var inter = scratch[0]
        var t_hit = inter.tHit
        var ray_dir = rd.to_simd()

        # Volume free-flight
        if has_med and Int(cur_med_idx) >= 0:
            var med = sd.mediums[Int(cur_med_idx)]
            var ff = sample_homogeneous_free_flight(med, t_hit, pcg)
            if ff.collided:
                var sp = ro + rd*ff.t_free
                # BDPT vertex beta: exp(-sig_t×t)/pdf_free × alb_s = 1/sig_t × alb_s = alb_s (for sig_t=1)
                var v = _null_vertex()
                v.pos = sp
                v.beta = flux * ff.albedo
                v.alb = ff.albedo
                v.is_surface = Int32(0); v.is_delta = Int32(0)
                v.pdf_fwd = ff.sig_t * exp(-ff.sig_t * ff.t_free)
                v.med_idx = cur_med_idx
                n_verts += 1
                _bdpt_store_lvc_vertex[use_gpu](v, lvc, lvc_cap, lvc_counter)
                # Continuation: flux = prev × alb_s (same as stored vertex beta)
                flux *= ff.albedo
                var u1 = pcg.next_float(); var u2 = pcg.next_float()
                var cosT = Float32(2)*u1 - Float32(1)
                var sinT = sqrt(max(Float32(0), Float32(1)-cosT*cosT))
                var phi  = Float32(2)*PI*u2
                rd = Vec3f(sinT*cos(phi), sinT*sin(phi), cosT)
                ro = sp + rd*Float32(0.0002)
                continue
            else:
                flux *= ff.transmittance

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[mat_idx]
        var hit = ro + rd*t_hit

        if mat.type == MatKind.diffuse or mat.type == MatKind.coated_diffuse or mat.type == MatKind.diffuse_transmit:
            var gn = _geom_normal(inter, sd.meshes, sd.instances)
            if dot(gn, ray_dir) > Float32(0): gn = gn * Float32(-1)
            var v = _null_vertex()
            v.pos = hit
            v.normal = vec3f(gn)
            v.beta = flux
            v.alb = mat.albedo
            v.is_surface = Int32(1); v.is_delta = Int32(0)
            v.pdf_fwd = Float32(1); v.med_idx = cur_med_idx
            n_verts += 1
            _bdpt_store_lvc_vertex[use_gpu](v, lvc, lvc_cap, lvc_counter)
            # Scatter
            var u1 = pcg.next_float(); var u2 = pcg.next_float()
            rd = vec3f(_cosine_hemisphere_sample(gn, u1, u2))
            ro = hit + rd*Float32(0.0002)
            flux *= mat.albedo

        elif mat.type == MatKind.conductor:
            var gn_c: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si_c = Int(inter.primId.id1)
                var sph_c = sd.spheres[si_c]
                gn_c = sphere_outward_normal(hit, sph_c.center).to_simd()
            else:
                gn_c = _geom_normal(inter, sd.meshes, sd.instances)
            if dot(gn_c, ray_dir) > Float32(0): gn_c = gn_c * Float32(-1)
            var wo_c = (-rd).to_simd()
            var frm_c = Frame.from_z(Vec3f(gn_c[0], gn_c[1], gn_c[2]))
            var gc_c = GeomContext(
                normal=gn_c, geo_normal=gn_c, hit_point=hit.to_simd(), wo=wo_c,
                tangent=SIMD[DType.float32, 3](frm_c.x.x, frm_c.x.y, frm_c.x.z),
                bitangent=SIMD[DType.float32, 3](frm_c.y.x, frm_c.y.y, frm_c.y.z),
                alb=mat.albedo, pixel_uv=Float32(0),
            )
            var uc1 = pcg.next_float(); var uc2 = pcg.next_float()
            var bs_c = bxdf_sample_conductor(gc_c, mat, uc1, uc2)
            if bs_c.is_valid == Int8(0):
                break
            if not bxdf_is_delta(bs_c.flags):
                var v = _null_vertex()
                v.pos = hit
                v.normal = vec3f(gn_c)
                v.beta = flux
                v.alb = mat.albedo
                v.is_surface = Int32(1); v.is_delta = Int32(0); v.mat_kind = Int32(1)
                v.pdf_bwd = max(mat.roughU, mat.roughV)
                v.wo = vec3f(wo_c)
                v.pdf_fwd = Float32(1)
                v.med_idx = cur_med_idx
                n_verts += 1
                _bdpt_store_lvc_vertex[use_gpu](v, lvc, lvc_cap, lvc_counter)
            flux *= bs_c.f
            rd = vec3f(bs_c.wi)
            ro = hit + rd*Float32(0.0002)

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            var gn: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si = Int(inter.primId.id1)
                var sph = sd.spheres[si]
                gn = sphere_outward_normal(hit, sph.center).to_simd()
            else:
                gn = _geom_normal(inter, sd.meshes, sd.instances)
            var (new_dir, new_org) = _dielectric_bounce(ray_dir, hit.to_simd(), gn, mat.albedo.r, n_lbounces, pcg)
            n_lbounces += 1
            if has_med:
                var new_idx = _bdpt_medium_update(ray_dir, inter, mat, sd, hit)
                if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
            rd = vec3f(new_dir)
            ro = point3f(new_org)

        elif mat.type == MatKind.interface:
            if has_med:
                var new_idx = _bdpt_medium_update(ray_dir, inter, mat, sd, hit)
                if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
            ro = hit + rd*Float32(0.0002)

        else:
            break

    return

# ── BSDF/phase evaluation at a vertex ────────────────────────────────────────

@always_inline
def _eval_conductor_ggx(
    n:     SIMD[DType.float32, 3],
    wo:    SIMD[DType.float32, 3],   # toward this vertex's own predecessor
    wi:    SIMD[DType.float32, 3],   # toward the other connected vertex
    alpha: Float32,
    f0: RGB,
) -> SIMD[DType.float32, 3]:
    """Isotropic GGX (Trowbridge-Reitz) conductor f(wo,wi) × |cos(wi,n)|, for an
    arbitrary (not self-sampled) direction pair — the BDPT connection case that
    bxdf_sample_conductor (a self-sampled-direction-only throughput multiplier)
    can't serve. Schlick Fresnel at the half-vector, height-correlated Smith G2."""
    var cos_o = dot(wo, n)
    var cos_i = dot(wi, n)
    if cos_o <= Float32(0) or cos_i <= Float32(0):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var wh = wo + wi
    var whl = dot(wh, wh)
    if whl <= Float32(0):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    wh = wh * (Float32(1) / sqrt(whl))
    var cos_h = dot(wh, n)
    var cos_wo_h = dot(wo, wh)
    if cos_wo_h < Float32(0): cos_wo_h = -cos_wo_h
    var d = ggx_D(cos_h, alpha)
    var g = ggx_G2(cos_o, cos_i, alpha)
    var one_m = Float32(1) - cos_wo_h
    var one_m2 = one_m * one_m
    var schlick = one_m2 * one_m2 * one_m
    var fr = f0 + (RGB(Float32(1)) - f0) * schlick
    var k = d * g / (Float32(4) * cos_o * cos_i) * cos_i
    return SIMD[DType.float32, 3](k * fr.r, k * fr.g, k * fr.b)

@always_inline
def _eval_vertex(
    v:   BDPTVertex,
    dir_to_other:  SIMD[DType.float32, 3],   # direction from v toward the other connected vertex
) -> SIMD[DType.float32, 3]:
    """Evaluate BSDF (or phase) × cos at vertex v toward dir_to_other."""
    if v.is_delta != Int32(0):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    if v.is_surface == Int32(0):
        # Volume scatter: isotropic phase function 1/(4π), no cosine term
        return SIMD[DType.float32, 3](v.alb.r*INV_FOUR_PI, v.alb.g*INV_FOUR_PI, v.alb.b*INV_FOUR_PI)
    var vn = v.normal.to_simd()
    if v.mat_kind == Int32(1):
        var vwo = v.wo.to_simd()
        return _eval_conductor_ggx(vn, vwo, dir_to_other, v.pdf_bwd, v.alb)
    # Surface: Lambertian f = alb/π × |cos(wo,n)|
    var cos_o = dot(dir_to_other, vn)
    if cos_o < Float32(0): cos_o = -cos_o
    return SIMD[DType.float32, 3](v.alb.r*INV_PI*cos_o, v.alb.g*INV_PI*cos_o, v.alb.b*INV_PI*cos_o)

# ── Geometry term ─────────────────────────────────────────────────────────────

@always_inline
def _geom_term(
    a: BDPTVertex, b: BDPTVertex,
) -> Float32:
    """Geometry factor G(a,b) = |cos_a| × |cos_b| / dist²."""
    var d3 = b.pos - a.pos
    var dist2 = d3.length_sq()
    if dist2 < Float32(1e-8): return Float32(0)
    var d = sqrt(dist2)
    var dir = d3.to_simd() / d
    var cos_a: Float32
    if a.is_surface == Int32(1):
        cos_a = dot(dir, a.normal.to_simd())
        if cos_a < Float32(0): cos_a = -cos_a
    else:
        cos_a = Float32(1)   # volume: no cosine
    var cos_b: Float32
    if b.is_surface == Int32(1):
        cos_b = dot(dir, b.normal.to_simd())
        if cos_b < Float32(0): cos_b = -cos_b
    else:
        cos_b = Float32(1)
    return cos_a * cos_b / dist2

# ── Connect one camera vertex to one light vertex ─────────────────────────────

def _connect(
    cv: BDPTVertex,  # camera-subpath vertex
    lv: BDPTVertex,  # light-subpath vertex (including light point itself)
    sd: SceneDescriptor2_C,
    has_med: Bool,
    scratch: UnsafePointer[Intersection_C, MutAnyOrigin],
) -> SIMD[DType.float32, 3]:
    """Evaluate the contribution of connecting cv to lv via a shadow ray.
    Returns raw colour contribution — caller (the LVC uniform-subsample
    estimator) scales by cache_size/k_connections; this is no longer an
    exhaustive same-length-strategy MIS weight (see bdpt_render's docstring)."""
    if cv.is_delta != Int32(0) or lv.is_delta != Int32(0):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))

    var d3 = lv.pos - cv.pos
    var dist2 = d3.length_sq()
    if dist2 < Float32(1e-8):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var dist = sqrt(dist2)

    # Determine medium for the shadow segment.
    # Use camera vertex's medium (both should agree in a well-defined scene).
    var seg_med = cv.med_idx
    var Tr = _visible_transmittance(cv.pos, lv.pos, seg_med, sd, scratch)
    if Tr[0] < Float32(1e-7) and Tr[1] < Float32(1e-7) and Tr[2] < Float32(1e-7):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))

    var dir = d3.to_simd() / dist
    var neg_dir = -dir

    # BSDF at camera vertex (toward light)
    var f_cam = _eval_vertex(cv, dir)
    # BSDF at light vertex (toward camera)
    var f_lgt: SIMD[DType.float32, 3]
    if lv.is_light == Int32(1):
        # Light emission: f_lgt = Le (emission radiance, no cosine here).
        # _geom_term already computes cos_l at the light surface, so don't multiply again.
        var ln = lv.normal.to_simd()
        var cos_l = dot(neg_dir, ln)  # emission normal vs direction toward cv
        if cos_l <= Float32(0):
            return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
        f_lgt = SIMD[DType.float32, 3](lv.alb.r, lv.alb.g, lv.alb.b)
    else:
        f_lgt = _eval_vertex(lv, neg_dir)

    # Geometry term G = |cos_cv| × |cos_lv| / dist²
    var G = _geom_term(cv, lv)

    var beta = cv.beta * lv.beta
    var contrib = f_cam * f_lgt * SIMD[DType.float32, 3](G, G, G) * Tr
    contrib[0] *= beta.r
    contrib[1] *= beta.g
    contrib[2] *= beta.b
    return contrib

# ── Main BDPT render ──────────────────────────────────────────────────────────

def bdpt_render(
    psc:      UnsafePointer[ParsedScene_Mojo, MutAnyOrigin],
    sd:       SceneDescriptor2_C,
    n_spp:    Int,
    no_denoise: Bool,
    verbose:  Bool,
) -> Int32:
    """Bidirectional Path Tracing main loop (Light Vertex Cache architecture —
    see the module docstring above) — one output EXR per pixel.

    Each spp sample rebuilds a fresh shared Light Vertex Cache from
    `n_light_paths` independent light subpaths (not paired to any one
    pixel), then traces every pixel's camera subpath and connects each of
    its non-delta vertices to `_BDPT_K_CONNECTIONS` random cache entries.
    This replaced the old per-(pixel,sample) design that retraced an
    independent, unshared light subpath for every single pixel×sample pair
    — and, with it, the old exhaustive same-total-length strategy-count MIS
    (which doesn't compose once light vertices come from a cache shared
    across the whole frame): see `_bdpt_connect_to_cache`'s docstring for
    the uniform-subsample-and-scale estimator that replaces it."""
    var fw = Int(psc[0].film_w)
    var fh = Int(psc[0].film_h)
    var n_pix = fw * fh
    var iso_scale = psc[0].film_iso / Float32(100)
    var max_comp  = psc[0].film_max_comp

    print("BDPT: " + String(fw) + "x" + String(fh) + "  " + String(n_spp) + " spp")

    var has_med = Int(sd.mediumCount) > 0

    # Determine starting medium for light subpaths (same logic as SPPM)
    var default_emit_med = Int32(-1)
    if has_med and Int(sd.mediumIfaceCount) > 0:
        for mi in range(Int(sd.mediumIfaceCount)):
            var iface = sd.mediumInterfaces[mi]
            if Int(iface.outside_medium_idx) >= 0:
                default_emit_med = iface.outside_medium_idx
                break

    # Output buffer: one RGB per pixel
    var buf = alloc[RGB](n_pix)
    for i in range(n_pix):
        buf[i] = RGB(Float32(0))

    var r2c = psc[0].raster_to_camera
    var c2w = psc[0].camera_to_world
    var base_seed = psc[0].rng_seed

    # One light path per pixel, for grid-size symmetry with the camera pass
    # (matters once Phase (b) launches both as GPU kernels over the same
    # grid) — capacity mirrors sppm_emit_photons_gpu's max_photons: fixed,
    # cap-and-drop, no realloc.
    var n_light_paths = n_pix
    var lvc_cap = n_light_paths * _BDPT_MAX_VERTS
    var lvc = alloc[BDPTVertex](max(lvc_cap, 1))
    var lvc_counter = alloc[Int32](1)
    var scratch = alloc[Intersection_C](1)

    for si in range(n_spp):
        # ── Phase 1: fill the shared Light Vertex Cache for this sample ──────
        lvc_counter[0] = Int32(0)
        for lp_idx in range(n_light_paths):
            var lpcg = PCG32(base_seed ^ UInt64(lp_idx * 6364136223846793005 + 1442695040888963407),
                              UInt64(si * 2654435761 + 1))
            _bdpt_trace_light_path[False](sd, lpcg, has_med, default_emit_med, scratch, lvc, lvc_cap, lvc_counter)
        var lvc_count = min(Int(lvc_counter[0]), lvc_cap)
        var scale = _bdpt_lvc_connection_scale(lvc_count, n_light_paths)

        # ── Phase 2: trace each pixel's camera path and connect ──────────────
        for pix in range(n_pix):
            var px = pix % fw; var py = pix // fw
            var cpcg = PCG32(base_seed ^ UInt64(pix * 6364136223846793005 + 1442695040888963407),
                              UInt64(si * 2654435761 + 1))
            var contrib = _bdpt_trace_camera_and_connect[False](
                r2c, c2w, px, py, sd, cpcg, has_med, scratch, lvc, lvc_count, scale)
            buf[pix] += contrib

        if verbose:
            print("BDPT: sample " + String(si + 1) + "/" + String(n_spp))

    scratch.free(); lvc.free(); lvc_counter.free()

    # Clamp and write image
    var inv_spp = iso_scale / Float32(n_spp)
    var pixels = alloc[Float32](n_pix * 3)
    for i in range(n_pix):
        var c = buf[i] * inv_spp
        if max_comp > Float32(0):
            c.r = c.r if c.r < max_comp else max_comp
            c.g = c.g if c.g < max_comp else max_comp
            c.b = c.b if c.b < max_comp else max_comp
        pixels[i*3]   = c.r
        pixels[i*3+1] = c.g
        pixels[i*3+2] = c.b
    buf.free()

    _ = write_image(pixels, psc[0].film_w, psc[0].film_h, psc[0].film_filename, Int32(32), Int32(32))
    pixels.free()
    return Int32(0)

# ── GPU port ───────────────────────────────────────────────────────────────
# Everything below reuses _bdpt_trace_light_path[True]/_bdpt_trace_camera_
# and_connect[True] verbatim — the SAME functions bdpt_render (CPU) calls
# with [False] above. This is deliberately unlike gpu_sppm.mojo's GPU port
# of sppm.mojo (a full line-by-line reimplementation of every bounce loop,
# noted in its own header comment) — the whole point of doing BDPT's port
# this way first is to prove the zero-duplication comptime[use_gpu] pattern
# (already used for the wavefront path tracer's shade_nee_core) scales to a
# full renderer, as a template for eventually retrofitting SPPM the same way.

@always_inline
def _mk_sd_full(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    meshCount: Int64,
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    materialCount: Int64,
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int64,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    sphereCount: Int64,
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    curveCount: Int64,
    mediums: UnsafePointer[Medium_C, MutAnyOrigin],
    mediumCount: Int64,
    mediumInterfaces: UnsafePointer[MediumInterface_C, MutAnyOrigin],
    mediumIfaceCount: Int64,
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    blasCount: Int64,
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    instanceCount: Int64,
) -> SceneDescriptor2_C:
    """Builds a complete SceneDescriptor2_C from raw device pointers so the
    SAME `sd.field`-based traversal code _bdpt_trace_light_path/
    _bdpt_trace_camera_and_connect already use on CPU works unmodified on
    GPU (SceneDescriptor2_C is TrivialRegisterPassable — cheap to construct
    per-thread, no allocation). Only the fields BDPT's shared functions
    actually dereference are filled from real device buffers; textures,
    distant/point/infinite lights, grids, and the light sampler CDF are
    never touched by bdpt.mojo's code paths (confirmed by grep) so they
    stay dangling/zero-count, same convention traverse_bvh2_core's own
    optional instancing args already use."""
    return SceneDescriptor2_C(
        bvh2Nodes=bvh2Nodes, primIds=primIds,
        meshes=meshes, meshCount=meshCount,
        materials=materials, materialCount=materialCount,
        areaLights=areaLights, areaLightCount=areaLightCount,
        textures=UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        textureCount=Int64(0),
        distantLights=UnsafePointer[DistantLight_C, MutAnyOrigin].unsafe_dangling(),
        distantLightCount=Int64(0),
        pointLights=UnsafePointer[PointLight_C, MutAnyOrigin].unsafe_dangling(),
        pointLightCount=Int64(0),
        infiniteLights=UnsafePointer[InfiniteLight_C, MutAnyOrigin].unsafe_dangling(),
        infiniteLightCount=Int64(0),
        spheres=spheres, sphereCount=sphereCount,
        curves=curves, curveCount=curveCount,
        mediums=mediums, mediumCount=mediumCount,
        mediumInterfaces=mediumInterfaces, mediumIfaceCount=mediumIfaceCount,
        grids=UnsafePointer[Grid_C, MutAnyOrigin].unsafe_dangling(),
        gridCount=Int64(0),
        lightSampler=LightSampler_C(cdf=UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(), n=Int32(0), _pad=Int32(0)),
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, blasCount=blasCount,
        instances=instances, instanceCount=instanceCount,
    )

# ── Kernels ───────────────────────────────────────────────────────────────

def _bdpt_emit_light_paths_gpu(
    lvc: UnsafePointer[BDPTVertex, MutAnyOrigin],
    lvc_cap: Int,
    lvc_counter: UnsafePointer[Int32, MutAnyOrigin],
    inter_scratch: UnsafePointer[Intersection_C, MutAnyOrigin],
    n_light_paths: Int,
    default_emit_med: Int32,
    seed: UInt64,
    pass_idx: Int,
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int64,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    sphereCount: Int64,
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    curveCount: Int64,
    mediums: UnsafePointer[Medium_C, MutAnyOrigin],
    mediumCount: Int64,
    mediumInterfaces: UnsafePointer[MediumInterface_C, MutAnyOrigin],
    mediumIfaceCount: Int64,
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    blasCount: Int64,
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    instanceCount: Int64,
):
    """One thread per light path. Thin wrapper: build sd, seed this thread's
    own PCG32, call the SAME _bdpt_trace_light_path bdpt_render's CPU driver
    calls with [False]. `has_med` isn't a kernel parameter (`Bool` isn't a
    `DevicePassable` type `enqueue_function` accepts) -- derived here from
    `mediumCount`, which already is."""
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k >= n_light_paths:
        return
    var sd = _mk_sd_full(
        bvh2Nodes, primIds, meshes, Int64(0), materials, Int64(0),
        areaLights, areaLightCount, spheres, sphereCount, curves, curveCount,
        mediums, mediumCount, mediumInterfaces, mediumIfaceCount,
        blasNodesArr, blasPrimIdsArr, blasCount, instances, instanceCount,
    )
    var has_med = mediumCount > Int64(0)
    var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + k), UInt64(7))
    var scratch = inter_scratch + k
    _bdpt_trace_light_path[True](sd, pcg, has_med, default_emit_med, scratch, lvc, lvc_cap, lvc_counter)

def _bdpt_camera_connect_gpu(
    accum: UnsafePointer[Float32, MutAnyOrigin],
    n_pix: Int,
    fw: Int,
    r2c: UnsafePointer[Float32, MutAnyOrigin],
    c2w: UnsafePointer[Float32, MutAnyOrigin],
    inter_scratch: UnsafePointer[Intersection_C, MutAnyOrigin],
    lvc: UnsafePointer[BDPTVertex, MutAnyOrigin],
    lvc_count: Int,
    scale: Float32,
    seed: UInt64,
    pass_idx: Int,
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int64,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    sphereCount: Int64,
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    curveCount: Int64,
    mediums: UnsafePointer[Medium_C, MutAnyOrigin],
    mediumCount: Int64,
    mediumInterfaces: UnsafePointer[MediumInterface_C, MutAnyOrigin],
    mediumIfaceCount: Int64,
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    blasCount: Int64,
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    instanceCount: Int64,
):
    """One thread per pixel. Thin wrapper: build sd, seed this thread's own
    PCG32 (same seed formula bdpt_render's CPU driver uses, keyed by pixel
    index), call the SAME _bdpt_trace_camera_and_connect with [True], then
    accumulate straight into this pixel's own slot of `accum` — race-free
    since every thread owns exactly one pixel, the same reasoning the live
    (non-queued) shadow-ray path in gpu.mojo's shade_*_gpu kernels already
    relies on. `has_med` isn't a kernel parameter (see
    _bdpt_emit_light_paths_gpu's docstring) -- derived from mediumCount."""
    var pix = Int(block_idx.x * block_dim.x + thread_idx.x)
    if pix >= n_pix:
        return
    var sd = _mk_sd_full(
        bvh2Nodes, primIds, meshes, Int64(0), materials, Int64(0),
        areaLights, areaLightCount, spheres, sphereCount, curves, curveCount,
        mediums, mediumCount, mediumInterfaces, mediumIfaceCount,
        blasNodesArr, blasPrimIdsArr, blasCount, instances, instanceCount,
    )
    var has_med = mediumCount > Int64(0)
    var px = pix % fw
    var py = pix // fw
    var pcg = PCG32(seed ^ UInt64(pix * 6364136223846793005 + 1442695040888963407),
                     UInt64(pass_idx * 2654435761 + 1))
    var scratch = inter_scratch + pix
    var contrib = _bdpt_trace_camera_and_connect[True](
        r2c, c2w, px, py, sd, pcg, has_med, scratch, lvc, lvc_count, scale)
    accum[pix*3]   += contrib.r
    accum[pix*3+1] += contrib.g
    accum[pix*3+2] += contrib.b

# ── Host driver ───────────────────────────────────────────────────────────

def bdpt_render_gpu(
    handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin],
    psc:      UnsafePointer[ParsedScene_Mojo, MutAnyOrigin],
    sd:       SceneDescriptor2_C,
    n_spp:    Int,
    no_denoise: Bool,
    verbose:  Bool,
) -> Int32:
    """GPU-accelerated Light Vertex Cache BDPT — same algorithm as
    bdpt_render (CPU), same shared _bdpt_trace_light_path/
    _bdpt_trace_camera_and_connect functions, parallelized: one thread per
    light path for the light pass, one thread per pixel for the camera+
    connect pass. Mirrors sppm_render_gpu's (gpu_sppm.mojo) per-pass
    reset-counter -> emit -> sync+readback+clamp -> consume shape."""
    var fw = Int(psc[0].film_w)
    var fh = Int(psc[0].film_h)
    var n_pix = fw * fh
    var iso_scale = psc[0].film_iso / Float32(100)
    var max_comp  = psc[0].film_max_comp

    print("BDPT (GPU): " + String(fw) + "x" + String(fh) + "  " + String(n_spp) + " spp")

    var has_med = Int(sd.mediumCount) > 0
    var default_emit_med = Int32(-1)
    if has_med and Int(sd.mediumIfaceCount) > 0:
        for mi in range(Int(sd.mediumIfaceCount)):
            var iface = sd.mediumInterfaces[mi]
            if Int(iface.outside_medium_idx) >= 0:
                default_emit_med = iface.outside_medium_idx
                break

    var n_light_paths = n_pix
    var lvc_cap = n_light_paths * _BDPT_MAX_VERTS
    var base_seed = psc[0].rng_seed

    var ret = Int32(0)
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            comptime block_size = 256

            var lvc_buf     = handle[].ctx.enqueue_create_buffer[DType.uint8](max(lvc_cap, 1) * size_of[BDPTVertex]())
            var counter_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](size_of[Int32]())
            var inter_light_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](max(n_light_paths, 1) * size_of[Intersection_C]())
            var inter_cam_buf   = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * size_of[Intersection_C]())
            var accum_buf   = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * 3 * size_of[Float32]())
            with accum_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr().bitcast[Float32]()
                for i in range(n_pix * 3):
                    dst[i] = Float32(0)

            var r2c_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
            with r2c_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = psc[0].raster_to_camera.bitcast[UInt8]()
                for i in range(16 * size_of[Float32]()):
                    dst[i] = src[i]
            var c2w_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
            with c2w_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = psc[0].camera_to_world.bitcast[UInt8]()
                for i in range(16 * size_of[Float32]()):
                    dst[i] = src[i]

            var lvc_ptr     = lvc_buf.unsafe_ptr().bitcast[BDPTVertex]()
            var counter_ptr = counter_buf.unsafe_ptr().bitcast[Int32]()
            var inter_light_ptr = inter_light_buf.unsafe_ptr().bitcast[Intersection_C]()
            var inter_cam_ptr   = inter_cam_buf.unsafe_ptr().bitcast[Intersection_C]()
            var accum_ptr   = accum_buf.unsafe_ptr().bitcast[Float32]()
            var r2c_ptr = r2c_buf.unsafe_ptr().bitcast[Float32]()
            var c2w_ptr = c2w_buf.unsafe_ptr().bitcast[Float32]()

            var bvh2Nodes = handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node]()
            var primIds = handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C]()
            var meshes = handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C]()
            var curves = handle[].curves_buf.unsafe_ptr().bitcast[Curve_C]()
            var blasNodesArr = handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutAnyOrigin]]()
            var blasPrimIdsArr = handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutAnyOrigin]]()
            var instances = handle[].instances_buf.unsafe_ptr().bitcast[Instance_C]()
            var materials = handle[].materials_buf.unsafe_ptr().bitcast[Material_C]()
            var mediums = handle[].mediums_buf.unsafe_ptr().bitcast[Medium_C]()
            var mediumInterfaces = handle[].medium_ifaces_buf.unsafe_ptr().bitcast[MediumInterface_C]()
            var spheres = handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C]()
            var areaLights = handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C]()
            var n_mediums = Int64(handle[].n_mediums)
            var n_medium_ifaces = Int64(handle[].n_medium_ifaces)
            var n_spheres = Int64(handle[].n_spheres)
            var n_curves = Int64(handle[].n_curves)
            var n_area_lights = Int64(handle[].n_area_lights)
            var n_blas = Int64(handle[].n_blas)
            var n_instances = Int64(handle[].n_instances)

            var grid_light = ceildiv(max(n_light_paths, 1), block_size)
            var grid_pix = ceildiv(n_pix, block_size)

            for si in range(n_spp):
                handle[].ctx.enqueue_function[sppm_reset_i32_gpu](
                    counter_ptr, grid_dim=1, block_dim=1)

                var pass_seed = base_seed ^ UInt64(si * 2654435761 + 1)
                handle[].ctx.enqueue_function[_bdpt_emit_light_paths_gpu](
                    lvc_ptr, lvc_cap, counter_ptr, inter_light_ptr, n_light_paths,
                    default_emit_med, pass_seed, si,
                    bvh2Nodes, primIds, meshes, materials,
                    areaLights, n_area_lights, spheres, n_spheres, curves, n_curves,
                    mediums, n_mediums, mediumInterfaces, n_medium_ifaces,
                    blasNodesArr, blasPrimIdsArr, n_blas, instances, n_instances,
                    grid_dim=grid_light, block_dim=block_size)

                handle[].ctx.synchronize()
                var lvc_count_raw: Int32
                with counter_buf.map_to_host() as host_buf:
                    lvc_count_raw = host_buf.unsafe_ptr().bitcast[Int32]()[0]
                var lvc_count = min(Int(lvc_count_raw), lvc_cap)
                var scale = _bdpt_lvc_connection_scale(lvc_count, n_light_paths)

                handle[].ctx.enqueue_function[_bdpt_camera_connect_gpu](
                    accum_ptr, n_pix, Int(psc[0].film_w), r2c_ptr, c2w_ptr, inter_cam_ptr,
                    lvc_ptr, lvc_count, scale, base_seed, si,
                    bvh2Nodes, primIds, meshes, materials,
                    areaLights, n_area_lights, spheres, n_spheres, curves, n_curves,
                    mediums, n_mediums, mediumInterfaces, n_medium_ifaces,
                    blasNodesArr, blasPrimIdsArr, n_blas, instances, n_instances,
                    grid_dim=grid_pix, block_dim=block_size)

                if verbose:
                    print("BDPT (GPU): sample " + String(si + 1) + "/" + String(n_spp))

            handle[].ctx.synchronize()

            var pixels = alloc[Float32](n_pix * 3)
            with accum_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr().bitcast[Float32]()
                var inv_spp = iso_scale / Float32(n_spp)
                for i in range(n_pix):
                    var r = src[i*3]   * inv_spp
                    var g = src[i*3+1] * inv_spp
                    var b = src[i*3+2] * inv_spp
                    if max_comp > Float32(0):
                        r = r if r < max_comp else max_comp
                        g = g if g < max_comp else max_comp
                        b = b if b < max_comp else max_comp
                    pixels[i*3] = r; pixels[i*3+1] = g; pixels[i*3+2] = b

            _ = write_image(pixels, psc[0].film_w, psc[0].film_h, psc[0].film_filename, Int32(32), Int32(32))
            pixels.free()
        except e:
            print("BDPT GPU render failed: " + String(e))
            ret = Int32(-1)
    else:
        print("BDPT GPU: no accelerator")
        ret = Int32(-1)
    return ret
