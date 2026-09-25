# Unit tests for pure MIS/vertex-evaluation math extracted from bdpt.mojo:
# solid-angle-to-area PDF conversion, the geometry term G(a,b), per-vertex
# BSDF/phase evaluation (_eval_vertex_spectral), and the GGX conductor
# connection BRDF (_eval_conductor_ggx_spectral) used when connecting a camera
# vertex to a light vertex
# via a shadow ray. (power_heuristic itself is tested in test_sampling.mojo;
# ggx_D/ggx_G2 primitives are tested in test_bxdf.mojo -- this file tests the
# combination logic bdpt.mojo layers on top of them, not those primitives
# themselves.) Also covers _bdpt_connect_to_cache's exhaustive-sum-over-the-
# paired-light-path structure (VCM Stage 2b's standard-Veach-pairing
# rewrite), using the shared TriangleSceneFixture for a real (if
# geometrically irrelevant) BVH to satisfy _connect's visibility check.

from std.math import abs
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, TestSuite
from gonzales.geometry import RGB, Point3f, Vec3f, INV_FOUR_PI, INV_PI
from gonzales.materials import LobeKind, MeasuredBRDF_C
from gonzales.render_state import GpuTexture_C, NormalSlopeMap_C
from gonzales.primitives import Intersection, Sphere, PrimId, Instance
from gonzales.media import Medium_C, MediumInterface_C, Grid_C, NvdbGrid_C
from gonzales.lights import LightSampler, AreaLight, DistantLight, PointLight, InfiniteLight
from gonzales.bvh import SceneDescriptor2_C, BVH2Node
from gonzales.bxdf import ggx_D, ggx_G2, ggx_albedo_avg, ggx_ms_shape, ggx_ms_tint
from gonzales.bdpt import (
    BDPTVertex, _pdf_solid_to_area, _eval_vertex_spectral,
    _eval_conductor_ggx_spectral, _bdpt_connect_to_cache,
)
from gonzales.spectrum import SampledWavelengths, SpectralSample, null_spectral_handle
from _scene_fixture import make_triangle_scene

comptime EPS: Float32 = 1e-3

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

def _simd_close(a: Vec3f, b: Vec3f) -> Bool:
    return _close(a[0], b[0]) and _close(a[1], b[1]) and _close(a[2], b[2])

# BDPT/VCM transport is spectral, so the vertex evaluators return a
# SpectralSample. These tests use a NULL spectral handle, under which
# spectrum.mojo's conversions carry plain R/G/B on lanes v0/v1/v2 (see
# rgb_to_spectral_sample's table-less fallback) -- so every closed form
# asserted below is still exactly the RGB one, read off the first three lanes.
comptime NULL_WL = SampledWavelengths(Float32(0.0), Float32(0.0), Float32(0.0),
                                      Float32(0.0), Float32(0.0))

def _spec_close(a: SpectralSample, b: Vec3f) -> Bool:
    return _close(a.v0, b[0]) and _close(a.v1, b[1]) and _close(a.v2, b[2])

def _spec_close_spec(a: SpectralSample, b: SpectralSample) -> Bool:
    return _close(a.v0, b.v0) and _close(a.v1, b.v1) and _close(a.v2, b.v2)

# Hero wavelengths chosen so lanes 0/1/2 fall in the RED/GREEN/BLUE bands
# respectively, under the sRGB-primary crossovers band-picking uses (blue
# below 490nm, green to 580nm, red above). That makes a band-picked RGB
# coefficient land as (r, g, b) on lanes 0/1/2 -- so a test can state its
# expectation in plain RGB while genuinely exercising the band-pick path.
# NULL_WL (all zeros) cannot: every lane would read as blue.
comptime BAND_WL = SampledWavelengths(Float32(600.0), Float32(550.0),
                                      Float32(450.0), Float32(700.0),
                                      Float32(1.0))

@always_inline
def _eval_v_bands(v: BDPTVertex, dir: Vec3f, sd: SceneDescriptor2_C) -> SpectralSample:
    var h = null_spectral_handle()
    return _eval_vertex_spectral(v, dir, sd, h.coeffs, h.res, h.cie_x, h.cie_y,
                                 h.cie_z, h.d65, BAND_WL)

@always_inline
def _eval_v(v: BDPTVertex, dir: Vec3f, sd: SceneDescriptor2_C) -> SpectralSample:
    var h = null_spectral_handle()
    return _eval_vertex_spectral(v, dir, sd, h.coeffs, h.res, h.cie_x, h.cie_y,
                                 h.cie_z, h.d65, NULL_WL)

@always_inline
def _eval_cond(n: Vec3f, wo: Vec3f, wi: Vec3f, alpha: Float32, f0: RGB) -> SpectralSample:
    var h = null_spectral_handle()
    return _eval_conductor_ggx_spectral(n, wo, wi, alpha, f0, h.coeffs, h.res,
                                        h.cie_x, h.cie_y, h.cie_z, h.d65, NULL_WL)

# ── shared fixture ────────────────────────────────────────────────────────────

def _make_vertex(pos: Point3f, normal: Vec3f, is_surface: Int32) -> BDPTVertex:
    return BDPTVertex(
        pos=pos, normal=normal, shading_normal=normal, beta=SpectralSample(Float32(0)), alb=RGB(Float32(0)),
        pdf_fwd=Float32(0), pdf_bwd=Float32(0),
        dVCM=Float32(0), dVC=Float32(0), dVM=Float32(0),
        is_surface=is_surface, is_delta=Int32(0), is_light=Int32(0),
        med_idx=Int32(-1), mat_kind=LobeKind.lambertian, wo=Vec3f(Float32(0)),
        mat_idx=Int32(-1), hair_curve_idx=Int32(-1), hair_h=Float32(0), hair_v=Float32(0),
        wavelengths=SampledWavelengths(Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)),
    )

def _dummy_sd() -> SceneDescriptor2_C:
    """A minimal (geometrically irrelevant) SceneDescriptor2_C for
    _eval_vertex_spectral calls that never exercise mat_kind=2 (hair) -- that's the
    only branch that dereferences sd.materials/sd.curves, so any valid
    SceneDescriptor2_C works for the Lambertian/conductor/volume tests below."""
    var fixture = make_triangle_scene([
        Point3f(100.0, 100.0, 100.0), Point3f(101.0, 100.0, 100.0), Point3f(100.0, 101.0, 100.0),
    ])
    return SceneDescriptor2_C(
        fixture.bvh_nodes, fixture.prim_ids, fixture.meshes, Int64(1),
        fixture.materials, Int64(1),
        Pointer[AreaLight, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[DistantLight, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[PointLight, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[InfiniteLight, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[Sphere, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        fixture.curves, Int64(0),
        Pointer[Medium_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[MediumInterface_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[Grid_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[NvdbGrid_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        LightSampler(Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(), Int32(0), Int32(0)),
        Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Pointer[PrimId, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        Int64(0),
        Pointer[Instance, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        null_spectral_handle(),
        Pointer[GpuTexture_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[NormalSlopeMap_C, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(), Float32(0), Float32(1), Int32(9),
        Float32(0), Float32(0), Float32(0), Float32(0), Float32(0),
    )

# ── _pdf_solid_to_area ────────────────────────────────────────────────────────
# p_A = p_omega * |cos_theta| / dist^2 (solid-angle -> area PDF conversion,
# used to put camera- and light-subpath vertices on a common measure for MIS).

def test_pdf_solid_to_area_matches_closed_form() raises:
    var result = _pdf_solid_to_area(Float32(2.5), Float32(0.6), Float32(4.0))
    assert_true(_close(result, Float32(2.5) * Float32(0.6) / Float32(4.0)))

def test_pdf_solid_to_area_uses_absolute_value_of_cosine() raises:
    """A negative cos_theta (vertex normal facing away from the connection
    direction) must still yield a positive area PDF -- the function takes
    |cos_theta|, not the signed cosine."""
    var result = _pdf_solid_to_area(Float32(2.5), Float32(-0.6), Float32(4.0))
    assert_true(_close(result, Float32(2.5) * Float32(0.6) / Float32(4.0)))

def test_pdf_solid_to_area_degenerate_distance_returns_zero() raises:
    var result = _pdf_solid_to_area(Float32(5.0), Float32(0.5), Float32(1e-10))
    assert_true(_close(result, Float32(0.0)))

# ── _eval_vertex_spectral ──────────────────────────────────────────────────────────────

def test_eval_vertex_delta_vertex_is_always_zero() raises:
    """Specular (mirror conductor / dielectric) vertices cannot be connected
    via a shadow ray -- _eval_vertex_spectral must return 0 regardless of mat_kind or
    albedo."""
    var v = BDPTVertex(
        pos=Point3f(Float32(0)), normal=Vec3f(0.0, 0.0, 1.0), shading_normal=Vec3f(0.0, 0.0, 1.0),
        beta=SpectralSample(Float32(0)), alb=RGB(Float32(1.0), Float32(1.0), Float32(1.0)),
        pdf_fwd=Float32(0), pdf_bwd=Float32(0),
        dVCM=Float32(0), dVC=Float32(0), dVM=Float32(0),
        is_surface=Int32(1), is_delta=Int32(1), is_light=Int32(0),
        med_idx=Int32(-1), mat_kind=LobeKind.lambertian, wo=Vec3f(Float32(0)),
        mat_idx=Int32(-1), hair_curve_idx=Int32(-1), hair_h=Float32(0), hair_v=Float32(0),
        wavelengths=SampledWavelengths(Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)),
    )
    var result = _eval_v(v, Vec3f(0.0, 0.0, 1.0), _dummy_sd())
    assert_true(_spec_close(result, Vec3f(0.0, 0.0, 0.0)))

def test_eval_vertex_volume_scatter_matches_isotropic_phase_function() raises:
    """A volume-scatter vertex (is_surface=0) uses the isotropic phase
    function alb/(4*pi) -- no cosine term at all, unlike the surface case."""
    var v = BDPTVertex(
        pos=Point3f(Float32(0)), normal=Vec3f(0.0, 1.0, 0.0), shading_normal=Vec3f(0.0, 1.0, 0.0),
        beta=SpectralSample(Float32(0)), alb=RGB(Float32(0.3), Float32(0.4), Float32(0.5)),
        pdf_fwd=Float32(0), pdf_bwd=Float32(0),
        dVCM=Float32(0), dVC=Float32(0), dVM=Float32(0),
        is_surface=Int32(0), is_delta=Int32(0), is_light=Int32(0),
        med_idx=Int32(-1), mat_kind=LobeKind.lambertian, wo=Vec3f(Float32(0)),
        mat_idx=Int32(-1), hair_curve_idx=Int32(-1), hair_h=Float32(0), hair_v=Float32(0),
        wavelengths=SampledWavelengths(Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)),
    )
    var dir = Vec3f(0.267261, 0.534522, 0.801784)  # arbitrary; ignored by volume path
    # BAND_WL, not NULL_WL: a volume vertex's alb is the single-scattering
    # albedo sigma_s/sigma_t, a per-channel COEFFICIENT, so it is band-picked
    # rather than pushed through the Jakob-Hanika reflectance upsampler (see
    # docs/02_spectra_and_color.md, "A coefficient is not a color"). With
    # all-zero wavelengths every lane would band-pick as blue.
    var result = _eval_v_bands(v, dir, _dummy_sd())
    assert_true(_spec_close(result, Vec3f(
        Float32(0.3) * INV_FOUR_PI, Float32(0.4) * INV_FOUR_PI, Float32(0.5) * INV_FOUR_PI)))

def test_eval_vertex_lambertian_matches_closed_form() raises:
    """Surface Lambertian: f = (alb/pi) * |cos(dir_to_other, normal)|."""
    var v = BDPTVertex(
        pos=Point3f(Float32(0)), normal=Vec3f(0.0, 0.0, 1.0), shading_normal=Vec3f(0.0, 0.0, 1.0),
        beta=SpectralSample(Float32(0)), alb=RGB(Float32(0.2), Float32(0.4), Float32(0.6)),
        pdf_fwd=Float32(0), pdf_bwd=Float32(0),
        dVCM=Float32(0), dVC=Float32(0), dVM=Float32(0),
        is_surface=Int32(1), is_delta=Int32(0), is_light=Int32(0),
        med_idx=Int32(-1), mat_kind=LobeKind.lambertian, wo=Vec3f(Float32(0)),
        mat_idx=Int32(-1), hair_curve_idx=Int32(-1), hair_h=Float32(0), hair_v=Float32(0),
        wavelengths=SampledWavelengths(Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)),
    )
    var dir = Vec3f(0.7071068, 0.0, 0.7071068)  # 45 degrees off the normal
    var result = _eval_v(v, dir, _dummy_sd())
    var cos_o = Float32(0.7071068)
    assert_true(_spec_close(result, Vec3f(
        Float32(0.2) * INV_PI * cos_o, Float32(0.4) * INV_PI * cos_o, Float32(0.6) * INV_PI * cos_o)))

def test_eval_vertex_conductor_dispatches_to_eval_conductor_ggx_with_own_fields() raises:
    """Mat_kind=1 vertices must route to the GGX conductor eval using the
    vertex's own normal/wo/pdf_bwd(=alpha)/alb(=F0) fields -- verified by
    comparing against a direct call to _eval_conductor_ggx_spectral with those same
    values, which pins down the field-to-argument wiring (not just the GGX
    math itself, which is covered by the dedicated test below)."""
    var n = Vec3f(0.0, 0.0, 1.0)
    var wo = Vec3f(0.0, 0.0, 1.0)
    var wi = Vec3f(0.0, 0.0, 1.0)
    var alpha = Float32(0.2)
    var f0 = RGB(Float32(0.5), Float32(0.6), Float32(0.7))
    var v = BDPTVertex(
        pos=Point3f(Float32(0)), normal=Vec3f(0.0, 0.0, 1.0), shading_normal=Vec3f(0.0, 0.0, 1.0),
        beta=SpectralSample(Float32(0)), alb=f0,
        pdf_fwd=Float32(0), pdf_bwd=alpha,
        dVCM=Float32(0), dVC=Float32(0), dVM=Float32(0),
        is_surface=Int32(1), is_delta=Int32(0), is_light=Int32(0),
        med_idx=Int32(-1), mat_kind=LobeKind.ggx, wo=Vec3f(0.0, 0.0, 1.0),
        mat_idx=Int32(-1), hair_curve_idx=Int32(-1), hair_h=Float32(0), hair_v=Float32(0),
        wavelengths=SampledWavelengths(Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)),
    )
    var expected = _eval_cond(n, wo, wi, alpha, f0)
    var result = _eval_v(v, wi, _dummy_sd())
    assert_true(_spec_close_spec(result, expected))

# ── _eval_conductor_ggx_spectral ───────────────────────────────────────────────────────

def test_eval_conductor_ggx_grazing_wo_returns_zero() raises:
    var n  = Vec3f(0.0, 0.0, 1.0)
    var wo = Vec3f(1.0, 0.0, 0.0)  # perpendicular to n -> cos_o = 0
    var wi = Vec3f(0.0, 0.0, 1.0)
    var result = _eval_cond(n, wo, wi, Float32(0.2), RGB(Float32(1.0)))
    assert_true(_spec_close(result, Vec3f(0.0, 0.0, 0.0)))

def test_eval_conductor_ggx_grazing_wi_returns_zero() raises:
    var n  = Vec3f(0.0, 0.0, 1.0)
    var wo = Vec3f(0.0, 0.0, 1.0)
    var wi = Vec3f(1.0, 0.0, 0.0)  # perpendicular to n -> cos_i = 0
    var result = _eval_cond(n, wo, wi, Float32(0.2), RGB(Float32(1.0)))
    assert_true(_spec_close(result, Vec3f(0.0, 0.0, 0.0)))

def test_eval_conductor_ggx_normal_incidence_matches_closed_form() raises:
    """At wo=wi=n (normal incidence, half-vector = n exactly), cos_wo_h=1 so
    the Schlick term (1-cos_wo_h)^5 vanishes and Fresnel reduces to exactly
    F0. The single-scattering factor is then D(1,alpha)*G2(1,1,alpha)/(4*1*1)
    -- checked by calling the same ggx_D/ggx_G2 primitives (already
    independently unit-tested in test_bxdf.mojo) and combining them exactly
    as _eval_conductor_ggx_spectral's k/fr formula does.

    Since Kulla-Conty landed there is a SECOND term. It is built here from
    the same ggx_ms_shape/ggx_ms_tint primitives rather than pasted as a
    number, so this stays a closed-form check: if the compensation lobe's
    shape or its Turquin colour changes, this test follows the formula
    instead of going red for a value nobody can re-derive.

    _eval_cond passes null_spectral_handle(), whose upsampler is the RGB
    passthrough (v0,v1,v2 = r,g,b), so comparing the first three spectral
    components against an RGB triple is exact rather than approximate."""
    var n = Vec3f(0.0, 0.0, 1.0)
    var alpha = Float32(0.2)
    var f0 = RGB(Float32(0.5), Float32(0.6), Float32(0.7))
    var d = ggx_D(Float32(1.0), alpha)
    var g = ggx_G2(Float32(1.0), Float32(1.0), alpha)
    var k = d * g / Float32(4.0)
    # cos_i = 1 here, so ggx_ms_shape's f_ms and the evaluator's f_ms*cos_i
    # coincide -- no cosine to carry.
    var eavg = ggx_albedo_avg(alpha)
    var shape = ggx_ms_shape(Float32(1.0), Float32(1.0), alpha)
    var expected = Vec3f(k * f0.r + shape * ggx_ms_tint(f0.r, eavg),
                         k * f0.g + shape * ggx_ms_tint(f0.g, eavg),
                         k * f0.b + shape * ggx_ms_tint(f0.b, eavg))
    var result = _eval_cond(n, n, n, alpha, f0)
    assert_true(_spec_close(result, expected))

# ── _bdpt_connect_to_cache (VCM Stage 2b: standard Veach pairing) ───────────
# Since Stage 2b.1, connection is an EXHAUSTIVE sum over the camera vertex's
# own deterministically-paired light path (`lvc[lp_idx*_BDPT_MAX_VERTS +
# local]` for `local in range(path_len)`), not a K-random-draw-from-a-
# shared-pool-then-rescale estimator -- no RNG, no scale factor. This test
# pins that "sum path_len connections, no rescale" structure: with a
# single-vertex path (path_len=1) the expected total is exactly
# _connect(cv, the_one_light_vertex, ..., mis_vm_weight_factor=0), computed
# here independently via _connect's own constituent pieces (_eval_vertex,
# the 1/d^2 falloff) rather than by re-deriving _connect's formula. mis_vm_weight_
# factor=0 keeps _connect's own real-MIS weight (both endpoints are
# diffuse/light-source here, so it would otherwise activate) at exactly 1,
# since cv/lv's dVCM/dVC/dVM are all 0 here -- see _connect's docstring.

def test_bdpt_connect_to_cache_sums_one_paired_light_path() raises:
    # A real (if geometrically irrelevant) BVH so _connect's internal
    # visibility check has something valid to traverse -- placed far from
    # the cv/lv segment below so the shadow ray is guaranteed unoccluded.
    var fixture = make_triangle_scene([
        Point3f(100.0, 100.0, 100.0), Point3f(101.0, 100.0, 100.0), Point3f(100.0, 101.0, 100.0),
    ])
    var sd = SceneDescriptor2_C(
        fixture.bvh_nodes, fixture.prim_ids, fixture.meshes, Int64(1),
        fixture.materials, Int64(1),
        Pointer[AreaLight, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[DistantLight, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[PointLight, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[InfiniteLight, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[Sphere, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        fixture.curves, Int64(0),
        Pointer[Medium_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[MediumInterface_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[Grid_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[NvdbGrid_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        LightSampler(Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(), Int32(0), Int32(0)),
        Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Pointer[PrimId, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        Int64(0),
        Pointer[Instance, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        null_spectral_handle(),
        Pointer[GpuTexture_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[NormalSlopeMap_C, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(), Float32(0), Float32(1), Int32(9),
        Float32(0), Float32(0), Float32(0), Float32(0), Float32(0),
    )

    var cv = BDPTVertex(
        pos=Point3f(5.0, 5.0, 10.0), normal=Vec3f(0.0, 0.0, 1.0), shading_normal=Vec3f(0.0, 0.0, 1.0),
        beta=SpectralSample(Float32(3.0)), alb=RGB(Float32(0.5)),
        pdf_fwd=Float32(0), pdf_bwd=Float32(0),
        dVCM=Float32(0), dVC=Float32(0), dVM=Float32(0),
        is_surface=Int32(1), is_delta=Int32(0), is_light=Int32(0),
        med_idx=Int32(-1), mat_kind=LobeKind.lambertian, wo=Vec3f(Float32(0)),
        mat_idx=Int32(-1), hair_curve_idx=Int32(-1), hair_h=Float32(0), hair_v=Float32(0),
        wavelengths=SampledWavelengths(Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)),
    )
    var lv = BDPTVertex(
        pos=Point3f(5.0, 5.0, 20.0), normal=Vec3f(0.0, 0.0, -1.0), shading_normal=Vec3f(0.0, 0.0, -1.0),
        beta=SpectralSample(Float32(2.0)), alb=RGB(Float32(1.0)),
        pdf_fwd=Float32(0.25), pdf_bwd=Float32(0),   # a real light origin: 1 / (area * n_lights)
        dVCM=Float32(0), dVC=Float32(0), dVM=Float32(0),
        is_surface=Int32(1), is_delta=Int32(0), is_light=Int32(1),
        med_idx=Int32(-1), mat_kind=LobeKind.lambertian, wo=Vec3f(Float32(0)),
        mat_idx=Int32(-1), hair_curve_idx=Int32(-1), hair_h=Float32(0), hair_v=Float32(0),
        wavelengths=SampledWavelengths(Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)),
    )
    var lvc = unsafe_alloc[BDPTVertex](1)
    lvc[unsafe_offset=0] = lv
    var scratch = unsafe_alloc[Intersection](1)

    var result = _bdpt_connect_to_cache(cv, sd, False, scratch, lvc, 0, 1, Float32(0), 1)

    # Independently compute the single-connection value _connect would
    # produce, then verify the exhaustive-sum-over-the-path wrapper against
    # it directly.
    var dir_to_light = Vec3f(0.0, 0.0, 1.0)
    var f_cam = _eval_v(cv, dir_to_light, sd)  # Lambertian: alb/pi * cos
    var f_lgt = Vec3f(lv.alb.r, lv.alb.g, lv.alb.b)  # is_light: Le, no cosine
    # _connect = f*cos at each endpoint (inside f_cam / the light's Le here,
    # both cosines 1) times the bare 1/d^2 between them -- no separate G term.
    var g = Float32(1.0) / Float32(100.0)
    var beta_prod = Float32(3.0) * Float32(2.0)
    # s=1 MIS (SmallVCM DirectIllumination): the only competitor with these
    # zero camera carries is the camera BSDF hitting the light, so
    # w = 1 / (1 + pdfA_bsdf / p_A), pdfA_bsdf = (cos/pi) * cos_l / d^2.
    var w_s1 = Float32(1.0) / (Float32(1.0) + (Float32(1.0) / Float32(3.14159265) / Float32(100.0)) / Float32(0.25))
    var expected = f_cam.v0 * f_lgt[0] * g * beta_prod * w_s1  # unoccluded, Tr=1; all channels equal here

    assert_true(_close(result.v0, expected))
    assert_true(_close(result.v1, expected))
    assert_true(_close(result.v2, expected))
    # `sd` holds raw pointers borrowed from `fixture` (bvh_nodes/prim_ids/
    # meshes/materials/curves) with no other owner -- `fixture` has no
    # syntactic use after constructing `sd` above, so Mojo's ASAP
    # destruction would otherwise free it (and everything `sd` points into)
    # before `_bdpt_connect_to_cache`'s internal BVH traversal ever runs.
    # Keep it alive through the whole test, not a style nicety.
    _ = fixture^

    lvc.unsafe_free(); scratch.unsafe_free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
