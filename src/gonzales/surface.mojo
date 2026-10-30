# A ray's hit on a scattering surface as every integrator sees it, and the BSDF there (pbrt's SurfaceInteraction).

from std.math import abs, min
from .geometry import offset_eps, face_toward, RGB, Point3f, Vec3f, vec3f, dot
from .materials import Material, MatKind
from .primitives import Intersection
from .bvh import SceneView, curve_hit_frame, curve_offset_eps, curve_tangent, fibre_cos
from .geom_normal import _geom_normal, _shading_normal_at
from .shading import _tex_lookup, _get_tri_verts, apply_surface_maps_at_hit, uv_footprint_at_hit
from .bxdf import BSDF
from .spectrum import SampledWavelengths, SpectralSample, rgb_bands_to_spectral_sample


@fieldwise_init
struct SurfaceHit(TrivialRegisterPassable):
    """Where a ray met a lobe material: its normals, texture colour and spawn offset."""
    var pos: Point3f
    var wo: Vec3f            # toward where the ray came from
    var gn: Vec3f            # shading normal, maps applied; faces wo unless the lobe needs it outward
    var gn_geo: Vec3f        # the same before maps: what a stored vertex keeps as its geometric normal
    var cos_in: Float32      # |cos| between wo and gn_geo
    var spawn_eps: Float32
    var on_curve: Bool
    var alb: RGB             # the material's colour here, textures applied
    var curve_idx: Int32
    var curve_h: Float32
    var curve_v: Float32

    @always_inline
    def bsdf(self, mat: Material, mat_idx: Int, adjoint: Bool = False) -> BSDF:
        """The material's BSDF here; `adjoint` on a light path."""
        return BSDF(mat.lobe(), True, mat.is_mirror(), self.gn, self.wo, self.alb, Int32(mat_idx), mat.lobe_param(),
                    Float32(1), self.curve_idx, self.curve_h, self.curve_v, False, adjoint)

    @always_inline
    def spawn(self, wi: Vec3f) -> Point3f:
        """A new ray's origin toward `wi`: off a curve along its normal, else a step along wi."""
        if self.on_curve:
            return self.pos + vec3f(self.gn_geo) * (self.spawn_eps if dot(wi, self.gn_geo) >= Float32(0) else -self.spawn_eps)
        return self.pos + vec3f(wi) * Float32(0.0002)


@always_inline
def sss_entry_tint[use_gpu: Bool](ref sd: SceneView, mat: Material, inter: Intersection, wl: SampledWavelengths) -> SpectralSample:
    """A textured subsurface walk's colour against the mean it was built from, capped so a bright texel stays no spike."""
    if mat.tex_idx < Int32(0):
        return SpectralSample(Float32(1))
    var (mesh, v0, v1, v2, ok) = _get_tri_verts(inter, sd.meshes)
    if not ok:
        return SpectralSample(Float32(1))
    var a = _tex_lookup[use_gpu](mat, inter, v0, v1, v2, mesh, sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
    var m = mat.sss_mean_refl
    return rgb_bands_to_spectral_sample(min(a.r / m.r, Float32(4)) if m.r > Float32(1e-4) else Float32(1),
                                        min(a.g / m.g, Float32(4)) if m.g > Float32(1e-4) else Float32(1),
                                        min(a.b / m.b, Float32(4)) if m.b > Float32(1e-4) else Float32(1), wl)


@always_inline
def surface_hit[use_gpu: Bool](ref sd: SceneView, inter: Intersection, mat: Material, hit: Point3f, ray_dir: Vec3f,
                               cone_w: Float32) -> SurfaceHit:
    """The hit at a lobe material. `cone_w` is the camera ray's footprint width for texture filtering, -1 for none."""
    # Geometry, not material: a curve hit has its own normal and spawn offset, and no texture or bump map.
    var on_curve = inter.primId.type == Int8(5)
    var spawn_eps = offset_eps(hit.x, hit.y, hit.z)
    var gn: Vec3f
    if on_curve:
        var cf = curve_hit_frame(sd.curves, Int(inter.primId.id1), inter.v, inter.u, (-ray_dir).to_simd())
        gn = cf[3]
        spawn_eps = curve_offset_eps(cf[4])
    elif mat.type == MatKind.dielectric:
        gn = _shading_normal_at(inter, sd.meshes, sd.instances, sd.spheres, hit)
    else:
        gn = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
    # A rough dielectric needs the outward normal: inside vs outside is which side of it wo lies on.
    var outward = mat.type == MatKind.dielectric
    if dot(gn, ray_dir) > Float32(0) and not outward: gn = gn * Float32(-1)
    var cos_in = fibre_cos(curve_tangent(sd.curves, Int(inter.primId.id1), inter.v), -ray_dir) if on_curve \
        else abs(dot(-ray_dir, gn))
    var alb = mat.albedo
    var (tex_mesh, tv0, tv1, tv2, tex_ok) = _get_tri_verts(inter, sd.meshes)
    if tex_ok and not on_curve:
        alb = _tex_lookup[use_gpu](mat, inter, tv0, tv1, tv2, tex_mesh, sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount),
            uv_footprint_at_hit(inter, sd.meshes, sd.instances, hit.to_simd(), ray_dir, cone_w, sd.camFp).width)
    var gn_geo = gn
    if not on_curve:
        gn = apply_surface_maps_at_hit[use_gpu](mat, inter, sd.meshes, sd.instances,
            gn, gn, hit.to_simd(), ray_dir, cone_w, sd.camFp,
            sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
    if not outward:
        gn = face_toward(gn, -ray_dir)   # pbrt two-sided reflection
    var curve_idx = Int32(inter.primId.id1) if on_curve else Int32(-1)
    return SurfaceHit(hit, -ray_dir, gn, gn_geo, cos_in, spawn_eps, on_curve, alb, curve_idx,
                      inter.u if on_curve else Float32(0), inter.v if on_curve else Float32(0))
