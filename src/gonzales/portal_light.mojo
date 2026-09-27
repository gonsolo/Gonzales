# pbrt-v4 portal lights: `LightSource "infinite" "point3 portal" [p0 p1 p2 p3]`
# restricts an environment light to the solid angle actually visible through a
# world-space quadrilateral -- how pbrt lets a textured sky light an interior
# through its real window openings without modelling glass geometry (a
# "-no-windowglass" scene). Found root-causing pbrt-v4-scenes/watercolor
# reading 2.6-3.1x too bright: gonzales silently dropped "portal" entirely
# (zero references anywhere), so both of that scene's window-portaled copies
# of the same envmap illuminated the WHOLE sphere instead of just their own
# window, uniformly over-brightening the room -- see
# project_portal_light_unsupported memory for the full diagnosis.
#
# pbrt-v4's own PortalImageInfiniteLight (lights.h/.cpp) reparametrizes the
# source image into a whole SEPARATE portal-frame tangent-plane mapping with
# its own windowed-piecewise-constant-2D importance sampler, built once at
# load time -- real, but substantial, infrastructure. gonzales instead only
# gates the ESCAPE side (a camera/BSDF ray that leaves the scene entirely,
# shading.mojo's miss handler): Le is paid out only when the ray's direction
# actually crosses the portal quad, exactly like a real window opening would
# let sky through and a wall would not. NEE toward this light is
# DELIBERATELY NOT portal-restricted -- it still draws from the ordinary
# whole-sphere env-map CDF and relies on the ordinary shadow-ray occlusion
# test (against the scene's real wall geometry, which has an actual hole cut
# for the window) to zero out directions that don't escape through it. Both
# paths were tried: uniform-area sampling of the portal quad for NEE too
# measured SELF-CONSISTENT in isolation (its implied solid angle matched an
# independent direction-sampling estimate to 4 significant figures) but
# under-lit watercolor end-to-end by ~2x at both 64 and 512 spp -- a flat
# bias, not noise, most likely because the artist-placed portal quad is
# measurably larger than the real geometric window hole, so uniform-area
# sampling wastes roughly half its draws on occluded wall behind the frame
# while still dividing by the full quad's area/pdf. The escape side has no
# such fallback (a ray that already left the scene cannot occlusion-test
# itself against anything), which is why it alone needs this file.
#
# Net effect on watercolor: mean brightness ratio vs the pbrt-v4 reference
# went from 2.622x (unrestricted escape) to 0.914x (escape gated, NEE left
# alone) at 256 spp -- see _sample_infinite_light_nee's docstring in
# bvh.mojo for the measurements this design choice rests on.

from std.math import sqrt
from .geometry import Point3f, Vec3f, RGB, dot, cross, vec3f, point3f


@always_inline
def portal_frame(p0: Point3f, p1: Point3f, p3: Point3f) -> Tuple[Vec3f, Vec3f, Vec3f, Float32]:
    """(edge1, edge2, unit normal, area) of the parallelogram p0,p1,_,p3 (p2 =
    p0+edge1+edge2 is not read -- pbrt's own portal quads are rectangles, and
    a general (non-planar-parallelogram) quad has no single well-defined
    normal/area to begin with)."""
    var e1 = p1 - p0
    var e2 = p3 - p0
    var n = cross(e1, e2)
    var area = n.length()
    var nn = n * (Float32(1.0) / area) if area > Float32(1e-12) else Vec3f(Float32(0), Float32(0), Float32(1))
    return (e1, e2, nn, area)


@always_inline
def portal_ray_crosses(ray_org: Point3f, ray_dir: Vec3f, p0: Point3f, e1: Vec3f, e2: Vec3f, n: Vec3f) -> Bool:
    """Does the ray from ray_org along ray_dir cross the portal quad (in
    front of the origin, t>0)? The only consumer is the escape side (a ray
    that has already left the scene, shading.mojo's miss handler): Le pays
    out only where a real window opening would actually be visible, exactly
    matching what a modelled wall-with-a-hole would do for any other light."""
    var denom = dot(ray_dir, n)
    if abs(denom) < Float32(1e-9):
        return False   # parallel to the portal's plane: measure zero, never hit
    var t = dot(p0 - ray_org, n) / denom
    if t <= Float32(1e-6):
        return False   # portal plane is behind (or at) the ray origin
    var hit = ray_org + ray_dir * t
    var d = hit - p0
    # e1/e2 need not be orthogonal to each other in general, but pbrt's portal
    # quads are axis-aligned rectangles, so they always are here -- a plain
    # projection (dot / length_sq) is exact and needs no 2x2 solve.
    var e1_sq = dot(e1, e1)
    var e2_sq = dot(e2, e2)
    if e1_sq < Float32(1e-12) or e2_sq < Float32(1e-12):
        return False
    var uu = dot(d, e1) / e1_sq
    var vv = dot(d, e2) / e2_sq
    return uu >= Float32(0) and uu <= Float32(1) and vv >= Float32(0) and vv <= Float32(1)
