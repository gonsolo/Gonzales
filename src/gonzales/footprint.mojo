# pbrt-v4's texture/bump footprint, shared by all three integrators.
#
# pbrt estimates, at every surface hit, how far p moves per pixel step (dpdx,
# dpdy) and turns that into (u, v) derivatives by least squares
# (SurfaceInteraction::ComputeDifferentials). Two sources for dpdx/dpdy:
#   - a camera ray, or a ray that has so far only been reflected/refracted
#     specularly, carries real ray differentials. Here: the ray cone, whose
#     width is the path length along that specular chain.
#   - any other ray (after a diffuse/glossy scatter, every light-subpath and
#     photon vertex) uses Camera::Approximate_dp_dxy: the footprint the camera
#     would see at p, from the minimum per-pixel direction differential.
# Both are scaled by max(1/8, 1/sqrt(spp)): the filter width a sample needs is
# the sample spacing, not the whole pixel.
from std.math import sqrt, abs, max, min
from .geometry import Vec3f, Point3f, dot, cross
from .primitives import TriangleMesh, Instance
from .transform import Mat4, transform_normal


@fieldwise_init
struct CameraFootprint(TrivialRegisterPassable):
    """Everything Approximate_dp_dxy needs, plus the specular-chain cone's
    spread. `spp_scale == 0` means no footprint at all (every map at LOD 0 with
    pbrt's fixed 0.0005 bump step)."""
    var pos: Vec3f
    var ax: Vec3f          # camera-space x, y, z axes in world (c2w columns)
    var ay: Vec3f
    var az: Vec3f
    var min_dx: Vec3f      # pbrt minDirDifferentialX/Y, in the ray's local frame
    var min_dy: Vec3f
    var spp_scale: Float32
    var cone_spread: Float32   # world width per unit length of the cone, spp-scaled

    @staticmethod
    def none() -> Self:
        var z = Vec3f(Float32(0), Float32(0), Float32(0))
        return Self(z, z, z, z, z, z, Float32(0), Float32(0))


@fieldwise_init
struct UVFootprint(TrivialRegisterPassable):
    var du: Float32      # bump finite-difference steps, pbrt .5*(|dudx|+|dudy|)
    var dv: Float32
    var width: Float32   # MIPMap filter width, 2*max|d(u,v)/d(x,y)|; 0 = LOD 0

    @staticmethod
    def none() -> Self:
        return Self(Float32(0), Float32(0), Float32(0))


@fieldwise_init
struct TriWorld(TrivialRegisterPassable):
    """A hit triangle's corners, and its vertex normals when the mesh has
    them (`has_n`), in WORLD space (instance transform applied)."""
    var p0: Vec3f
    var p1: Vec3f
    var p2: Vec3f
    var n0: Vec3f
    var n1: Vec3f
    var n2: Vec3f
    var has_n: Bool


@always_inline
def tri_world(
    mesh: TriangleMesh, v0: Int, v1: Int, v2: Int,
    instance_idx: Int32, instances: Pointer[Instance, MutUntrackedOrigin],
) -> TriWorld:
    var p0 = Vec3f(mesh.points[unsafe_offset=v0*4], mesh.points[unsafe_offset=v0*4+1], mesh.points[unsafe_offset=v0*4+2])
    var p1 = Vec3f(mesh.points[unsafe_offset=v1*4], mesh.points[unsafe_offset=v1*4+1], mesh.points[unsafe_offset=v1*4+2])
    var p2 = Vec3f(mesh.points[unsafe_offset=v2*4], mesh.points[unsafe_offset=v2*4+1], mesh.points[unsafe_offset=v2*4+2])
    var z = Vec3f(Float32(0), Float32(0), Float32(0))
    var n0 = z; var n1 = z; var n2 = z
    var has_n = Int(mesh.normals) > 4
    if has_n:
        n0 = Vec3f(mesh.normals[unsafe_offset=v0*3], mesh.normals[unsafe_offset=v0*3+1], mesh.normals[unsafe_offset=v0*3+2])
        n1 = Vec3f(mesh.normals[unsafe_offset=v1*3], mesh.normals[unsafe_offset=v1*3+1], mesh.normals[unsafe_offset=v1*3+2])
        n2 = Vec3f(mesh.normals[unsafe_offset=v2*3], mesh.normals[unsafe_offset=v2*3+1], mesh.normals[unsafe_offset=v2*3+2])
    if instance_idx >= Int32(0):
        var inst = instances[unsafe_offset=Int(instance_idx)]
        var m = Mat4(inst.objToWorld)
        var q0 = m.transform_point(Point3f(p0[0], p0[1], p0[2]))
        var q1 = m.transform_point(Point3f(p1[0], p1[1], p1[2]))
        var q2 = m.transform_point(Point3f(p2[0], p2[1], p2[2]))
        p0 = Vec3f(q0.x, q0.y, q0.z); p1 = Vec3f(q1.x, q1.y, q1.z); p2 = Vec3f(q2.x, q2.y, q2.z)
        if has_n:
            var inv = Mat4(inst.worldToObj)
            n0 = transform_normal(inv, n0); n1 = transform_normal(inv, n1); n2 = transform_normal(inv, n2)
    return TriWorld(p0, p1, p2, n0, n1, n2, has_n)


@always_inline
def _rotate_from_to_apply(frm: Vec3f, to: Vec3f, v: Vec3f, inverse: Bool) -> Vec3f:
    """pbrt's RotateFromTo(frm, to) applied to v (or its inverse = transpose)."""
    var refl: Vec3f
    if abs(frm[0]) < Float32(0.72) and abs(to[0]) < Float32(0.72):
        refl = Vec3f(Float32(1), Float32(0), Float32(0))
    elif abs(frm[1]) < Float32(0.72) and abs(to[1]) < Float32(0.72):
        refl = Vec3f(Float32(0), Float32(1), Float32(0))
    else:
        refl = Vec3f(Float32(0), Float32(0), Float32(1))
    var u = refl - frm
    var w = refl - to
    var uu = dot(u, u); var ww = dot(w, w); var uw = dot(u, w)
    var out = SIMD[DType.float32, 4](0)
    for i in range(3):
        var acc = Float32(0)
        for j in range(3):
            var a = i if not inverse else j
            var b = j if not inverse else i
            var r = (Float32(1) if a == b else Float32(0)) - Float32(2) / uu * u[a] * u[b] \
                - Float32(2) / ww * w[a] * w[b] + Float32(4) * uw / (uu * ww) * w[a] * u[b]
            acc += r * v[j]
        out[i] = acc
    return Vec3f(out[0], out[1], out[2])


@always_inline
def _to_cam(cam: CameraFootprint, v: Vec3f) -> Vec3f:
    return Vec3f(dot(v, cam.ax) / dot(cam.ax, cam.ax), dot(v, cam.ay) / dot(cam.ay, cam.ay),
                 dot(v, cam.az) / dot(cam.az, cam.az))


@always_inline
def _from_cam(cam: CameraFootprint, v: Vec3f) -> Vec3f:
    return cam.ax * v[0] + cam.ay * v[1] + cam.az * v[2]


@always_inline
def dp_dxy_approx_camera(cam: CameraFootprint, p: Vec3f, n: Vec3f) -> Tuple[Vec3f, Vec3f]:
    """pbrt CameraBase::Approximate_dp_dxy (pinhole/thin lens: minPosDifferential
    is 0)."""
    var z = Vec3f(Float32(0), Float32(0), Float32(0))
    var pc = _to_cam(cam, p - cam.pos)
    var plen = sqrt(dot(pc, pc))
    if plen <= Float32(0):
        return (z, z)
    var dirc = pc * (Float32(1) / plen)
    var zax = Vec3f(Float32(0), Float32(0), Float32(1))
    var nc = _to_cam(cam, n)
    var p_dz = Vec3f(Float32(0), Float32(0), plen)
    var n_dz = _rotate_from_to_apply(dirc, zax, nc, False)
    var d = n_dz[2] * p_dz[2]
    var xd = zax + cam.min_dx
    var yd = zax + cam.min_dy
    var dnx = dot(n_dz, xd); var dny = dot(n_dz, yd)
    if dnx == Float32(0) or dny == Float32(0):
        return (z, z)
    var px = xd * (d / dnx)
    var py = yd * (d / dny)
    var dpdx = _from_cam(cam, _rotate_from_to_apply(dirc, zax, px - p_dz, True)) * cam.spp_scale
    var dpdy = _from_cam(cam, _rotate_from_to_apply(dirc, zax, py - p_dz, True)) * cam.spp_scale
    return (dpdx, dpdy)


@always_inline
def dp_dxy_cone(cam: CameraFootprint, n: Vec3f, ray_dir: Vec3f, cone_w: Float32) -> Tuple[Vec3f, Vec3f]:
    """Ray differentials of a cone of width `cone_w` at the hit, projected onto
    the tangent plane: offset rays along the camera's x/y axes (made
    perpendicular to the ray), intersected with the plane through p."""
    var z = Vec3f(Float32(0), Float32(0), Float32(0))
    var dn = dot(n, ray_dir)
    if dn == Float32(0):
        return (z, z)
    var ex = cam.ax - ray_dir * dot(cam.ax, ray_dir)
    var lx = sqrt(dot(ex, ex))
    if lx <= Float32(1e-8):
        ex = cross(ray_dir, cam.ay); lx = sqrt(dot(ex, ex))
    if lx <= Float32(0):
        return (z, z)
    ex = ex * (Float32(1) / lx)
    var ey = cross(ray_dir, ex)
    var ox = ex * cone_w
    var oy = ey * cone_w
    var dpdx = ox - ray_dir * (dot(n, ox) / dn)
    var dpdy = oy - ray_dir * (dot(n, oy) / dn)
    return (dpdx, dpdy)


@always_inline
def _clamp_deriv(x: Float32) -> Float32:
    if x != x or abs(x) > Float32(3.0e38):
        return Float32(0)
    return max(Float32(-1e8), min(Float32(1e8), x))


@always_inline
def uv_footprint(dpdu: Vec3f, dpdv: Vec3f, dpdx: Vec3f, dpdy: Vec3f) -> UVFootprint:
    """pbrt's least-squares (u, v) derivatives (ComputeDifferentials), then the
    bump step (BumpMap) and the MIPMap filter width (MIPMap::Filter)."""
    var ata00 = dot(dpdu, dpdu); var ata01 = dot(dpdu, dpdv); var ata11 = dot(dpdv, dpdv)
    var det = ata00 * ata11 - ata01 * ata01
    var inv_det = Float32(0) if det == Float32(0) else Float32(1) / det
    if inv_det != inv_det or abs(inv_det) > Float32(3.0e38):
        inv_det = Float32(0)
    var atb0x = dot(dpdu, dpdx); var atb1x = dot(dpdv, dpdx)
    var atb0y = dot(dpdu, dpdy); var atb1y = dot(dpdv, dpdy)
    var dudx = _clamp_deriv((ata11 * atb0x - ata01 * atb1x) * inv_det)
    var dvdx = _clamp_deriv((ata00 * atb1x - ata01 * atb0x) * inv_det)
    var dudy = _clamp_deriv((ata11 * atb0y - ata01 * atb1y) * inv_det)
    var dvdy = _clamp_deriv((ata00 * atb1y - ata01 * atb0y) * inv_det)
    var du = Float32(0.5) * (abs(dudx) + abs(dudy))
    var dv = Float32(0.5) * (abs(dvdx) + abs(dvdy))
    var width = Float32(2) * max(max(abs(dudx), abs(dudy)), max(abs(dvdx), abs(dvdy)))
    return UVFootprint(du, dv, width)


@always_inline
def tri_dpduv(tri: TriWorld, mesh: TriangleMesh, v0: Int, v1: Int, v2: Int) -> Tuple[Vec3f, Vec3f, Bool]:
    """The triangle's dp/du, dp/dv (pbrt Triangle::InteractionFromIntersection)."""
    var z = Vec3f(Float32(0), Float32(0), Float32(0))
    if Int(mesh.uvs) <= 4:
        return (z, z, False)
    var du02 = mesh.uvs[unsafe_offset=v0*2] - mesh.uvs[unsafe_offset=v2*2]
    var dv02 = mesh.uvs[unsafe_offset=v0*2+1] - mesh.uvs[unsafe_offset=v2*2+1]
    var du12 = mesh.uvs[unsafe_offset=v1*2] - mesh.uvs[unsafe_offset=v2*2]
    var dv12 = mesh.uvs[unsafe_offset=v1*2+1] - mesh.uvs[unsafe_offset=v2*2+1]
    var dp02 = tri.p0 - tri.p2
    var dp12 = tri.p1 - tri.p2
    var det = du02 * dv12 - dv02 * du12
    if abs(det) < Float32(1e-9):
        return (z, z, False)
    var inv = Float32(1) / det
    var dpdu = (dp02 * dv12 - dp12 * dv02) * inv
    var dpdv = (dp12 * du02 - dp02 * du12) * inv
    return (dpdu, dpdv, True)


@always_inline
def hit_uv_footprint(
    cam: CameraFootprint, tri: TriWorld, mesh: TriangleMesh, v0: Int, v1: Int, v2: Int,
    hit: Vec3f, ray_dir: Vec3f, cone_w: Float32,
) -> UVFootprint:
    """`cone_w >= 0`: the ray still carries its camera differentials (camera
    ray or an all-specular chain) and this is the cone's width here. `cone_w <
    0`: it does not, so use the camera approximation."""
    if cam.spp_scale <= Float32(0):
        return UVFootprint.none()
    var (dpdu, dpdv, ok) = tri_dpduv(tri, mesh, v0, v1, v2)
    if not ok:
        return UVFootprint.none()
    var ng = cross(tri.p0 - tri.p2, tri.p1 - tri.p2)
    var ngl = sqrt(dot(ng, ng))
    if ngl <= Float32(0):
        return UVFootprint.none()
    ng = ng * (Float32(1) / ngl)
    if cone_w >= Float32(0):
        var dc = dp_dxy_cone(cam, ng, ray_dir, cone_w)
        return uv_footprint(dpdu, dpdv, dc[0], dc[1])
    var da = dp_dxy_approx_camera(cam, hit, ng)
    return uv_footprint(dpdu, dpdv, da[0], da[1])


@always_inline
def _coordinate_system_local(v: Vec3f, d: Vec3f) -> Vec3f:
    """Frame::FromZ(d).ToLocal(v), with pbrt's CoordinateSystem basis."""
    var sign = Float32(1) if d[2] >= Float32(0) else Float32(-1)
    var a = Float32(-1) / (sign + d[2])
    var b = d[0] * d[1] * a
    var x = Vec3f(Float32(1) + sign * d[0] * d[0] * a, sign * b, -sign * d[0])
    var y = Vec3f(b, sign + d[1] * d[1] * a, -d[1])
    return Vec3f(dot(v, x), dot(v, y), dot(v, d))


def _raster_to_camera(r2c: Pointer[Float32, MutUntrackedOrigin], x: Float32, y: Float32) -> Vec3f:
    var cx = r2c[unsafe_offset=0]*x + r2c[unsafe_offset=4]*y + r2c[unsafe_offset=12]
    var cy = r2c[unsafe_offset=1]*x + r2c[unsafe_offset=5]*y + r2c[unsafe_offset=13]
    var cz = r2c[unsafe_offset=2]*x + r2c[unsafe_offset=6]*y + r2c[unsafe_offset=14]
    var cw = r2c[unsafe_offset=3]*x + r2c[unsafe_offset=7]*y + r2c[unsafe_offset=15]
    if cw != Float32(0) and cw != Float32(1):
        cx /= cw; cy /= cw; cz /= cw
    return Vec3f(cx, cy, cz)


@always_inline
def _normalized(v: Vec3f) -> Vec3f:
    var l = sqrt(dot(v, v))
    return v * (Float32(1) / l) if l > Float32(0) else v


def camera_footprint(
    r2c: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    film_w: Int, film_h: Int, spp: Int,
) -> CameraFootprint:
    """Host-side: pbrt's CameraBase::FindMinimumDifferentials (512 samples down
    the film diagonal, lens centre) plus the spp scale and the cone spread."""
    var big = Float32(3.0e38)
    var min_dx = Vec3f(big, big, big)
    var min_dy = Vec3f(big, big, big)
    var n = 512
    for i in range(n):
        var fx = Float32(i) / Float32(n - 1) * Float32(film_w)
        var fy = Float32(i) / Float32(n - 1) * Float32(film_h)
        var pc = _raster_to_camera(r2c, fx, fy)
        var dxc = _raster_to_camera(r2c, fx + Float32(1), fy) - pc
        var dyc = _raster_to_camera(r2c, fx, fy + Float32(1)) - pc
        var d = _normalized(pc)
        var rx = _normalized(pc + dxc)
        var ry = _normalized(pc + dyc)
        var df = _coordinate_system_local(d, d)
        var dxf = _normalized(_coordinate_system_local(rx, d))
        var dyf = _normalized(_coordinate_system_local(ry, d))
        if dot(dxf - df, dxf - df) < dot(min_dx, min_dx):
            min_dx = dxf - df
        if dot(dyf - df, dyf - df) < dot(min_dy, min_dy):
            min_dy = dyf - df
    var spp_scale = max(Float32(0.125), Float32(1) / sqrt(Float32(max(spp, 1))))
    # Cone spread: the per-pixel direction differential at the film centre.
    var cx = Float32(film_w) * Float32(0.5); var cy = Float32(film_h) * Float32(0.5)
    var pcc = _raster_to_camera(r2c, cx, cy)
    var dcx = _normalized(_raster_to_camera(r2c, cx + Float32(1), cy)) - _normalized(pcc)
    var spread = sqrt(dot(dcx, dcx)) * spp_scale
    return CameraFootprint(
        Vec3f(c2w[unsafe_offset=12], c2w[unsafe_offset=13], c2w[unsafe_offset=14]),
        Vec3f(c2w[unsafe_offset=0], c2w[unsafe_offset=1], c2w[unsafe_offset=2]),
        Vec3f(c2w[unsafe_offset=4], c2w[unsafe_offset=5], c2w[unsafe_offset=6]),
        Vec3f(c2w[unsafe_offset=8], c2w[unsafe_offset=9], c2w[unsafe_offset=10]),
        min_dx, min_dy, spp_scale, spread)
