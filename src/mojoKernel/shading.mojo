from std.math import sqrt, cos, sin, floor, acos, atan2
from std.ffi import external_call
from std.memory import alloc
from .geometry import RGB, SampledSpectrum, Point3f, Vec3f, Ray_C, Intersection_C, PrimId_C, TriangleMesh_C, Material_C, AreaLight_C, Sphere_C, DistantLight_C, PointLight_C, InfiniteLight_C, PathState_C, GpuTexture_C, ShadowTask_C, dot, cross, Frame, safe_sqrt, reflect, refract, schlick_fresnel, fr_dielectric, PI, TWO_PI, INV_PI, INV_FOUR_PI
from .rng import PCG32
from .bvh import BVH2Node, SceneDescriptor2_C, any_hit_bvh2_core, ray_sphere_hit
from .sampling import power_heuristic, sample_cosine_hemisphere, sample_ggx_vndf

@always_inline
def _shading_normal(
    mesh: TriangleMesh_C,
    v0: Int, v1: Int, v2: Int,
    bu: Float32, bv: Float32,
    geo_normal: SIMD[DType.float32, 3],
) -> SIMD[DType.float32, 3]:
    """Interpolate per-vertex shading normals with barycentric (bu, bv).
    Falls back to the geometric normal if the mesh has no shading normals.
    The result is aligned to the same hemisphere as geo_normal (which the
    caller has already oriented against the incoming ray)."""
    if Int(mesh.normals) <= 1:
        return geo_normal
    var w0 = Float32(1.0) - bu - bv
    var n0 = SIMD[DType.float32, 3](mesh.normals[v0*3], mesh.normals[v0*3+1], mesh.normals[v0*3+2])
    var n1 = SIMD[DType.float32, 3](mesh.normals[v1*3], mesh.normals[v1*3+1], mesh.normals[v1*3+2])
    var n2 = SIMD[DType.float32, 3](mesh.normals[v2*3], mesh.normals[v2*3+1], mesh.normals[v2*3+2])
    var sn = n0 * w0 + n1 * bu + n2 * bv
    var slen = dot(sn, sn)
    if slen <= Float32(1e-12):
        return geo_normal
    sn = sn * (Float32(1.0) / sqrt(slen))
    if dot(sn, geo_normal) < Float32(0.0):
        sn = -sn
    return sn

@always_inline
def _equal_area_sphere_to_square(dx: Float32, dy: Float32, dz: Float32) -> SIMD[DType.float32, 2]:
    """Convert a unit direction vector to [0,1]^2 UV using PBRT v4's equal-area
    octahedral mapping (Clarberg 2008)."""
    var x = dx if dx >= Float32(0) else -dx
    var y = dy if dy >= Float32(0) else -dy
    var z = dz if dz >= Float32(0) else -dz
    # Compute radius r = sqrt(1 - |z|)
    var r = sqrt(max(Float32(0), Float32(1) - z))
    # Compute atan(b/a)*2/pi via polynomial approximation
    var a = max(x, y)
    var b: Float32
    if a == Float32(0):
        b = Float32(0)
    else:
        b = min(x, y) / a
    # Minimax polynomial for atan(b)*2/pi
    var t1 = Float32(0.406758566246788489601959989e-5)
    var t2 = Float32(0.636226545274016134946890922156)
    var t3 = Float32(0.61572017898280213493197203466e-2)
    var t4 = Float32(-0.247333733281268944196501420480)
    var t5 = Float32(0.881770664775316294736387951347e-1)
    var t6 = Float32(0.419038818029165735901852432784e-1)
    var t7 = Float32(-0.251390972343483509333252996350e-1)
    var phi = t1 + b*(t2 + b*(t3 + b*(t4 + b*(t5 + b*(t6 + b*t7)))))
    # If x < y, we're in the 45-90 degree range
    if x < y:
        phi = Float32(1) - phi
    # Find (u, v) from (r, phi)
    var v = phi * r
    var u = r - v
    # Southern hemisphere: mirror
    if dz < Float32(0):
        var tmp = u
        u = Float32(1) - v
        v = Float32(1) - tmp
    # Apply sign from original (x, y)
    if dx < Float32(0): u = -u
    if dy < Float32(0): v = -v
    # Transform from [-1,1] to [0,1]
    u = Float32(0.5) * (u + Float32(1))
    v = Float32(0.5) * (v + Float32(1))
    return SIMD[DType.float32, 2](u, v)

@always_inline
def _equal_area_square_to_sphere(u: Float32, v: Float32) -> SIMD[DType.float32, 3]:
    """Inverse of _equal_area_sphere_to_square: [0,1]^2 UV -> unit sphere direction."""
    var uu = Float32(2) * u - Float32(1)
    var vv = Float32(2) * v - Float32(1)
    var up = uu if uu >= Float32(0) else -uu
    var vp = vv if vv >= Float32(0) else -vv
    var signed_dist = Float32(1) - (up + vp)
    var d = signed_dist if signed_dist >= Float32(0) else -signed_dist
    var r = Float32(1) - d
    var phi: Float32
    if r == Float32(0):
        phi = Float32(0)
    elif up >= vp:
        phi = (vp / r) * PI / Float32(4)
    else:
        phi = ((vp - up) / r + Float32(1)) * PI / Float32(4)
    var r2 = r * r
    var z = Float32(1) - r2
    if signed_dist < Float32(0): z = -z
    var cp = cos(phi)
    var sp = sin(phi)
    if uu < Float32(0): cp = -cp
    if vv < Float32(0): sp = -sp
    var xy_scale = r * sqrt(max(Float32(0), Float32(2) - r2))
    return SIMD[DType.float32, 3](cp * xy_scale, sp * xy_scale, z)

@always_inline
def _lower_bound(arr: UnsafePointer[Float32, MutAnyOrigin], lo: Int, hi: Int, val: Float32) -> Int:
    """Returns first index i in [lo, hi) s.t. arr[i] >= val. Returns hi if all < val."""
    var l = lo; var h = hi
    while l < h:
        var m = (l + h) >> 1
        if arr[m] < val:
            l = m + 1
        else:
            h = m
    return l

@always_inline
def _srgb_to_linear(c: Float32) -> Float32:
    if c <= Float32(0.04045):
        return c / Float32(12.92)
    else:
        return Float32(((c + Float32(0.055)) / Float32(1.055)) ** Float32(2.4))

@always_inline
# Bilinear sample of ONE mip level: `off` = float offset of the level in
# tex.data, (lw, lh) = that level's dimensions. Pixel centres at +0.5, wrap.
@always_inline
def _sample_level(data: UnsafePointer[Float32, MutAnyOrigin], off: Int, lw: Int, lh: Int, u: Float32, v: Float32) -> RGB:
    var s = u - Float32(Int(u))
    if s < Float32(0.0): s += Float32(1.0)
    var t = v - Float32(Int(v))
    if t < Float32(0.0): t += Float32(1.0)
    var fx = s * Float32(lw) - Float32(0.5)
    var fy = t * Float32(lh) - Float32(0.5)
    var x0 = Int(floor(fx)); var y0 = Int(floor(fy))
    var wx = fx - Float32(x0); var wy = fy - Float32(y0)
    var x0w = ((x0 % lw) + lw) % lw
    var y0w = ((y0 % lh) + lh) % lh
    var x1w = (x0w + 1) % lw
    var y1w = (y0w + 1) % lh
    var i00 = off + (y0w * lw + x0w) * 3
    var i10 = off + (y0w * lw + x1w) * 3
    var i01 = off + (y1w * lw + x0w) * 3
    var i11 = off + (y1w * lw + x1w) * 3
    var w00 = (Float32(1.0) - wx) * (Float32(1.0) - wy)
    var w10 = wx * (Float32(1.0) - wy)
    var w01 = (Float32(1.0) - wx) * wy
    var w11 = wx * wy
    return RGB(
        data[i00]   * w00 + data[i10]   * w10 + data[i01]   * w01 + data[i11]   * w11,
        data[i00+1] * w00 + data[i10+1] * w10 + data[i01+1] * w01 + data[i11+1] * w11,
        data[i00+2] * w00 + data[i10+2] * w10 + data[i01+2] * w01 + data[i11+2] * w11,
    )

# Trilinear mip sample. lod 0 = base level (full res); higher = coarser.
# With a 1-level texture (no pyramid) this is plain bilinear on the base.
@always_inline
def _sample_tex(tex: GpuTexture_C, u: Float32, v: Float32, lod: Float32 = Float32(0.0)) -> RGB:
    var nl = Int(tex.n_levels)
    if nl <= 1:
        return _sample_level(tex.data, 0, Int(tex.width), Int(tex.height), u, v)
    var clamped = lod
    if clamped < Float32(0.0): clamped = Float32(0.0)
    var maxl = Float32(nl - 1)
    if clamped > maxl: clamped = maxl
    var l0 = Int(floor(clamped))
    var f = clamped - Float32(l0)
    # Walk to level l0, tracking its float offset and dims.
    var off = 0; var w = Int(tex.width); var h = Int(tex.height)
    for _k in range(l0):
        off += w * h * 3
        w = max(1, w // 2); h = max(1, h // 2)
    var c0 = _sample_level(tex.data, off, w, h, u, v)
    if f <= Float32(0.0) or l0 >= nl - 1:
        return c0
    var off1 = off + w * h * 3
    var w1 = max(1, w // 2); var h1 = max(1, h // 2)
    var c1 = _sample_level(tex.data, off1, w1, h1, u, v)
    return c0 + (c1 - c0) * f

# Unified 2D-texture fetch — the single use_gpu seam for texture sampling.
# GPU reads the uploaded GpuTexture_C table; CPU reads via OIIO by filename.
# (u, v) are the interpolated, NOT-yet-V-flipped coords; this applies pbrt's
# V-flip (1 - v) and (CPU) wrap. raw=True skips the sRGB decode (normal maps).
# Sets `found` False when there is no texture/data (caller uses its fallback).
@always_inline
def sample_texture[use_gpu: Bool](
    tex_idx: Int,
    u: Float32, v: Float32,
    raw: Bool,
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin],
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
    mut found: Bool,
) -> RGB:
    found = False
    if tex_idx < 0:
        return RGB(Float32(0.0), Float32(0.0), Float32(0.0))
    var su = u
    var tv = Float32(1.0) - v  # pbrt V-flip: V=0 at top
    comptime if use_gpu:
        if tex_idx < n_textures:
            var tex = textures[tex_idx]
            if Int(tex.width) > 0:
                found = True
                return _sample_tex(tex, su, tv)
    else:
        if Int(tex_filenames) > 1:
            var filename = tex_filenames[tex_idx]
            if Int(filename) > 1:
                su = su - Float32(Int(su))
                if su < Float32(0.0): su += Float32(1.0)
                tv = tv - Float32(Int(tv))
                if tv < Float32(0.0): tv += Float32(1.0)
                var tr = alloc[Float32](3)
                tr[0] = Float32(0.0); tr[1] = Float32(0.0); tr[2] = Float32(0.0)
                _ = external_call["texture", Bool,
                    UnsafePointer[UInt8, MutAnyOrigin], Float32, Float32,
                    UnsafePointer[Float32, MutAnyOrigin]](filename, su, tv, tr)
                var rr = tr[0]; var gg = tr[1]; var bb = tr[2]
                tr.free()
                found = True
                if raw:
                    return RGB(rr, gg, bb)
                return RGB(_srgb_to_linear(rr), _srgb_to_linear(gg), _srgb_to_linear(bb))
    return RGB(Float32(0.0), Float32(0.0), Float32(0.0))

@always_inline
def _tex_lookup[use_gpu: Bool](
    mat: Material_C,
    inter: Intersection_C,
    v0: Int, v1: Int, v2: Int,
    mesh: TriangleMesh_C,
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin],
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
) -> RGB:
    if mat.tex_idx < Int32(0) or Int(mesh.uvs) <= 1:
        return mat.albedo
    var w0 = Float32(1.0) - inter.u - inter.v
    var su = w0*mesh.uvs[v0*2]   + inter.u*mesh.uvs[v1*2]   + inter.v*mesh.uvs[v2*2]
    var tv = w0*mesh.uvs[v0*2+1] + inter.u*mesh.uvs[v1*2+1] + inter.v*mesh.uvs[v2*2+1]
    var found = False
    var rgb = sample_texture[use_gpu](Int(mat.tex_idx), su, tv, False, tex_filenames, textures, n_textures, found)
    if found:
        return rgb
    return mat.albedo


@always_inline
def shade_core(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    tid: Int,
):
    var path_ptr = paths + tid
    if path_ptr[].active == 0:
        return

    var inter = intersections[tid]
    if inter.hit == 0:
        path_ptr[].active = 0
        return

    var mat_idx = Int(inter.primId.materialIndex)
    var mat = materials[mat_idx]

    if mat.type == 2:
        path_ptr[].estimate += path_ptr[].throughput * mat.emission
        path_ptr[].active = 0
        return

    if mat.type == 1:
        # Construct Normal
        var mesh_idx: Int
        var base_vidx: Int
        if inter.primId.type == 0:
            mesh_idx = Int(inter.primId.id1)
            base_vidx = Int(inter.primId.id2)
        elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
            mesh_idx = Int(inter.primId.id2 >> 32)
            base_vidx = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
        else:
            path_ptr[].active = 0
            return

        var mesh = meshes[mesh_idx]
        var v0_idx = Int(mesh.vertexIndices[base_vidx])
        var v1_idx = Int(mesh.vertexIndices[base_vidx + 1])
        var v2_idx = Int(mesh.vertexIndices[base_vidx + 2])

        var p0 = SIMD[DType.float32, 3](mesh.points[v0_idx * 4], mesh.points[v0_idx * 4 + 1], mesh.points[v0_idx * 4 + 2])
        var p1 = SIMD[DType.float32, 3](mesh.points[v1_idx * 4], mesh.points[v1_idx * 4 + 1], mesh.points[v1_idx * 4 + 2])
        var p2 = SIMD[DType.float32, 3](mesh.points[v2_idx * 4], mesh.points[v2_idx * 4 + 1], mesh.points[v2_idx * 4 + 2])

        var edge1 = p1 - p0
        var edge2 = p2 - p0
        var normal = cross(edge1, edge2)
        var nlen = dot(normal, normal)
        if nlen > 0:
            normal = normal * (1.0 / sqrt(nlen))

        # Orient normal towards ray
        var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
        if dot(normal, ray_dir) > 0:
            normal = normal * -1.0

        # Cosine weighted hemisphere sampling
        var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
        var u1 = pcg.next_float()
        var u2 = pcg.next_float()
        path_ptr[].pcgState = pcg.state

        var r = sqrt(u1)
        var theta = 2.0 * PI * u2
        var x = r * cos(theta)
        var y = r * sin(theta)
        var z2 = 1.0 - u1
        var z = sqrt(z2 if z2 > 0.0 else Float32(0.0))

        # Build tangent basis
        var sign = Float32(1.0) if normal[2] >= 0.0 else Float32(-1.0)
        var a = Float32(-1.0) / (sign + normal[2])
        var b = normal[0] * normal[1] * a
        var tangent = SIMD[DType.float32, 3](Float32(1.0) + sign * normal[0] * normal[0] * a, sign * b, -sign * normal[0])
        var bitangent = SIMD[DType.float32, 3](b, sign + normal[1] * normal[1] * a, -normal[1])

        var dir = tangent * x + bitangent * y + normal * z
        var dlen = dot(dir, dir)
        if dlen > 0:
            dir = dir * (1.0 / sqrt(dlen))

        # Update Ray
        var org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z) + ray_dir * inter.tHit + normal * 0.0001
        path_ptr[].ray = Ray_C(Point3f(org[0], org[1], org[2]), Vec3f(dir[0], dir[1], dir[2]))

        # Update Throughput (albedo)
        path_ptr[].throughput *= mat.albedo
    else:
        # Unknown material type — deactivate to prevent infinite loops
        path_ptr[].active = 0


# ── DiffuseTransmission branch ────────────────────────────────────────────────
@always_inline
def shade_diffuse_transmission[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    path_idx: Int,
    inter: Intersection_C,
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int,
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin],
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
    shadow_tasks: UnsafePointer[ShadowTask_C, MutAnyOrigin],
):
    var mat = materials[Int(inter.primId.materialIndex)]
    var mesh_idx: Int
    var base_vidx: Int
    if inter.primId.type == 0:
        mesh_idx = Int(inter.primId.id1)
        base_vidx = Int(inter.primId.id2)
    elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
        mesh_idx = Int(inter.primId.id2 >> 32)
        base_vidx = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
    else:
        path_ptr[].active = 0
        return

    var mesh = meshes[mesh_idx]
    var v0 = Int(mesh.vertexIndices[base_vidx])
    var v1 = Int(mesh.vertexIndices[base_vidx + 1])
    var v2 = Int(mesh.vertexIndices[base_vidx + 2])
    var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])

    var normal = cross(p1 - p0, p2 - p0)
    var nlen = dot(normal, normal)
    if nlen > Float32(0.0):
        normal = normal * (Float32(1.0) / sqrt(nlen))

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    # Orient normal toward incoming ray
    if dot(normal, ray_dir) > Float32(0.0):
        normal = -normal

    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)

    # Balance heuristic: choose reflect vs transmit proportional to luminance
    var refl = mat.albedo
    var trans = mat.emission
    var pr = refl.luma()
    var pt = trans.luma()
    var total = pr + pt
    if total <= Float32(0.0):
        path_ptr[].active = 0
        return

    var choose_reflect = pcg.next_float() < pr / total
    # Bounce normal: same side for reflection, opposite for transmission
    var bounce_normal = normal if choose_reflect else -normal
    # Albedo for this lobe (used in NEE estimator)
    var lobe_alb = refl if choose_reflect else trans
    # Selection-probability compensation weight
    var lobe_w = total / (pr if choose_reflect else pt)

    var hit_point = ray_org + ray_dir * inter.tHit + bounce_normal * Float32(0.0001)

    # ── NEE direct light sampling (MIS weighted) ───────────────────────────────
    if areaLightCount > 0:
        var light_idx = Int(pcg.next_uint() % UInt32(areaLightCount))
        var al = areaLights[light_idx]
        var lmesh = meshes[Int(al.meshIdx)]
        var lti = Int(pcg.next_uint() % UInt32(max(Int(al.n_tris), 1)))
        var lb = lti * 3
        var lv0 = Int(lmesh.vertexIndices[lb])
        var lv1 = Int(lmesh.vertexIndices[lb + 1])
        var lv2 = Int(lmesh.vertexIndices[lb + 2])
        var lp0 = SIMD[DType.float32, 3](lmesh.points[lv0*4], lmesh.points[lv0*4+1], lmesh.points[lv0*4+2])
        var lp1 = SIMD[DType.float32, 3](lmesh.points[lv1*4], lmesh.points[lv1*4+1], lmesh.points[lv1*4+2])
        var lp2 = SIMD[DType.float32, 3](lmesh.points[lv2*4], lmesh.points[lv2*4+1], lmesh.points[lv2*4+2])
        var r1 = pcg.next_float()
        var r2 = pcg.next_float()
        var sqrt_r1 = sqrt(r1)
        var light_point = lp0 * (Float32(1.0) - sqrt_r1) + lp1 * (sqrt_r1 * (Float32(1.0) - r2)) + lp2 * (sqrt_r1 * r2)
        var lcross = cross(lp1 - lp0, lp2 - lp0)
        var light_normal = lcross
        var lcross_len = dot(lcross, lcross)
        if lcross_len > Float32(0.0):
            light_normal = lcross * (Float32(1.0) / sqrt(lcross_len))
        var to_light = light_point - hit_point
        var dist_sq = dot(to_light, to_light)
        var dist = sqrt(dist_sq)
        if dist > Float32(0.0001) and al.total_area > Float32(0.0):
            var shadow_dir = to_light * (Float32(1.0) / dist)
            var cos_s = dot(bounce_normal, shadow_dir)
            var cos_l = -dot(light_normal, shadow_dir)
            if cos_s > Float32(0.0) and cos_l > Float32(0.0):
                var pi = PI
                var pdf_light = dist_sq / (cos_l * al.total_area * Float32(areaLightCount))
                var pdf_bsdf_dt = cos_s / pi
                var w_dt = power_heuristic(pdf_light, pdf_bsdf_dt)
                # f = lobe_alb/π * lobe_w (lobe selection weight already folded in)
                var weight_dt = lobe_alb * al.emission * (cos_s * w_dt * lobe_w / (pdf_light * pi))
                var contrib = path_ptr[].throughput * weight_dt
                comptime if enqueue_shadow:
                    shadow_tasks[path_idx] = ShadowTask_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]), dist * Float32(0.9999), RGB(contrib.r, contrib.g, contrib.b), Int32(1), Int32(0))
                else:
                    var shadow_ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
                    if not any_hit_bvh2_core(bvh2Nodes, primIds, meshes, shadow_ray, dist * Float32(0.9999)):
                        path_ptr[].estimate += contrib

    # ── BSDF scatter ───────────────────────────────────────────────────────────
    var u1 = pcg.next_float()
    var u2 = pcg.next_float()
    var r = sqrt(u1)
    var theta = TWO_PI * u2
    var sx = r * cos(theta)
    var sy = r * sin(theta)
    var z2 = Float32(1.0) - u1
    var sz = sqrt(z2 if z2 > Float32(0.0) else Float32(0.0))

    var sign = Float32(1.0) if bounce_normal[2] >= Float32(0.0) else Float32(-1.0)
    var aa = Float32(-1.0) / (sign + bounce_normal[2])
    var bb = bounce_normal[0] * bounce_normal[1] * aa
    var tangent  = SIMD[DType.float32, 3](Float32(1.0) + sign * bounce_normal[0] * bounce_normal[0] * aa, sign * bb, -sign * bounce_normal[0])
    var bitangent = SIMD[DType.float32, 3](bb, sign + bounce_normal[1] * bounce_normal[1] * aa, -bounce_normal[1])

    var dir = tangent * sx + bitangent * sy + bounce_normal * sz
    var dlen = dot(dir, dir)
    if dlen > Float32(0.0):
        dir = dir * (Float32(1.0) / sqrt(dlen))

    path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(dir[0], dir[1], dir[2]))

    # Throughput: lobe_alb / pdf_bsdf * lobe_selection_weight
    # = lobe_alb / (cos/π) * (total/p_lobe) → lobe_alb * π/cos * lobe_w
    # But cosine-hemisphere importance sampling gives cos/π cancel:
    # f * cos / pdf = (lobe_alb/π) * cos / (cos/π) = lobe_alb
    # Then multiply by lobe_w to compensate for stochastic lobe selection.
    path_ptr[].throughput *= lobe_alb * lobe_w

    # Store BSDF pdf for next-bounce MIS (cosine hemisphere: cos/π)
    var cos_sc = dot(dir, bounce_normal)
    path_ptr[].lastBsdfPdf = (cos_sc if cos_sc > Float32(0.0) else Float32(0.0)) / PI
    path_ptr[].specularBounce = Int8(0)

    if path_ptr[].bounce == 0:
        path_ptr[].albedo = mat.albedo
    path_ptr[].bounce += 1

    # Russian roulette after first bounce
    if path_ptr[].bounce > 1:
        var lum = path_ptr[].throughput.luma()
        var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
        if pcg.next_float() < q:
            path_ptr[].active = 0
        else:
            path_ptr[].throughput *= Float32(1.0) / (Float32(1.0) - q)

    path_ptr[].pcgState = pcg.state


# ── CoatedDiffuse (plastic) branch ───────────────────────────────────────────
@always_inline
def shade_coated_diffuse[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    path_idx: Int,
    inter: Intersection_C,
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    mat: Material_C,
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    infiniteLightCount: Int,
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin],
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
    shadow_tasks: UnsafePointer[ShadowTask_C, MutAnyOrigin],
):
    var mesh_idx: Int
    var base_vidx: Int
    if inter.primId.type == 0:
        mesh_idx = Int(inter.primId.id1)
        base_vidx = Int(inter.primId.id2)
    elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
        mesh_idx = Int(inter.primId.id2 >> 32)
        base_vidx = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
    else:
        path_ptr[].active = 0
        return

    var mesh = meshes[mesh_idx]
    var v0 = Int(mesh.vertexIndices[base_vidx])
    var v1 = Int(mesh.vertexIndices[base_vidx + 1])
    var v2 = Int(mesh.vertexIndices[base_vidx + 2])
    var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])

    var alb = _tex_lookup[use_gpu](mat, inter, v0, v1, v2, mesh, tex_filenames, textures, n_textures)

    var normal = cross(p1 - p0, p2 - p0)
    var nlen = dot(normal, normal)
    if nlen > Float32(0.0):
        normal = normal * (Float32(1.0) / sqrt(nlen))

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    if dot(normal, ray_dir) > Float32(0.0):
        normal = -normal

    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)
    # Use interpolated shading normal (geometric normal still drives hit-point offset)
    normal = _shading_normal(mesh, v0, v1, v2, inter.u, inter.v, normal)

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)

    if path_ptr[].bounce == 0:
        path_ptr[].albedo = alb

    # ── Layered BSDF: smooth dielectric coat over a Lambertian base ──────────
    # Stochastic random walk (PBRT LayeredBxDF, CoatedDiffuse). The coat's air
    # interface reflects a glossy lobe (exact dielectric Fresnel) and transmits
    # the rest into the coat, where light scatters off the diffuse base and is
    # partially recycled by (total internal) reflection at the coat underside.
    # That multiple scattering is what brightens and saturates the base colour.
    # Fresnel at each interface is handled by the reflect/transmit probability
    # split, so the throughput accumulator beta only gathers the base albedo.
    var ior = mat.emission.r            # coat IOR (η_coat/η_air), set at parse
    var inv_ior = Float32(1.0) / ior
    # Coat GGX roughness (remaproughness=false ⇒ roughness IS alpha). 0 ⇒ smooth
    # mirror coat (e.g. car paint 0.001); larger ⇒ soft sheen (e.g. tyres 0.4).
    var coat_alpha = max(mat.roughU, mat.roughV)
    var is_rough_coat = coat_alpha > Float32(0.001)
    var wo = SIMD[DType.float32, 3](-ray_dir[0], -ray_dir[1], -ray_dir[2])  # toward viewer

    # Tangent frame (Frisvad) around the shading normal for hemisphere sampling.
    var sign = Float32(1.0) if normal[2] >= Float32(0.0) else Float32(-1.0)
    var a = Float32(-1.0) / (sign + normal[2])
    var b = normal[0] * normal[1] * a
    var tangent = SIMD[DType.float32, 3](Float32(1.0) + sign * normal[0] * normal[0] * a, sign * b, -sign * normal[0])
    var bitangent = SIMD[DType.float32, 3](b, sign + normal[1] * normal[1] * a, -normal[1])

    # Microfacet normal at the coat's air interface: the surface normal when
    # smooth, else a GGX visible-normal sample (Heitz VNDF). Fresnel is taken
    # at this microfacet; the VNDF G2/G1 weight is approximated as 1 (matching
    # the conductor path), so the throughput accumulator stays albedo-only.
    var wm = normal
    if is_rough_coat:
        var wo_l = Vec3f(dot(wo, tangent), dot(wo, bitangent), dot(wo, normal))
        var wm_l = sample_ggx_vndf(wo_l, coat_alpha, coat_alpha, pcg.next_float(), pcg.next_float())
        wm = tangent * wm_l.x + bitangent * wm_l.y + normal * wm_l.z
        var wmlen = dot(wm, wm)
        if wmlen > Float32(0.0):
            wm = wm * (Float32(1.0) / sqrt(wmlen))
    var cos_wm = dot(wo, wm)
    var f_entry = fr_dielectric(cos_wm, ior)

    if pcg.next_float() < f_entry:
        # Glossy reflection off the coat (rough ⇒ GGX lobe, smooth ⇒ mirror).
        # Single-strategy lobe (no NEE) ⇒ specularBounce=1 takes full light on
        # miss and the exact pdf is irrelevant.
        var refl = wm * (Float32(2.0) * cos_wm) - wo
        var rlen = dot(refl, refl)
        if rlen > Float32(0.0):
            refl = refl * (Float32(1.0) / sqrt(rlen))
        if dot(refl, normal) <= Float32(0.0):
            path_ptr[].active = 0          # reflected below the surface — discard
            path_ptr[].pcgState = pcg.state
            return
        path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(refl[0], refl[1], refl[2]))
        path_ptr[].specularBounce = Int8(1)
        path_ptr[].lastBsdfPdf = Float32(0.0)
        path_ptr[].bounce += 1
        path_ptr[].pcgState = pcg.state
        return

    # Transmitted into the coat: random-walk the base/coat-underside layers.
    var beta = RGB(Float32(1.0), Float32(1.0), Float32(1.0))
    var exited = False
    var exit_dir = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(0.0))
    var did_nee = False
    var did_env_nee = False

    comptime MAX_COAT_DEPTH = 10
    for _ in range(MAX_COAT_DEPTH):
        # ── Diffuse base: NEE (single scatter) once, weighted by the coat's
        #    transmittance for the incoming light direction. ──
        if not did_nee and areaLightCount > 0:
            did_nee = True
            var light_idx = Int(pcg.next_uint() % UInt32(areaLightCount))
            var al = areaLights[light_idx]
            var lmesh = meshes[Int(al.meshIdx)]
            var lti = Int(pcg.next_uint() % UInt32(max(Int(al.n_tris), 1)))
            var lb = lti * 3
            var lv0 = Int(lmesh.vertexIndices[lb])
            var lv1 = Int(lmesh.vertexIndices[lb + 1])
            var lv2 = Int(lmesh.vertexIndices[lb + 2])
            var lp0 = SIMD[DType.float32, 3](lmesh.points[lv0*4], lmesh.points[lv0*4+1], lmesh.points[lv0*4+2])
            var lp1 = SIMD[DType.float32, 3](lmesh.points[lv1*4], lmesh.points[lv1*4+1], lmesh.points[lv1*4+2])
            var lp2 = SIMD[DType.float32, 3](lmesh.points[lv2*4], lmesh.points[lv2*4+1], lmesh.points[lv2*4+2])
            var r1 = pcg.next_float()
            var r2 = pcg.next_float()
            var sqrt_r1 = sqrt(r1)
            var light_point = lp0 * (Float32(1.0) - sqrt_r1) + lp1 * (sqrt_r1 * (Float32(1.0) - r2)) + lp2 * (sqrt_r1 * r2)
            var lcross = cross(lp1 - lp0, lp2 - lp0)
            var light_normal = lcross
            var lcross_len = dot(lcross, lcross)
            if lcross_len > Float32(0.0):
                light_normal = lcross * (Float32(1.0) / sqrt(lcross_len))
            var to_light = light_point - hit_point
            var dist_sq = dot(to_light, to_light)
            var dist = sqrt(dist_sq)
            if dist > Float32(0.0001) and al.total_area > Float32(0.0):
                var shadow_dir = to_light * (Float32(1.0) / dist)
                var cos_s = dot(normal, shadow_dir)
                var cos_l = -dot(light_normal, shadow_dir)
                if cos_s > Float32(0.0) and cos_l > Float32(0.0):
                    # Light must transmit through the coat to reach the base.
                    var t_light = Float32(1.0) - fr_dielectric(cos_s, ior)
                    var pdf_light_cd = dist_sq / (cos_l * al.total_area * Float32(areaLightCount))
                    var pdf_bsdf_cd = cos_s / PI
                    var w_cd = power_heuristic(pdf_light_cd, pdf_bsdf_cd)
                    var weight_cd = alb * al.emission * (cos_s * t_light * w_cd / (pdf_light_cd * PI))
                    var contrib = path_ptr[].throughput * weight_cd
                    comptime if enqueue_shadow:
                        shadow_tasks[path_idx] = ShadowTask_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]), dist * Float32(0.9999), RGB(contrib.r, contrib.g, contrib.b), Int32(1), Int32(0))
                    else:
                        var shadow_ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
                        if not any_hit_bvh2_core(bvh2Nodes, primIds, meshes, shadow_ray, dist * Float32(0.9999)):
                            path_ptr[].estimate += contrib

        # ── Env-map (infinite light) NEE at the base, once. Light enters via
        #    the coat so the contribution is weighted by 1 - F(cos_env). The
        #    view-side coat transmittance is implicit in reaching this branch. ──
        if not did_env_nee and infiniteLightCount > 0:
            did_env_nee = True
            for inf_i in range(infiniteLightCount):
                var ilight = infiniteLights[inf_i]
                var w2l = ilight.world_to_light
                var env_dir: SIMD[DType.float32, 3]
                var env_rgb: RGB
                var pdf_light: Float32
                if ilight.tex_idx >= Int32(0) and Int(ilight.pixels_ptr) > 1 and Int(ilight.cdf_ptr) > 1 and ilight.cdf_w > Int32(0):
                    var iw = Int(ilight.cdf_w); var ih = Int(ilight.cdf_h)
                    var u1_env = pcg.next_float()
                    var u2_env = pcg.next_float()
                    var row_idx = _lower_bound(ilight.cdf_ptr, 0, ih, u1_env)
                    row_idx = min(row_idx, ih - 1)
                    var dp_row = ilight.cdf_ptr[row_idx + 1] - ilight.cdf_ptr[row_idx]
                    var cond_base = (ih + 1) + row_idx * (iw + 1)
                    var col_idx = _lower_bound(ilight.cdf_ptr, cond_base, cond_base + iw, u2_env) - cond_base
                    col_idx = min(col_idx, iw - 1)
                    var dp_col = ilight.cdf_ptr[cond_base + col_idx + 1] - ilight.cdf_ptr[cond_base + col_idx]
                    var sample_u = (Float32(col_idx) + Float32(0.5)) / Float32(iw)
                    var sample_v = (Float32(row_idx) + Float32(0.5)) / Float32(ih)
                    var local_d = _equal_area_square_to_sphere(sample_u, sample_v)
                    var wd_x = w2l[0]*local_d[0] + w2l[1]*local_d[1] + w2l[2]*local_d[2]
                    var wd_y = w2l[4]*local_d[0] + w2l[5]*local_d[1] + w2l[6]*local_d[2]
                    var wd_z = w2l[8]*local_d[0] + w2l[9]*local_d[1] + w2l[10]*local_d[2]
                    env_dir = SIMD[DType.float32, 3](wd_x, wd_y, wd_z)
                    var px = min(iw - 1, max(0, Int(sample_u * Float32(iw))))
                    var py = min(ih - 1, max(0, Int(sample_v * Float32(ih))))
                    var pr = ilight.pixels_ptr[(py*iw+px)*3+0]
                    var pg = ilight.pixels_ptr[(py*iw+px)*3+1]
                    var pb = ilight.pixels_ptr[(py*iw+px)*3+2]
                    env_rgb = RGB(pr, pg, pb) * ilight.scale
                    if dp_row > Float32(0) and dp_col > Float32(0):
                        pdf_light = dp_row * dp_col * Float32(iw * ih) * INV_FOUR_PI
                    else:
                        pdf_light = INV_FOUR_PI
                else:
                    var u1_env = pcg.next_float()
                    var u2_env = pcg.next_float()
                    var r_env = sqrt(u1_env)
                    var theta_env = TWO_PI * u2_env
                    var x_env = r_env * cos(theta_env)
                    var y_env = r_env * sin(theta_env)
                    var z2_env = Float32(1.0) - u1_env
                    var z_env = sqrt(z2_env if z2_env > Float32(0.0) else Float32(0.0))
                    env_dir = tangent * x_env + bitangent * y_env + normal * z_env
                    var env_dlen = dot(env_dir, env_dir)
                    if env_dlen > Float32(0.0):
                        env_dir = env_dir * (Float32(1.0) / sqrt(env_dlen))
                    env_rgb = ilight.scale
                    pdf_light = INV_FOUR_PI
                var cos_env = dot(normal, env_dir)
                if cos_env > Float32(0.0) and not env_rgb.is_black() and pdf_light > Float32(0.0):
                    var t_env = Float32(1.0) - fr_dielectric(cos_env, ior)
                    var pdf_bsdf_nee = cos_env / PI
                    var mis_w = power_heuristic(pdf_light, pdf_bsdf_nee)
                    var contrib_e = path_ptr[].throughput * alb * env_rgb * (cos_env * t_env / (PI * pdf_light)) * mis_w
                    var t_max_env = Float32(100000.0)
                    comptime if enqueue_shadow:
                        shadow_tasks[path_idx] = ShadowTask_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(env_dir[0], env_dir[1], env_dir[2]), t_max_env, RGB(contrib_e.r, contrib_e.g, contrib_e.b), Int32(1), Int32(0))
                    else:
                        var shadow_ray_e = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(env_dir[0], env_dir[1], env_dir[2]))
                        if not any_hit_bvh2_core(bvh2Nodes, primIds, meshes, shadow_ray_e, t_max_env):
                            path_ptr[].estimate += contrib_e

        # Lambertian base: sample a cosine-weighted up-going direction.
        var u1 = pcg.next_float()
        var u2 = pcg.next_float()
        var r = sqrt(u1)
        var theta = TWO_PI * u2
        var x = r * cos(theta)
        var y = r * sin(theta)
        var z2 = Float32(1.0) - u1
        var z = sqrt(z2 if z2 > Float32(0.0) else Float32(0.0))
        var w_up = tangent * x + bitangent * y + normal * z
        var wlen = dot(w_up, w_up)
        if wlen > Float32(0.0):
            w_up = w_up * (Float32(1.0) / sqrt(wlen))
        beta *= alb

        # Coat underside: transmit out (exit) or reflect back (recycle).
        # The interface microfacet is the surface normal when smooth, else a
        # GGX VNDF sample seen from w_up — this softens the exit direction.
        var wm_e = normal
        if is_rough_coat:
            var wup_l = Vec3f(dot(w_up, tangent), dot(w_up, bitangent), dot(w_up, normal))
            var wm_e_l = sample_ggx_vndf(wup_l, coat_alpha, coat_alpha, pcg.next_float(), pcg.next_float())
            wm_e = tangent * wm_e_l.x + bitangent * wm_e_l.y + normal * wm_e_l.z
            var wmelen = dot(wm_e, wm_e)
            if wmelen > Float32(0.0):
                wm_e = wm_e * (Float32(1.0) / sqrt(wmelen))
        var cos_up = dot(w_up, wm_e)
        var f_exit = fr_dielectric(cos_up, inv_ior)
        if pcg.next_float() < (Float32(1.0) - f_exit):
            var rr = refract(Vec3f(-w_up[0], -w_up[1], -w_up[2]),
                             Vec3f(-wm_e[0], -wm_e[1], -wm_e[2]), ior)
            if rr[0]:
                var wt = rr[1]
                exit_dir = SIMD[DType.float32, 3](wt.x, wt.y, wt.z)
                var elen = dot(exit_dir, exit_dir)
                if elen > Float32(0.0):
                    exit_dir = exit_dir * (Float32(1.0) / sqrt(elen))
                # A rough microfacet can refract below the surface; recycle then.
                if dot(exit_dir, normal) > Float32(0.0):
                    exited = True
                    break
            # Refraction failed / exited below surface → fall through and recycle.
        # Internal reflection: light recycled; next iteration re-samples base.

    if not exited:
        path_ptr[].active = 0
        path_ptr[].pcgState = pcg.state
        return

    path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(exit_dir[0], exit_dir[1], exit_dir[2]))
    # NEE-only for direct lighting: the layered exit ray's true pdf is
    # intractable (refracted, multi-bounce) so it can't be MIS-combined with
    # NEE. Setting lastBsdfPdf = 0 makes the miss/emitter handlers drop this
    # ray's *direct* light contribution (mis_weight → 0); NEE supplies direct
    # lighting while the exit ray still carries indirect bounces. Avoids the
    # double-count between the multi-scatter exit and single-scatter NEE.
    path_ptr[].lastBsdfPdf = Float32(0.0)
    path_ptr[].specularBounce = Int8(0)
    path_ptr[].throughput *= beta
    path_ptr[].bounce += 1

    # Russian roulette after first bounce
    if path_ptr[].bounce > 1:
        var lum = path_ptr[].throughput.luma()
        var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
        if pcg.next_float() < q:
            path_ptr[].active = 0
        else:
            path_ptr[].throughput *= Float32(1.0) / (Float32(1.0) - q)

    path_ptr[].pcgState = pcg.state


# ── Dielectric (glass) branch ─────────────────────────────────────────────────
@always_inline
def shade_dielectric(
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    mat: Material_C,
):
    var mesh_idx: Int
    var base_vidx: Int
    if inter.primId.type == 0:
        mesh_idx = Int(inter.primId.id1)
        base_vidx = Int(inter.primId.id2)
    elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
        mesh_idx = Int(inter.primId.id2 >> 32)
        base_vidx = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
    else:
        path_ptr[].active = 0
        return

    var mesh = meshes[mesh_idx]
    var v0 = Int(mesh.vertexIndices[base_vidx])
    var v1 = Int(mesh.vertexIndices[base_vidx + 1])
    var v2 = Int(mesh.vertexIndices[base_vidx + 2])
    var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])

    var geom_normal = cross(p1 - p0, p2 - p0)
    var nlen = dot(geom_normal, geom_normal)
    if nlen > Float32(0.0):
        geom_normal = geom_normal * (Float32(1.0) / sqrt(nlen))

    # Smooth shading normal, oriented to the geometric/winding normal (PBRT's
    # FaceForward(Ns, Ng)): the winding — flipped at parse time by
    # ReverseOrientation — is the authoritative outside direction the dielectric
    # uses to pick entering vs exiting. (Using the raw vertex normal instead
    # breaks meshes whose vertex normals point inward, causing spurious TIR.)
    geom_normal = _shading_normal(mesh, v0, v1, v2, inter.u, inter.v, geom_normal)

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)

    var ior = mat.albedo.r
    # Orient the normal to face the incoming ray (cos_i > 0). Whether the
    # geometric normal already faced the ray tells us entering vs exiting.
    var facing = dot(ray_dir, geom_normal) < Float32(0.0)
    var entering = facing
    # A camera/primary ray (bounce 0) from an exterior camera always enters the
    # glass from air. Some meshes in this model have inward-facing normals (no
    # ReverseOrientation) which would otherwise be read as "exiting" and total-
    # internal-reflect the envmap. Trust the physics for the first bounce.
    if path_ptr[].bounce == 0:
        entering = True
    var normal = geom_normal if facing else -geom_normal  # faces the ray
    var eta = (Float32(1.0) / ior) if entering else ior

    var cos_i = -dot(ray_dir, normal)
    var sin2_t = eta * eta * (Float32(1.0) - cos_i * cos_i)
    var tir = sin2_t > Float32(1.0)

    # Exact unpolarized dielectric Fresnel (eta here is η_i/η_t, so pass its
    # reciprocal as the relative IOR). Returns 1.0 on total internal reflection.
    var fresnel = fr_dielectric(cos_i, Float32(1.0) / eta)

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)

    if tir or pcg.next_float() < fresnel:
        # Reflect: r = d + 2*cos_i*n
        var refl = ray_dir + normal * (Float32(2.0) * cos_i)
        var rlen = dot(refl, refl)
        if rlen > Float32(0.0):
            refl = refl * (Float32(1.0) / sqrt(rlen))
        var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)
        path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(refl[0], refl[1], refl[2]))
    else:
        # Refract: t = eta*d + (eta*cos_i - sqrt(1 - sin2_t))*n
        var cos_t = sqrt(Float32(1.0) - sin2_t)
        var refr = ray_dir * eta + normal * (eta * cos_i - cos_t)
        var rlen = dot(refr, refr)
        if rlen > Float32(0.0):
            refr = refr * (Float32(1.0) / sqrt(rlen))
        var hit_point = ray_org + ray_dir * inter.tHit - normal * Float32(0.0001)
        path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(refr[0], refr[1], refr[2]))

    if path_ptr[].bounce == 0:
        path_ptr[].albedo = RGB(Float32(1), Float32(1), Float32(1))
    path_ptr[].bounce += 1

    # Russian roulette after first bounce (throughput unchanged for ideal glass)
    if path_ptr[].bounce > 1:
        var lum = path_ptr[].throughput.luma()
        var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
        if pcg.next_float() < q:
            path_ptr[].active = 0
        else:
            path_ptr[].throughput *= Float32(1.0) / (Float32(1.0) - q)

    path_ptr[].pcgState = pcg.state
    path_ptr[].specularBounce = Int8(1)   # dielectric is a delta BSDF
    path_ptr[].lastBsdfPdf = Float32(0.0) # delta — pdf undefined for cosine hemisphere

# Thin dielectric (type 9): one-sided glass — Fresnel selects reflect or transmit,
# but transmitted ray is NOT refracted (direction unchanged). Models window glass,
# soap bubbles, thin films. IOR stored in mat.albedo.r like regular dielectric.
@always_inline
def shade_thin_dielectric(
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    mat: Material_C,
):
    var mesh_idx: Int
    var base_vidx: Int
    if inter.primId.type == 0:
        mesh_idx = Int(inter.primId.id1)
        base_vidx = Int(inter.primId.id2)
    elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
        mesh_idx = Int(inter.primId.id2 >> 32)
        base_vidx = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
    else:
        path_ptr[].active = 0
        return

    var mesh = meshes[mesh_idx]
    var v0 = Int(mesh.vertexIndices[base_vidx])
    var v1 = Int(mesh.vertexIndices[base_vidx + 1])
    var v2 = Int(mesh.vertexIndices[base_vidx + 2])
    var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])

    var geom_normal = cross(p1 - p0, p2 - p0)
    var nlen = dot(geom_normal, geom_normal)
    if nlen > Float32(0.0):
        geom_normal = geom_normal * (Float32(1.0) / sqrt(nlen))

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)

    # For thin dielectric, always use outward-facing normal
    var entering = dot(ray_dir, geom_normal) < Float32(0.0)
    var normal = geom_normal if entering else -geom_normal
    var cos_i = max(Float32(0.0), -dot(ray_dir, normal))
    var ior = mat.albedo.r

    # Exact single-interface dielectric Fresnel, then compound the two slab
    # interfaces: R' = R + (1-R)^2 R / (1 - R^2) = 2R / (1 + R)  (PBRT thin glass).
    var r_single = fr_dielectric(cos_i, ior)
    var fresnel = r_single
    if r_single < Float32(1.0):
        fresnel = Float32(2.0) * r_single / (Float32(1.0) + r_single)

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)

    if pcg.next_float() < fresnel:
        # Reflect
        var refl = ray_dir + normal * (Float32(2.0) * cos_i)
        var rlen = dot(refl, refl)
        if rlen > Float32(0.0):
            refl = refl * (Float32(1.0) / sqrt(rlen))
        var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)
        path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(refl[0], refl[1], refl[2]))
    else:
        # Transmit — same direction, offset past the surface
        var hit_point = ray_org + ray_dir * inter.tHit - normal * Float32(0.0001)
        path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(ray_dir[0], ray_dir[1], ray_dir[2]))

    if path_ptr[].bounce == 0:
        path_ptr[].albedo = RGB(Float32(1), Float32(1), Float32(1))
    path_ptr[].bounce += 1
    if path_ptr[].bounce > 1:
        var lum = path_ptr[].throughput.luma()
        var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
        if pcg.next_float() < q:
            path_ptr[].active = 0
        else:
            path_ptr[].throughput *= Float32(1.0) / (Float32(1.0) - q)
    path_ptr[].pcgState = pcg.state
    path_ptr[].specularBounce = Int8(1)
    path_ptr[].lastBsdfPdf = Float32(0.0)


# ── Conductor (mirror + GGX microfacet) branch ────────────────────────────────
@always_inline
def shade_conductor(
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    mat: Material_C,
):
    var mesh_idx: Int
    var base_vidx: Int
    if inter.primId.type == 0:
        mesh_idx = Int(inter.primId.id1)
        base_vidx = Int(inter.primId.id2)
    elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
        mesh_idx = Int(inter.primId.id2 >> 32)
        base_vidx = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
    else:
        path_ptr[].active = 0
        return

    var mesh = meshes[mesh_idx]
    var v0 = Int(mesh.vertexIndices[base_vidx])
    var v1 = Int(mesh.vertexIndices[base_vidx + 1])
    var v2 = Int(mesh.vertexIndices[base_vidx + 2])
    var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])

    var normal = cross(p1 - p0, p2 - p0)
    var nlen = dot(normal, normal)
    if nlen > Float32(0.0):
        normal = normal * (Float32(1.0) / sqrt(nlen))

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    if dot(normal, ray_dir) > Float32(0.0):
        normal = -normal

    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)
    # Use interpolated shading normal for smooth specular reflections
    normal = _shading_normal(mesh, v0, v1, v2, inter.u, inter.v, normal)

    var alpha_x = max(mat.roughU * mat.roughU, Float32(0.0001))
    var alpha_y = max(mat.roughV * mat.roughV, Float32(0.0001))
    var is_rough = mat.roughU > Float32(0.001) or mat.roughV > Float32(0.001)

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var scatter_dir: SIMD[DType.float32, 3]
    # Per-channel conductor Fresnel reflectance (F0 = mat.albedo, brightens to
    # white at grazing via Schlick). Used directly as the throughput multiplier.
    var fresnel_rgb: RGB

    if not is_rough:
        # Perfect specular reflection
        scatter_dir = ray_dir - normal * (Float32(2.0) * dot(ray_dir, normal))
        var rlen = dot(scatter_dir, scatter_dir)
        if rlen > Float32(0.0):
            scatter_dir = scatter_dir * (Float32(1.0) / sqrt(rlen))
        var cos_i = max(Float32(0.0), -dot(ray_dir, normal))
        var one_m = Float32(1.0) - cos_i
        var schlick = one_m * one_m * one_m * one_m * one_m
        var white = RGB(Float32(1.0), Float32(1.0), Float32(1.0))
        fresnel_rgb = mat.albedo + (white - mat.albedo) * schlick
    else:
        # True anisotropic GGX VNDF (Heitz 2018)
        # Derive UV tangent frame for anisotropy direction
        var t1: SIMD[DType.float32, 3]
        var t2: SIMD[DType.float32, 3]
        if Int(mesh.uvs) > 1 and alpha_x != alpha_y:
            # UV-gradient tangent: tangent = (duv2.v*dp1 - duv1.v*dp2) / det
            var dp1 = p1 - p0; var dp2 = p2 - p0
            var u0f = mesh.uvs[v0*2]; var v0f = mesh.uvs[v0*2+1]
            var u1f = mesh.uvs[v1*2]; var v1f = mesh.uvs[v1*2+1]
            var u2f = mesh.uvs[v2*2]; var v2f = mesh.uvs[v2*2+1]
            var du1 = u1f - u0f; var dv1 = v1f - v0f
            var du2 = u2f - u0f; var dv2 = v2f - v0f
            var det = du1 * dv2 - du2 * dv1
            if det != Float32(0.0):
                var inv_det = Float32(1.0) / det
                t1 = (dp1 * dv2 - dp2 * dv1) * inv_det
                var tlen = dot(t1, t1)
                if tlen > Float32(0.0): t1 = t1 * (Float32(1.0) / sqrt(tlen))
                t2 = cross(normal, t1)
                var t2len = dot(t2, t2)
                if t2len > Float32(0.0): t2 = t2 * (Float32(1.0) / sqrt(t2len))
            else:
                var sign_n = Float32(1.0) if normal[2] >= Float32(0.0) else Float32(-1.0)
                var an = Float32(-1.0) / (sign_n + normal[2])
                var bn = normal[0] * normal[1] * an
                t1 = SIMD[DType.float32, 3](Float32(1.0) + sign_n*normal[0]*normal[0]*an, sign_n*bn, -sign_n*normal[0])
                t2 = SIMD[DType.float32, 3](bn, sign_n + normal[1]*normal[1]*an, -normal[1])
        else:
            # Frisvad arbitrary tangent frame
            var sign_n = Float32(1.0) if normal[2] >= Float32(0.0) else Float32(-1.0)
            var an = Float32(-1.0) / (sign_n + normal[2])
            var bn = normal[0] * normal[1] * an
            t1 = SIMD[DType.float32, 3](Float32(1.0) + sign_n*normal[0]*normal[0]*an, sign_n*bn, -sign_n*normal[0])
            t2 = SIMD[DType.float32, 3](bn, sign_n + normal[1]*normal[1]*an, -normal[1])

        # wo in local anisotropic frame
        var wo = -ray_dir
        var wo_x = dot(wo, t1); var wo_y = dot(wo, t2); var wo_z = dot(wo, normal)

        # Stretch wo by (alpha_x, alpha_y) — anisotropic
        var wos = SIMD[DType.float32, 3](wo_x * alpha_x, wo_y * alpha_y, wo_z)
        var wos_len = sqrt(dot(wos, wos))
        var vh = wos * (Float32(1.0) / wos_len) if wos_len > Float32(0.0) else normal

        # Orthonormal basis around vh
        var sign_vh = Float32(1.0) if vh[2] >= Float32(0.0) else Float32(-1.0)
        var av = Float32(-1.0) / (sign_vh + vh[2])
        var bv = vh[0] * vh[1] * av
        var bt1 = SIMD[DType.float32, 3](Float32(1.0) + sign_vh*vh[0]*vh[0]*av, sign_vh*bv, -sign_vh*vh[0])
        var bt2 = SIMD[DType.float32, 3](bv, sign_vh + vh[1]*vh[1]*av, -vh[1])

        # Sample visible hemisphere disk
        var u1 = pcg.next_float(); var u2 = pcg.next_float()
        var r_disk = sqrt(u1)
        var phi = TWO_PI * u2
        var tx = r_disk * cos(phi); var ty_raw = r_disk * sin(phi)
        var s_corr = Float32(0.5) * (Float32(1.0) + vh[2])
        var ty = tx * sqrt(Float32(1.0) - s_corr) + ty_raw * sqrt(s_corr)
        var tz2 = Float32(1.0) - tx*tx - ty*ty
        var tz = sqrt(tz2 if tz2 > Float32(0.0) else Float32(0.0))
        var nh_local = bt1 * tx + bt2 * ty + vh * tz

        # Unstretch with per-axis alpha — anisotropic
        var wh_local = SIMD[DType.float32, 3](alpha_x * nh_local[0], alpha_y * nh_local[1], max(Float32(0.0), nh_local[2]))
        var wh_len = dot(wh_local, wh_local)
        var wh_unit = wh_local * (Float32(1.0) / sqrt(wh_len)) if wh_len > Float32(0.0) else normal

        # Half-vector back to world space
        var wh_world = t1 * wh_unit[0] + t2 * wh_unit[1] + normal * wh_unit[2]
        var wh_wlen = dot(wh_world, wh_world)
        if wh_wlen > Float32(0.0):
            wh_world = wh_world * (Float32(1.0) / sqrt(wh_wlen))

        # Reflect wo around wh
        var wo_dot_wh = dot(wo, wh_world)
        scatter_dir = wh_world * (Float32(2.0) * wo_dot_wh) - wo
        var sd_len = dot(scatter_dir, scatter_dir)
        if sd_len > Float32(0.0):
            scatter_dir = scatter_dir * (Float32(1.0) / sqrt(sd_len))

        # Per-channel Schlick Fresnel for conductor using albedo as F0
        var cos_wh = max(Float32(0.0), wo_dot_wh)
        var one_m = Float32(1.0) - cos_wh
        var one_m2 = one_m * one_m
        var schlick = one_m2 * one_m2 * one_m
        var white = RGB(Float32(1.0), Float32(1.0), Float32(1.0))
        fresnel_rgb = mat.albedo + (white - mat.albedo) * schlick

        if dot(scatter_dir, normal) <= Float32(0.0):
            path_ptr[].active = 0
            path_ptr[].pcgState = pcg.state
            return

    path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(scatter_dir[0], scatter_dir[1], scatter_dir[2]))
    if path_ptr[].bounce == 0:
        path_ptr[].albedo = mat.albedo
    path_ptr[].throughput *= fresnel_rgb
    path_ptr[].specularBounce = Int8(1)
    path_ptr[].lastBsdfPdf = Float32(0.0)
    path_ptr[].bounce += 1

    # Russian roulette after first bounce
    if path_ptr[].bounce > 1:
        var lum = path_ptr[].throughput.luma()
        var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
        if pcg.next_float() < q:
            path_ptr[].active = 0
        else:
            path_ptr[].throughput *= Float32(1.0) / (Float32(1.0) - q)
    path_ptr[].pcgState = pcg.state


# CoatedConductor: dielectric clearcoat over GGX conductor.
# Schlick Fresnel at the air/coat interface: F_schlick(cos_theta, 0, 1, ior)
# selects between coat specular reflection (F) and conducting GGX layer (1-F).
# This is an energy-conserving two-lobe approximation of pbrt's LayeredBxDF.
@always_inline
def shade_coated_conductor(
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    mat: Material_C,
):
    var mesh_idx: Int
    var base_vidx: Int
    if inter.primId.type == 0:
        mesh_idx = Int(inter.primId.id1)
        base_vidx = Int(inter.primId.id2)
    elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
        mesh_idx = Int(inter.primId.id2 >> 32)
        base_vidx = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
    else:
        path_ptr[].active = 0
        return

    var mesh = meshes[mesh_idx]
    var v0 = Int(mesh.vertexIndices[base_vidx])
    var v1 = Int(mesh.vertexIndices[base_vidx + 1])
    var v2 = Int(mesh.vertexIndices[base_vidx + 2])
    var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])
    var normal = cross(p1 - p0, p2 - p0)
    var nlen = dot(normal, normal)
    if nlen > Float32(0.0):
        normal = normal * (Float32(1.0) / sqrt(nlen))
    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    if dot(normal, ray_dir) > Float32(0.0):
        normal = -normal
    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var cos_theta = max(Float32(0.0), -dot(ray_dir, normal))
    var ior = mat.emission.r if mat.emission.r > Float32(1.0) else Float32(1.5)

    # Schlick Fresnel approximation for coat interface (air→dielectric)
    var r0 = (ior - Float32(1.0)) / (ior + Float32(1.0))
    r0 = r0 * r0
    var one_m = Float32(1.0) - cos_theta
    var one_m2 = one_m * one_m
    var f_coat = r0 + (Float32(1.0) - r0) * one_m2 * one_m2 * one_m

    var scatter_dir: SIMD[DType.float32, 3]
    var tput_scale: RGB

    if pcg.next_float() < f_coat:
        # Coat reflection: perfect specular off the coat surface
        scatter_dir = ray_dir - normal * (Float32(2.0) * dot(ray_dir, normal))
        var sd_len = dot(scatter_dir, scatter_dir)
        if sd_len > Float32(0.0):
            scatter_dir = scatter_dir * (Float32(1.0) / sqrt(sd_len))
        tput_scale = RGB(Float32(1.0), Float32(1.0), Float32(1.0))  # coat is clear
        path_ptr[].specularBounce = Int8(1)
    else:
        # Conductor lobe: GGX VNDF (reuse conductor logic with mat.roughU/roughV)
        var alpha_u = max(mat.roughU * mat.roughU, Float32(0.0001))
        var alpha_v = max(mat.roughV * mat.roughV, Float32(0.0001))
        var alpha = (alpha_u + alpha_v) * Float32(0.5)
        var sign_n = Float32(1.0) if normal[2] >= Float32(0.0) else Float32(-1.0)
        var an = Float32(-1.0) / (sign_n + normal[2])
        var bn = normal[0] * normal[1] * an
        var t1 = SIMD[DType.float32, 3](Float32(1.0) + sign_n * normal[0]*normal[0]*an, sign_n*bn, -sign_n*normal[0])
        var t2 = SIMD[DType.float32, 3](bn, sign_n + normal[1]*normal[1]*an, -normal[1])
        var wo = -ray_dir
        var wo_x = dot(wo, t1); var wo_y = dot(wo, t2); var wo_z = dot(wo, normal)
        var wos = SIMD[DType.float32, 3](wo_x * alpha, wo_y * alpha, wo_z)
        var wos_len = sqrt(dot(wos, wos))
        var vh = wos * (Float32(1.0) / wos_len) if wos_len > Float32(0.0) else normal
        var sign_vh = Float32(1.0) if vh[2] >= Float32(0.0) else Float32(-1.0)
        var av = Float32(-1.0) / (sign_vh + vh[2])
        var bv = vh[0] * vh[1] * av
        var bt1 = SIMD[DType.float32, 3](Float32(1.0) + sign_vh*vh[0]*vh[0]*av, sign_vh*bv, -sign_vh*vh[0])
        var bt2 = SIMD[DType.float32, 3](bv, sign_vh + vh[1]*vh[1]*av, -vh[1])
        var u1 = pcg.next_float(); var u2 = pcg.next_float()
        var r_disk = sqrt(u1)
        var phi = TWO_PI * u2
        var tx = r_disk * cos(phi); var ty_raw = r_disk * sin(phi)
        var s_corr = Float32(0.5) * (Float32(1.0) + vh[2])
        var ty = tx * sqrt(Float32(1.0) - s_corr) + ty_raw * sqrt(s_corr)
        var tz2 = Float32(1.0) - tx*tx - ty*ty
        var tz = sqrt(tz2 if tz2 > Float32(0.0) else Float32(0.0))
        var nh_local = bt1 * tx + bt2 * ty + vh * tz
        var wh_local = SIMD[DType.float32, 3](alpha * nh_local[0], alpha * nh_local[1], max(Float32(0.0), nh_local[2]))
        var wh_len = dot(wh_local, wh_local)
        var wh_unit = wh_local * (Float32(1.0) / sqrt(wh_len)) if wh_len > Float32(0.0) else normal
        var wh_world = t1 * wh_unit[0] + t2 * wh_unit[1] + normal * wh_unit[2]
        var wh_wlen = dot(wh_world, wh_world)
        if wh_wlen > Float32(0.0):
            wh_world = wh_world * (Float32(1.0) / sqrt(wh_wlen))
        var wo_dot_wh = dot(wo, wh_world)
        scatter_dir = wh_world * (Float32(2.0) * wo_dot_wh) - wo
        var sd_len = dot(scatter_dir, scatter_dir)
        if sd_len > Float32(0.0):
            scatter_dir = scatter_dir * (Float32(1.0) / sqrt(sd_len))
        if dot(scatter_dir, normal) <= Float32(0.0):
            path_ptr[].active = 0
            path_ptr[].pcgState = pcg.state
            return
        var cos_wh = max(Float32(0.0), wo_dot_wh)
        var one_m3 = Float32(1.0) - cos_wh
        var one_m4 = one_m3 * one_m3
        var schlick = one_m4 * one_m4 * one_m3
        var f0_luma = mat.albedo.luma()
        var f_metal = f0_luma + (Float32(1.0) - f0_luma) * schlick
        tput_scale = mat.albedo * f_metal * (Float32(1.0) - f_coat)
        path_ptr[].specularBounce = Int8(1)

    path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(scatter_dir[0], scatter_dir[1], scatter_dir[2]))
    if path_ptr[].bounce == 0:
        path_ptr[].albedo = mat.albedo
    path_ptr[].throughput *= tput_scale
    path_ptr[].lastBsdfPdf = Float32(0.0)
    path_ptr[].bounce += 1
    if path_ptr[].bounce > 1:
        var lum = path_ptr[].throughput.luma()
        var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
        if pcg.next_float() < q:
            path_ptr[].active = 0
        else:
            path_ptr[].throughput *= Float32(1.0) / (Float32(1.0) - q)
    path_ptr[].pcgState = pcg.state


# Mix material: randomly select one of two sub-materials using amount as probability.
# Sub-material indices are packed into mat.tex_idx: low 16 bits = idx1, high 16 bits = idx2.
# mat.roughU = blend amount (probability of picking mat2).
@always_inline
def shade_mix[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    path_idx: Int,
    inter: Intersection_C,
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    infiniteLightCount: Int,
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin],
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
    shadow_tasks: UnsafePointer[ShadowTask_C, MutAnyOrigin],
    mat: Material_C,
):
    var packed = mat.tex_idx
    var idx1 = Int(packed & Int32(0xFFFF))
    var idx2 = Int((packed >> 16) & Int32(0xFFFF))
    var amount = mat.roughU  # blend factor: 0 = all mat1, 1 = all mat2
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var chosen_idx: Int
    if pcg.next_float() < amount:
        chosen_idx = idx2
    else:
        chosen_idx = idx1
    path_ptr[].pcgState = pcg.state
    # Recurse: shade with the chosen sub-material
    # Guard against self-referential mix (cycle) — if sub-mat is also mix,
    # fall through to diffuse to avoid infinite recursion.
    var sub_mat = materials[chosen_idx]
    if sub_mat.type == Int8(8):
        sub_mat.type = Int8(1)  # fallback to diffuse
    if sub_mat.type == Int8(3):
        shade_conductor(path_ptr, inter, meshes, sub_mat)
    elif sub_mat.type == Int8(4):
        shade_dielectric(path_ptr, inter, meshes, sub_mat)
    elif sub_mat.type == Int8(7):
        shade_coated_conductor(path_ptr, inter, meshes, sub_mat)
    elif sub_mat.type == Int8(5):
        shade_coated_diffuse[use_gpu, enqueue_shadow](
            path_ptr, path_idx, inter, bvh2Nodes, primIds, meshes, sub_mat,
            areaLights, areaLightCount, infiniteLights, infiniteLightCount,
            tex_filenames, textures, n_textures, shadow_tasks)
    elif sub_mat.type == Int8(6):
        shade_diffuse_transmission[use_gpu, enqueue_shadow](
            path_ptr, path_idx, inter, bvh2Nodes, primIds, meshes, materials,
            areaLights, areaLightCount, tex_filenames, textures, n_textures, shadow_tasks)
    else:
        # Diffuse fallback (type 1 or unknown)
        sub_mat.type = Int8(1)
        # re-use diffuse path via a minimal stub
        path_ptr[].active = Int8(0)  # will be reset by diffuse path — set inactive, diffuse handles restart
        # Actually just let shade_nee_core handle it below; override type in materials array would
        # corrupt shared data. Instead, manually inline a simple lambertian bounce.
        var mesh_idx: Int
        var base_vidx: Int
        if inter.primId.type == 0:
            mesh_idx = Int(inter.primId.id1)
            base_vidx = Int(inter.primId.id2)
        elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
            mesh_idx = Int(inter.primId.id2 >> 32)
            base_vidx = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
        else:
            path_ptr[].active = 0
            return
        var mesh = meshes[mesh_idx]
        var v0 = Int(mesh.vertexIndices[base_vidx])
        var v1 = Int(mesh.vertexIndices[base_vidx + 1])
        var v2 = Int(mesh.vertexIndices[base_vidx + 2])
        var pp0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
        var pp1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
        var pp2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])
        var normal = cross(pp1 - pp0, pp2 - pp0)
        var nlen = dot(normal, normal)
        if nlen > Float32(0.0):
            normal = normal * (Float32(1.0) / sqrt(nlen))
        var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
        if dot(normal, ray_dir) > Float32(0.0):
            normal = -normal
        var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
        var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)
        var pcg2 = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
        var u1 = pcg2.next_float(); var u2 = pcg2.next_float()
        var phi = TWO_PI * u2
        var st = sqrt(u1)
        var ct = sqrt(Float32(1.0) - u1)
        var sign_n2 = Float32(1.0) if normal[2] >= Float32(0.0) else Float32(-1.0)
        var an2 = Float32(-1.0) / (sign_n2 + normal[2])
        var bn2 = normal[0] * normal[1] * an2
        var t1b = SIMD[DType.float32, 3](Float32(1.0) + sign_n2*normal[0]*normal[0]*an2, sign_n2*bn2, -sign_n2*normal[0])
        var t2b = SIMD[DType.float32, 3](bn2, sign_n2 + normal[1]*normal[1]*an2, -normal[1])
        var lx = cos(phi) * st; var ly = sin(phi) * st
        var scatter_dir = t1b * lx + t2b * ly + normal * ct
        var sd_len = dot(scatter_dir, scatter_dir)
        if sd_len > Float32(0.0):
            scatter_dir = scatter_dir * (Float32(1.0) / sqrt(sd_len))
        path_ptr[].active = Int8(1)
        path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(scatter_dir[0], scatter_dir[1], scatter_dir[2]))
        if path_ptr[].bounce == 0:
            path_ptr[].albedo = sub_mat.albedo
        path_ptr[].throughput *= sub_mat.albedo
        path_ptr[].specularBounce = Int8(0)
        path_ptr[].lastBsdfPdf = INV_PI
        path_ptr[].bounce += 1
        if path_ptr[].bounce > 1:
            var lum = path_ptr[].throughput.luma()
            var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
            if pcg2.next_float() < q:
                path_ptr[].active = 0
            else:
                path_ptr[].throughput *= Float32(1.0) / (Float32(1.0) - q)
        path_ptr[].pcgState = pcg2.state

# Apply tangent-space normal map: samples the texture, decodes [-1,1] normal,
# rotates it into world space via UV-gradient tangent frame.
# Returns geom_normal unchanged when normal_tex_idx < 0 or no UVs.
@always_inline
def _apply_normal_map[use_gpu: Bool](
    mat: Material_C,
    v0: Int, v1: Int, v2: Int,
    mesh: TriangleMesh_C,
    inter: Intersection_C,
    geom_normal: SIMD[DType.float32, 3],
    p0: SIMD[DType.float32, 3],
    p1: SIMD[DType.float32, 3],
    p2: SIMD[DType.float32, 3],
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin],
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
) -> SIMD[DType.float32, 3]:
    if mat.normal_tex_idx < Int32(0) or Int(mesh.uvs) <= 1:
        return geom_normal
    # Compute barycentric UV coordinates
    var dp1 = p1 - p0; var dp2 = p2 - p0
    var u0f = mesh.uvs[v0*2]; var v0f = mesh.uvs[v0*2+1]
    var u1f = mesh.uvs[v1*2]; var v1f = mesh.uvs[v1*2+1]
    var u2f = mesh.uvs[v2*2]; var v2f = mesh.uvs[v2*2+1]
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
    var ns = sample_texture[use_gpu](Int(mat.normal_tex_idx), uv_u, uv_v, True, tex_filenames, textures, n_textures, found)
    if found:
        var nx = ns.r * Float32(2.0) - Float32(1.0)
        var ny = ns.g * Float32(2.0) - Float32(1.0)
        var nz = ns.b * Float32(2.0) - Float32(1.0)
        # Rotate tangent-space normal to world space
        var world_n = tangent * nx + bitangent * ny + geom_normal * nz
        var wn_len = dot(world_n, world_n)
        if wn_len > Float32(0.0):
            return world_n * (Float32(1.0) / sqrt(wn_len))
    return geom_normal


# Unified NEE core — comptime-specialized for CPU (use_gpu=False) and GPU (use_gpu=True).
# Texture lookup uses OIIO external_call on CPU and device-resident GpuTexture_C on GPU.
@always_inline
def shade_nee_core[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    path_idx: Int,
    inter: Intersection_C,
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int,
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin],
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
    shadow_tasks: UnsafePointer[ShadowTask_C, MutAnyOrigin],
    distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin],
    distantLightCount: Int,
    pointLights: UnsafePointer[PointLight_C, MutAnyOrigin],
    pointLightCount: Int,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    infiniteLightCount: Int,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    sphereCount: Int,
):
    # ── Miss handler: ray escaped — add infinite light and deactivate ──────────
    if inter.hit == 0:
        var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
        for inf_i in range(infiniteLightCount):
            var ilight = infiniteLights[inf_i]
            # Transform world-space ray direction into light's local frame
            var w2l = ilight.world_to_light
            var ld_x = w2l[0]*ray_dir[0] + w2l[4]*ray_dir[1] + w2l[8]*ray_dir[2]
            var ld_y = w2l[1]*ray_dir[0] + w2l[5]*ray_dir[1] + w2l[9]*ray_dir[2]
            var ld_z = w2l[2]*ray_dir[0] + w2l[6]*ray_dir[1] + w2l[10]*ray_dir[2]
            var local_dir = SIMD[DType.float32, 3](ld_x, ld_y, ld_z)
            var env_rgb: RGB
            if ilight.tex_idx >= Int32(0) and Int(ilight.pixels_ptr) > 1 and ilight.cdf_w > Int32(0):
                # Bilinear lookup in GPU/CPU-resident pixels (cdf_w × cdf_h, 3 floats/pixel)
                var iw = Int(ilight.cdf_w); var ih = Int(ilight.cdf_h)
                var ea_uv = _equal_area_sphere_to_square(local_dir[0], local_dir[1], local_dir[2])
                var u = ea_uv[0]; var v = ea_uv[1]
                var fx = u * Float32(iw) - Float32(0.5)
                var fy = v * Float32(ih) - Float32(0.5)
                var x0 = Int(max(Float32(0), min(Float32(iw - 1), floor(fx))))
                var y0 = Int(max(Float32(0), min(Float32(ih - 1), floor(fy))))
                var x1 = min(x0 + 1, iw - 1)
                var y1 = min(y0 + 1, ih - 1)
                var wx = fx - Float32(x0); var wy = fy - Float32(y0)
                var r00 = ilight.pixels_ptr[(y0*iw+x0)*3+0]; var g00 = ilight.pixels_ptr[(y0*iw+x0)*3+1]; var b00 = ilight.pixels_ptr[(y0*iw+x0)*3+2]
                var r10 = ilight.pixels_ptr[(y0*iw+x1)*3+0]; var g10 = ilight.pixels_ptr[(y0*iw+x1)*3+1]; var b10 = ilight.pixels_ptr[(y0*iw+x1)*3+2]
                var r01 = ilight.pixels_ptr[(y1*iw+x0)*3+0]; var g01 = ilight.pixels_ptr[(y1*iw+x0)*3+1]; var b01 = ilight.pixels_ptr[(y1*iw+x0)*3+2]
                var r11 = ilight.pixels_ptr[(y1*iw+x1)*3+0]; var g11 = ilight.pixels_ptr[(y1*iw+x1)*3+1]; var b11 = ilight.pixels_ptr[(y1*iw+x1)*3+2]
                var tr = (Float32(1)-wx)*(Float32(1)-wy)*r00 + wx*(Float32(1)-wy)*r10 + (Float32(1)-wx)*wy*r01 + wx*wy*r11
                var tg = (Float32(1)-wx)*(Float32(1)-wy)*g00 + wx*(Float32(1)-wy)*g10 + (Float32(1)-wx)*wy*g01 + wx*wy*g11
                var tb = (Float32(1)-wx)*(Float32(1)-wy)*b00 + wx*(Float32(1)-wy)*b10 + (Float32(1)-wx)*wy*b01 + wx*wy*b11
                env_rgb = RGB(tr, tg, tb) * ilight.scale
            else:
                comptime if not use_gpu:
                    if ilight.tex_idx >= Int32(0):
                        var fname = tex_filenames[Int(ilight.tex_idx)]
                        var ea_uv2 = _equal_area_sphere_to_square(local_dir[0], local_dir[1], local_dir[2])
                        var u = ea_uv2[0]; var v = ea_uv2[1]
                        var tr = alloc[Float32](3)
                        tr[0] = Float32(0.0); tr[1] = Float32(0.0); tr[2] = Float32(0.0)
                        _ = external_call["texture", Bool,
                            UnsafePointer[UInt8, MutAnyOrigin], Float32, Float32,
                            UnsafePointer[Float32, MutAnyOrigin]](fname, u, v, tr)
                        env_rgb = RGB(tr[0], tr[1], tr[2]) * ilight.scale
                        tr.free()
                    else:
                        env_rgb = ilight.scale
                else:
                    env_rgb = ilight.scale
            var mis_weight = Float32(1.0)
            if path_ptr[].specularBounce == Int8(0) and path_ptr[].bounce > 0:
                var pdf_bsdf = path_ptr[].lastBsdfPdf
                var pdf_light = INV_FOUR_PI
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
                        var dp_row = ilight.cdf_ptr[py + 1] - ilight.cdf_ptr[py]
                        var dp_col_base = row_cdf_base + px
                        var dp_col = ilight.cdf_ptr[dp_col_base + 1] - ilight.cdf_ptr[dp_col_base]
                        if dp_row > Float32(0.0):
                            pdf_light = dp_row * dp_col * Float32(iw) * Float32(ih) * INV_FOUR_PI
                mis_weight = power_heuristic(pdf_bsdf, pdf_light)
            path_ptr[].estimate += path_ptr[].throughput * env_rgb * mis_weight
        path_ptr[].active = 0
        return

    var mat = materials[Int(inter.primId.materialIndex)]

    # ── Analytical sphere hit: collect emission and terminate (Null material) ──
    if inter.primId.type == Int8(4) and sphereCount > 0:
        var sph_idx = Int(inter.primId.id1)
        var sph = spheres[sph_idx]
        if sph.isAreaLight == Int8(1):
            if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
                path_ptr[].estimate += path_ptr[].throughput * sph.emission
            else:
                # MIS: BSDF pdf vs sphere solid-angle pdf from previous shading point
                var pdf_bsdf = path_ptr[].lastBsdfPdf
                var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
                var to_c_x = sph.center.x - ray_org[0]
                var to_c_y = sph.center.y - ray_org[1]
                var to_c_z = sph.center.z - ray_org[2]
                var dc_sq = to_c_x*to_c_x + to_c_y*to_c_y + to_c_z*to_c_z
                var sin2_max = sph.radius * sph.radius / dc_sq
                if sin2_max < Float32(1.0) and pdf_bsdf > Float32(0.0):
                    var cos_max = sqrt(Float32(1.0) - sin2_max)
                    var solid_angle = TWO_PI * (Float32(1.0) - cos_max)
                    var pdf_light = Float32(1.0) / (solid_angle * Float32(max(sphereCount, 1)))
                    var w = power_heuristic(pdf_bsdf, pdf_light)
                    path_ptr[].estimate += path_ptr[].throughput * sph.emission * w
        path_ptr[].active = 0
        return

    if inter.primId.type == Int8(3):
        # Area light triangle hit — use emission from AreaLight_C directly so
        # NamedMaterial area lights (mat.type == 1) also emit correctly.
        var al_idx = Int(inter.primId.id1)
        var al = areaLights[al_idx]
        var emission = al.emission
        if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
            path_ptr[].estimate += path_ptr[].throughput * emission
        else:
            var pdf_bsdf = path_ptr[].lastBsdfPdf
            if pdf_bsdf > Float32(0.0):
                var lmesh_idx = Int(inter.primId.id2 >> 32)
                var lbase   = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
                var lmesh   = meshes[lmesh_idx]
                var lv0 = Int(lmesh.vertexIndices[lbase])
                var lv1 = Int(lmesh.vertexIndices[lbase + 1])
                var lv2 = Int(lmesh.vertexIndices[lbase + 2])
                var lp0 = SIMD[DType.float32, 3](lmesh.points[lv0*4], lmesh.points[lv0*4+1], lmesh.points[lv0*4+2])
                var lp1 = SIMD[DType.float32, 3](lmesh.points[lv1*4], lmesh.points[lv1*4+1], lmesh.points[lv1*4+2])
                var lp2 = SIMD[DType.float32, 3](lmesh.points[lv2*4], lmesh.points[lv2*4+1], lmesh.points[lv2*4+2])
                var lnorm = cross(lp1 - lp0, lp2 - lp0)
                var lnlen = dot(lnorm, lnorm)
                if lnlen > Float32(0.0):
                    lnorm = lnorm * (Float32(1.0) / sqrt(lnlen))
                var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
                var cos_l = -dot(lnorm, ray_dir)
                var dist  = inter.tHit
                var dist2 = dist * dist
                if cos_l > Float32(0.0) and al.total_area > Float32(0.0):
                    var pdf_light = dist2 / (cos_l * al.total_area * Float32(areaLightCount))
                    var w = power_heuristic(pdf_bsdf, pdf_light)
                    path_ptr[].estimate += path_ptr[].throughput * emission * w
        path_ptr[].active = 0
        return

    if mat.type == 3:
        shade_conductor(path_ptr, inter, meshes, mat)
        return

    if mat.type == 4:
        shade_dielectric(path_ptr, inter, meshes, mat)
        return

    if mat.type == 5:
        shade_coated_diffuse[use_gpu, enqueue_shadow](path_ptr, path_idx, inter, bvh2Nodes, primIds, meshes, mat, areaLights, areaLightCount, infiniteLights, infiniteLightCount, tex_filenames, textures, n_textures, shadow_tasks)
        return

    if mat.type == 6:
        shade_diffuse_transmission[use_gpu, enqueue_shadow](
            path_ptr, path_idx, inter, bvh2Nodes, primIds, meshes, materials,
            areaLights, areaLightCount, tex_filenames, textures, n_textures, shadow_tasks)
        return

    if mat.type == 7:
        shade_coated_conductor(path_ptr, inter, meshes, mat)
        return

    if mat.type == 8:
        shade_mix[use_gpu, enqueue_shadow](
            path_ptr, path_idx, inter, bvh2Nodes, primIds, meshes, materials,
            areaLights, areaLightCount, infiniteLights, infiniteLightCount,
            tex_filenames, textures, n_textures, shadow_tasks, mat)
        return

    if mat.type == 9:
        shade_thin_dielectric(path_ptr, inter, meshes, mat)
        return

    if mat.type != 1:
        path_ptr[].active = 0
        return

    var mesh_idx: Int
    var base_vidx: Int
    if inter.primId.type == 0:
        mesh_idx = Int(inter.primId.id1)
        base_vidx = Int(inter.primId.id2)
    elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
        mesh_idx = Int(inter.primId.id2 >> 32)
        base_vidx = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
    else:
        path_ptr[].active = 0
        return

    var mesh = meshes[mesh_idx]
    var v0 = Int(mesh.vertexIndices[base_vidx])
    var v1 = Int(mesh.vertexIndices[base_vidx + 1])
    var v2 = Int(mesh.vertexIndices[base_vidx + 2])
    var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])
    var normal = cross(p1 - p0, p2 - p0)
    var nlen = dot(normal, normal)
    if nlen > Float32(0.0):
        normal = normal * (Float32(1.0) / sqrt(nlen))

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    if dot(normal, ray_dir) > Float32(0.0):
        normal = -normal
    # Geometric normal oriented to the camera side; the shading/bumped normal is
    # kept on this side below (NOT flipped to the view ray).
    var ng_ff = normal

    # Use interpolated shading normal as the base for smooth diffuse shading
    normal = _shading_normal(mesh, v0, v1, v2, inter.u, inter.v, normal)

    # Apply normal map if present
    normal = _apply_normal_map[use_gpu](mat, v0, v1, v2, mesh, inter, normal, p0, p1, p2,
        tex_filenames, textures, n_textures)
    # Faceforward the bumped normal to the geometric normal — never to the view
    # ray. Flipping to the ray inverts bumps that tilt away from the camera at
    # grazing angles, washing out the relief (pbrt faceforwards to Ng).
    if dot(normal, ng_ff) < Float32(0.0):
        normal = -normal

    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)

    var alb = _tex_lookup[use_gpu](mat, inter, v0, v1, v2, mesh, tex_filenames, textures, n_textures)

    # pbrt's diffuse BRDF is zero when the viewer and the lit direction are in
    # opposite hemispheres of the SHADING normal (SameHemisphere). A normal map
    # can tilt the shading normal past the viewer at grazing angles; those faces
    # reflect no direct light (without this, the away side of each bump is wrongly
    # lit — the top/bottom split across each bead). They still receive ambient
    # from a uniform environment light (albedo * L), which is pbrt's dim grey
    # there — so add that rather than going black, then stop.
    if dot(normal, ray_dir) >= Float32(0.0):
        for inf_i in range(infiniteLightCount):
            var il = infiniteLights[inf_i]
            if il.tex_idx < Int32(0):
                path_ptr[].estimate += path_ptr[].throughput * (alb * il.scale)
        path_ptr[].active = 0
        return

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)

    if areaLightCount > 0:
        var light_idx = Int(pcg.next_uint() % UInt32(areaLightCount))
        var al = areaLights[light_idx]
        var lmesh = meshes[Int(al.meshIdx)]
        var lti = Int(pcg.next_uint() % UInt32(max(Int(al.n_tris), 1)))
        var lb = lti * 3
        var lv0 = Int(lmesh.vertexIndices[lb])
        var lv1 = Int(lmesh.vertexIndices[lb + 1])
        var lv2 = Int(lmesh.vertexIndices[lb + 2])
        var lp0 = SIMD[DType.float32, 3](lmesh.points[lv0*4], lmesh.points[lv0*4+1], lmesh.points[lv0*4+2])
        var lp1 = SIMD[DType.float32, 3](lmesh.points[lv1*4], lmesh.points[lv1*4+1], lmesh.points[lv1*4+2])
        var lp2 = SIMD[DType.float32, 3](lmesh.points[lv2*4], lmesh.points[lv2*4+1], lmesh.points[lv2*4+2])
        var r1 = pcg.next_float()
        var r2 = pcg.next_float()
        var sqrt_r1 = sqrt(r1)
        var light_point = lp0 * (Float32(1.0) - sqrt_r1) + lp1 * (sqrt_r1 * (Float32(1.0) - r2)) + lp2 * (sqrt_r1 * r2)
        var lcross = cross(lp1 - lp0, lp2 - lp0)
        var light_normal = lcross
        var lcross_len = dot(lcross, lcross)
        if lcross_len > Float32(0.0):
            light_normal = lcross * (Float32(1.0) / sqrt(lcross_len))
        var to_light = light_point - hit_point
        var dist_sq = dot(to_light, to_light)
        var dist = sqrt(dist_sq)
        if dist > Float32(0.0001) and al.total_area > Float32(0.0):
            var shadow_dir = to_light * (Float32(1.0) / dist)
            var cos_s = dot(normal, shadow_dir)
            var cos_l = -dot(light_normal, shadow_dir)
            if cos_s > Float32(0.0) and cos_l > Float32(0.0):
                var pdf_light = dist_sq / (cos_l * al.total_area * Float32(areaLightCount))
                var pi = PI
                var pdf_bsdf_nee = cos_s / pi
                var w_nee = power_heuristic(pdf_light, pdf_bsdf_nee)
                var weight = alb * al.emission * (cos_s * w_nee / (pdf_light * pi))
                var contrib = path_ptr[].throughput * weight
                comptime if enqueue_shadow:
                    shadow_tasks[path_idx] = ShadowTask_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]), dist * Float32(0.9999), RGB(contrib.r, contrib.g, contrib.b), Int32(1), Int32(0))
                else:
                    var shadow_ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
                    if not any_hit_bvh2_core(bvh2Nodes, primIds, meshes, shadow_ray, dist * Float32(0.9999)):
                        path_ptr[].estimate += contrib

    # ── Distant light NEE (delta light: MIS weight = 1) ──────────────────────
    for dl_i in range(distantLightCount):
        var dl = distantLights[dl_i]
        var ldir = SIMD[DType.float32, 3](dl.direction.x, dl.direction.y, dl.direction.z)  # direction toward scene (away from light)
        var to_light = -ldir  # direction from hit point toward the light
        var cos_s = dot(normal, to_light)
        if cos_s > Float32(0.0):
            # f = alb/pi, no geometry term (parallel rays), pdf = delta -> weight = 1
            var pi = PI
            var contrib = path_ptr[].throughput * alb * dl.emission * (cos_s / pi)
            # Shadow ray at very long distance (scene diameter ~1000)
            var t_max = Float32(2000.0)
            comptime if enqueue_shadow:
                shadow_tasks[path_idx] = ShadowTask_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(to_light[0], to_light[1], to_light[2]), t_max, RGB(contrib.r, contrib.g, contrib.b), Int32(1), Int32(0))
            else:
                var shadow_ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(to_light[0], to_light[1], to_light[2]))
                if not any_hit_bvh2_core(bvh2Nodes, primIds, meshes, shadow_ray, t_max):
                    path_ptr[].estimate += contrib

    # ── Point light NEE (delta light: MIS weight = 1) ────────────────────────
    for pl_i in range(pointLightCount):
        var pl = pointLights[pl_i]
        var lpos = SIMD[DType.float32, 3](pl.position.x, pl.position.y, pl.position.z)
        var to_light = lpos - hit_point
        var dist_sq = dot(to_light, to_light)
        var dist = sqrt(dist_sq)
        if dist > Float32(0.0001):
            var ldir = to_light * (Float32(1.0) / dist)
            var cos_s = dot(normal, ldir)
            if cos_s > Float32(0.0):
                var pi = PI
                # f = alb/pi, geometry = cos_s, pdf = delta -> weight = 1
                # radiance = intensity / dist²
                var contrib = path_ptr[].throughput * alb * pl.intensity * (cos_s / (pi * dist_sq))
                comptime if enqueue_shadow:
                    shadow_tasks[path_idx] = ShadowTask_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(ldir[0], ldir[1], ldir[2]), dist * Float32(0.9999), RGB(contrib.r, contrib.g, contrib.b), Int32(1), Int32(0))
                else:
                    var shadow_ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(ldir[0], ldir[1], ldir[2]))
                    if not any_hit_bvh2_core(bvh2Nodes, primIds, meshes, shadow_ray, dist * Float32(0.9999)):
                        path_ptr[].estimate += contrib

    # ── Sphere light NEE (solid-angle cone sampling) ──────────────────────────
    for sph_i in range(sphereCount):
        var sph = spheres[sph_i]
        if sph.isAreaLight == Int8(0):
            continue
        var to_cx = sph.center.x - hit_point[0]
        var to_cy = sph.center.y - hit_point[1]
        var to_cz = sph.center.z - hit_point[2]
        var dc_sq = to_cx*to_cx + to_cy*to_cy + to_cz*to_cz
        var dc = sqrt(dc_sq)
        if dc < sph.radius or dc_sq == Float32(0.0):
            continue
        var sin2_max = sph.radius * sph.radius / dc_sq
        if sin2_max >= Float32(1.0):
            continue
        var cos_max = sqrt(Float32(1.0) - sin2_max)
        # Build local ONB with z-axis toward sphere center
        var zc = SIMD[DType.float32, 3](to_cx / dc, to_cy / dc, to_cz / dc)
        var xc: SIMD[DType.float32, 3]
        if (zc[0] if zc[0] >= Float32(0.0) else -zc[0]) < Float32(0.9):
            xc = cross(zc, SIMD[DType.float32, 3](Float32(1), Float32(0), Float32(0)))
        else:
            xc = cross(zc, SIMD[DType.float32, 3](Float32(0), Float32(1), Float32(0)))
        var xlen = dot(xc, xc)
        if xlen > Float32(0.0):
            xc = xc * (Float32(1.0) / sqrt(xlen))
        var yc = cross(zc, xc)
        # Uniform cone sampling toward sphere
        var r1s = pcg.next_float()
        var r2s = pcg.next_float()
        var cos_th = Float32(1.0) - r1s * (Float32(1.0) - cos_max)
        var sin_th = sqrt(max(Float32(0.0), Float32(1.0) - cos_th * cos_th))
        var phis = TWO_PI * r2s
        var shadow_dir = xc * (sin_th * cos(phis)) + yc * (sin_th * sin(phis)) + zc * cos_th
        var sdlen = dot(shadow_dir, shadow_dir)
        if sdlen > Float32(0.0):
            shadow_dir = shadow_dir * (Float32(1.0) / sqrt(sdlen))
        var cos_s = dot(normal, shadow_dir)
        if cos_s > Float32(0.0):
            var solid_angle = TWO_PI * (Float32(1.0) - cos_max)
            var n_sphere_lights = Float32(max(sphereCount, 1))
            var pdf_light = Float32(1.0) / (solid_angle * n_sphere_lights)
            var pdf_bsdf_nee = cos_s / PI
            var w_nee = power_heuristic(pdf_light, pdf_bsdf_nee)
            var weight = alb * sph.emission * (cos_s * w_nee / (pdf_light * PI))
            var contrib = path_ptr[].throughput * weight
            comptime if enqueue_shadow:
                shadow_tasks[path_idx] = ShadowTask_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]), dc * Float32(0.9999), RGB(contrib.r, contrib.g, contrib.b), Int32(1), Int32(0))
            else:
                var shadow_ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
                if not any_hit_bvh2_core(bvh2Nodes, primIds, meshes, shadow_ray, dc * Float32(0.9999)):
                    path_ptr[].estimate += contrib

    # ── Infinite (env-map) light NEE ──────────────────────────────────────────
    for inf_i in range(infiniteLightCount):
        var ilight = infiniteLights[inf_i]
        var w2l = ilight.world_to_light
        var env_dir: SIMD[DType.float32, 3]
        var env_rgb: RGB
        var pdf_light: Float32

        if ilight.tex_idx >= Int32(0) and Int(ilight.pixels_ptr) > 1 and Int(ilight.cdf_ptr) > 1 and ilight.cdf_w > Int32(0):
            # ── CDF importance sampling ──────────────────────────────────────
            var iw = Int(ilight.cdf_w); var ih = Int(ilight.cdf_h)
            var u1_env = pcg.next_float()
            var u2_env = pcg.next_float()

            # 1. Sample row from marginal CDF
            var row_idx = _lower_bound(ilight.cdf_ptr, 0, ih, u1_env)
            row_idx = min(row_idx, ih - 1)
            var dp_row = ilight.cdf_ptr[row_idx + 1] - ilight.cdf_ptr[row_idx]

            # 2. Sample column from conditional CDF for this row
            var cond_base = (ih + 1) + row_idx * (iw + 1)
            var col_idx = _lower_bound(ilight.cdf_ptr, cond_base, cond_base + iw, u2_env) - cond_base
            col_idx = min(col_idx, iw - 1)
            var dp_col = ilight.cdf_ptr[cond_base + col_idx + 1] - ilight.cdf_ptr[cond_base + col_idx]

            # 3. Texel center UV → light-space direction
            var sample_u = (Float32(col_idx) + Float32(0.5)) / Float32(iw)
            var sample_v = (Float32(row_idx) + Float32(0.5)) / Float32(ih)
            var local_d = _equal_area_square_to_sphere(sample_u, sample_v)

            # 4. Rotate to world space: L2W = transpose(W2L) for pure rotation
            var wd_x = w2l[0]*local_d[0] + w2l[1]*local_d[1] + w2l[2]*local_d[2]
            var wd_y = w2l[4]*local_d[0] + w2l[5]*local_d[1] + w2l[6]*local_d[2]
            var wd_z = w2l[8]*local_d[0] + w2l[9]*local_d[1] + w2l[10]*local_d[2]
            env_dir = SIMD[DType.float32, 3](wd_x, wd_y, wd_z)

            # 5. Lookup env-map value at sampled pixel
            var px = min(iw - 1, max(0, Int(sample_u * Float32(iw))))
            var py = min(ih - 1, max(0, Int(sample_v * Float32(ih))))
            var pr = ilight.pixels_ptr[(py*iw+px)*3+0]
            var pg = ilight.pixels_ptr[(py*iw+px)*3+1]
            var pb = ilight.pixels_ptr[(py*iw+px)*3+2]
            env_rgb = RGB(pr, pg, pb) * ilight.scale

            # 6. PDF in solid angle: dp_row * dp_col * (iw * ih) / (4π)
            if dp_row > Float32(0) and dp_col > Float32(0):
                pdf_light = dp_row * dp_col * Float32(iw * ih) * INV_FOUR_PI
            else:
                pdf_light = INV_FOUR_PI
        else:
            # ── Fallback: cosine-weighted hemisphere (no CDF) ─────────────────
            var u1_env = pcg.next_float()
            var u2_env = pcg.next_float()
            var r_env = sqrt(u1_env)
            var theta_env = TWO_PI * u2_env
            var x_env = r_env * cos(theta_env)
            var y_env = r_env * sin(theta_env)
            var z2_env = Float32(1.0) - u1_env
            var z_env = sqrt(z2_env if z2_env > Float32(0.0) else Float32(0.0))
            var sign_env = Float32(1.0) if normal[2] >= Float32(0.0) else Float32(-1.0)
            var a_env = Float32(-1.0) / (sign_env + normal[2])
            var b_env = normal[0] * normal[1] * a_env
            var tangent_env = SIMD[DType.float32, 3](Float32(1.0) + sign_env * normal[0] * normal[0] * a_env, sign_env * b_env, -sign_env * normal[0])
            var bitangent_env = SIMD[DType.float32, 3](b_env, sign_env + normal[1] * normal[1] * a_env, -normal[1])
            env_dir = tangent_env * x_env + bitangent_env * y_env + normal * z_env
            var env_dlen = dot(env_dir, env_dir)
            if env_dlen > Float32(0.0):
                env_dir = env_dir * (Float32(1.0) / sqrt(env_dlen))
            env_rgb = ilight.scale
            pdf_light = INV_FOUR_PI

        # ── MIS + shadow ray ──────────────────────────────────────────────────
        var cos_env = dot(normal, env_dir)
        if cos_env > Float32(0.0) and not env_rgb.is_black() and pdf_light > Float32(0.0):
            var pdf_bsdf_nee = cos_env / PI
            var mis_w = power_heuristic(pdf_light, pdf_bsdf_nee)
            var contrib = path_ptr[].throughput * alb * env_rgb * (cos_env / (PI * pdf_light)) * mis_w
            var t_max_env = Float32(100000.0)
            comptime if enqueue_shadow:
                shadow_tasks[path_idx] = ShadowTask_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(env_dir[0], env_dir[1], env_dir[2]), t_max_env, RGB(contrib.r, contrib.g, contrib.b), Int32(1), Int32(0))
            else:
                var shadow_ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(env_dir[0], env_dir[1], env_dir[2]))
                if not any_hit_bvh2_core(bvh2Nodes, primIds, meshes, shadow_ray, t_max_env):
                    path_ptr[].estimate += contrib


    var u1 = pcg.next_float()
    var u2 = pcg.next_float()
    var r = sqrt(u1)
    var theta = TWO_PI * u2
    var x = r * cos(theta)
    var y = r * sin(theta)
    var z2 = Float32(1.0) - u1
    var z = sqrt(z2 if z2 > Float32(0.0) else Float32(0.0))
    var sign = Float32(1.0) if normal[2] >= Float32(0.0) else Float32(-1.0)
    var a = Float32(-1.0) / (sign + normal[2])
    var b = normal[0] * normal[1] * a
    var tangent = SIMD[DType.float32, 3](Float32(1.0) + sign * normal[0] * normal[0] * a, sign * b, -sign * normal[0])
    var bitangent = SIMD[DType.float32, 3](b, sign + normal[1] * normal[1] * a, -normal[1])
    var dir = tangent * x + bitangent * y + normal * z
    var dlen = dot(dir, dir)
    if dlen > Float32(0.0):
        dir = dir * (Float32(1.0) / sqrt(dlen))

    path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(dir[0], dir[1], dir[2]))
    if path_ptr[].bounce == 0:
        path_ptr[].albedo = alb
    # Store BSDF pdf for MIS weighting if the next bounce hits an emitter.
    # For cosine-weighted hemisphere: pdf = cos(theta) / pi = dot(dir, normal) / pi
    var cos_scatter = dot(dir, normal)
    path_ptr[].lastBsdfPdf = (cos_scatter if cos_scatter > Float32(0.0) else Float32(0.0)) / PI
    path_ptr[].specularBounce = Int8(0)
    path_ptr[].throughput *= alb
    path_ptr[].bounce += 1

    if path_ptr[].bounce > 1:
        var lum = path_ptr[].throughput.luma()
        var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
        if pcg.next_float() < q:
            path_ptr[].active = 0
        else:
            path_ptr[].throughput *= Float32(1.0) / (Float32(1.0) - q)

    path_ptr[].pcgState = pcg.state


@always_inline
def shade_core_cpu_nee(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int,
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin],
    tid: Int,
    distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin],
    distantLightCount: Int,
    pointLights: UnsafePointer[PointLight_C, MutAnyOrigin],
    pointLightCount: Int,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    infiniteLightCount: Int,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    sphereCount: Int,
):
    var path_ptr = paths + tid
    if path_ptr[].active == 0:
        return
    var inter = intersections[tid]
    shade_nee_core[False, False](path_ptr, tid, inter, bvh2Nodes, primIds, meshes, materials, areaLights, areaLightCount,
        tex_filenames, UnsafePointer[GpuTexture_C, MutAnyOrigin].unsafe_dangling(), 0,
        UnsafePointer[ShadowTask_C, MutAnyOrigin].unsafe_dangling(),
        distantLights, distantLightCount, pointLights, pointLightCount,
        infiniteLights, infiniteLightCount,
        spheres, sphereCount)
