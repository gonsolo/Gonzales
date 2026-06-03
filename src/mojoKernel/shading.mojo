from std.math import sqrt, cos, sin, floor
from std.ffi import external_call
from std.memory import alloc
from .geometry import RGB, Ray_C, Intersection_C, PrimId_C, TriangleMesh_C, Material_C, AreaLight_C, PathState_C, GpuTexture_C, dot, cross
from .rng import PCG32
from .bvh import BVH2Node, SceneDescriptor2_C, any_hit_bvh2_core

@always_inline
fn _srgb_to_linear(c: Float32) -> Float32:
    if c <= Float32(0.04045):
        return c / Float32(12.92)
    else:
        return Float32(((c + Float32(0.055)) / Float32(1.055)) ** Float32(2.4))

@always_inline
fn _sample_tex(tex: GpuTexture_C, u: Float32, v: Float32) -> RGB:
    var tw = Int(tex.width); var th = Int(tex.height)
    var s = u - Float32(Int(u))
    if s < Float32(0.0): s += Float32(1.0)
    var t = v - Float32(Int(v))
    if t < Float32(0.0): t += Float32(1.0)
    var px = min(Int(s * Float32(tw)), tw - 1)
    var py = min(Int(t * Float32(th)), th - 1)
    var idx = (py * tw + px) * 3
    return RGB(tex.data[idx], tex.data[idx+1], tex.data[idx+2])

@always_inline
fn _tex_lookup[use_gpu: Bool](
    mat: Material_C,
    inter: Intersection_C,
    v0: Int, v1: Int, v2: Int,
    mesh: TriangleMesh_C,
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin],
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
) -> RGB:
    var ti = Int(mat.tex_idx)
    @parameter
    if use_gpu:
        if ti >= 0 and ti < n_textures:
            var tex = textures[ti]
            if Int(tex.width) > 0:
                var w0 = Float32(1.0) - inter.u - inter.v
                var su = w0*mesh.uvs[v0*2]   + inter.u*mesh.uvs[v1*2]   + inter.v*mesh.uvs[v2*2]
                var tv = w0*mesh.uvs[v0*2+1] + inter.u*mesh.uvs[v1*2+1] + inter.v*mesh.uvs[v2*2+1]
                return _sample_tex(tex, su, tv)
    else:
        if ti >= 0 and tex_filenames:
            var filename = tex_filenames[ti]
            if filename and mesh.uvs:
                var w0 = Float32(1.0) - inter.u - inter.v
                var su = w0*mesh.uvs[v0*2]   + inter.u*mesh.uvs[v1*2]   + inter.v*mesh.uvs[v2*2]
                var tv = w0*mesh.uvs[v0*2+1] + inter.u*mesh.uvs[v1*2+1] + inter.v*mesh.uvs[v2*2+1]
                su = su - Float32(Int(su))
                if su < Float32(0.0): su += Float32(1.0)
                tv = tv - Float32(Int(tv))
                if tv < Float32(0.0): tv += Float32(1.0)
                var tr = alloc[Float32](3)
                tr[0] = Float32(0.0); tr[1] = Float32(0.0); tr[2] = Float32(0.0)
                _ = external_call["texture", Bool,
                    UnsafePointer[UInt8, MutAnyOrigin], Float32, Float32,
                    UnsafePointer[Float32, MutAnyOrigin]](filename, su, tv, tr)
                var result = RGB(_srgb_to_linear(tr[0]), _srgb_to_linear(tr[1]), _srgb_to_linear(tr[2]))
                tr.free()
                return result
    return mat.albedo


@always_inline
fn shade_core(
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
        var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.dirX, path_ptr[].ray.dirY, path_ptr[].ray.dirZ)
        if dot(normal, ray_dir) > 0:
            normal = normal * -1.0

        # Cosine weighted hemisphere sampling
        var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
        var u1 = pcg.next_float()
        var u2 = pcg.next_float()
        path_ptr[].pcgState = pcg.state

        var r = sqrt(u1)
        var theta = 2.0 * Float32(3.14159265359) * u2
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
        var org = SIMD[DType.float32, 3](path_ptr[].ray.orgX, path_ptr[].ray.orgY, path_ptr[].ray.orgZ) + ray_dir * inter.tHit + normal * 0.0001
        path_ptr[].ray = Ray_C(org[0], org[1], org[2], dir[0], dir[1], dir[2])

        # Update Throughput (albedo)
        path_ptr[].throughput *= mat.albedo
    else:
        # Unknown material type — deactivate to prevent infinite loops
        path_ptr[].active = 0


# ── DiffuseTransmission branch ────────────────────────────────────────────────
@always_inline
fn shade_diffuse_transmission(
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

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.dirX, path_ptr[].ray.dirY, path_ptr[].ray.dirZ)
    # Orient normal toward incoming ray
    if dot(normal, ray_dir) > Float32(0.0):
        normal = -normal

    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.orgX, path_ptr[].ray.orgY, path_ptr[].ray.orgZ)
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

    # Cosine-weighted hemisphere around bounce_normal
    var u1 = pcg.next_float()
    var u2 = pcg.next_float()
    var r = sqrt(u1)
    var theta = Float32(2.0) * Float32(3.14159265359) * u2
    var x = r * cos(theta)
    var y = r * sin(theta)
    var z2 = Float32(1.0) - u1
    var z = sqrt(z2 if z2 > Float32(0.0) else Float32(0.0))

    var sign = Float32(1.0) if bounce_normal[2] >= Float32(0.0) else Float32(-1.0)
    var a = Float32(-1.0) / (sign + bounce_normal[2])
    var b = bounce_normal[0] * bounce_normal[1] * a
    var tangent  = SIMD[DType.float32, 3](Float32(1.0) + sign * bounce_normal[0] * bounce_normal[0] * a, sign * b, -sign * bounce_normal[0])
    var bitangent = SIMD[DType.float32, 3](b, sign + bounce_normal[1] * bounce_normal[1] * a, -bounce_normal[1])

    var dir = tangent * x + bitangent * y + bounce_normal * z
    var dlen = dot(dir, dir)
    if dlen > Float32(0.0):
        dir = dir * (Float32(1.0) / sqrt(dlen))

    # Offset along bounce_normal to avoid self-intersection
    var hit_point = ray_org + ray_dir * inter.tHit + bounce_normal * Float32(0.0001)
    path_ptr[].ray = Ray_C(hit_point[0], hit_point[1], hit_point[2], dir[0], dir[1], dir[2])

    # Throughput weight = color / selection_probability = color * total / p_choice
    if choose_reflect:
        var w = total / pr
        path_ptr[].throughput *= refl * w
    else:
        var w = total / pt
        path_ptr[].throughput *= trans * w

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
fn shade_coated_diffuse[use_gpu: Bool](
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    mat: Material_C,
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int,
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin],
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
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

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.dirX, path_ptr[].ray.dirY, path_ptr[].ray.dirZ)
    if dot(normal, ray_dir) > Float32(0.0):
        normal = -normal

    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.orgX, path_ptr[].ray.orgY, path_ptr[].ray.orgZ)
    var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)

    # Schlick Fresnel for the dielectric coating (IOR stored in emission.r)
    var ior = mat.emission.r
    var cos_i = -dot(ray_dir, normal)
    var r0 = (Float32(1.0) - ior) / (Float32(1.0) + ior)
    r0 = r0 * r0
    var one_minus = Float32(1.0) - cos_i
    var one_minus2 = one_minus * one_minus
    var fresnel = r0 + (Float32(1.0) - r0) * one_minus2 * one_minus2 * one_minus

    if pcg.next_float() < fresnel:
        # Specular reflection from coating — throughput unchanged
        var refl = ray_dir + normal * (Float32(2.0) * cos_i)
        var rlen = dot(refl, refl)
        if rlen > Float32(0.0):
            refl = refl * (Float32(1.0) / sqrt(rlen))
        path_ptr[].ray = Ray_C(hit_point[0], hit_point[1], hit_point[2], refl[0], refl[1], refl[2])
    else:
        # Diffuse bounce through coating — NEE direct light sampling
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
                    var shadow_ray = Ray_C(hit_point[0], hit_point[1], hit_point[2],
                                          shadow_dir[0], shadow_dir[1], shadow_dir[2])
                    if not any_hit_bvh2_core(bvh2Nodes, primIds, meshes, shadow_ray, dist * Float32(0.9999)):
                        var weight = cos_s * cos_l * al.total_area * Float32(areaLightCount) / (dist_sq * Float32(3.14159265359))
                        path_ptr[].estimate += path_ptr[].throughput * alb * al.emission * weight

        var u1 = pcg.next_float()
        var u2 = pcg.next_float()
        var r = sqrt(u1)
        var theta = Float32(2.0) * Float32(3.14159265359) * u2
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

        path_ptr[].ray = Ray_C(hit_point[0], hit_point[1], hit_point[2], dir[0], dir[1], dir[2])
        path_ptr[].throughput *= alb

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
fn shade_dielectric(
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

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.dirX, path_ptr[].ray.dirY, path_ptr[].ray.dirZ)
    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.orgX, path_ptr[].ray.orgY, path_ptr[].ray.orgZ)

    var ior = mat.albedo.r
    var entering = dot(ray_dir, geom_normal) < Float32(0.0)
    var normal = geom_normal if entering else -geom_normal
    var eta = (Float32(1.0) / ior) if entering else ior

    var cos_i = -dot(ray_dir, normal)
    var sin2_t = eta * eta * (Float32(1.0) - cos_i * cos_i)
    var tir = sin2_t > Float32(1.0)

    # Schlick Fresnel
    var r0 = (Float32(1.0) - ior) / (Float32(1.0) + ior)
    r0 = r0 * r0
    var one_minus = Float32(1.0) - cos_i
    var one_minus2 = one_minus * one_minus
    var fresnel = r0 + (Float32(1.0) - r0) * one_minus2 * one_minus2 * one_minus

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)

    if tir or pcg.next_float() < fresnel:
        # Reflect: r = d + 2*cos_i*n
        var refl = ray_dir + normal * (Float32(2.0) * cos_i)
        var rlen = dot(refl, refl)
        if rlen > Float32(0.0):
            refl = refl * (Float32(1.0) / sqrt(rlen))
        var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)
        path_ptr[].ray = Ray_C(hit_point[0], hit_point[1], hit_point[2], refl[0], refl[1], refl[2])
    else:
        # Refract: t = eta*d + (eta*cos_i - sqrt(1 - sin2_t))*n
        var cos_t = sqrt(Float32(1.0) - sin2_t)
        var refr = ray_dir * eta + normal * (eta * cos_i - cos_t)
        var rlen = dot(refr, refr)
        if rlen > Float32(0.0):
            refr = refr * (Float32(1.0) / sqrt(rlen))
        var hit_point = ray_org + ray_dir * inter.tHit - normal * Float32(0.0001)
        path_ptr[].ray = Ray_C(hit_point[0], hit_point[1], hit_point[2], refr[0], refr[1], refr[2])

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


# ── Conductor (perfect mirror) branch ────────────────────────────────────────
@always_inline
fn shade_conductor(
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

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.dirX, path_ptr[].ray.dirY, path_ptr[].ray.dirZ)
    if dot(normal, ray_dir) > Float32(0.0):
        normal = -normal

    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.orgX, path_ptr[].ray.orgY, path_ptr[].ray.orgZ)
    var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)

    # Perfect specular reflection: r = d - 2(d·n)n
    var refl = ray_dir - normal * (Float32(2.0) * dot(ray_dir, normal))
    var rlen = dot(refl, refl)
    if rlen > Float32(0.0):
        refl = refl * (Float32(1.0) / sqrt(rlen))

    path_ptr[].ray = Ray_C(hit_point[0], hit_point[1], hit_point[2], refl[0], refl[1], refl[2])
    path_ptr[].throughput *= mat.albedo
    path_ptr[].bounce += 1

    # Russian roulette after first bounce
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    if path_ptr[].bounce > 1:
        var lum = path_ptr[].throughput.luma()
        var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
        if pcg.next_float() < q:
            path_ptr[].active = 0
        else:
            path_ptr[].throughput *= Float32(1.0) / (Float32(1.0) - q)
    path_ptr[].pcgState = pcg.state


# Unified NEE core — comptime-specialized for CPU (use_gpu=False) and GPU (use_gpu=True).
# Texture lookup uses OIIO external_call on CPU and device-resident GpuTexture_C on GPU.
@always_inline
fn shade_nee_core[use_gpu: Bool](
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
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
):
    var mat = materials[Int(inter.primId.materialIndex)]

    if mat.type == 2:
        if path_ptr[].bounce == 0:
            path_ptr[].estimate += path_ptr[].throughput * mat.emission
        path_ptr[].active = 0
        return

    if mat.type == 3:
        shade_conductor(path_ptr, inter, meshes, mat)
        return

    if mat.type == 4:
        shade_dielectric(path_ptr, inter, meshes, mat)
        return

    if mat.type == 5:
        shade_coated_diffuse[use_gpu](path_ptr, inter, bvh2Nodes, primIds, meshes, mat, areaLights, areaLightCount, tex_filenames, textures, n_textures)
        return

    if mat.type == 6:
        shade_diffuse_transmission(path_ptr, inter, meshes, mat)
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

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.dirX, path_ptr[].ray.dirY, path_ptr[].ray.dirZ)
    if dot(normal, ray_dir) > Float32(0.0):
        normal = -normal

    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.orgX, path_ptr[].ray.orgY, path_ptr[].ray.orgZ)
    var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)

    var alb = _tex_lookup[use_gpu](mat, inter, v0, v1, v2, mesh, tex_filenames, textures, n_textures)

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
                var shadow_ray = Ray_C(hit_point[0], hit_point[1], hit_point[2],
                                      shadow_dir[0], shadow_dir[1], shadow_dir[2])
                if not any_hit_bvh2_core(bvh2Nodes, primIds, meshes, shadow_ray, dist * Float32(0.9999)):
                    var weight = cos_s * cos_l * al.total_area * Float32(areaLightCount) / (dist_sq * Float32(3.14159265359))
                    path_ptr[].estimate += path_ptr[].throughput * alb * al.emission * weight

    var u1 = pcg.next_float()
    var u2 = pcg.next_float()
    var r = sqrt(u1)
    var theta = Float32(2.0) * Float32(3.14159265359) * u2
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

    path_ptr[].ray = Ray_C(hit_point[0], hit_point[1], hit_point[2], dir[0], dir[1], dir[2])
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
fn shade_core_cpu_nee(
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
):
    var path_ptr = paths + tid
    if path_ptr[].active == 0:
        return
    var inter = intersections[tid]
    if inter.hit == 0:
        path_ptr[].active = 0
        return
    shade_nee_core[False](path_ptr, inter, bvh2Nodes, primIds, meshes, materials, areaLights, areaLightCount,
        tex_filenames, UnsafePointer[GpuTexture_C, MutAnyOrigin](), 0)


@export
fn mojo_cpu_shade_batch(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    count: Int64,
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
):
    var n = Int(count)
    for tid in range(n):
        shade_core(paths, intersections, meshes, materials, tid)
