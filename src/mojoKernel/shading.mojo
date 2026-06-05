from std.math import sqrt, cos, sin, floor, acos, atan2
from std.ffi import external_call
from std.memory import alloc
from .geometry import RGB, Point3f, Vec3f, Ray_C, Intersection_C, PrimId_C, TriangleMesh_C, Material_C, AreaLight_C, DistantLight_C, PointLight_C, InfiniteLight_C, PathState_C, GpuTexture_C, ShadowTask_C, dot, cross
from .rng import PCG32
from .bvh import BVH2Node, SceneDescriptor2_C, any_hit_bvh2_core

# Power heuristic (β=2) for two-strategy MIS.
@always_inline
def power_heuristic(pdf_f: Float32, pdf_g: Float32) -> Float32:
    var f2 = pdf_f * pdf_f
    var g2 = pdf_g * pdf_g
    var denom = f2 + g2
    if denom <= Float32(0.0):
        return Float32(0.0)
    return f2 / denom

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
        var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
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
                var pi = Float32(3.14159265359)
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
    var theta = Float32(2.0) * Float32(3.14159265359) * u2
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
    path_ptr[].lastBsdfPdf = (cos_sc if cos_sc > Float32(0.0) else Float32(0.0)) / Float32(3.14159265359)
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
fn shade_coated_diffuse[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    path_idx: Int,
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
        path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(refl[0], refl[1], refl[2]))
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
                    var pdf_light_cd = dist_sq / (cos_l * al.total_area * Float32(areaLightCount))
                    var pi_cd = Float32(3.14159265359)
                    var pdf_bsdf_cd = cos_s / pi_cd
                    var w_cd = power_heuristic(pdf_light_cd, pdf_bsdf_cd)
                    var weight_cd = alb * al.emission * (cos_s * w_cd / (pdf_light_cd * pi_cd))
                    var contrib = path_ptr[].throughput * weight_cd
                    @parameter
                    if enqueue_shadow:
                        shadow_tasks[path_idx] = ShadowTask_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]), dist * Float32(0.9999), RGB(contrib.r, contrib.g, contrib.b), Int32(1), Int32(0))
                    else:
                        var shadow_ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
                        if not any_hit_bvh2_core(bvh2Nodes, primIds, meshes, shadow_ray, dist * Float32(0.9999)):
                            path_ptr[].estimate += contrib

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

        path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(dir[0], dir[1], dir[2]))
        # Store BSDF pdf for next-bounce MIS (cosine hemisphere: cos/pi).
        var cos_sc = dot(dir, normal)
        path_ptr[].lastBsdfPdf = (cos_sc if cos_sc > Float32(0.0) else Float32(0.0)) / Float32(3.14159265359)
        path_ptr[].specularBounce = Int8(0)
        path_ptr[].throughput *= alb

    if path_ptr[].bounce == 0:
        path_ptr[].albedo = alb
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

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)

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
fn shade_thin_dielectric(
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

    # Two-way Fresnel for thin slab: account for internal reflection too
    # Effective reflectance = F + (1-F)*F*(1-F) / (1 - F^2) ≈ 2F/(1+F)
    # Simplification: just use single-interface Schlick
    var r0 = (ior - Float32(1.0)) / (ior + Float32(1.0))
    r0 = r0 * r0
    var one_m = Float32(1.0) - cos_i
    var one_m2 = one_m * one_m
    var fresnel = r0 + (Float32(1.0) - r0) * one_m2 * one_m2 * one_m

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

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    if dot(normal, ray_dir) > Float32(0.0):
        normal = -normal

    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)

    var alpha_x = max(mat.roughU * mat.roughU, Float32(0.0001))
    var alpha_y = max(mat.roughV * mat.roughV, Float32(0.0001))
    var is_rough = mat.roughU > Float32(0.001) or mat.roughV > Float32(0.001)

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var scatter_dir: SIMD[DType.float32, 3]
    var fresnel_weight: Float32

    if not is_rough:
        # Perfect specular reflection
        scatter_dir = ray_dir - normal * (Float32(2.0) * dot(ray_dir, normal))
        var rlen = dot(scatter_dir, scatter_dir)
        if rlen > Float32(0.0):
            scatter_dir = scatter_dir * (Float32(1.0) / sqrt(rlen))
        fresnel_weight = Float32(1.0)
    else:
        # True anisotropic GGX VNDF (Heitz 2018)
        # Derive UV tangent frame for anisotropy direction
        var t1: SIMD[DType.float32, 3]
        var t2: SIMD[DType.float32, 3]
        if mesh.uvs and alpha_x != alpha_y:
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
        var phi = Float32(6.28318530718) * u2
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

        # Schlick Fresnel for conductor using albedo as F0
        var cos_wh = max(Float32(0.0), wo_dot_wh)
        var one_m = Float32(1.0) - cos_wh
        var one_m2 = one_m * one_m
        var schlick = one_m2 * one_m2 * one_m
        var f0_luma = mat.albedo.luma()
        fresnel_weight = f0_luma + (Float32(1.0) - f0_luma) * schlick

        if dot(scatter_dir, normal) <= Float32(0.0):
            path_ptr[].active = 0
            path_ptr[].pcgState = pcg.state
            return

    path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(scatter_dir[0], scatter_dir[1], scatter_dir[2]))
    if path_ptr[].bounce == 0:
        path_ptr[].albedo = mat.albedo
    path_ptr[].throughput *= mat.albedo * fresnel_weight
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
fn shade_coated_conductor(
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
        var phi = Float32(6.28318530718) * u2
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
fn shade_mix[use_gpu: Bool, enqueue_shadow: Bool](
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
            areaLights, areaLightCount, tex_filenames, textures, n_textures, shadow_tasks)
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
        var phi = Float32(6.28318530718) * u2
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
        path_ptr[].lastBsdfPdf = Float32(1.0) / Float32(3.14159265359)
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
fn _apply_normal_map[use_gpu: Bool](
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
    if mat.normal_tex_idx < Int32(0) or not mesh.uvs:
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
    # Tangent and bitangent from UV gradients
    var tangent = (dp1 * dv2 - dp2 * dv1) * inv_det
    var tlen = dot(tangent, tangent)
    if tlen <= Float32(0.0): return geom_normal
    tangent = tangent * (Float32(1.0) / sqrt(tlen))
    var bitangent = cross(geom_normal, tangent)
    var blen = dot(bitangent, bitangent)
    if blen <= Float32(0.0): return geom_normal
    bitangent = bitangent * (Float32(1.0) / sqrt(blen))
    # Interpolate UV at hit point (use centroid as approximation — barycentrics not stored)
    var uv_u = (u0f + u1f + u2f) * Float32(0.333333)
    var uv_v = (v0f + v1f + v2f) * Float32(0.333333)
    # Use the normal_tex_idx texture directly via OIIO
    @parameter
    if not use_gpu:
        var nfname = tex_filenames[Int(mat.normal_tex_idx)]
        var tr = alloc[Float32](3)
        tr[0] = Float32(0.5); tr[1] = Float32(0.5); tr[2] = Float32(1.0)
        _ = external_call["texture", Bool,
            UnsafePointer[UInt8, MutAnyOrigin], Float32, Float32,
            UnsafePointer[Float32, MutAnyOrigin]](nfname, uv_u, uv_v, tr)
        # Decode: [0,1] → [-1,1]
        var nx = tr[0] * Float32(2.0) - Float32(1.0)
        var ny = tr[1] * Float32(2.0) - Float32(1.0)
        var nz = tr[2] * Float32(2.0) - Float32(1.0)
        tr.free()
        # Rotate tangent-space normal to world space
        var world_n = tangent * nx + bitangent * ny + geom_normal * nz
        var wn_len = dot(world_n, world_n)
        if wn_len > Float32(0.0):
            return world_n * (Float32(1.0) / sqrt(wn_len))
    return geom_normal


# Unified NEE core — comptime-specialized for CPU (use_gpu=False) and GPU (use_gpu=True).
# Texture lookup uses OIIO external_call on CPU and device-resident GpuTexture_C on GPU.
@always_inline
fn shade_nee_core[use_gpu: Bool, enqueue_shadow: Bool](
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
):
    # ── Miss handler: ray escaped — add infinite light and deactivate ──────────
    if inter.hit == 0:
        var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
        for inf_i in range(infiniteLightCount):
            var ilight = infiniteLights[inf_i]
            var env_rgb: RGB
            @parameter
            if not use_gpu:
                if ilight.tex_idx >= Int32(0):
                    var fname = tex_filenames[Int(ilight.tex_idx)]
                    var u = (atan2(ray_dir[2], ray_dir[0]) + Float32(3.14159265359)) / Float32(6.28318530718)
                    var v = acos(max(Float32(-1.0), min(Float32(1.0), ray_dir[1]))) / Float32(3.14159265359)
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
                var pdf_light = Float32(1.0) / (Float32(4.0) * Float32(3.14159265359))
                @parameter
                if not use_gpu:
                    # Use CDF-based pdf when available (env-map importance sampling)
                    if ilight.cdf_w > Int32(0) and ilight.cdf_h > Int32(0):
                        var iw = Int(ilight.cdf_w); var ih = Int(ilight.cdf_h)
                        var u = (atan2(ray_dir[2], ray_dir[0]) + Float32(3.14159265359)) / Float32(6.28318530718)
                        var v = acos(max(Float32(-1.0), min(Float32(1.0), ray_dir[1]))) / Float32(3.14159265359)
                        var px = Int(min(Float32(iw - 1), max(Float32(0.0), u * Float32(iw))))
                        var py = Int(min(Float32(ih - 1), max(Float32(0.0), v * Float32(ih))))
                        var marginal_base = ih + 1
                        var row_cdf_base = marginal_base + py * (iw + 1)
                        # pdf of this texel in the CDF
                        var dp_row = ilight.cdf_ptr[py + 1] - ilight.cdf_ptr[py]
                        var dp_col_base = row_cdf_base + px
                        var dp_col = ilight.cdf_ptr[dp_col_base + 1] - ilight.cdf_ptr[dp_col_base]
                        var sin_theta = sin(Float32(3.14159265359) * (Float32(py) + Float32(0.5)) / Float32(ih))
                        if sin_theta > Float32(0.0) and dp_row > Float32(0.0):
                            pdf_light = (dp_row * dp_col * Float32(iw) * Float32(ih)) / (Float32(2.0) * Float32(3.14159265359) * Float32(3.14159265359) * sin_theta)
                mis_weight = power_heuristic(pdf_bsdf, pdf_light)
            path_ptr[].estimate += path_ptr[].throughput * env_rgb * mis_weight
        path_ptr[].active = 0
        return

    var mat = materials[Int(inter.primId.materialIndex)]

    if mat.type == 2:
        # Emissive surface hit.
        if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
            # Camera ray or must-follow specular chain — always add full emission.
            path_ptr[].estimate += path_ptr[].throughput * mat.emission
        else:
            # Arrived via BSDF scatter from the previous bounce.
            # Apply MIS weight using the stored BSDF pdf and the light's solid-angle pdf.
            var pdf_bsdf = path_ptr[].lastBsdfPdf
            if pdf_bsdf > Float32(0.0) and inter.primId.type == Int8(3):
                var al_idx = Int(inter.primId.id1)
                var al = areaLights[al_idx]
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
                    path_ptr[].estimate += path_ptr[].throughput * mat.emission * w
        path_ptr[].active = 0
        return

    if mat.type == 3:
        shade_conductor(path_ptr, inter, meshes, mat)
        return

    if mat.type == 4:
        shade_dielectric(path_ptr, inter, meshes, mat)
        return

    if mat.type == 5:
        shade_coated_diffuse[use_gpu, enqueue_shadow](path_ptr, path_idx, inter, bvh2Nodes, primIds, meshes, mat, areaLights, areaLightCount, tex_filenames, textures, n_textures, shadow_tasks)
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
            areaLights, areaLightCount, tex_filenames, textures, n_textures, shadow_tasks, mat)
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

    # Apply normal map if present (CPU only; GPU path uses geometric normal)
    normal = _apply_normal_map[use_gpu](mat, v0, v1, v2, mesh, inter, normal, p0, p1, p2,
        tex_filenames, textures, n_textures)
    if dot(normal, ray_dir) > Float32(0.0):
        normal = -normal

    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
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
                var pdf_light = dist_sq / (cos_l * al.total_area * Float32(areaLightCount))
                var pi = Float32(3.14159265359)
                var pdf_bsdf_nee = cos_s / pi
                var w_nee = power_heuristic(pdf_light, pdf_bsdf_nee)
                var weight = alb * al.emission * (cos_s * w_nee / (pdf_light * pi))
                var contrib = path_ptr[].throughput * weight
                @parameter
                if enqueue_shadow:
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
            var pi = Float32(3.14159265359)
            var contrib = path_ptr[].throughput * alb * dl.emission * (cos_s / pi)
            # Shadow ray at very long distance (scene diameter ~1000)
            var t_max = Float32(2000.0)
            @parameter
            if enqueue_shadow:
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
                var pi = Float32(3.14159265359)
                # f = alb/pi, geometry = cos_s, pdf = delta -> weight = 1
                # radiance = intensity / dist²
                var contrib = path_ptr[].throughput * alb * pl.intensity * (cos_s / (pi * dist_sq))
                @parameter
                if enqueue_shadow:
                    shadow_tasks[path_idx] = ShadowTask_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(ldir[0], ldir[1], ldir[2]), dist * Float32(0.9999), RGB(contrib.r, contrib.g, contrib.b), Int32(1), Int32(0))
                else:
                    var shadow_ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(ldir[0], ldir[1], ldir[2]))
                    if not any_hit_bvh2_core(bvh2Nodes, primIds, meshes, shadow_ray, dist * Float32(0.9999)):
                        path_ptr[].estimate += contrib

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

    path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(dir[0], dir[1], dir[2]))
    if path_ptr[].bounce == 0:
        path_ptr[].albedo = alb
    # Store BSDF pdf for MIS weighting if the next bounce hits an emitter.
    # For cosine-weighted hemisphere: pdf = cos(theta) / pi = dot(dir, normal) / pi
    var cos_scatter = dot(dir, normal)
    path_ptr[].lastBsdfPdf = (cos_scatter if cos_scatter > Float32(0.0) else Float32(0.0)) / Float32(3.14159265359)
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
    distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin],
    distantLightCount: Int,
    pointLights: UnsafePointer[PointLight_C, MutAnyOrigin],
    pointLightCount: Int,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    infiniteLightCount: Int,
):
    var path_ptr = paths + tid
    if path_ptr[].active == 0:
        return
    var inter = intersections[tid]
    shade_nee_core[False, False](path_ptr, 0, inter, bvh2Nodes, primIds, meshes, materials, areaLights, areaLightCount,
        tex_filenames, UnsafePointer[GpuTexture_C, MutAnyOrigin](), 0,
        UnsafePointer[ShadowTask_C, MutAnyOrigin](),
        distantLights, distantLightCount, pointLights, pointLightCount,
        infiniteLights, infiniteLightCount)


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
