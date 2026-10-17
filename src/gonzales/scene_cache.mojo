"""A binary snapshot of a parsed scene, so a big scene (the Moana island: 7.5 minutes of text) loads in seconds.

GONZALES_SCENE_CACHE=<file>: if the file exists and was written for the same scene file (path and size) by this
format, the scene is read from it; otherwise the scene is parsed and the file is written. Delete the file whenever
the parser or the scene's data changes: only the scene file's path and size, and the merge/prune settings, are checked.

The snapshot is the ParsedScene_Mojo struct as it stands in memory, followed by every array its pointers name.
Scenes with media, distant or point lights, normal maps, alpha cut-outs or measured materials are not cached.
"""
from std.ffi import external_call
from std.os import getenv
from std.memory.alloc import unsafe_alloc
from std.sys.info import size_of, num_performance_cores
from std.time import perf_counter_ns
from .geometry import _is_real_ptr
from .bvh import BVH2Node, BVH4
from .curves import Curve
from .lights import AreaLight, InfiniteLight, LightSampler, LIGHT_BOUNDS_STRIDE
from .materials import Material
from .pbrt_parser import ParsedScene_Mojo, _plan_ptex_budget, mojo_parsed_free
from .primitives import Instance, PrimId, Sphere, TriangleMesh, mesh_points_floats
from .render_state import GpuTexture, NormalSlopeMap, normal_slope_map_none
from .host_textures import TexPrefetch
from .fast_io import read_parallel, PARALLEL_MIN

comptime _MAGIC: Int = 0x474E5A53434143
comptime _FORMAT: Int = 9

comptime _Ptr = Pointer[UInt8, MutUntrackedOrigin]


def _fopen(path: _Ptr, write: Bool) -> Int:
    var mode = unsafe_alloc[UInt8](3)
    mode[unsafe_offset=0] = UInt8(119) if write else UInt8(114)   # w / r
    mode[unsafe_offset=1] = UInt8(98)                              # b
    mode[unsafe_offset=2] = UInt8(0)
    var f = external_call["fopen", Int, _Ptr, _Ptr](path, mode)
    mode.unsafe_free()
    return f


def _fclose(f: Int):
    _ = external_call["fclose", Int32, Int](f)


def _put[T: AnyType](f: Int, p: Pointer[T, MutUntrackedOrigin], n: Int):
    if n > 0:
        _ = external_call["fwrite", Int, _Ptr, Int, Int, Int](p.unsafe_bitcast[UInt8](), 1, n * size_of[T](), f)


def _get[T: AnyType](f: Int, n: Int) -> Pointer[T, MutUntrackedOrigin]:
    """n elements read into an allocation of at least one, which is zero when n is 0 (an empty BVH's root)."""
    if n * size_of[T]() >= 100 * 1024 * 1024 and getenv("GONZALES_SCENE_CACHE_LOG") != "":
        print("scene cache: array of", n * size_of[T]() // (1024 * 1024), "MB,", size_of[T](), "bytes an element")
    var p = unsafe_alloc[T](max(n, 1))
    var raw = p.unsafe_bitcast[UInt8]()
    if n * size_of[T]() >= PARALLEL_MIN:
        # Big arrays come in on every core; the stdio position is moved past them by hand.
        var pos = external_call["ftell", Int, Int](f)
        read_parallel(external_call["fileno", Int32, Int](f), raw, n * size_of[T](), pos)
        _ = external_call["fseek", Int32, Int, Int, Int32](f, pos + n * size_of[T](), Int32(0))
    elif n > 0:
        _ = external_call["fread", Int, _Ptr, Int, Int, Int](raw, 1, n * size_of[T](), f)
    else:
        for k in range(size_of[T]()):
            raw[unsafe_offset=k] = UInt8(0)
    return p


def _put_int(f: Int, v: Int):
    var p = unsafe_alloc[Int](1)
    p[unsafe_offset=0] = v
    _put[Int](f, p, 1)
    p.unsafe_free()


def _get_int(f: Int) -> Int:
    var p = _get[Int](f, 1)
    var v = p[unsafe_offset=0]
    p.unsafe_free()
    return v


def _cstr_len(p: _Ptr) -> Int:
    var n = 0
    while p[unsafe_offset=n] != UInt8(0):
        n += 1
    return n


def _put_cstr(f: Int, p: _Ptr):
    var n = _cstr_len(p)
    _put_int(f, n)
    _put[UInt8](f, p, n)


def _get_cstr(f: Int) -> _Ptr:
    var n = _get_int(f)
    var p = unsafe_alloc[UInt8](n + 1)
    if n > 0:
        _ = external_call["fread", Int, _Ptr, Int, Int, Int](p, 1, n, f)
    p[unsafe_offset=n] = UInt8(0)
    return p


def _file_size(path: _Ptr) -> Int:
    var f = _fopen(path, False)
    if f == 0:
        return -1
    _ = external_call["fseek", Int32, Int, Int, Int32](f, 0, Int32(2))
    var n = external_call["ftell", Int, Int](f)
    _fclose(f)
    return n


def _unsupported(ref s: ParsedScene_Mojo) -> String:
    if s.medium_count > 0 or s.grid_count > 0 or s.nvdb_grid_count > 0:
        return "media"
    if s.distant_count > 0 or s.point_count > 0:
        return "distant or point lights"
    if s.measured_count > 0:
        return "measured materials"
    if s.bvh4.is_real():
        return "a BVH4"
    for ti in range(Int(s.tex_count)):
        if s.nmaps[unsafe_offset=ti].res > Int32(0):
            return "normal maps"
    for mi in range(Int(s.mesh_count)):
        if s.meshes[unsafe_offset=mi].alpha_w > Int32(0):
            return "alpha cut-outs"
    return String("")


def scene_cache_variant() -> Int:
    """The settings that change what the parser builds from the same file: template merging and the RT pruning."""
    var v = 1 if getenv("GONZALES_MERGE_TEMPLATES", "1") == "0" else 0
    var prune = 0
    try:
        prune = Int(getenv("GONZALES_RT_PRUNE", "0"))
    except:
        prune = 0
    return v + 2 * prune


# True when the file is a cache of this format made for another scene: it is kept (delete it to replace it).
def _is_cache_of_other_scene(cache_path: _Ptr, scene_path: _Ptr) -> Bool:
    var f = _fopen(cache_path, False)
    if f == 0:
        return False
    var other = False
    if _get_int(f) == _MAGIC and _get_int(f) == _FORMAT and _get_int(f) == size_of[ParsedScene_Mojo]():
        var saved_path = _get_cstr(f)
        var same = _cstr_len(saved_path) == _cstr_len(scene_path)
        if same:
            for k in range(_cstr_len(scene_path)):
                if saved_path[unsafe_offset=k] != scene_path[unsafe_offset=k]:
                    same = False
        saved_path.unsafe_free()
        other = not same or _get_int(f) != _file_size(scene_path)
    _fclose(f)
    return other


def scene_cache_save(psc: Pointer[ParsedScene_Mojo, MutUntrackedOrigin], cache_path: _Ptr, scene_path: _Ptr) -> Bool:
    var why = _unsupported(psc[unsafe_offset=0])
    if why.byte_length() > 0:
        print("scene cache: not written, the scene has " + why)
        return False
    if _is_cache_of_other_scene(cache_path, scene_path):
        print("scene cache: the file belongs to another scene, not overwriting it")
        return False
    var t0 = perf_counter_ns()
    var f = _fopen(cache_path, True)
    if f == 0:
        print("scene cache: cannot write it")
        return False
    ref s = psc[unsafe_offset=0]
    _put_int(f, _MAGIC); _put_int(f, _FORMAT); _put_int(f, size_of[ParsedScene_Mojo]())
    _put_cstr(f, scene_path); _put_int(f, _file_size(scene_path)); _put_int(f, scene_cache_variant())
    _put[ParsedScene_Mojo](f, psc, 1)

    _put[Float32](f, s.raster_to_camera, 16); _put[Float32](f, s.camera_to_world, 16)
    _put_cstr(f, s.film_filename)
    _put[Material](f, s.materials, Int(s.material_count))

    _put[AreaLight](f, s.area_lights, Int(s.area_light_count))
    for i in range(Int(s.area_light_count)):
        var al = s.area_lights[unsafe_offset=i]
        var has = al.kind == Int8(0) and _is_real_ptr(al.tri_cdf)
        _put_int(f, 1 if has else 0)
        if has:
            _put[Float32](f, al.tri_cdf, Int(al.n_tris))

    var nm = Int(s.mesh_count)
    _put[TriangleMesh](f, s.meshes, nm)
    _put[Int32](f, s.mesh_n_verts, nm); _put[Int32](f, s.mesh_n_tris, nm)
    _put[Int32](f, s.mesh_uv_n_verts, nm); _put[Int32](f, s.mesh_nrm_n_verts, nm)
    for i in range(nm):
        var m = s.meshes[unsafe_offset=i]
        var nv = Int(s.mesh_n_verts[unsafe_offset=i]); var nt = Int(s.mesh_n_tris[unsafe_offset=i])
        var per_face = nt // 2 if m.quads != Int32(0) else nt
        _put[Float32](f, m.points, mesh_points_floats(nv, m.quads))
        _put[Int32](f, m.faceIndices, per_face)
        _put[Int32](f, m.vertexIndices, 1 if m.quads != Int32(0) else 3 * nt)
        _put[Float32](f, m.uvs, 2 * Int(s.mesh_uv_n_verts[unsafe_offset=i]))
        _put[Float32](f, m.normals, Int(s.mesh_nrm_n_verts[unsafe_offset=i]))
        var has_mat = _is_real_ptr(m.materials)
        _put_int(f, 1 if has_mat else 0)
        if has_mat:
            _put[Int32](f, m.materials, per_face)

    _put[BVH2Node](f, s.bvh_nodes, max(Int(s.bvh_node_count), 1)); _put[PrimId](f, s.prim_ids, max(Int(s.prim_count), 1))
    var shared = Int(s.bvh_nodes_cpu) == Int(s.bvh_nodes)
    _put_int(f, 1 if shared else 0)
    if not shared:
        _put[BVH2Node](f, s.bvh_nodes_cpu, max(Int(s.bvh_node_count_cpu), 1))
        _put[PrimId](f, s.prim_ids_cpu, max(Int(s.prim_count_cpu), 1))

    var nb = Int(s.blas_total)
    _put[Int32](f, s.blas_node_counts, nb); _put[Int32](f, s.blas_primid_counts, nb)
    for bi in range(nb):
        _put[BVH2Node](f, s.blas_nodes_arr[unsafe_offset=bi], max(Int(s.blas_node_counts[unsafe_offset=bi]), 1))
        _put[PrimId](f, s.blas_primids_arr[unsafe_offset=bi], max(Int(s.blas_primid_counts[unsafe_offset=bi]), 1))
    _put[Instance](f, s.instances, max(Int(s.instance_count), 1))
    _put[Int32](f, s.template_mesh_start, max(Int(s.blas_count), 1)); _put[Int32](f, s.template_mesh_end, max(Int(s.blas_count), 1))
    _put[Curve](f, s.curves, Int(s.curve_count)); _put[Sphere](f, s.spheres, Int(s.sphere_count))

    _put[Float32](f, s.light_sampler.cdf, max(Int(s.light_sampler.n) + 1, 2) + (Int(s.light_sampler.n) * LIGHT_BOUNDS_STRIDE if s.light_sampler.has_bounds != Int32(0) else 0))
    _put[Float32](f, s.light_pick_cdf, max(Int(s.light_pick_n) + 1, 2))

    _put[InfiniteLight](f, s.infinite_lights, Int(s.infinite_count))
    for i in range(Int(s.infinite_count)):
        var il = s.infinite_lights[unsafe_offset=i]
        var has_map = il.cdf_w > Int32(0) and _is_real_ptr(il.cdf_ptr)
        _put_int(f, 1 if has_map else 0)
        if has_map:
            var w = Int(il.cdf_w); var h = Int(il.cdf_h)
            _put[Float32](f, il.cdf_ptr, (h + 1) + h * (w + 1))
            var has_px = _is_real_ptr(il.pixels_ptr)
            _put_int(f, 1 if has_px else 0)
            if has_px:
                _put[Float32](f, il.pixels_ptr, w * h * 3)
        _put[Float32](f, il.world_to_light, 16)

    # The textures the decoder threads were given, so a load can start the same ones.
    var pf = s.tex_prefetch
    var npf = pf[unsafe_offset=0].n if _is_real_ptr(pf) else 0
    _put_int(f, npf)
    for i in range(npf):
        _put_cstr(f, pf[unsafe_offset=0].files[unsafe_offset=i])
        _put_int(f, 1 if pf[unsafe_offset=0].raw[unsafe_offset=i] else 0)
    _put_int(f, Int(s.tex_count))
    for ti in range(Int(s.tex_count)):
        _put_cstr(f, s.tex_filenames[unsafe_offset=ti])
    _put_int(f, _MAGIC)
    _fclose(f)
    print("scene cache: written in " + String(Float64(perf_counter_ns() - t0) / 1.0e9) + " s")
    return True


def scene_cache_load(cache_path: _Ptr, scene_path: _Ptr) -> Pointer[ParsedScene_Mojo, MutUntrackedOrigin]:
    """The cached scene, or a dangling pointer (not _is_real_ptr) when there is no usable cache."""
    var none = Pointer[ParsedScene_Mojo, MutUntrackedOrigin].unsafe_dangling()
    var f = _fopen(cache_path, False)
    if f == 0:
        return none
    var t0 = perf_counter_ns()
    if _get_int(f) != _MAGIC or _get_int(f) != _FORMAT or _get_int(f) != size_of[ParsedScene_Mojo]():
        _fclose(f)
        print("scene cache: stale format, parsing the scene")
        return none
    var saved_path = _get_cstr(f)
    var same = _cstr_len(saved_path) == _cstr_len(scene_path)
    if same:
        for k in range(_cstr_len(scene_path)):
            if saved_path[unsafe_offset=k] != scene_path[unsafe_offset=k]:
                same = False
    saved_path.unsafe_free()
    if not same or _get_int(f) != _file_size(scene_path):
        _fclose(f)
        print("scene cache: made for another scene, parsing it")
        return none
    if _get_int(f) != scene_cache_variant():
        _fclose(f)
        print("scene cache: made with other merge/prune settings, parsing the scene")
        return none

    var psc = _get[ParsedScene_Mojo](f, 1)
    ref s = psc[unsafe_offset=0]
    s.raster_to_camera = _get[Float32](f, 16); s.camera_to_world = _get[Float32](f, 16)
    s.film_filename = _get_cstr(f)
    s.materials = _get[Material](f, Int(s.material_count))

    s.area_lights = _get[AreaLight](f, Int(s.area_light_count))
    for i in range(Int(s.area_light_count)):
        if _get_int(f) == 1:
            s.area_lights[unsafe_offset=i].tri_cdf = _get[Float32](f, Int(s.area_lights[unsafe_offset=i].n_tris))
        else:
            s.area_lights[unsafe_offset=i].tri_cdf = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()

    var nm = Int(s.mesh_count)
    s.meshes = _get[TriangleMesh](f, nm)
    s.mesh_n_verts = _get[Int32](f, nm); s.mesh_n_tris = _get[Int32](f, nm)
    s.mesh_uv_n_verts = _get[Int32](f, nm); s.mesh_nrm_n_verts = _get[Int32](f, nm)
    s.mesh_pts = unsafe_alloc[Pointer[Float32, MutUntrackedOrigin]](max(nm, 1))
    s.mesh_vis = unsafe_alloc[Pointer[Int32, MutUntrackedOrigin]](max(nm, 1))
    s.mesh_fis = unsafe_alloc[Pointer[Int32, MutUntrackedOrigin]](max(nm, 1))
    for i in range(nm):
        var nv = Int(s.mesh_n_verts[unsafe_offset=i]); var nt = Int(s.mesh_n_tris[unsafe_offset=i])
        var quads = s.meshes[unsafe_offset=i].quads
        var per_face = nt // 2 if quads != Int32(0) else nt
        s.meshes[unsafe_offset=i].points = _get[Float32](f, mesh_points_floats(nv, quads))
        s.meshes[unsafe_offset=i].faceIndices = _get[Int32](f, per_face)
        s.meshes[unsafe_offset=i].vertexIndices = _get[Int32](f, 1 if quads != Int32(0) else 3 * nt)
        var uvn = Int(s.mesh_uv_n_verts[unsafe_offset=i]); var nrn = Int(s.mesh_nrm_n_verts[unsafe_offset=i])
        if uvn > 0:
            s.meshes[unsafe_offset=i].uvs = _get[Float32](f, 2 * uvn)
        else:
            s.meshes[unsafe_offset=i].uvs = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
        if nrn > 0:
            s.meshes[unsafe_offset=i].normals = _get[Float32](f, nrn)
        else:
            s.meshes[unsafe_offset=i].normals = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
        s.meshes[unsafe_offset=i].alpha = Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling()
        if _get_int(f) == 1:
            s.meshes[unsafe_offset=i].materials = _get[Int32](f, per_face)
        else:
            s.meshes[unsafe_offset=i].materials = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
        s.mesh_pts[unsafe_offset=i] = s.meshes[unsafe_offset=i].points
        s.mesh_vis[unsafe_offset=i] = s.meshes[unsafe_offset=i].vertexIndices
        s.mesh_fis[unsafe_offset=i] = s.meshes[unsafe_offset=i].faceIndices

    s.bvh_nodes = _get[BVH2Node](f, max(Int(s.bvh_node_count), 1)); s.prim_ids = _get[PrimId](f, max(Int(s.prim_count), 1))
    if _get_int(f) == 1:
        s.bvh_nodes_cpu = s.bvh_nodes; s.prim_ids_cpu = s.prim_ids
    else:
        s.bvh_nodes_cpu = _get[BVH2Node](f, max(Int(s.bvh_node_count_cpu), 1))
        s.prim_ids_cpu = _get[PrimId](f, max(Int(s.prim_count_cpu), 1))
    s.bvh4 = BVH4.none()

    var nb = Int(s.blas_total)
    s.blas_node_counts = _get[Int32](f, nb); s.blas_primid_counts = _get[Int32](f, nb)
    s.blas_nodes_arr = unsafe_alloc[Pointer[BVH2Node, MutUntrackedOrigin]](nb + 1)
    s.blas_primids_arr = unsafe_alloc[Pointer[PrimId, MutUntrackedOrigin]](nb + 1)
    for bi in range(nb):
        s.blas_nodes_arr[unsafe_offset=bi] = _get[BVH2Node](f, max(Int(s.blas_node_counts[unsafe_offset=bi]), 1))
        s.blas_primids_arr[unsafe_offset=bi] = _get[PrimId](f, max(Int(s.blas_primid_counts[unsafe_offset=bi]), 1))
    s.instances = _get[Instance](f, max(Int(s.instance_count), 1))
    s.template_mesh_start = _get[Int32](f, max(Int(s.blas_count), 1)); s.template_mesh_end = _get[Int32](f, max(Int(s.blas_count), 1))
    s.curves = _get[Curve](f, Int(s.curve_count)); s.spheres = _get[Sphere](f, Int(s.sphere_count))

    var ls_cdf = _get[Float32](f, max(Int(s.light_sampler.n) + 1, 2) + (Int(s.light_sampler.n) * LIGHT_BOUNDS_STRIDE if s.light_sampler.has_bounds != Int32(0) else 0))
    s.light_sampler = LightSampler(ls_cdf, s.light_sampler.n, s.light_sampler.has_bounds)
    s.light_pick_cdf = _get[Float32](f, max(Int(s.light_pick_n) + 1, 2))

    s.infinite_lights = _get[InfiniteLight](f, Int(s.infinite_count))
    for i in range(Int(s.infinite_count)):
        if _get_int(f) == 1:
            var w = Int(s.infinite_lights[unsafe_offset=i].cdf_w); var h = Int(s.infinite_lights[unsafe_offset=i].cdf_h)
            s.infinite_lights[unsafe_offset=i].cdf_ptr = _get[Float32](f, (h + 1) + h * (w + 1))
            if _get_int(f) == 1:
                # Released with free() by mojo_parsed_free (free_texture_rgb), so it comes from malloc.
                var px = external_call["malloc", Pointer[Float32, MutUntrackedOrigin], Int](w * h * 3 * 4)
                _ = external_call["fread", Int, _Ptr, Int, Int, Int](px.unsafe_bitcast[UInt8](), 1, w * h * 3 * 4, f)
                s.infinite_lights[unsafe_offset=i].pixels_ptr = px
            else:
                s.infinite_lights[unsafe_offset=i].pixels_ptr = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
        else:
            s.infinite_lights[unsafe_offset=i].cdf_ptr = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
            s.infinite_lights[unsafe_offset=i].pixels_ptr = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
        s.infinite_lights[unsafe_offset=i].world_to_light = _get[Float32](f, 16)

    var npf = _get_int(f)
    var names = List[String]()
    var raws = List[Bool]()
    for i in range(npf):
        var cs = _get_cstr(f)
        names.append(String(unsafe_from_utf8_ptr=cs.as_imm()))
        cs.unsafe_free()
        raws.append(_get_int(f) == 1)
    var n_tex = _get_int(f)
    s.tex_filenames = unsafe_alloc[_Ptr](max(n_tex, 1))
    for ti in range(n_tex):
        s.tex_filenames[unsafe_offset=ti] = _get_cstr(f)
    s.nmaps = Pointer[NormalSlopeMap, MutUntrackedOrigin].unsafe_dangling()
    if n_tex > 0:
        s.nmaps = unsafe_alloc[NormalSlopeMap](n_tex)
        for ti in range(n_tex):
            s.nmaps[unsafe_offset=ti] = normal_slope_map_none()
    var ok = _get_int(f) == _MAGIC
    _fclose(f)
    if not ok:
        print("scene cache: truncated, parsing the scene")
        mojo_parsed_free(psc)
        return none

    # What parsing leaves for later: a fresh seed, and the decoder threads on the textures the scene names.
    s.rng_seed = UInt64(perf_counter_ns())
    s.host_textures = Pointer[GpuTexture, MutUntrackedOrigin].unsafe_dangling()
    s.tex_prefetch = Pointer[TexPrefetch, MutUntrackedOrigin].unsafe_dangling()
    _plan_ptex_budget(names)
    var tex_pf = unsafe_alloc[TexPrefetch](1)
    tex_pf.unsafe_write(TexPrefetch())
    tex_pf[unsafe_offset=0].start(names, raws, min(12, num_performance_cores()))
    tex_pf[unsafe_offset=0].join()
    s.tex_prefetch = tex_pf
    print("scene cache: loaded in " + String(Float64(perf_counter_ns() - t0) / 1.0e9) + " s")
    return psc
