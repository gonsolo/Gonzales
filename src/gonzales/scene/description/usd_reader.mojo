# USD scenes through Pixar's libusd (its Python bindings): data/usd_extract.py flattens the stage into arrays, this
# module turns them into a SceneParseState as the pbrt and Mitsuba front ends do. Prototypes become templates, instanceable
# prims and point-instancer points become instances. Needs the system Python with pxr (MOJO_PYTHON_LIBRARY) and, for
# libusd's jemalloc, LD_PRELOAD=libjemalloc.so.2.
from std.memory.alloc import unsafe_alloc
from std.ffi import external_call
from std.python import Python, PythonObject
from gonzales.core.env import env_str
from gonzales.core.geometry import RGB
from gonzales.core.transform import matrix_invert
from gonzales.scene.appearance.materials import MatKind
from gonzales.scene.description.parse_types import SceneParseState, NamedMaterial
from gonzales.scene.description.scene_builder import store_mesh
from gonzales.scene.description.pbrt_parser import ParsedScene_Mojo, finalize_scene, _psc_finish_object_def, _psc_emit_object_instance


def _f32(arr: PythonObject) raises -> Pointer[Float32, MutUntrackedOrigin]:
    return Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=arr.ctypes.data))


def _set_ctm(s: Pointer[SceneParseState, MutUntrackedOrigin], m: Pointer[Float32, MutUntrackedOrigin]):
    for k in range(16):
        s[unsafe_offset=0].ctm[k] = m[unsafe_offset=k]


def _set_identity(s: Pointer[SceneParseState, MutUntrackedOrigin]):
    for k in range(16):
        s[unsafe_offset=0].ctm[k] = Float32(1) if k % 5 == 0 else Float32(0)


def _usd_camera(s: Pointer[SceneParseState, MutUntrackedOrigin], cam: Pointer[Float32, MutUntrackedOrigin], fov: Float32):
    # A USD camera looks down -z, gonzales' (as pbrt's) down +z: flip the z axis; cam2w_raw holds world-to-camera.
    var c2w = unsafe_alloc[Float32](16)
    for k in range(16):
        c2w[unsafe_offset=k] = -cam[unsafe_offset=k] if k >= 8 and k < 11 else cam[unsafe_offset=k]
    var w2c = unsafe_alloc[Float32](16)
    _ = matrix_invert(c2w, w2c)
    for k in range(16):
        s[unsafe_offset=0].cam2w_raw[k] = w2c[unsafe_offset=k]
    s[unsafe_offset=0].camera_fov = fov
    c2w.unsafe_free(); w2c.unsafe_free()


def _usd_sky(s: Pointer[SceneParseState, MutUntrackedOrigin], rgb: RGB):
    s[unsafe_offset=0].inf_tex_idx.append(Int32(-1))
    s[unsafe_offset=0].inf_rgb.append(rgb.r)
    s[unsafe_offset=0].inf_rgb.append(rgb.g)
    s[unsafe_offset=0].inf_rgb.append(rgb.b)
    for k in range(16):
        s[unsafe_offset=0].inf_ctm.append(Float32(1) if k % 5 == 0 else Float32(0))
    s[unsafe_offset=0].inf_has_portal.append(Int32(0))
    for _ in range(12):
        s[unsafe_offset=0].inf_portal.append(Float32(0))


def _usd_texture(s: Pointer[SceneParseState, MutUntrackedOrigin], path: String) -> Int32:
    s[unsafe_offset=0].tex_names.append(String("__usd"))
    s[unsafe_offset=0].tex_files.append(path)
    return Int32(len(s[unsafe_offset=0].tex_files) - 1)


def _usd_fill(s: Pointer[SceneParseState, MutUntrackedOrigin], d: PythonObject) raises:
    var mats = d["materials"]
    var rgb = _f32(mats)
    for m in range(Int(py=len(mats))):
        var nm = NamedMaterial(String("usd") + String(m))
        nm.kind = MatKind.diffuse
        nm.albedo = RGB(rgb[unsafe_offset=3 * m], rgb[unsafe_offset=3 * m + 1], rgb[unsafe_offset=3 * m + 2])
        var color = String(py=d["mat_color"][m])
        if color.byte_length() > 0:
            nm.tex_idx = _usd_texture(s, color)
            nm.albedo = RGB(Float32(1))
        var normal = String(py=d["mat_normal"][m])
        if normal.byte_length() > 0:
            nm.normal_tex_idx = _usd_texture(s, normal)
        s[unsafe_offset=0].named_materials.append(nm^)

    var points = _f32(d["points"])
    var indices = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=d["indices"].ctypes.data))
    var uvs = _f32(d["uvs"])
    var xf = _f32(d["mesh_xf"])
    var rows_obj = d["meshes"]
    var rows = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=rows_obj.ctypes.data))
    var n_meshes = Int(py=len(rows_obj))
    var n_tmpl = Int(py=d["n_templates"])
    s[unsafe_offset=0].cur_attr.is_alight = False

    # World meshes, then each template's meshes between ObjectBegin/ObjectEnd as the pbrt parser records them.
    for t in range(-1, n_tmpl):
        if t >= 0:
            s[unsafe_offset=0].pending_object_name = String("usd_t") + String(t)
            s[unsafe_offset=0].pending_object_start = Int32(len(s[unsafe_offset=0].meshes))
            s[unsafe_offset=0].pending_object_curve_start = Int32(len(s[unsafe_offset=0].curves_mat))
            for k in range(16):
                s[unsafe_offset=0].pending_object_ctm[k] = Float32(1) if k % 5 == 0 else Float32(0)
        for i in range(n_meshes):
            var r = rows.unsafe_offset(7 * i)
            if Int(r[unsafe_offset=5]) != t:
                continue
            _set_ctm(s, xf.unsafe_offset(16 * i))
            s[unsafe_offset=0].cur_attr.mat_idx = Int32(r[unsafe_offset=4])
            store_mesh(s, points.unsafe_offset(3 * Int(r[unsafe_offset=0])), indices.unsafe_offset(Int(r[unsafe_offset=2])),
                       Int32(r[unsafe_offset=1]), Int32(r[unsafe_offset=3]))
            if r[unsafe_offset=6] != 0:
                ref mesh = s[unsafe_offset=0].meshes[len(s[unsafe_offset=0].meshes) - 1]
                var uv = uvs.unsafe_offset(2 * Int(r[unsafe_offset=0]))
                mesh.uvs.reserve(2 * Int(r[unsafe_offset=1]))
                for k in range(2 * Int(r[unsafe_offset=1])):
                    mesh.uvs.append(uv[unsafe_offset=k])
        if t >= 0:
            _set_identity(s)
            _psc_finish_object_def(s)

    var inst_tmpl_obj = d["inst_tmpl"]
    var inst_tmpl = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=inst_tmpl_obj.ctypes.data))
    var inst_xf = _f32(d["inst_xf"])
    for k in range(Int(py=len(inst_tmpl_obj))):
        _set_ctm(s, inst_xf.unsafe_offset(16 * k))
        _psc_emit_object_instance(s, String("usd_t") + String(inst_tmpl[unsafe_offset=k]))
    _set_identity(s)

    var cam = d["camera"]
    if Int(py=len(cam)) == 16:
        _usd_camera(s, _f32(cam), Float32(Float64(py=d["fov"])))
    _usd_lights(s, d)


def _usd_lights(s: Pointer[SceneParseState, MutUntrackedOrigin], d: PythonObject) raises:
    var black = NamedMaterial(String("usd_light"))
    black.kind = MatKind.diffuse
    black.albedo = RGB(Float32(0))
    var light_mat = Int32(len(s[unsafe_offset=0].named_materials))
    s[unsafe_offset=0].named_materials.append(black^)

    var rows_obj = d["emit_rows"]
    var rows = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=rows_obj.ctypes.data))
    var pts = _f32(d["emit_pts"])
    var idx = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=d["emit_idx"].ctypes.data))
    var ergb = _f32(d["emit_rgb"])
    _set_identity(s)
    s[unsafe_offset=0].cur_attr.mat_idx = light_mat
    s[unsafe_offset=0].cur_attr.is_alight = True
    for i in range(Int(py=len(rows_obj))):
        var r = rows.unsafe_offset(4 * i)
        s[unsafe_offset=0].cur_attr.al_rgb = RGB(ergb[unsafe_offset=3 * i], ergb[unsafe_offset=3 * i + 1], ergb[unsafe_offset=3 * i + 2])
        store_mesh(s, pts.unsafe_offset(3 * Int(r[unsafe_offset=0])), idx.unsafe_offset(Int(r[unsafe_offset=2])),
                   Int32(r[unsafe_offset=1]), Int32(r[unsafe_offset=3]))
    s[unsafe_offset=0].cur_attr.is_alight = False

    var sph_obj = d["spheres"]
    var sph = _f32(sph_obj)
    for i in range(Int(py=len(sph_obj))):
        var q = sph.unsafe_offset(7 * i)
        s[unsafe_offset=0].spheres_cx.append(q[unsafe_offset=0])
        s[unsafe_offset=0].spheres_cy.append(q[unsafe_offset=1])
        s[unsafe_offset=0].spheres_cz.append(q[unsafe_offset=2])
        s[unsafe_offset=0].spheres_r.append(q[unsafe_offset=3])
        s[unsafe_offset=0].spheres_mat.append(light_mat)
        s[unsafe_offset=0].spheres_inside_med.append(s[unsafe_offset=0].cur_attr.inside_medium)
        s[unsafe_offset=0].spheres_outside_med.append(s[unsafe_offset=0].cur_attr.outside_medium)
        s[unsafe_offset=0].spheres_al.append(True)
        s[unsafe_offset=0].spheres_rgb.append(RGB(q[unsafe_offset=4], q[unsafe_offset=5], q[unsafe_offset=6]))

    var dist_obj = d["distant"]
    var dist = _f32(dist_obj)
    for i in range(Int(py=len(dist_obj))):
        for c in range(3):
            s[unsafe_offset=0].distant_dirs.append(dist[unsafe_offset=6 * i + c])
            s[unsafe_offset=0].distant_rgbs.append(dist[unsafe_offset=6 * i + 3 + c])

    var sky = _f32(d["sky"])
    var lit = Int(py=d["n_domes"]) > 0 or Int(py=len(rows_obj)) + Int(py=len(sph_obj)) + Int(py=len(dist_obj)) > 0
    if Int(py=d["n_domes"]) > 0:
        _usd_sky(s, RGB(sky[unsafe_offset=0], sky[unsafe_offset=1], sky[unsafe_offset=2]))
    elif not lit:
        _usd_sky(s, RGB(Float32(1)))   # an unlit stage: a white sky to see it


def mojo_parse_usd_scene(path: Pointer[UInt8, MutUntrackedOrigin], verbose: Bool = False) -> Pointer[ParsedScene_Mojo, MutUntrackedOrigin]:
    external_call["createTextureSystem", NoneType]()
    var path_str = String(unsafe_from_utf8_ptr=path.as_imm())
    var s_ptr = unsafe_alloc[SceneParseState](1)
    s_ptr.unsafe_write(SceneParseState())
    try:
        Python.add_to_path(env_str["GONZALES_DATA_DIR"]("src/gonzales/data"))
        var ux = Python.import_module("usd_extract")
        var aspect = Float64(s_ptr[unsafe_offset=0].film_w) / Float64(s_ptr[unsafe_offset=0].film_h)
        var d = ux.extract(path_str, env_str["GONZALES_USD_CAMERA"](""), aspect, env_str["GONZALES_USD_TIME"](""))
        _usd_fill(s_ptr, d)
        if verbose:
            print("USD:", Int(py=len(d["meshes"])), "meshes,", Int(py=d["n_templates"]), "templates,",
                  Int(py=len(d["inst_tmpl"])), "instances")
    except e:
        print("Error: cannot read USD scene", path_str + ":", e)
        _ = s_ptr.unsafe_take_pointee()
        s_ptr.unsafe_free()
        return Pointer[ParsedScene_Mojo, MutUntrackedOrigin].unsafe_dangling()
    var psc = unsafe_alloc[ParsedScene_Mojo](1)
    finalize_scene(s_ptr, psc, verbose)
    _ = s_ptr.unsafe_take_pointee()
    s_ptr.unsafe_free()
    return psc
