# C API of the Gonzales shared library, called by the Hydra render delegate.
from std.ffi import external_call
from std.memory.alloc import unsafe_alloc
from std.memory import unsafe_memcpy
from std.os import getenv, setenv
from std.time import perf_counter_ns
from std.runtime import initialize_runtime
from max.gpu.host import DeviceContext
from gonzales.geometry import RGB, _is_real_ptr
from gonzales.materials import MatKind
from gonzales.parse_types import SceneParseState, NamedMaterial
from gonzales.pbrt_parser import ParsedScene_Mojo, finalize_scene, mojo_apply_overrides
from gonzales.pipeline import (_generate_sobol_matrices, RenderSession, SessionPtr, session_new, session_set_camera, session_set_samples, session_set_lens, session_set_filter_glossy, session_set_adaptive, session_set_clamp, session_set_bounces,
                               session_render, session_resolve, session_free)
from gonzales.scene_builder import store_mesh
from gonzales.spectrum import load_spectral_context, spectral_handle, SpectralContext

comptime F32Ptr = Pointer[Float32, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime ScenePtr = Pointer[SceneParseState, MutUntrackedOrigin]

struct Renderer(Movable):
    var sobol: Pointer[UInt32, MutUntrackedOrigin]
    var spectral: SpectralContext
    var ctx: Optional[DeviceContext]

    def __init__(out self, sobol: Pointer[UInt32, MutUntrackedOrigin], var spectral: SpectralContext, var ctx: Optional[DeviceContext]):
        self.sobol = sobol
        self.spectral = spectral^
        self.ctx = ctx^

comptime RendererPtr = Pointer[Renderer, MutUntrackedOrigin]

@export
def gz_renderer_new(data_dir: Pointer[UInt8, MutUntrackedOrigin]) abi("C") -> RendererPtr:
    # Without the caching allocator a freed session gives its memory back to the driver; renders allocate nothing.
    _ = setenv("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE", "0", False)
    initialize_runtime()  # Mojo's CPU worker pool; an executable's main starts it
    var dir = String(unsafe_from_utf8_ptr=data_dir.as_imm())
    var sobol = _generate_sobol_matrices(dir + "/new-joe-kuo-6.21201")
    var spectral = load_spectral_context(dir)
    if not sobol or not spectral[0]:
        print("Gonzales: cannot load tables from", dir)
        return RendererPtr.unsafe_dangling()
    var ctx: Optional[DeviceContext] = None
    try:
        ctx = DeviceContext()
    except e:
        print("Gonzales: no GPU:", e)
        return RendererPtr.unsafe_dangling()
    external_call["createTextureSystem", NoneType]()
    _ = external_call["ptex_set_paging", NoneType, Int32](Int32(1))
    var r = unsafe_alloc[Renderer](1)
    r.unsafe_write(Renderer(sobol.value(), spectral[1].copy(), ctx^))
    return r

@export
def gz_scene_new() abi("C") -> ScenePtr:
    var s = unsafe_alloc[SceneParseState](1)
    s.unsafe_write(SceneParseState())
    return s

def _register_texture(s: ScenePtr, name: String, path: Pointer[UInt8, MutUntrackedOrigin]) -> Int32:
    if Int(path) == 0:
        return Int32(-1)
    s[unsafe_offset=0].tex_names.append(name)
    s[unsafe_offset=0].tex_files.append(String(unsafe_from_utf8_ptr=path.as_imm()))
    return Int32(len(s[unsafe_offset=0].tex_names) - 1)

@export
def gz_scene_add_diffuse(s: ScenePtr, r: Float32, g: Float32, b: Float32, texture: Pointer[UInt8, MutUntrackedOrigin],
                         normal_map: Pointer[UInt8, MutUntrackedOrigin]) abi("C") -> Int32:
    """`texture` (sRGB colour) and `normal_map` (tangent space) are image paths or null."""
    var name = String("m") + String(len(s[unsafe_offset=0].named_materials))
    var nm = NamedMaterial(name)
    nm.kind = MatKind.diffuse
    nm.albedo = RGB(r, g, b)
    nm.tex_idx = _register_texture(s, name, texture)
    nm.normal_tex_idx = _register_texture(s, String("__normalmap"), normal_map)
    s[unsafe_offset=0].named_materials.append(nm^)
    return Int32(len(s[unsafe_offset=0].named_materials) - 1)

@export
def gz_scene_add_coated(s: ScenePtr, r: Float32, g: Float32, b: Float32, texture: Pointer[UInt8, MutUntrackedOrigin],
                        normal_map: Pointer[UInt8, MutUntrackedOrigin], alpha: Float32, ior: Float32) abi("C") -> Int32:
    """Cycles' Principled BSDF: a GGX dielectric specular layer of roughness `alpha` and index `ior` over a diffuse base."""
    var name = String("m") + String(len(s[unsafe_offset=0].named_materials))
    var nm = NamedMaterial(name)
    nm.kind = MatKind.coated_diffuse
    nm.albedo = RGB(r, g, b)
    nm.tex_idx = _register_texture(s, name, texture)
    nm.normal_tex_idx = _register_texture(s, String("__normalmap"), normal_map)
    nm.roughness_u = alpha
    nm.roughness_v = alpha
    nm.ior = ior
    nm.principled = True
    s[unsafe_offset=0].named_materials.append(nm^)
    return Int32(len(s[unsafe_offset=0].named_materials) - 1)

@export
def gz_scene_add_mesh(s: ScenePtr, xform: F32Ptr, points: F32Ptr, n_verts: Int32, indices: I32Ptr, n_tris: Int32,
                      normals: F32Ptr, uvs: F32Ptr, material: Int32) abi("C"):
    """`xform` is object-to-world in USD's GfMatrix4d layout, which is Gonzales's CTM layout.
    `normals` (world space, unit length) and `uvs` are per vertex, or null."""
    for i in range(16):
        s[unsafe_offset=0].ctm[i] = xform[unsafe_offset=i]
    s[unsafe_offset=0].cur_attr.mat_idx = material
    store_mesh(s, points, indices, n_verts, n_tris)
    ref m = s[unsafe_offset=0].meshes[len(s[unsafe_offset=0].meshes) - 1]
    if Int(normals) != 0:
        for i in range(Int(n_verts) * 3):
            m.normals.append(normals[unsafe_offset=i])
    if Int(uvs) != 0:
        for i in range(Int(n_verts) * 2):
            m.uvs.append(uvs[unsafe_offset=i])

@export
def gz_scene_add_area_light(s: ScenePtr, xform: F32Ptr, points: F32Ptr, n_verts: Int32,
                            indices: I32Ptr, n_tris: Int32, r: Float32, g: Float32, b: Float32) abi("C"):
    """A one-sided emitter of radiance (r, g, b), transparent like a Cycles lamp."""
    for i in range(16):
        s[unsafe_offset=0].ctm[i] = xform[unsafe_offset=i]
    s[unsafe_offset=0].cur_attr.mat_idx = Int32(-1)
    s[unsafe_offset=0].cur_attr.is_alight = True
    s[unsafe_offset=0].cur_attr.al_transparent = True
    s[unsafe_offset=0].cur_attr.al_rgb = RGB(r, g, b)
    store_mesh(s, points, indices, n_verts, n_tris)
    s[unsafe_offset=0].cur_attr.is_alight = False
    s[unsafe_offset=0].cur_attr.al_transparent = False

def _add_infinite(s: ScenePtr, tex_idx: Int32, r: Float32, g: Float32, b: Float32):
    s[unsafe_offset=0].inf_tex_idx.append(tex_idx)
    s[unsafe_offset=0].inf_rgb.append(r); s[unsafe_offset=0].inf_rgb.append(g); s[unsafe_offset=0].inf_rgb.append(b)
    for ci in range(16):
        s[unsafe_offset=0].inf_ctm.append(Float32(1) if ci % 5 == 0 else Float32(0))
    s[unsafe_offset=0].inf_has_portal.append(Int32(0))
    for _ in range(12):
        s[unsafe_offset=0].inf_portal.append(Float32(0))

@export
def gz_scene_add_dome(s: ScenePtr, r: Float32, g: Float32, b: Float32) abi("C"):
    _add_infinite(s, Int32(-1), r, g, b)

@export
def gz_scene_add_dome_map(s: ScenePtr, path: Pointer[UInt8, MutUntrackedOrigin], r: Float32, g: Float32, b: Float32) abi("C"):
    """`path` is an equal-area square image in world space."""
    s[unsafe_offset=0].tex_names.append(String("__inf"))
    s[unsafe_offset=0].tex_files.append(String(unsafe_from_utf8_ptr=path.as_imm()))
    _add_infinite(s, Int32(len(s[unsafe_offset=0].tex_names) - 1), r, g, b)

@export
def gz_scene_add_distant(s: ScenePtr, dx: Float32, dy: Float32, dz: Float32,
                         r: Float32, g: Float32, b: Float32) abi("C"):
    """(dx, dy, dz) is the direction the light travels."""
    s[unsafe_offset=0].distant_dirs.append(dx); s[unsafe_offset=0].distant_dirs.append(dy); s[unsafe_offset=0].distant_dirs.append(dz)
    s[unsafe_offset=0].distant_rgbs.append(r); s[unsafe_offset=0].distant_rgbs.append(g); s[unsafe_offset=0].distant_rgbs.append(b)

@export
def gz_scene_set_camera(s: ScenePtr, world_to_camera: F32Ptr, fov_deg: Float32, width: Int32, height: Int32) abi("C"):
    """`world_to_camera` in Gonzales's camera convention (+z forward); `fov_deg` spans the shorter image axis."""
    for i in range(16):
        s[unsafe_offset=0].cam2w_raw[i] = world_to_camera[unsafe_offset=i]
    s[unsafe_offset=0].camera_fov = fov_deg
    s[unsafe_offset=0].film_w = width
    s[unsafe_offset=0].film_h = height

@export
def gz_scene_set_lens(s: ScenePtr, radius: Float32, focal_distance: Float32) abi("C"):
    """A thin lens in scene units; radius 0 is a pinhole."""
    s[unsafe_offset=0].lens_radius = radius
    s[unsafe_offset=0].focal_distance = focal_distance

@export
def gz_session_new(r: RendererPtr, s: ScenePtr, spp: Int32, max_depth: Int32) abi("C") -> OptionalPointer[RenderSession, MutUntrackedOrigin]:
    """Consumes the scene and puts it on the GPU; null if that fails."""
    s[unsafe_offset=0].max_depth = max_depth
    var psc = unsafe_alloc[ParsedScene_Mojo](1)
    var t0 = perf_counter_ns()
    finalize_scene(s, psc, False, curves_apart=getenv("GONZALES_RTCORE") != "" and getenv("GONZALES_CURVE_BVH") != "0")
    if getenv("GONZALES_TIMING") != "":
        print("Gonzales:   finalize", Float64(perf_counter_ns() - t0) / 1.0e9, "s")
    _ = s.unsafe_take_pointee()
    s.unsafe_free()
    mojo_apply_overrides(psc, spp, Int32(0), Int32(0))
    var session = session_new(psc, r[unsafe_offset=0].sobol, spectral_handle(r[unsafe_offset=0].spectral), r[unsafe_offset=0].ctx)
    if _is_real_ptr(session):
        return session
    return None

@export
def gz_session_set_camera(s: SessionPtr, world_to_camera: F32Ptr, fov_deg: Float32, width: Int32, height: Int32) abi("C") -> Int32:
    """As gz_scene_set_camera; starts the film over."""
    try:
        session_set_camera(s, world_to_camera, fov_deg, width, height)
        return Int32(0)
    except e:
        print("Gonzales: camera failed:", e)
        return Int32(-1)

@export
def gz_session_set_lens(s: SessionPtr, radius: Float32, focal_distance: Float32) abi("C"):
    """As gz_scene_set_lens; a change starts the film over."""
    session_set_lens(s, radius, focal_distance)

@export
def gz_session_set_clamp(s: SessionPtr, clamp: Float32) abi("C"):
    """Cycles' Clamp Indirect: a limit on each indirect light contribution; 0 is off."""
    session_set_clamp(s, clamp)

@export
def gz_session_set_bounces(s: SessionPtr, diffuse: Int32, glossy: Int32, transmission: Int32) abi("C"):
    """Cycles' diffuse, glossy and transmission bounce limits, with its Russian roulette."""
    session_set_bounces(s, diffuse, glossy, transmission)

@export
def gz_session_set_adaptive(s: SessionPtr, threshold: Float32, min_samples: Int32) abi("C"):
    """Cycles' adaptive sampling: noise threshold (0 is off) and minimum samples (0: Cycles' automatic count)."""
    session_set_adaptive(s, threshold, min_samples)

@export
def gz_session_set_filter_glossy(s: SessionPtr, filter: Float32) abi("C"):
    """Cycles' Filter Glossy: glossy lobes widen after rough bounces; 0 is off."""
    session_set_filter_glossy(s, filter)

@export
def gz_session_set_samples(s: SessionPtr, spp: Int32, max_depth: Int32) abi("C"):
    """The converged sample count and the path depth; a change starts the film over."""
    session_set_samples(s, spp, max_depth)

@export
def gz_session_render(s: SessionPtr, n_spp: Int32) abi("C") -> Int32:
    """Adds up to n_spp samples per pixel; returns the samples in the film, -1 on failure."""
    try:
        return Int32(session_render(s, Int(n_spp)))
    except e:
        print("Gonzales: render failed:", e)
        return Int32(-1)

@export
def gz_session_resolve(s: SessionPtr, out_rgb: F32Ptr, denoise: Int32) abi("C") -> Int32:
    """Writes width*height RGB floats, top row first."""
    try:
        session_resolve(s, out_rgb, denoise != Int32(0))
        return Int32(0)
    except e:
        print("Gonzales: resolve failed:", e)
        return Int32(-1)

@export
def gz_session_free(s: SessionPtr) abi("C"):
    session_free(s)

@export
def gz_scene_add_point(s: ScenePtr, x: Float32, y: Float32, z: Float32, r: Float32, g: Float32, b: Float32) abi("C"):
    """(r, g, b) is the radiant intensity."""
    s[unsafe_offset=0].point_pos.append(x); s[unsafe_offset=0].point_pos.append(y); s[unsafe_offset=0].point_pos.append(z)
    s[unsafe_offset=0].point_rgbs.append(r); s[unsafe_offset=0].point_rgbs.append(g); s[unsafe_offset=0].point_rgbs.append(b)

@export
def gz_scene_add_sphere_light(s: ScenePtr, x: Float32, y: Float32, z: Float32, radius: Float32,
                              r: Float32, g: Float32, b: Float32) abi("C"):
    """(r, g, b) is the radiance."""
    s[unsafe_offset=0].spheres_cx.append(x)
    s[unsafe_offset=0].spheres_cy.append(y)
    s[unsafe_offset=0].spheres_cz.append(z)
    s[unsafe_offset=0].spheres_r.append(radius)
    s[unsafe_offset=0].spheres_mat.append(Int32(-1))
    s[unsafe_offset=0].spheres_inside_med.append(Int32(-1))
    s[unsafe_offset=0].spheres_outside_med.append(Int32(-1))
    s[unsafe_offset=0].spheres_al.append(True)
    s[unsafe_offset=0].spheres_rgb.append(RGB(r, g, b))

@export
def gz_scene_add_hair(s: ScenePtr, sigma_r: Float32, sigma_g: Float32, sigma_b: Float32,
                      beta_m: Float32, beta_n: Float32, ior: Float32, alpha: Float32) abi("C") -> Int32:
    """Hair: pbrt's BSDF from absorption, roughnesses, index of refraction and scale tilt in degrees."""
    var nm = NamedMaterial(String("m") + String(len(s[unsafe_offset=0].named_materials)))
    nm.kind = MatKind.hair
    nm.albedo = RGB(sigma_r, sigma_g, sigma_b)
    nm.ior = ior
    nm.roughness_u = beta_m
    nm.roughness_v = beta_n
    nm.hair_alpha = alpha
    s[unsafe_offset=0].named_materials.append(nm^)
    return Int32(len(s[unsafe_offset=0].named_materials) - 1)

@export
def gz_scene_set_subsurface(s: ScenePtr, material: Int32, weight: Float32, radius_r: Float32, radius_g: Float32,
                            radius_b: Float32, anisotropy: Float32, van_de_hulst: Int32) abi("C"):
    """Gives a coated diffuse a Cycles-style random-walk base: its share, mean free path per channel and anisotropy."""
    ref nm = s[unsafe_offset=0].named_materials[Int(material)]
    nm.sss_weight = min(max(weight, Float32(0)), Float32(1))
    nm.sss_mfp = RGB(radius_r, radius_g, radius_b)
    nm.sss_g = min(max(anisotropy, Float32(-0.9)), Float32(0.9))
    nm.sss_van_de_hulst = van_de_hulst

@export
def gz_scene_set_emission(s: ScenePtr, material: Int32, texture: Pointer[UInt8, MutUntrackedOrigin]) abi("C"):
    """Gives a material emitted radiance from an image of linear RGB (Blender's Principled emission colour times strength)."""
    ref nm = s[unsafe_offset=0].named_materials[Int(material)]
    nm.emit_tex_idx = _register_texture(s, nm.name + "_emission", texture)

@export
def gz_scene_add_dielectric(s: ScenePtr, ior: Float32, alpha: Float32) abi("C") -> Int32:
    """Glass with GGX roughness `alpha` (0 is smooth)."""
    var nm = NamedMaterial(String("m") + String(len(s[unsafe_offset=0].named_materials)))
    nm.kind = MatKind.dielectric
    nm.ior = ior
    nm.roughness_u = alpha
    nm.roughness_v = alpha
    s[unsafe_offset=0].named_materials.append(nm^)
    return Int32(len(s[unsafe_offset=0].named_materials) - 1)

@export
def gz_scene_add_curves(s: ScenePtr, control_points: F32Ptr, widths: F32Ptr, n_segments: Int32, material: Int32,
                        materials: I32Ptr) abi("C"):
    """Round cubic B-spline segments in world space: 4 control points and 2 diameters each.
    `materials` (per segment) overrides `material` when not null."""
    ref st = s[unsafe_offset=0]
    var n = Int(n_segments)
    var c0 = len(st.curves_w0)
    st.curves_cp.resize(len(st.curves_cp) + n * 12, Float32(0))
    unsafe_memcpy(dest=st.curves_cp.unsafe_ptr().unsafe_offset(c0 * 12), src=control_points, count=n * 12)
    st.curves_w0.resize(c0 + n, Float32(0)); st.curves_w1.resize(c0 + n, Float32(0)); st.curves_mat.resize(c0 + n, material)
    st.curves_flat.resize(c0 + n, Int32(0)); st.curves_al.resize(c0 + n, False); st.curves_al_rgb.resize(c0 + n, RGB(Float32(0)))
    var w0 = st.curves_w0.unsafe_ptr().unsafe_offset(c0)
    var w1 = st.curves_w1.unsafe_ptr().unsafe_offset(c0)
    var mt = st.curves_mat.unsafe_ptr().unsafe_offset(c0)
    for i in range(n):
        w0[unsafe_offset=i] = widths[unsafe_offset=2 * i]
        w1[unsafe_offset=i] = widths[unsafe_offset=2 * i + 1]
        if Int(materials) != 0:
            mt[unsafe_offset=i] = materials[unsafe_offset=i]
