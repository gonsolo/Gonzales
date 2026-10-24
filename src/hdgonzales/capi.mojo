# C API of the Gonzales shared library, called by the Hydra render delegate.
from std.ffi import external_call
from std.memory.alloc import unsafe_alloc
from std.os import setenv
from std.runtime import initialize_runtime
from max.gpu.host import DeviceContext
from gonzales.geometry import RGB, _is_real_ptr
from gonzales.materials import MatKind
from gonzales.parse_types import SceneParseState, NamedMaterial
from gonzales.pbrt_parser import ParsedScene_Mojo, finalize_scene, mojo_apply_overrides
from gonzales.pipeline import (_generate_sobol_matrices, RenderSession, SessionPtr, session_new, session_set_camera, session_set_samples,
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
    """A one-sided emitter of radiance (r, g, b)."""
    for i in range(16):
        s[unsafe_offset=0].ctm[i] = xform[unsafe_offset=i]
    s[unsafe_offset=0].cur_attr.mat_idx = Int32(-1)
    s[unsafe_offset=0].cur_attr.is_alight = True
    s[unsafe_offset=0].cur_attr.al_rgb = RGB(r, g, b)
    store_mesh(s, points, indices, n_verts, n_tris)
    s[unsafe_offset=0].cur_attr.is_alight = False

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
def gz_session_new(r: RendererPtr, s: ScenePtr, spp: Int32, max_depth: Int32) abi("C") -> OptionalPointer[RenderSession, MutUntrackedOrigin]:
    """Consumes the scene and puts it on the GPU; null if that fails."""
    s[unsafe_offset=0].max_depth = max_depth
    var psc = unsafe_alloc[ParsedScene_Mojo](1)
    finalize_scene(s, psc, False)
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
                      beta_m: Float32, beta_n: Float32) abi("C") -> Int32:
    """pbrt's hair BSDF from its absorption coefficient and roughnesses."""
    var nm = NamedMaterial(String("m") + String(len(s[unsafe_offset=0].named_materials)))
    nm.kind = MatKind.hair
    nm.albedo = RGB(sigma_r, sigma_g, sigma_b)
    nm.ior = Float32(1.55)
    nm.roughness_u = beta_m
    nm.roughness_v = beta_n
    s[unsafe_offset=0].named_materials.append(nm^)
    return Int32(len(s[unsafe_offset=0].named_materials) - 1)

@export
def gz_scene_add_curves(s: ScenePtr, control_points: F32Ptr, widths: F32Ptr, n_segments: Int32, material: Int32) abi("C"):
    """Round cubic B-spline segments in world space: 4 control points and 2 diameters each."""
    ref st = s[unsafe_offset=0]
    st.curves_cp.reserve(len(st.curves_cp) + Int(n_segments) * 12)
    for i in range(Int(n_segments) * 12):
        st.curves_cp.append(control_points[unsafe_offset=i])
    for i in range(Int(n_segments)):
        st.curves_w0.append(widths[unsafe_offset=2 * i])
        st.curves_w1.append(widths[unsafe_offset=2 * i + 1])
        st.curves_mat.append(material)
        st.curves_flat.append(Int32(0))
        st.curves_al.append(False)
        st.curves_al_rgb.append(RGB(Float32(0)))
