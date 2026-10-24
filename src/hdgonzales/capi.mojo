# C API of the Gonzales shared library, called by the Hydra render delegate.
from std.ffi import external_call
from std.memory.alloc import unsafe_alloc
from std.runtime import initialize_runtime
from max.gpu.host import DeviceContext
from gonzales.geometry import RGB
from gonzales.materials import MatKind
from gonzales.parse_types import SceneParseState, NamedMaterial
from gonzales.pbrt_parser import ParsedScene_Mojo, finalize_scene
from gonzales.pipeline import _generate_sobol_matrices, parse_and_render
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

@export
def gz_scene_add_diffuse(s: ScenePtr, r: Float32, g: Float32, b: Float32) abi("C") -> Int32:
    var nm = NamedMaterial(String("m") + String(len(s[unsafe_offset=0].named_materials)))
    nm.kind = MatKind.diffuse
    nm.albedo = RGB(r, g, b)
    s[unsafe_offset=0].named_materials.append(nm^)
    return Int32(len(s[unsafe_offset=0].named_materials) - 1)

@export
def gz_scene_add_mesh(s: ScenePtr, xform: F32Ptr, points: F32Ptr, n_verts: Int32,
                      indices: I32Ptr, n_tris: Int32, uvs: F32Ptr, material: Int32) abi("C"):
    """`xform` is object-to-world in USD's GfMatrix4d layout, which is Gonzales's CTM layout."""
    for i in range(16):
        s[unsafe_offset=0].ctm[i] = xform[unsafe_offset=i]
    s[unsafe_offset=0].cur_attr.mat_idx = material
    store_mesh(s, points, indices, n_verts, n_tris)
    if Int(uvs) != 0:
        ref m = s[unsafe_offset=0].meshes[len(s[unsafe_offset=0].meshes) - 1]
        for i in range(Int(n_verts) * 2):
            m.uvs.append(uvs[unsafe_offset=i])

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
def gz_render(r: RendererPtr, s: ScenePtr, spp: Int32, max_depth: Int32, out_rgb: F32Ptr) abi("C") -> Int32:
    """Renders and consumes the scene; writes width*height RGB floats."""
    s[unsafe_offset=0].max_depth = max_depth
    var psc = unsafe_alloc[ParsedScene_Mojo](1)
    finalize_scene(s, psc, False)
    _ = s.unsafe_take_pointee()
    s.unsafe_free()
    var spectral = spectral_handle(r[unsafe_offset=0].spectral)
    try:
        return parse_and_render(Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(),
                                r[unsafe_offset=0].sobol, True, spectral=spectral, preparsed=psc,
                                gpu_ctx=r[unsafe_offset=0].ctx, spp_override=spp, out_rgb=out_rgb)
    except e:
        print("Gonzales: render failed:", e)
        return Int32(-1)

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
