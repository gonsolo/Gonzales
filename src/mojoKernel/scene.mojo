from std.collections import List
from std.ffi import external_call
from .geometry import (
    RGB, Point3f, Vec3f,
    Material_C, AreaLight_C, TriangleMesh_C, PrimId_C,
    Sphere_C, DistantLight_C, PointLight_C, InfiniteLight_C,
    Medium_C, MediumInterface_C,
)
from .bvh import BVH2Node, SceneDescriptor2_C

# ── Scene IR ──────────────────────────────────────────────────────────────────
#
# `Scene` is the stable intermediate representation between the parser and the
# renderer — analogous to LLVM IR. It owns all scene data via `List[T]` fields
# (auto-freed when it goes out of scope) and exposes raw pointers only at FFI
# call sites via `.unsafe_ptr()`.
#
# Public API:
#   from gonzales.scene import Scene
#   from gonzales.parsing import parse_pbrt
#   var scene = parse_pbrt("my_scene.pbrt")
#   var film  = render(scene, RenderOptions(...))

# ── Camera / film settings ────────────────────────────────────────────────────

@fieldwise_init
struct FilmSettings(Copyable, Movable):
    var width:      Int32
    var height:     Int32
    var iso:        Float32
    var max_comp:   Float32
    var filename:   String
    var filter_sigma:     Float32
    var filter_support_x: Float32
    var filter_support_y: Float32
    var filter_norm_x:    Float32
    var filter_norm_y:    Float32
    var filter_weight:    Float32
    var filter_type:      Int32

@fieldwise_init
struct CameraSettings(Copyable, Movable):
    var raster_to_camera: InlineArray[Float32, 16]  # column-major 4×4
    var camera_to_world:  InlineArray[Float32, 16]  # column-major 4×4
    var fov_degrees:      Float32

@fieldwise_init
struct SamplerSettings(Copyable, Movable):
    var samples_per_pixel: Int32
    var log2_spp:          Int32
    var n_base4_digits:    Int32
    var max_depth:         Int32
    var rng_seed:          UInt64

# ── Scene ─────────────────────────────────────────────────────────────────────

struct Scene(Movable):
    """The parsed scene: owns all geometry, materials, lights, and settings.

    This is the IR layer between the parser and the renderer. Create it with
    `parse_pbrt(path)`. Pass it to `render(scene, opts)` or `gpu_render(...)`.
    All memory is automatically released when `Scene` goes out of scope.
    """

    var camera:  CameraSettings
    var film:    FilmSettings
    var sampler: SamplerSettings

    # Geometry
    var materials:    List[Material_C]
    var meshes:       List[TriangleMesh_C]   # wire format kept for GPU upload
    var bvh_nodes:    List[BVH2Node]
    var prim_ids:     List[PrimId_C]

    # Per-mesh geometry arrays (parallel to `meshes`)
    var mesh_points: List[List[Float32]]     # 4 floats per vertex (xyz + pad)
    var mesh_vis:    List[List[Int64]]       # vertex indices (flat)
    var mesh_fis:    List[List[Int64]]       # face indices (flat)
    var mesh_uvs:    List[List[Float32]]     # UV coords (2 floats per vertex)

    # Lights
    var area_lights:     List[AreaLight_C]
    var distant_lights:  List[DistantLight_C]
    var point_lights:    List[PointLight_C]
    var infinite_lights: List[InfiniteLight_C]
    var spheres:         List[Sphere_C]

    # Media
    var mediums:      List[Medium_C]
    var medium_ifaces: List[MediumInterface_C]

    # Textures (file paths, loaded on demand by the renderer)
    var tex_filenames: List[String]

    fn __init__(out self,
        camera: CameraSettings,
        film: FilmSettings,
        sampler: SamplerSettings,
        materials: List[Material_C],
        meshes: List[TriangleMesh_C],
        bvh_nodes: List[BVH2Node],
        prim_ids: List[PrimId_C],
        mesh_points: List[List[Float32]],
        mesh_vis: List[List[Int64]],
        mesh_fis: List[List[Int64]],
        mesh_uvs: List[List[Float32]],
        area_lights: List[AreaLight_C],
        distant_lights: List[DistantLight_C],
        point_lights: List[PointLight_C],
        infinite_lights: List[InfiniteLight_C],
        spheres: List[Sphere_C],
        mediums: List[Medium_C],
        medium_ifaces: List[MediumInterface_C],
        tex_filenames: List[String],
    ):
        self.camera  = camera^
        self.film    = film^
        self.sampler = sampler^
        self.materials    = materials^
        self.meshes       = meshes^
        self.bvh_nodes    = bvh_nodes^
        self.prim_ids     = prim_ids^
        self.mesh_points  = mesh_points^
        self.mesh_vis     = mesh_vis^
        self.mesh_fis     = mesh_fis^
        self.mesh_uvs     = mesh_uvs^
        self.area_lights  = area_lights^
        self.distant_lights  = distant_lights^
        self.point_lights    = point_lights^
        self.infinite_lights = infinite_lights^
        self.spheres      = spheres^
        self.mediums      = mediums^
        self.medium_ifaces = medium_ifaces^
        self.tex_filenames = tex_filenames^

    fn __del__(owned self):
        # InfiniteLight_C has OIIO-managed pixel data that needs explicit free.
        for i in range(len(self.infinite_lights)):
            var il = self.infinite_lights[i]
            if Int(il.cdf_ptr) != 0:
                il.cdf_ptr.free()
            _ = external_call["free_texture_rgb", Int32,
                UnsafePointer[Float32, MutAnyOrigin]](il.pixels_ptr)
        # All List[T] fields are freed automatically after __del__ body.

    fn n_pixels(self) -> Int:
        """Convenience: total pixel count."""
        return Int(self.film.width) * Int(self.film.height)

    fn scene_descriptor(self) -> SceneDescriptor2_C:
        """Build the CPU-renderer scene handle (raw pointers into our Lists).
        Valid only while `self` is live.
        """
        return SceneDescriptor2_C(
            bvh2Nodes         = self.bvh_nodes.unsafe_ptr(),
            primIds           = self.prim_ids.unsafe_ptr(),
            meshes            = self.meshes.unsafe_ptr(),
            meshCount         = Int64(len(self.meshes)),
            materials         = self.materials.unsafe_ptr(),
            materialCount     = Int64(len(self.materials)),
            areaLights        = self.area_lights.unsafe_ptr(),
            areaLightCount    = Int64(len(self.area_lights)),
            textures          = UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
            textureCount      = Int64(len(self.tex_filenames)),
            distantLights     = self.distant_lights.unsafe_ptr(),
            distantLightCount = Int64(len(self.distant_lights)),
            pointLights       = self.point_lights.unsafe_ptr(),
            pointLightCount   = Int64(len(self.point_lights)),
            infiniteLights    = self.infinite_lights.unsafe_ptr(),
            infiniteLightCount = Int64(len(self.infinite_lights)),
            spheres           = self.spheres.unsafe_ptr(),
            sphereCount       = Int64(len(self.spheres)),
            mediums           = self.mediums.unsafe_ptr(),
            mediumCount       = Int64(len(self.mediums)),
            mediumInterfaces  = self.medium_ifaces.unsafe_ptr(),
            mediumIfaceCount  = Int64(len(self.medium_ifaces)),
        )
