from .geometry import RGB

# ── ParsedScene constants ─────────────────────────────────────────────────────

comptime PSC_MAX_MESHES = 1024
comptime PSC_MAX_NAMED  = 64
comptime PSC_CTM_DEPTH  = 16
comptime PSC_ATTR_DEPTH = 8
comptime PSC_NAME_MAX   = 64
comptime PSC_FILE_MAX   = 256
comptime PSC_MAX_TEX    = 64

# ── Hair curve tessellation constants ─────────────────────────────────────────
# B-spline curves (Shape "curve") are tessellated into cross-ribbon triangles.
# HAIR_EVAL_N points are sampled uniformly along each strand (B-spline),
# giving HAIR_EVAL_N-1 ribbon segments. Each segment becomes 2 perpendicular
# quads (4 triangles) forming a cross-shaped profile visible from any angle.
comptime HAIR_EVAL_N    = 8          # sample points along each strand
comptime HAIR_MAX_VTX   = 15_000_000 # lazy buffer: max accumulated hair vertices
comptime HAIR_MAX_TRI   = 10_000_000 # lazy buffer: max accumulated hair triangles

# ── Internal parse state ──────────────────────────────────────────────────────
# These types are private to the parsing subsystem.

@fieldwise_init
struct AttributeState(Copyable, ImplicitlyCopyable, Movable):
    var mat_idx:        Int32
    var is_alight:      Bool
    var al_rgb:         RGB
    var inside_medium:  Int32
    var outside_medium: Int32
    var reverse_orient: Bool   # PBRT ReverseOrientation: flip surface normals

struct NamedMaterial(Copyable, ImplicitlyCopyable, Movable):
    var name:           String
    var albedo:         RGB
    var kind:           Int8
    var ior:            Float32
    var roughness_u:    Float32
    var roughness_v:    Float32
    var tex_idx:        Int32
    var normal_tex_idx: Int32
    var mix_name1:      String
    var mix_name2:      String
    var mix_amount:     Float32
    var transmittance:  RGB

    def __init__(out self, name: String):
        self.name           = name
        self.albedo         = RGB(Float32(0.8), Float32(0.8), Float32(0.8))
        self.kind           = Int8(0)
        self.ior            = Float32(1.5)
        self.roughness_u    = Float32(0)
        self.roughness_v    = Float32(0)
        self.tex_idx        = Int32(-1)
        self.normal_tex_idx = Int32(-1)
        self.mix_name1      = String("")
        self.mix_name2      = String("")
        self.mix_amount     = Float32(0.5)
        self.transmittance  = RGB(Float32(1), Float32(1), Float32(1))

struct MeshAccum(Copyable, Movable):
    var points:         List[Float32]  # 4 floats per vertex (xyz + pad)
    var vert_idxs:      List[Int64]    # flat vertex indices
    var face_idxs:      List[Int64]    # flat face indices
    var uvs:            List[Float32]  # 2 floats per vertex (or empty)
    var normals:        List[Float32]  # 3 floats per vertex (or empty) — world-space shading normals
    var mat_idx:        Int32
    var is_area_light:  Bool
    var al_rgb:         RGB
    var inside_medium:  Int32
    var outside_medium: Int32

    def __init__(out self, mat_idx: Int32, inside_medium: Int32, outside_medium: Int32):
        self.points        = List[Float32]()
        self.vert_idxs     = List[Int64]()
        self.face_idxs     = List[Int64]()
        self.uvs           = List[Float32]()
        self.normals       = List[Float32]()
        self.mat_idx       = mat_idx
        self.is_area_light = False
        self.al_rgb        = RGB(Float32(0), Float32(0), Float32(0))
        self.inside_medium  = inside_medium
        self.outside_medium = outside_medium

struct SceneParseState(Movable):
    # Current transform matrix and stack
    var ctm:       InlineArray[Float32, 16]
    var ctm_stack: List[InlineArray[Float32, 16]]

    # Attribute stack (material, area-light, medium per nesting level)
    var attr_stack: List[AttributeState]
    var cur_attr:   AttributeState

    # Named materials (MakeNamedMaterial / Material directives)
    var named_materials: List[NamedMaterial]

    # Mesh accumulation
    var meshes: List[MeshAccum]

    # Hair curve accumulator (lazily populated on first Shape "curve")
    var hair:         Optional[MeshAccum]

    # Non-area lights
    var distant_dirs: List[Float32]   # 3 floats per light
    var distant_rgbs: List[Float32]   # 3 floats per light
    var point_pos:    List[Float32]   # 3 floats per light
    var point_rgbs:   List[Float32]   # 3 floats per light
    var inf_tex_idx:  List[Int32]     # 1 per infinite light
    var inf_rgb:      List[Float32]   # 3 floats per light
    var inf_ctm:      List[Float32]   # 16 floats per light

    # Analytical sphere primitives
    var spheres_cx:  List[Float32]
    var spheres_cy:  List[Float32]
    var spheres_cz:  List[Float32]
    var spheres_r:   List[Float32]
    var spheres_mat: List[Int32]
    var spheres_al:  List[Bool]
    var spheres_rgb: List[RGB]
    var spheres_inside_med:  List[Int32]
    var spheres_outside_med: List[Int32]

    # Homogeneous media
    var med_names: List[String]
    var med_sa:    List[Float32]   # 3 floats per medium
    var med_ss:    List[Float32]   # 3 floats per medium
    var med_g:     List[Float32]   # 1 per medium

    # Medium interfaces
    var miface_inside:  List[Int32]
    var miface_outside: List[Int32]
    var miface_mat:     List[Int32]

    # Textures
    var tex_names: List[String]
    var tex_files: List[String]
    # Constant textures: name -> RGB value (3 floats per entry, parallel to names)
    var const_tex_names: List[String]
    var const_tex_rgb: List[Float32]

    # Film / camera / sampler settings
    var film_w:           Int32
    var film_h:           Int32
    var film_iso:         Float32
    var film_max_comp:    Float32
    var film_filename:    String
    var filter_sigma:     Float32
    var filter_support_x: Float32
    var filter_support_y: Float32
    var filter_type:      Int32      # 0=gaussian 1=triangle 2=box
    var samples_per_pixel: Int32
    var camera_fov:       Float32
    var cam2w_raw:        InlineArray[Float32, 16]
    var max_depth:        Int32
    var scene_dir:        String
    var object_depth:     Int32

    def __init__(out self):
        # Identity CTM
        self.ctm = InlineArray[Float32, 16](fill=Float32(0))
        self.ctm[0] = Float32(1); self.ctm[5] = Float32(1)
        self.ctm[10] = Float32(1); self.ctm[15] = Float32(1)
        self.ctm_stack = List[InlineArray[Float32, 16]]()

        self.cur_attr  = AttributeState(Int32(-1), False,
                             RGB(Float32(0),Float32(0),Float32(0)),
                             Int32(-1), Int32(-1), False)
        self.attr_stack = List[AttributeState]()

        self.named_materials = List[NamedMaterial]()
        self.meshes          = List[MeshAccum]()
        self.hair            = None

        self.distant_dirs = List[Float32]()
        self.distant_rgbs = List[Float32]()
        self.point_pos    = List[Float32]()
        self.point_rgbs   = List[Float32]()
        self.inf_tex_idx  = List[Int32]()
        self.inf_rgb      = List[Float32]()
        self.inf_ctm      = List[Float32]()

        self.spheres_cx  = List[Float32]()
        self.spheres_cy  = List[Float32]()
        self.spheres_cz  = List[Float32]()
        self.spheres_r   = List[Float32]()
        self.spheres_mat = List[Int32]()
        self.spheres_al  = List[Bool]()
        self.spheres_rgb = List[RGB]()
        self.spheres_inside_med  = List[Int32]()
        self.spheres_outside_med = List[Int32]()

        self.med_names = List[String]()
        self.med_sa    = List[Float32]()
        self.med_ss    = List[Float32]()
        self.med_g     = List[Float32]()

        self.miface_inside  = List[Int32]()
        self.miface_outside = List[Int32]()
        self.miface_mat     = List[Int32]()

        self.tex_names = List[String]()
        self.tex_files = List[String]()
        self.const_tex_names = List[String]()
        self.const_tex_rgb = List[Float32]()

        self.film_w           = Int32(512)
        self.film_h           = Int32(512)
        self.film_iso         = Float32(100)
        self.film_max_comp    = Float32(0)
        self.film_filename    = String("gonzales.exr")
        self.filter_sigma     = Float32(0.5)
        self.filter_support_x = Float32(1.5)
        self.filter_support_y = Float32(1.5)
        self.filter_type      = Int32(0)
        self.samples_per_pixel = Int32(1)
        self.camera_fov       = Float32(30)
        self.cam2w_raw        = InlineArray[Float32, 16](fill=Float32(0))
        self.cam2w_raw[0] = Float32(1); self.cam2w_raw[5] = Float32(1)
        self.cam2w_raw[10] = Float32(1); self.cam2w_raw[15] = Float32(1)
        self.max_depth  = Int32(5)
        self.scene_dir  = String("")
        self.object_depth = Int32(0)


# ── CTM stack helpers (operate on SceneParseState only) ──────────────────────

def ctm_push(mut s: SceneParseState):
    s.ctm_stack.append(s.ctm)

def ctm_pop(mut s: SceneParseState):
    if len(s.ctm_stack) > 0:
        s.ctm = s.ctm_stack[len(s.ctm_stack) - 1]
        _ = s.ctm_stack.pop()

def scene_parse_state_new() -> SceneParseState:
    return SceneParseState()
