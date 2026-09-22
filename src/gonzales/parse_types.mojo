from .geometry import RGB
from std.os.path import exists


def scene_path(scene_dir: String, name: String, what: String) -> String:
    """THE way a filename written in a scene file becomes a path on disk.

    An ABSOLUTE name is used as-is; a relative one is resolved against the
    scene's own directory, as pbrt does. Every asset site used to write
    `scene_dir + name` by hand -- nine of them, some by raw byte copying -- so
    an absolute path became `<scene_dir>//abs/path`, failed to load, and the
    asset silently fell back to a flat default. For an environment map that
    means a uniform white sky: a test scene lit by barcelona's sky.exr through
    an absolute path rendered ~3.9x brighter than pbrt in all three
    integrators, with nothing in the log but OIIO's own stderr noise.

    It also checks the file exists and WARNS if not, naming what it was for
    (`what`, e.g. "environment map") -- at parse time, so the warning fires
    the same way for every asset type, integrator and backend, instead of
    depending on whichever loader happens to touch the file later."""
    var path = name if name.startswith("/") else scene_dir + name
    if name.byte_length() > 0 and not exists(path):
        print("Warning: " + what + " file not found: '" + path
              + "' -- it will render as a flat default instead.")
    return path

# ── ParsedScene constants ─────────────────────────────────────────────────────

# Must be >= the buffer _psc_collect_params uses for string-typed parameter
# VALUES (512, lexer.mojo) -- a texture/material/object NAME registered
# through a smaller buffer here than a "texture reflectance"-style reference
# to that same name truncates the registration but not the reference,
# breaking the string-equality lookup silently (no error, just a fallback to
# a default value). Real-world case: bistro_cafe's 66-char
# "Pavement_Cobblestone_Wet_Leaves_BLENDSHADER_BaseColor" texture name
# truncated to 63 chars at registration, so its own material's "texture
# reflectance" reference (the untruncated 66-char string) never matched --
# the ground plane silently fell back to flat 0.5 gray, inflating bounce
# lighting across the whole scene (~2x too bright vs pbrt-v4).
comptime PSC_NAME_MAX   = 512
comptime PSC_FILE_MAX   = 256


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
    var rough_tex_idx:  Int32
    # PBRT "displacement": bump map, resolved to an imagemap texture index
    # (through a "scale" texture wrapper if present -- see handle_texture's
    # "scale" class) + the scalar height multiplier applied to that
    # texture's raw [0,1] value. -1 = no bump map.
    var bump_tex_idx:   Int32
    var bump_scale:     Float32
    # PBRT "scale"/"mix" texture graph over the reflectance imagemap, folded
    # into one affine correction: value = tex_bias + tex_scale * texel, per
    # channel. Identity is scale=1, bias=0. Resolved by material_builder's
    # _resolve_affine_rgb; applied in shading.mojo's _tex_lookup.
    var tex_scale:      RGB
    var sss_mean_refl:  RGB
    var tex_bias:       RGB
    # UV scale applied to an imagemap `tex_idx` texture's mesh UVs at parse
    # time (Mitsuba's `<transform name="to_uv"><scale .../></transform>`,
    # e.g. a tiled floor texture) -- NOT the same field as checker_uscale/
    # vscale below (those are read at shading time for the procedural
    # checkerboard pattern generator; this pair is only consulted once, by
    # whichever shape-building code populates a mesh's raw UV list).
    var tex_uscale:     Float32
    var tex_vscale:     Float32
    var mix_name1:      String
    var mix_name2:      String
    var mix_amount:     Float32
    var transmittance:  RGB
    # Procedural checkerboard params, valid only when tex_idx == -2.
    var checker_tex1:   RGB
    var checker_tex2:   RGB
    var checker_uscale: Float32
    var checker_vscale: Float32
    # "measured" material: resolved absolute path to its ".bsdf" tensor file,
    # or "" if this isn't a measured material (or it had no "filename"
    # param). The actual load + dedup-by-path happens once, at final-scene-
    # build time (mirrors curves/spheres's own "accumulate raw data during
    # parse, materialize packed C arrays once at the end" convention) — see
    # measured_bsdf.mojo's load_measured_brdf_full and its call site.
    var measured_bsdf_path: String
    # `Material "subsurface"`: index of the homogeneous interior medium built
    # for this material, or -1. The surface itself is an ordinary dielectric;
    # this is what makes it subsurface. Stored per-material (not just applied
    # to the live attribute state) so `NamedMaterial "x"` re-attaches the same
    # interior every time it is activated, exactly as an explicit
    # `MediumInterface "x_interior" ""` would.
    var sss_medium_idx: Int32

    def __init__(out self, name: String):
        self.name           = name
        self.albedo         = RGB(Float32(0.8), Float32(0.8), Float32(0.8))
        self.kind           = Int8(0)
        self.ior            = Float32(1.5)
        self.roughness_u    = Float32(0)
        self.roughness_v    = Float32(0)
        self.tex_idx        = Int32(-1)
        self.normal_tex_idx = Int32(-1)
        self.rough_tex_idx  = Int32(-1)
        self.bump_tex_idx   = Int32(-1)
        self.bump_scale     = Float32(1)
        self.tex_scale      = RGB(Float32(1))
        self.sss_mean_refl  = RGB(Float32(1))
        self.tex_bias       = RGB(Float32(0))
        self.tex_uscale     = Float32(1)
        self.tex_vscale     = Float32(1)
        self.mix_name1      = String("")
        self.mix_name2      = String("")
        self.mix_amount     = Float32(0.5)
        self.transmittance  = RGB(Float32(1))
        self.checker_tex1   = RGB(Float32(1))
        self.checker_tex2   = RGB(Float32(0))
        self.checker_uscale = Float32(1)
        self.checker_vscale = Float32(1)
        self.measured_bsdf_path = String("")
        self.sss_medium_idx = Int32(-1)

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
    # True for meshes captured between ObjectBegin/ObjectEnd (an instancing
    # template): baked at the CTM active during that block, NOT at each
    # placement's CTM. finalize_scene excludes these from the ordinary
    # top-level primitive list — they're only reachable via a per-template
    # BLAS referenced by Instance_C placements (see pbrt_parser.mojo).
    var is_object_template: Bool

    def __init__(out self, mat_idx: Int32, inside_medium: Int32, outside_medium: Int32):
        self.points        = List[Float32]()
        self.vert_idxs     = List[Int64]()
        self.face_idxs     = List[Int64]()
        self.uvs           = List[Float32]()
        self.normals       = List[Float32]()
        self.mat_idx       = mat_idx
        self.is_area_light = False
        self.al_rgb        = RGB(Float32(0))
        self.inside_medium  = inside_medium
        self.outside_medium = outside_medium
        self.is_object_template = False

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

    # Native curve primitives (Shape "curve" — one entry per local B-spline
    # segment, no tessellation). 3 floats per control point, 4 control points
    # per curve: cp[i] = (curves_cp[12*i+3*k+0..2] for k in 0..3).
    var curves_cp:  List[Float32]
    var curves_w0:  List[Float32]
    var curves_w1:  List[Float32]
    var curves_mat: List[Int32]
    # Per-curve area-light flag/emission — mirrors MeshAccum.is_area_light/
    # al_rgb (triangles) and spheres_al/spheres_rgb (spheres), set from the
    # enclosing AttributeBegin/AreaLightSource block the same way.
    var curves_al:     List[Bool]
    var curves_al_rgb: List[RGB]

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

    # Homogeneous / heterogeneous media
    var med_names:   List[String]
    var med_sa:      List[Float32]   # 3 floats per medium
    var med_ss:      List[Float32]   # 3 floats per medium
    var med_g:       List[Float32]   # 1 per medium
    var med_grid_idx: List[Int32]    # 1 per medium; -1 = homogeneous, else index into grid_* below
    var med_nvdb_idx: List[Int32]    # 1 per medium; -1 = none, else index into nvdb_* below
    var med_nvdb_temp_idx: List[Int32]  # 1 per medium; -1 = not emissive, else index into nvdb_*
    var med_le_scale: List[Float32]     # 1 per medium (pbrt "Lescale")
    var med_temp_offset: List[Float32]  # 1 per medium ("temperatureoffset"/"temperaturecutoff")
    var med_temp_scale: List[Float32]   # 1 per medium ("temperaturescale")
    var med_is_sss:  List[Int32]        # 1 per medium; 1 = subsurface interior (Medium_C.is_sss)

    # Heterogeneous density grids ("uniformgrid" media). One record per grid;
    # med_grid_idx above points into these by index.
    var grid_nx: List[Int32]
    var grid_ny: List[Int32]
    var grid_nz: List[Int32]
    var grid_p0: List[Float32]       # 3 floats per grid
    var grid_p1: List[Float32]       # 3 floats per grid
    var grid_ctm: List[Float32]      # 16 floats per grid (world_to_medium built from this at finalize)
    var grid_density: List[Float32]  # flattened, grid_density_base[i]..+nx*ny*nz per grid
    var grid_density_base: List[Int32]

    # Procedural "cloud" media whose scene file omitted "p0"/"p1": pbrt's own
    # CloudMedium::Density evaluates the noise function at the RAW medium-
    # space point (scaled only by "frequency"), never remapped through
    # p0/p1 -- that pair is a pure ray-marching bound, unrelated to where
    # the noise pattern itself sits. Gonzales instead BAKES the noise into a
    # dense [p0,p1] grid (see handle_named_medium's is_cloud branch), so
    # p0/p1 here does double duty as "where in world space the baked voxels
    # actually are" -- get that wrong (e.g. pbrt's own default p0=(0,0,0)/
    # p1=(1,1,1), a 1-unit box) and most of a scene's actual medium-bounding
    # shape (typically several units across) falls outside it, reading back
    # zero density (grid_sample_density's hard 0/1 UV cutoff) instead of
    # real noise -- the "soft mushy blob with no structure" symptom this
    # fixes. One record per grid (parallel to grid_nx/grid_p0/...): 1 if
    # this grid's bounds are still the synthetic default and need to be
    # resized to the first shape found using it as an inside-medium (see
    # handle_sphere_shape), 0 once resolved or never pending.
    var grid_cloud_pending:    List[Int32]
    var grid_cloud_density:    List[Float32]  # "density" param, 1 per grid
    var grid_cloud_wispiness:  List[Float32]  # "wispiness" param, 1 per grid
    var grid_cloud_frequency:  List[Float32]  # "frequency" param, 1 per grid

    # Sparse density grids ("nanovdb" media). One record per grid; the
    # actual .nvdb file is loaded (via the C bridge) at finalize_scene time,
    # not here -- only the filename and CTM are captured during parsing.
    # med_nvdb_idx above points into these by index.
    var nvdb_filenames: List[String]
    var nvdb_gridnames: List[String] # grid to read from that file ("" = first grid by index)
    var nvdb_ctm: List[Float32]      # 16 floats per grid (world_to_medium built from this at finalize)

    # Textures
    var tex_names: List[String]
    var tex_files: List[String]
    # Constant textures: name -> RGB value (3 floats per entry, parallel to names)
    var const_tex_names: List[String]
    var const_tex_rgb: List[Float32]
    # Procedural checkerboard textures: name -> (tex1 RGB, tex2 RGB, uscale, vscale),
    # parallel to names. Evaluated analytically per-shading-point (not baked to an
    # image) since the checker frequency is resolution-independent.
    var checker_tex_names:  List[String]
    var checker_tex1:       List[Float32]  # 3 floats per entry
    var checker_tex2:       List[Float32]  # 3 floats per entry
    var checker_uscale:     List[Float32]  # 1 float per entry
    var checker_vscale:     List[Float32]  # 1 float per entry
    # "scale" float textures: name -> (base texture name, scalar multiplier),
    # parallel to names. PBRT wraps a bump-map imagemap in one of these to
    # apply a height scale (e.g. `Texture "x" "float" "scale" "texture tex"
    # ["base"] "float scale" [0.005]`) -- resolved to an actual imagemap
    # index by material_builder.mojo's displacement handling, not stored as
    # its own texture kind in Material_C.
    # "scale" textures. BOTH operands are independently either a nested
    # texture name or a literal, exactly like "mix" below: kroken's book
    # covers are `"rgb tex" [c] "texture scale" [t]` (a constant tinted by a
    # texture), the mirror image of killeroos' `"texture tex" [t]
    # "float scale" [s]`. The *_name entry is "" when that operand is a
    # literal; the parallel *_rgb / *_scale entry then carries it.
    var scale_tex_names:    List[String]
    var scale_tex_base:     List[String]
    var scale_tex_base_rgb: List[Float32]  # 3 floats per entry
    var scale_tex_scale:    List[Float32]
    var scale_tex_scale_name: List[String]
    # "mix" textures: pbrt evaluates these as (1 - amount)*tex1 + amount*tex2
    # (textures.h's {Float,Spectrum}MixTexture::Evaluate). Each of tex1/tex2/
    # amount is independently either a nested texture NAME or a literal
    # constant, so all three are stored both ways: the *_name entry is "" when
    # that slot is a constant, and the parallel *_rgb / *_val entry carries the
    # constant. material_builder.mojo folds the whole thing into an affine
    # `bias + scale * texture` form -- see _resolve_affine_rgb there for which
    # combinations are representable and which fall back with a warning.
    var mix_tex_names:      List[String]
    var mix_tex1_name:      List[String]
    var mix_tex1_rgb:       List[Float32]  # 3 floats per entry
    var mix_tex2_name:      List[String]
    var mix_tex2_rgb:       List[Float32]  # 3 floats per entry
    var mix_amount_name:    List[String]
    var mix_amount_val:     List[Float32]

    # Film / camera / sampler settings
    var film_w:           Int32
    var film_h:           Int32
    var film_iso:         Float32
    var film_exposuretime: Float32
    var film_whitebalance: Float32   # colour temperature in K; 0 = disabled (pbrt default)
    var film_sensor:      String     # pbrt "sensor"; only "cie1931" is modelled
    var film_max_comp:    Float32
    var film_filename:    String
    # Film "float cropwindow" [x0 x1 y0 y1] — normalized fractional bounds
    # (0..1) of film_w/film_h to actually render/output, pbrt's own
    # convention. Defaults to the full frame (0,1,0,1) when unspecified.
    var crop_x0: Float32
    var crop_x1: Float32
    var crop_y0: Float32
    var crop_y1: Float32
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

    # SPPM integrator params (Integrator "sppm" "float radius"/"integer
    # photonsperiteration"), read from the scene when present. Sentinels
    # (-1) mean "not specified" — the caller falls back to a CLI flag or
    # a pbrt-matching default (photonsperiteration defaults to film_w*film_h
    # when unspecified, same as pbrt-v4's own SPPM integrator).
    var sppm_radius:            Float32
    var sppm_photons_per_iter:  Int32

    # ── ObjectBegin/ObjectEnd/ObjectInstance (template + placement capture) ──
    # Recorded templates: parallel arrays, one entry per ObjectBegin/ObjectEnd
    # block. Mesh range [object_mesh_start[i], object_mesh_end[i]) indexes into
    # `meshes` above (those entries have is_object_template=True).
    var object_names:      List[String]
    var object_mesh_start: List[Int32]
    var object_mesh_end:   List[Int32]
    var object_ctm:        List[Float32]  # 16 floats per template: CTM at that ObjectBegin (Mdef)
    # Pending (currently-open) ObjectBegin block, valid only while object_depth > 0.
    var pending_object_name:  String
    var pending_object_start: Int32
    var pending_object_ctm:   InlineArray[Float32, 16]
    # Recorded placements: parallel arrays, one entry per ObjectInstance.
    # template_idx indexes object_names/object_mesh_start/object_mesh_end
    # (and, 1:1 in the same order, the BLAS built from each template at
    # finalize_scene time). obj_to_world/world_to_obj are 16 floats each,
    # flattened (entry i occupies [i*16, i*16+16)).
    var instance_template_idx: List[Int32]
    var instance_obj_to_world: List[Float32]
    var instance_world_to_obj: List[Float32]

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
        self.curves_cp  = List[Float32]()
        self.curves_w0  = List[Float32]()
        self.curves_w1  = List[Float32]()
        self.curves_mat = List[Int32]()
        self.curves_al     = List[Bool]()
        self.curves_al_rgb = List[RGB]()

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

        self.med_names    = List[String]()
        self.med_sa       = List[Float32]()
        self.med_ss       = List[Float32]()
        self.med_g        = List[Float32]()
        self.med_grid_idx = List[Int32]()
        self.med_nvdb_idx = List[Int32]()
        self.med_nvdb_temp_idx = List[Int32]()
        self.med_le_scale = List[Float32]()
        self.med_temp_offset = List[Float32]()
        self.med_temp_scale = List[Float32]()
        self.med_is_sss   = List[Int32]()

        self.grid_nx = List[Int32]()
        self.grid_ny = List[Int32]()
        self.grid_nz = List[Int32]()
        self.grid_p0 = List[Float32]()
        self.grid_p1 = List[Float32]()
        self.grid_ctm = List[Float32]()
        self.grid_density = List[Float32]()
        self.grid_density_base = List[Int32]()
        self.grid_cloud_pending = List[Int32]()
        self.grid_cloud_density = List[Float32]()
        self.grid_cloud_wispiness = List[Float32]()
        self.grid_cloud_frequency = List[Float32]()

        self.nvdb_filenames = List[String]()
        self.nvdb_gridnames = List[String]()
        self.nvdb_ctm = List[Float32]()

        self.tex_names = List[String]()
        self.tex_files = List[String]()
        self.const_tex_names = List[String]()
        self.const_tex_rgb = List[Float32]()
        self.checker_tex_names = List[String]()
        self.scale_tex_names = List[String]()
        self.scale_tex_base = List[String]()
        self.scale_tex_base_rgb = List[Float32]()
        self.scale_tex_scale = List[Float32]()
        self.scale_tex_scale_name = List[String]()
        self.mix_tex_names = List[String]()
        self.mix_tex1_name = List[String]()
        self.mix_tex1_rgb = List[Float32]()
        self.mix_tex2_name = List[String]()
        self.mix_tex2_rgb = List[Float32]()
        self.mix_amount_name = List[String]()
        self.mix_amount_val = List[Float32]()
        self.checker_tex1 = List[Float32]()
        self.checker_tex2 = List[Float32]()
        self.checker_uscale = List[Float32]()
        self.checker_vscale = List[Float32]()

        self.film_w           = Int32(512)
        self.film_h           = Int32(512)
        self.film_iso         = Float32(100)
        self.film_exposuretime = Float32(1.0)
        self.film_whitebalance = Float32(0.0)
        self.film_sensor = String("cie1931")
        self.film_max_comp    = Float32(0)
        self.film_filename    = String("gonzales.exr")
        self.crop_x0 = Float32(0.0)
        self.crop_x1 = Float32(1.0)
        self.crop_y0 = Float32(0.0)
        self.crop_y1 = Float32(1.0)
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
        self.sppm_radius = Float32(-1)
        self.sppm_photons_per_iter = Int32(-1)

        self.object_names      = List[String]()
        self.object_mesh_start = List[Int32]()
        self.object_mesh_end   = List[Int32]()
        self.object_ctm        = List[Float32]()
        self.pending_object_name  = String("")
        self.pending_object_start = Int32(0)
        self.pending_object_ctm   = InlineArray[Float32, 16](fill=Float32(0))
        self.instance_template_idx = List[Int32]()
        self.instance_obj_to_world = List[Float32]()
        self.instance_world_to_obj = List[Float32]()


# ── CTM stack helpers (operate on SceneParseState only) ──────────────────────

def ctm_push(mut s: SceneParseState):
    s.ctm_stack.append(s.ctm.copy())

def ctm_pop(mut s: SceneParseState):
    if len(s.ctm_stack) > 0:
        s.ctm = s.ctm_stack.pop()

def scene_parse_state_new() -> SceneParseState:
    return SceneParseState()
