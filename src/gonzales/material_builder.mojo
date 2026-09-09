from std.memory import alloc
from std.math import sqrt, exp, max
from .lexer import (PbrtScanner, scanner_parse_quoted_string, _psc_collect_params, ParameterDictionary)
from .parse_types import NamedMaterial, SceneParseState, PSC_NAME_MAX
from .geometry import RGB, MatKind
from .measured_bsdf import load_measured_bsdf_reflectance
from .spd import load_spd_rgb, named_metal_rgb, named_glass_ior


@always_inline
def _mb_float_or_rgb(params: ParameterDictionary, name: StringLiteral, default: RGB) -> RGB:
    """A param that may be written as a bare float (replicated to all three
    channels) or an rgb triple. Local twin of pbrt_parser's
    _psc_get_float_or_rgb -- duplicated rather than imported because
    pbrt_parser imports THIS file, so the dependency only runs one way."""
    var f = params.get_floats(name)
    if len(f) >= 3:
        return RGB(f[0], f[1], f[2])
    elif len(f) == 1:
        return RGB(f[0])
    return default


@always_inline
def _sss_alpha_from_reflectance(a: Float32) -> Float32:
    """Single-scattering albedo that makes a random walk reproduce diffuse
    reflectance `a` (Christensen & Burley 2015, the fit Cycles uses to set up
    random-walk subsurface). pbrt instead inverts its tabulated BSSRDF; this
    targets the same quantity by a different route -- see the "reflectance"
    branch in the subsurface block below."""
    var x = min(max(a, Float32(0)), Float32(1))
    return Float32(1) - exp(Float32(-5.09406) * x
                            + Float32(2.61188) * x * x
                            - Float32(4.31805) * x * x * x)

def _psc_handle_make_named_material(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                                   s: UnsafePointer[SceneParseState, MutExternalOrigin],
                                   inline_type: Bool = False):
    """Builds a NamedMaterial from a `Material`/`MakeNamedMaterial` directive.
    Scans every parameter ONCE, generically, into a ParameterDictionary
    (_psc_collect_params -- lexer.mojo), then interprets specific named
    parameters by querying it -- mirrors real pbrt's own ParameterDictionary
    design instead of hand-scanning each (name, type) combination inline.
    See project_parser_architecture memory for why this replaced the old
    single-pass special-cased scan."""
    var mat_name = alloc[UInt8](PSC_NAME_MAX)
    _ = scanner_parse_quoted_string(handle, mat_name, PSC_NAME_MAX)

    var params = _psc_collect_params(handle)

    var rgb = RGB(Float32(0.5))
    # transmittance for DiffuseTransmission (default 0.25 per PBRT)
    var trans_rgb = params.get_rgb("transmittance", RGB(Float32(0.25)))
    # named-spectrum conductor optical constants (R/G/B at 630/530/450 nm)
    var metal_eta = RGB(Float32(0.5))
    var metal_k = RGB(Float32(0.5))
    var has_spectral_conductor = False
    # "measured" (tabulated .bsdf) material — approximated as a rough
    # conductor whose F0 is the mean of the file's own "luminance" tensor;
    # see measured_bsdf.mojo for why this isn't the full spectral BxDF.
    var is_measured = False
    # "subsurface" — a dielectric surface bounding a scattering interior
    # medium (real random-walk SSS). Its "string name" preset, sigma_a/
    # sigma_s, scale and mfp params need their own resolution pass since no
    # other material type uses them; see the subsurface block further down.
    var is_subsurface = False
    var mat_type = MatKind.diffuse
    var mat_ior = Float32(1.5)
    var mat_roughU = Float32(0.0)
    var mat_roughV = Float32(0.0)

    # Determine material kind from either the inline-directive keyword
    # (`Material "conductor" ...`, where the "name" IS the type) or the
    # "type" param (`MakeNamedMaterial "foo" "string type" "conductor"`).
    # Both paths dispatch the exact same recognized-type set and warning --
    # collapsed from two separately-written elif chains in the old code into
    # one, since the dictionary model no longer needs a separate live-scan
    # branch per directive shape.
    var type_str = String("diffuse")
    var have_type_str = False
    if inline_type:
        type_str = String(unsafe_from_utf8_ptr=mat_name.as_immutable())
        have_type_str = True
    elif params.has("type"):
        type_str = params.get_string("type", "diffuse")
        have_type_str = True

    if have_type_str:
        if type_str == "conductor":
            mat_type = MatKind.conductor
        elif type_str == "dielectric":
            mat_type = MatKind.dielectric
        elif type_str == "coateddiffuse":
            mat_type = MatKind.coated_diffuse
        elif type_str == "diffusetransmission":
            mat_type = MatKind.diffuse_transmit
        elif type_str == "coatedconductor":
            mat_type = MatKind.coated_conductor
        elif type_str == "mix":
            mat_type = MatKind.mix
        elif type_str == "thindielectric":
            mat_type = MatKind.thin_dielectric
        elif type_str == "hair":
            mat_type = MatKind.hair
        elif type_str == "interface":
            mat_type = MatKind.interface
        elif type_str == "diffuse":
            mat_type = MatKind.diffuse
        elif type_str == "measured":
            mat_type = MatKind.conductor
            is_measured = True
            mat_roughU = Float32(0.1); mat_roughV = Float32(0.1)
        elif type_str == "subsurface":
            # Real subsurface scattering, by random walk: the SURFACE is an
            # ordinary dielectric (exactly what pbrt's SubsurfaceMaterial
            # builds -- a DielectricBxDF), and the object's INTERIOR is a
            # homogeneous participating medium that light actually scatters
            # through. See the sigma-resolution block further down, which
            # registers that medium and records it in nm.sss_medium_idx.
            #
            # Where this differs from pbrt: pbrt resolves the interior with a
            # TabulatedBSSRDF (photon-beam-diffusion tables, an approximation
            # that assumes a semi-infinite planar slab), while this walks the
            # medium for real. The random walk converges to ground truth and
            # gets thin-geometry translucency the diffusion approximation
            # cannot, at the cost of many scattering events per path -- which
            # is why Medium_C.is_sss exists (those events must not be charged
            # to the path's maxdepth budget). Same approach as Cycles.
            mat_type = MatKind.dielectric
            is_subsurface = True
        else:
            # Unrecognized material type — used to fall back to a flat
            # 50%-grey diffuse in total silence, which made scenes using it
            # look wrong with no clue why. mat_type already defaults to
            # MatKind.diffuse above, so this just adds the warning.
            print("Warning: unsupported material type '" + type_str + "' — rendering as flat 50%-grey diffuse. Supported: diffuse, conductor, dielectric, thindielectric, coateddiffuse, coatedconductor, diffusetransmission, mix, hair, interface, measured (approximate), subsurface.")
            mat_type = MatKind.diffuse

    # "eta"/"k": dielectric IOR (a scalar) and conductor Fresnel constants (an
    # RGB triple OR a named-spectrum string OR an inline numeric spectrum
    # array, both already reduced to a single scalar mean by
    # _psc_scan_spectrum_scalar during collection) all arrive as the SAME
    # dictionary entry differing only in how many floats it holds -- >=3 is
    # an explicit RGB triple (conductor-only), exactly 1 is a scalar (used
    # for both mat_ior, meaningful only for dielectric, and metal_eta/k,
    # meaningful only for conductor -- harmless cross-assignment either way
    # since a material can't be both kinds), and 0 floats means it was a
    # named-spectrum string, looked up by common-metal prefix.
    # `coatedconductor` names its conductor constants "conductor.eta"/
    # "conductor.k" rather than plain "eta"/"k" (pbrt namespaces the inner
    # layer's params). Neither name was handled before, so every
    # coatedconductor in the corpus fell back to the 0.5/0.5 default --
    # killeroo-coated-gold rendered as dark chrome instead of gold.
    # ParameterDictionary's accessors take a StringLiteral (compile-time) key,
    # so the alias can't be selected by a runtime string -- resolve both
    # spellings into locals here, then interpret them once below.
    var eta_f = List[Float32]()
    var eta_name = String("")
    var has_eta = False
    if params.has("eta"):
        has_eta = True
        eta_f = params.get_floats("eta")
        eta_name = params.get_string("eta", "")
    elif params.has("conductor.eta"):
        has_eta = True
        eta_f = params.get_floats("conductor.eta")
        eta_name = params.get_string("conductor.eta", "")

    var k_f = List[Float32]()
    var k_name = String("")
    var has_k = False
    if params.has("k"):
        has_k = True
        k_f = params.get_floats("k")
        k_name = params.get_string("k", "")
    elif params.has("conductor.k"):
        has_k = True
        k_f = params.get_floats("conductor.k")
        k_name = params.get_string("conductor.k", "")

    if has_eta:
        if len(eta_f) >= 3:
            metal_eta = RGB(eta_f[0], eta_f[1], eta_f[2])
            has_spectral_conductor = True
        elif len(eta_f) == 1:
            mat_ior = eta_f[0]
            metal_eta = RGB(eta_f[0])
            has_spectral_conductor = True
        else:
            # Zero floats => the value was a string: either a pbrt built-in
            # named spectrum ("metal-Au-eta", "glass-BK7") or a path to a
            # .spd file. Both were previously ignored without a word.
            var (m_eta, m_ok) = named_metal_rgb(eta_name, False)
            if m_ok:
                metal_eta = m_eta
                has_spectral_conductor = True
            else:
                var (g_ior, g_ok) = named_glass_ior(eta_name)
                if g_ok:
                    mat_ior = g_ior
                elif eta_name.endswith(".spd"):
                    var (f_eta, f_ok) = load_spd_rgb(s[0].scene_dir + eta_name)
                    if f_ok:
                        metal_eta = f_eta
                        has_spectral_conductor = True
                    else:
                        print("SPD load FAILED (cannot open/parse), material '"
                              + String(unsafe_from_utf8_ptr=mat_name.as_immutable())
                              + "' eta falls back to 0.5:", s[0].scene_dir + eta_name)
                elif eta_name != "":
                    print("Warning: unknown named spectrum '" + eta_name
                          + "' for eta — falling back to 0.5. Supported: metal-{Ag,Al,Au,Cu,CuZn,TiO2,MgO}-*, glass-{BK7,BAF10,FK51A,LASF9,F5,F10,F11}, or a .spd file path.")
    if params.has("intIOR"):
        # Float-only alias for dielectric eta; no RGB/spectrum/named form.
        var iior_f = params.get_floats("intIOR")
        if len(iior_f) > 0:
            mat_ior = iior_f[0]
    if has_k:
        if len(k_f) >= 3:
            metal_k = RGB(k_f[0], k_f[1], k_f[2])
            has_spectral_conductor = True
        elif len(k_f) == 1:
            metal_k = RGB(k_f[0])
            has_spectral_conductor = True
        else:
            var (m_k, mk_ok) = named_metal_rgb(k_name, True)
            if mk_ok:
                metal_k = m_k
                has_spectral_conductor = True
            elif k_name.endswith(".spd"):
                var (f_k, fk_ok) = load_spd_rgb(s[0].scene_dir + k_name)
                if fk_ok:
                    metal_k = f_k
                    has_spectral_conductor = True
                else:
                    print("SPD load FAILED (cannot open/parse), material '"
                          + String(unsafe_from_utf8_ptr=mat_name.as_immutable())
                          + "' k falls back to 0.5:", s[0].scene_dir + k_name)
            elif k_name != "":
                print("Warning: unknown named spectrum '" + k_name
                      + "' for k — falling back to 0.5. Supported: metal-{Ag,Al,Au,Cu,CuZn,TiO2,MgO}-*, or a .spd file path.")

    # "reflectance": either an RGB/float value, OR a texture reference --
    # looked up in tex_names (imagemap) first, then constant textures, then
    # procedural checkerboard textures, then through one level of "scale"
    # indirection to an imagemap (`Texture "sgrid" "spectrum" "scale"
    # "texture tex" ["grid"] "float scale" [0.5]`, the shape killeroos and
    # ~49 other spectrum-texture declarations in the corpus use). Only one
    # level is resolved, matching the bump path's own convention below; a
    # scale-of-a-scale isn't a pattern seen in practice.
    var tex_idx_for_mat = Int32(-1)
    var tex_scale_for_mat = Float32(1)
    var checker_tex1 = RGB(Float32(1))
    var checker_tex2 = RGB(Float32(0))
    var checker_uscale = Float32(1)
    var checker_vscale = Float32(1)
    if params.has("reflectance"):
        var refl_f = params.get_floats("reflectance")
        if len(refl_f) >= 3:
            rgb = RGB(refl_f[0], refl_f[1], refl_f[2])
        else:
            var tex_name = params.get_string("reflectance", "")
            if tex_name != "":
                var matched_tex = False
                for ti in range(len(s[0].tex_names)):
                    if s[0].tex_names[ti] == tex_name:
                        tex_idx_for_mat = Int32(ti)
                        matched_tex = True
                        break
                if not matched_tex:
                    for ci in range(len(s[0].const_tex_names)):
                        if s[0].const_tex_names[ci] == tex_name:
                            rgb = RGB(s[0].const_tex_rgb[ci*3+0], s[0].const_tex_rgb[ci*3+1], s[0].const_tex_rgb[ci*3+2])
                            matched_tex = True
                            break
                if not matched_tex:
                    for ki in range(len(s[0].checker_tex_names)):
                        if s[0].checker_tex_names[ki] == tex_name:
                            # -2 marks the material as using the embedded
                            # procedural checkerboard fields below (see
                            # shading.mojo's _tex_lookup) rather than the
                            # imagemap texture table.
                            tex_idx_for_mat = Int32(-2)
                            checker_tex1 = RGB(s[0].checker_tex1[ki*3+0], s[0].checker_tex1[ki*3+1], s[0].checker_tex1[ki*3+2])
                            checker_tex2 = RGB(s[0].checker_tex2[ki*3+0], s[0].checker_tex2[ki*3+1], s[0].checker_tex2[ki*3+2])
                            checker_uscale = s[0].checker_uscale[ki]
                            checker_vscale = s[0].checker_vscale[ki]
                            matched_tex = True
                            break
                if not matched_tex:
                    for si in range(len(s[0].scale_tex_names)):
                        if s[0].scale_tex_names[si] == tex_name:
                            var base_name = s[0].scale_tex_base[si]
                            for ti in range(len(s[0].tex_names)):
                                if s[0].tex_names[ti] == base_name:
                                    tex_idx_for_mat = Int32(ti)
                                    tex_scale_for_mat = s[0].scale_tex_scale[si]
                                    matched_tex = True
                                    break
                            if not matched_tex:
                                # A "scale" wrapping something that isn't a
                                # plain imagemap (a constant, a checkerboard,
                                # or another scale). Fold the multiplier into
                                # the flat albedo rather than dropping it.
                                for ci in range(len(s[0].const_tex_names)):
                                    if s[0].const_tex_names[ci] == base_name:
                                        var sc = s[0].scale_tex_scale[si]
                                        rgb = RGB(s[0].const_tex_rgb[ci*3+0] * sc,
                                                  s[0].const_tex_rgb[ci*3+1] * sc,
                                                  s[0].const_tex_rgb[ci*3+2] * sc)
                                        matched_tex = True
                                        break
                            if not matched_tex:
                                print("Warning: texture '" + tex_name
                                      + "' is a \"scale\" of '" + base_name
                                      + "', which is not a supported base texture"
                                      + " — falling back to flat albedo.")
                                matched_tex = True
                            break
                if not matched_tex:
                    print("Warning: material references undefined texture '"
                          + tex_name + "' for reflectance — falling back to"
                          + " flat albedo. (Unsupported texture classes are"
                          + " reported by handle_texture at parse time.)")

    # "L": some scenes set a material's base color via this name instead of
    # "reflectance" -- overrides if present (same target, same as the RGB
    # form of "reflectance" above).
    if params.has("L"):
        var l_f = params.get_floats("L")
        if len(l_f) >= 3:
            rgb = RGB(l_f[0], l_f[1], l_f[2])

    # "roughness"/"uroughness"/"vroughness": each may be a float value OR
    # (isotropic "roughness"/"uroughness" only) a texture reference. Applying
    # "roughness" first and letting "uroughness"/"vroughness" override
    # matches the more-specific-wins convention real pbrt's own
    # GetFloat("uroughness", roughness_default) accessor chain uses,
    # independent of the params' declaration order in the scene file.
    var rough_tex_idx_for_mat = Int32(-1)
    var rough_f = params.get_floats("roughness")
    if len(rough_f) > 0:
        mat_roughU = rough_f[0]; mat_roughV = rough_f[0]
    else:
        # Isotropic only: applies to both roughU/roughV (matches every
        # texture-roughness usage seen in this scene corpus so far — a
        # separate "texture vroughness" would need its own slot, not added
        # since nothing uses it yet). Imagemap textures only (no constant-
        # texture/checkerboard fallback, unlike "reflectance" above — a
        # roughness value read off a procedural checkerboard or flat-color
        # texture isn't a pattern seen in practice).
        var rough_tex = params.get_string("roughness", "")
        if rough_tex != "":
            for ti in range(len(s[0].tex_names)):
                if s[0].tex_names[ti] == rough_tex:
                    rough_tex_idx_for_mat = Int32(ti)
                    break
    var urough_f = params.get_floats("uroughness")
    if len(urough_f) > 0:
        mat_roughU = urough_f[0]
    else:
        var urough_tex = params.get_string("uroughness", "")
        if urough_tex != "":
            for ti in range(len(s[0].tex_names)):
                if s[0].tex_names[ti] == urough_tex:
                    rough_tex_idx_for_mat = Int32(ti)
                    break
    var vrough_f = params.get_floats("vroughness")
    if len(vrough_f) > 0:
        mat_roughV = vrough_f[0]

    # pbrt default: remaproughness=true means roughU/V are a perceptual
    # roughness remapped to GGX alpha via sqrt(); false means the parsed
    # value already IS alpha and must be used as-is (see RoughnessToAlpha
    # in the pbrt-v4 book, section 9.6.1).
    var mat_remap_roughness = params.get_bool("remaproughness", True)

    # Hair material melanin parameters / direct sigma_a.
    var eumelanin = params.get_float("eumelanin", Float32(0.0))
    var pheomelanin = params.get_float("pheomelanin", Float32(0.0))
    var has_sigma_a = params.has("sigma_a")
    var sigma_a_rgb = params.get_rgb("sigma_a", RGB(Float32(-1)))

    # "measured": read the tensor file's mean "luminance" as an achromatic
    # Fresnel F0 approximation (see measured_bsdf.mojo for the real-BxDF gap).
    # Also record the resolved path so the final-scene-build step (Stage 1 of
    # the real-MeasuredBxDF port) can load+dedup the full tensor data; the
    # approximate conductor fallback below stays live until material_builder
    # is flipped to route through MatKind.measured (Stage 2).
    var measured_bsdf_path = String("")
    if is_measured and params.has("filename"):
        var bsdf_path = s[0].scene_dir + params.get_string("filename", "")
        measured_bsdf_path = bsdf_path
        var (bsdf_ok, mean_lum) = load_measured_bsdf_reflectance(bsdf_path)
        if bsdf_ok:
            # Clamp away from the extremes — a raw tensor mean can read near
            # 0 or above 1 at the tabulated grid's edges/highlights, neither
            # of which is a sane Fresnel F0 for the conductor approximation
            # this material is being rendered as.
            var refl = min(Float32(0.95), max(Float32(0.03), mean_lum))
            rgb = RGB(refl)
        else:
            print("Warning: could not read measured BRDF file '" + bsdf_path + "' — using default grey")

    # "subsurface": resolve the interior medium's scattering coefficients and
    # register it, so the dielectric surface selected above actually bounds a
    # scattering volume. Mirrors pbrt's SubsurfaceMaterial::Create
    # (materials.cpp:498) exactly, including its four mutually-exclusive ways
    # of specifying the properties and their precedence.
    var sss_medium_idx = Int32(-1)
    if is_subsurface:
        # pbrt defaults eta to 1.33 for subsurface (not 1.5 as elsewhere).
        if not params.has("eta") and not params.has("intIOR"):
            mat_ior = Float32(1.33)
        var sss_g = params.get_float("g", Float32(0))
        # Coefficients are in mm^-1; "scale" converts to the scene's own unit
        # (sssdragon uses scale 50). Applied to both, as pbrt does.
        var sss_scale = params.get_float("scale", Float32(1))
        var sig_s = RGB(Float32(0))
        var sig_a = RGB(Float32(0))
        var have_sigmas = False

        var preset_name = params.get_string("name", "")
        if preset_name != "":
            # 1. By name. pbrt's SubsurfaceParameterTable (media.cpp:81), from
            # Jensen/Marschner/Levoy/Hanrahan 2001 "A Practical Model for
            # Subsurface Light Transport". The table stores REDUCED scattering
            # coefficients, so pbrt forces g=0 with them -- do the same, with
            # the same warning, or the anisotropy would be applied twice.
            var found = True
            if   preset_name == "Apple":      sig_s = RGB(Float32(2.29), Float32(2.39), Float32(1.97)); sig_a = RGB(Float32(0.0030), Float32(0.0034), Float32(0.046))
            elif preset_name == "Chicken1":   sig_s = RGB(Float32(0.15), Float32(0.21), Float32(0.38)); sig_a = RGB(Float32(0.015), Float32(0.077), Float32(0.19))
            elif preset_name == "Chicken2":   sig_s = RGB(Float32(0.19), Float32(0.25), Float32(0.32)); sig_a = RGB(Float32(0.018), Float32(0.088), Float32(0.20))
            elif preset_name == "Cream":      sig_s = RGB(Float32(7.38), Float32(5.47), Float32(3.15)); sig_a = RGB(Float32(0.0002), Float32(0.0028), Float32(0.0163))
            elif preset_name == "Ketchup":    sig_s = RGB(Float32(0.18), Float32(0.07), Float32(0.03)); sig_a = RGB(Float32(0.061), Float32(0.97), Float32(1.45))
            elif preset_name == "Marble":     sig_s = RGB(Float32(2.19), Float32(2.62), Float32(3.00)); sig_a = RGB(Float32(0.0021), Float32(0.0041), Float32(0.0071))
            elif preset_name == "Potato":     sig_s = RGB(Float32(0.68), Float32(0.70), Float32(0.55)); sig_a = RGB(Float32(0.0024), Float32(0.0090), Float32(0.12))
            elif preset_name == "Skimmilk":   sig_s = RGB(Float32(0.70), Float32(1.22), Float32(1.90)); sig_a = RGB(Float32(0.0014), Float32(0.0025), Float32(0.0142))
            elif preset_name == "Skin1":      sig_s = RGB(Float32(0.74), Float32(0.88), Float32(1.01)); sig_a = RGB(Float32(0.032), Float32(0.17), Float32(0.48))
            elif preset_name == "Skin2":      sig_s = RGB(Float32(1.09), Float32(1.59), Float32(1.79)); sig_a = RGB(Float32(0.013), Float32(0.070), Float32(0.145))
            elif preset_name == "Spectralon": sig_s = RGB(Float32(11.6), Float32(20.4), Float32(14.9)); sig_a = RGB(Float32(0.0), Float32(0.0), Float32(0.0))
            elif preset_name == "Wholemilk":  sig_s = RGB(Float32(2.55), Float32(3.21), Float32(3.77)); sig_a = RGB(Float32(0.0011), Float32(0.0024), Float32(0.014))
            else:
                found = False
            if found:
                have_sigmas = True
                if sss_g != Float32(0):
                    print("Warning: subsurface preset '" + preset_name + "' specifies REDUCED scattering coefficients — ignoring \"g\" (matching pbrt).")
                sss_g = Float32(0)
            else:
                print("Warning: unknown subsurface preset '" + preset_name + "' — falling back to pbrt's default coefficients. Known presets: Apple, Chicken1, Chicken2, Cream, Ketchup, Marble, Potato, Skimmilk, Skin1, Skin2, Spectralon, Wholemilk.")
        elif params.has("sigma_a") or params.has("sigma_s"):
            # 2. sigma_a and sigma_s directly. pbrt makes it an error to give
            # one without the other; warn and fill the missing one from the
            # default rather than aborting the whole render.
            if not (params.has("sigma_a") and params.has("sigma_s")):
                print("Warning: subsurface material gives only one of \"sigma_a\"/\"sigma_s\" — pbrt requires both; using the default for the missing one.")
            sig_a = _mb_float_or_rgb(params, "sigma_a", RGB(Float32(0.0011), Float32(0.0024), Float32(0.014)))
            sig_s = _mb_float_or_rgb(params, "sigma_s", RGB(Float32(2.55), Float32(3.21), Float32(3.77)))
            have_sigmas = True
        elif params.has("reflectance"):
            # 3. Diffuse reflectance + mean free path. pbrt inverts its
            # tabulated BSSRDF (SubsurfaceFromDiffuse) to find the sigmas that
            # reproduce a given diffuse albedo; that inversion needs the
            # photon-beam-diffusion tables this renderer does not build. Use
            # the standard random-walk inversion instead -- Christensen &
            # Burley 2015's single-scattering-albedo fit, what Cycles uses for
            # exactly this purpose. It targets the same quantity by a
            # different route, so expect agreement in character but not to the
            # last digit against pbrt on a `reflectance`-specified scene.
            var refl = params.get_rgb("reflectance", RGB(Float32(0.5)))
            var mfp = _mb_float_or_rgb(params, "mfp", RGB(Float32(1)))
            print("Note: subsurface \"reflectance\"/\"mfp\" inverted with the Christensen-Burley fit, not pbrt's tabulated SubsurfaceFromDiffuse — close in character, not bit-comparable.")
            var ar = _sss_alpha_from_reflectance(refl.r)
            var ag = _sss_alpha_from_reflectance(refl.g)
            var ab = _sss_alpha_from_reflectance(refl.b)
            # sigma_t = 1/mfp, split into scattering/absorption by the albedo.
            var tr = Float32(1) / max(mfp.r, Float32(1e-6))
            var tg = Float32(1) / max(mfp.g, Float32(1e-6))
            var tb = Float32(1) / max(mfp.b, Float32(1e-6))
            sig_s = RGB(ar * tr, ag * tg, ab * tb)
            sig_a = RGB((Float32(1) - ar) * tr, (Float32(1) - ag) * tg, (Float32(1) - ab) * tb)
            have_sigmas = True

        if not have_sigmas:
            # 4. Nothing specified — pbrt's own defaults (a milk-like medium).
            sig_a = RGB(Float32(0.0011), Float32(0.0024), Float32(0.014))
            sig_s = RGB(Float32(2.55), Float32(3.21), Float32(3.77))

        # Register the interior as an ordinary homogeneous medium. Everything
        # downstream -- free-flight sampling, phase-function scattering,
        # volume NEE, the medium-crossing bookkeeping that swaps
        # current_medium_idx when a refracted ray passes through the surface
        # -- is the participating-media machinery that already exists, and is
        # entirely unaware this particular medium happens to be an object's
        # interior.
        sss_medium_idx = Int32(len(s[0].med_names))
        s[0].med_names.append(String("__sss_interior"))
        s[0].med_sa.append(sig_a.r * sss_scale); s[0].med_sa.append(sig_a.g * sss_scale); s[0].med_sa.append(sig_a.b * sss_scale)
        s[0].med_ss.append(sig_s.r * sss_scale); s[0].med_ss.append(sig_s.g * sss_scale); s[0].med_ss.append(sig_s.b * sss_scale)
        s[0].med_g.append(sss_g)
        s[0].med_grid_idx.append(Int32(-1))
        s[0].med_nvdb_idx.append(Int32(-1))
        s[0].med_nvdb_temp_idx.append(Int32(-1))
        s[0].med_le_scale.append(Float32(0)); s[0].med_temp_offset.append(Float32(0)); s[0].med_temp_scale.append(Float32(1))
        s[0].med_is_sss.append(Int32(1))

        # An inline `Material "subsurface"` takes effect immediately, so bind
        # the interior to the live attribute state here -- shapes declared
        # after it pick it up exactly as they would an explicit
        # `MediumInterface "interior" ""`. The MakeNamedMaterial form binds
        # later instead, when `NamedMaterial` activates it.
        if inline_type:
            s[0].cur_attr.inside_medium = sss_medium_idx

    # "normalmap"/"bumpmap": register the file as an (unnamed) imagemap
    # texture and point the material's normal_tex_idx at it — same path as a
    # texture reference.
    var normal_tex_idx_for_mat = Int32(-1)
    var normalmap_file = params.get_string("normalmap", "")
    if normalmap_file == "":
        normalmap_file = params.get_string("bumpmap", "")
    if normalmap_file != "":
        var nm_file = s[0].scene_dir + normalmap_file
        normal_tex_idx_for_mat = Int32(len(s[0].tex_names))
        s[0].tex_names.append(String("__normalmap"))
        s[0].tex_files.append(nm_file)

    # "displacement": bump map. The named texture is resolved to an
    # imagemap index either directly (a plain `Texture "x" "float"
    # "imagemap" ...`) or through one level of "scale" indirection (`Texture
    # "x" "float" "scale" "texture tex" ["base"] "float scale" [s]`, the
    # shape every bump map in this scene corpus actually uses -- see
    # handle_texture's "scale" class). Only one level of indirection is
    # resolved; a scale-of-a-scale isn't a pattern seen in practice.
    var bump_tex_idx_for_mat = Int32(-1)
    var bump_scale_for_mat = Float32(1)
    var disp_tex = params.get_string("displacement", "")
    if disp_tex != "":
        var matched_disp = False
        for ti in range(len(s[0].tex_names)):
            if s[0].tex_names[ti] == disp_tex:
                bump_tex_idx_for_mat = Int32(ti)
                matched_disp = True
                break
        if not matched_disp:
            for si in range(len(s[0].scale_tex_names)):
                if s[0].scale_tex_names[si] == disp_tex:
                    var base_name = s[0].scale_tex_base[si]
                    for ti in range(len(s[0].tex_names)):
                        if s[0].tex_names[ti] == base_name:
                            bump_tex_idx_for_mat = Int32(ti)
                            bump_scale_for_mat = s[0].scale_tex_scale[si]
                            matched_disp = True
                            break
                    break

    # "mix": blend amount and the two component material names.
    var mix_amount = params.get_float("amount", Float32(0.5))
    var mix_names = params.get_strings("materials")
    var mix_name1 = String("")
    var mix_name2 = String("")
    if len(mix_names) > 0:
        mix_name1 = mix_names[0]
    if len(mix_names) > 1:
        mix_name2 = mix_names[1]

    # Store into named_materials List
    var nm = NamedMaterial(String(unsafe_from_utf8_ptr=mat_name.as_immutable()))
    # For named-spectrum conductors: compute Fresnel F0 per channel
    if has_spectral_conductor and (mat_type == MatKind.conductor or mat_type == MatKind.coated_conductor):
        var f0r = ((metal_eta.r-Float32(1.0))*(metal_eta.r-Float32(1.0)) + metal_k.r*metal_k.r) / \
                  ((metal_eta.r+Float32(1.0))*(metal_eta.r+Float32(1.0)) + metal_k.r*metal_k.r)
        var f0g = ((metal_eta.g-Float32(1.0))*(metal_eta.g-Float32(1.0)) + metal_k.g*metal_k.g) / \
                  ((metal_eta.g+Float32(1.0))*(metal_eta.g+Float32(1.0)) + metal_k.g*metal_k.g)
        var f0b = ((metal_eta.b-Float32(1.0))*(metal_eta.b-Float32(1.0)) + metal_k.b*metal_k.b) / \
                  ((metal_eta.b+Float32(1.0))*(metal_eta.b+Float32(1.0)) + metal_k.b*metal_k.b)
        nm.albedo = RGB(f0r, f0g, f0b)
    elif mat_type == MatKind.hair:
        var ce = eumelanin; var cp = pheomelanin
        if has_sigma_a:
            nm.albedo = sigma_a_rgb
        else:
            nm.albedo = RGB(
                ce * Float32(0.419) + cp * Float32(0.187),
                ce * Float32(0.697) + cp * Float32(0.400),
                ce * Float32(1.370) + cp * Float32(1.050),
            )
    else:
        nm.albedo = rgb
    nm.transmittance  = trans_rgb
    nm.kind           = mat_type
    nm.ior            = mat_ior
    # Resolve roughU/V to the actual GGX alpha here, once, so every shading call
    # site can use mat.roughU/roughV directly. Only meaningful for the BSDF
    # kinds that use roughU/V as an alpha (conductor, dielectric, coated_diffuse's
    # coat, coated_conductor) — mix's "amount" and hair's beta_m/beta_n reuse the
    # same Material_C fields for unrelated values and must pass through untouched.
    if mat_type == MatKind.conductor or mat_type == MatKind.dielectric or \
       mat_type == MatKind.coated_diffuse or mat_type == MatKind.coated_conductor:
        if mat_remap_roughness:
            mat_roughU = sqrt(mat_roughU) if mat_roughU > Float32(0.0) else Float32(0.0)
            mat_roughV = sqrt(mat_roughV) if mat_roughV > Float32(0.0) else Float32(0.0)
        # else: roughU/V already IS alpha (pbrt RoughnessToAlpha semantics) — use as-is.
    nm.roughness_u    = mat_roughU
    nm.roughness_v    = mat_roughV
    nm.tex_idx        = tex_idx_for_mat
    nm.tex_scale      = tex_scale_for_mat
    nm.normal_tex_idx = normal_tex_idx_for_mat
    nm.rough_tex_idx  = rough_tex_idx_for_mat
    nm.bump_tex_idx   = bump_tex_idx_for_mat
    nm.bump_scale     = bump_scale_for_mat
    nm.checker_tex1   = checker_tex1
    nm.checker_tex2   = checker_tex2
    nm.checker_uscale = checker_uscale
    nm.checker_vscale = checker_vscale
    nm.mix_name1      = mix_name1
    nm.mix_name2      = mix_name2
    nm.mix_amount     = mix_amount
    nm.measured_bsdf_path = measured_bsdf_path
    nm.sss_medium_idx = sss_medium_idx
    s[0].named_materials.append(nm^)

    mat_name.free()

def _psc_handle_named_material(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                               s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    var mat_name = alloc[UInt8](PSC_NAME_MAX)
    _ = scanner_parse_quoted_string(handle, mat_name, PSC_NAME_MAX)
    s[0].cur_attr.mat_idx = Int32(-1)
    var name_str = String(unsafe_from_utf8_ptr=mat_name.as_immutable())
    for i in range(len(s[0].named_materials)):
        if s[0].named_materials[i].name == name_str:
            s[0].cur_attr.mat_idx = Int32(i)
            # A `subsurface` material carries an interior medium with it (see
            # _psc_handle_make_named_material). Activating the material has to
            # bind that interior to the live attribute state too, or shapes
            # using it would get the dielectric shell with vacuum inside and
            # render as clear glass. The inline `Material` form binds at
            # declaration instead, since it takes effect immediately.
            if s[0].named_materials[i].sss_medium_idx >= Int32(0):
                s[0].cur_attr.inside_medium = s[0].named_materials[i].sss_medium_idx
            break
    mat_name.free()
