from std.memory import alloc
from std.math import sqrt
from .lexer import (PbrtScanner, scanner_parse_quoted_string, _psc_collect_params)
from .parse_types import NamedMaterial, SceneParseState, PSC_NAME_MAX
from .geometry import RGB, MatKind
from .measured_bsdf import load_measured_bsdf_reflectance

def _psc_handle_make_named_material(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                                   s: UnsafePointer[SceneParseState, MutAnyOrigin],
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
    # "subsurface" — approximated as coateddiffuse; "string name" (a named
    # measured-scattering preset, e.g. "Skin1") needs its own lookup since no
    # other material type uses that param name. See measured_bsdf.mojo-style
    # scoping note at the subsurface branch below for why this isn't a real
    # BSSRDF.
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
            # Approximated as a coateddiffuse (specular coat + Lambertian
            # base) — real subsurface needs a volumetric random walk /
            # BSSRDF (lateral light transport under the surface, translucency
            # through thin geometry), which this does NOT reproduce. This
            # just gets the base color roughly right (see the "name" param
            # handler below) so the material isn't flat grey; reuses
            # coateddiffuse's existing reflectance-texture/roughness/eta
            # handling below for free since subsurface uses the same param
            # names for those.
            mat_type = MatKind.coated_diffuse
            is_subsurface = True
        else:
            # Unrecognized material type — used to fall back to a flat
            # 50%-grey diffuse in total silence, which made scenes using it
            # look wrong with no clue why. mat_type already defaults to
            # MatKind.diffuse above, so this just adds the warning.
            print("Warning: unsupported material type '" + type_str + "' — rendering as flat 50%-grey diffuse. Supported: diffuse, conductor, dielectric, thindielectric, coateddiffuse, coatedconductor, diffusetransmission, mix, hair, interface, measured (approximate), subsurface (approximate).")
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
    if params.has("eta"):
        var eta_f = params.get_floats("eta")
        if len(eta_f) >= 3:
            metal_eta = RGB(eta_f[0], eta_f[1], eta_f[2])
            has_spectral_conductor = True
        elif len(eta_f) == 1:
            mat_ior = eta_f[0]
            metal_eta = RGB(eta_f[0])
            has_spectral_conductor = True
        else:
            # Precomputed Fresnel F0 = ((eta-1)^2+k^2)/((eta+1)^2+k^2) for common metals
            # Channels: R≈630nm, G≈530nm, B≈450nm  (from NIST/Filament spectral data)
            var eta_name = params.get_string("eta", "")
            if eta_name.startswith("metal-Ag"):
                metal_eta = RGB(Float32(0.136), Float32(0.130), Float32(0.144))
            elif eta_name.startswith("metal-Al"):
                metal_eta = RGB(Float32(1.300), Float32(0.826), Float32(0.644))
            elif eta_name.startswith("metal-Au"):
                metal_eta = RGB(Float32(0.194), Float32(0.608), Float32(1.426))
            elif eta_name.startswith("metal-Cu"):
                metal_eta = RGB(Float32(0.272), Float32(1.120), Float32(1.160))
            has_spectral_conductor = True
    if params.has("intIOR"):
        # Float-only alias for dielectric eta; no RGB/spectrum/named form.
        var iior_f = params.get_floats("intIOR")
        if len(iior_f) > 0:
            mat_ior = iior_f[0]
    if params.has("k"):
        var k_f = params.get_floats("k")
        if len(k_f) >= 3:
            metal_k = RGB(k_f[0], k_f[1], k_f[2])
            has_spectral_conductor = True
        elif len(k_f) == 1:
            metal_k = RGB(k_f[0])
            has_spectral_conductor = True
        else:
            var k_name = params.get_string("k", "")
            if k_name.startswith("metal-Ag"):
                metal_k = RGB(Float32(3.880), Float32(3.070), Float32(2.560))
            elif k_name.startswith("metal-Al"):
                metal_k = RGB(Float32(7.480), Float32(6.280), Float32(5.580))
            elif k_name.startswith("metal-Au"):
                metal_k = RGB(Float32(3.060), Float32(2.120), Float32(1.846))
            elif k_name.startswith("metal-Cu"):
                metal_k = RGB(Float32(3.240), Float32(2.605), Float32(2.433))
            has_spectral_conductor = True

    # "reflectance": either an RGB/float value, OR a texture reference --
    # looked up in tex_names (imagemap) first, then constant textures, then
    # procedural checkerboard textures.
    var tex_idx_for_mat = Int32(-1)
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
                            break

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

    # "subsurface": named measured-scattering preset (Jensen/Marschner/Levoy/
    # Hanrahan 2001, "A Practical Model for Subsurface Light Transport" — the
    # same table pbrt's GetMediumScatteringProperties uses). Only the presets
    # actually seen in this scene corpus (sssdragon's "Skin1") plus its
    # common companion "Skin2" are included — add more from pbrt's media.cpp
    # SubsurfaceParameterTable if another shows up. Approximates the base
    # reflectance as each channel's single-scattering albedo
    # sigma_s'/(sigma_s'+sigma_a) — not the true dipole diffuse reflectance,
    # but a reasonable, cheap proxy (and, notably, no substitute for the real
    # lateral subsurface light transport this material is completely
    # missing).
    if is_subsurface and params.has("name"):
        var preset_name = params.get_string("name", "")
        if preset_name == "Skin1":
            rgb = RGB(Float32(0.9585), Float32(0.8381), Float32(0.6779))
        elif preset_name == "Skin2":
            rgb = RGB(Float32(0.9882), Float32(0.9578), Float32(0.9250))
        else:
            print("Warning: unrecognized subsurface preset '" + preset_name + "' — using default grey reflectance")

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
    nm.normal_tex_idx = normal_tex_idx_for_mat
    nm.rough_tex_idx  = rough_tex_idx_for_mat
    nm.checker_tex1   = checker_tex1
    nm.checker_tex2   = checker_tex2
    nm.checker_uscale = checker_uscale
    nm.checker_vscale = checker_vscale
    nm.mix_name1      = mix_name1
    nm.mix_name2      = mix_name2
    nm.mix_amount     = mix_amount
    nm.measured_bsdf_path = measured_bsdf_path
    s[0].named_materials.append(nm^)

    mat_name.free()

def _psc_handle_named_material(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                               s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var mat_name = alloc[UInt8](PSC_NAME_MAX)
    _ = scanner_parse_quoted_string(handle, mat_name, PSC_NAME_MAX)
    s[0].cur_attr.mat_idx = Int32(-1)
    var name_str = String(unsafe_from_utf8_ptr=mat_name.as_immutable())
    for i in range(len(s[0].named_materials)):
        if s[0].named_materials[i].name == name_str:
            s[0].cur_attr.mat_idx = Int32(i)
            break
    mat_name.free()
