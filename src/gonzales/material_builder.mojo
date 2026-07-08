from std.memory import alloc
from std.math import exp, sqrt
from .lexer import (PbrtScanner, scanner_parse_quoted_string,
                    scanner_scan_char, scanner_scan_float, ParamScanner,
                    _psc_streq, _psc_strncpy, _psc_strncmp,
                    _psc_scan_rgb, _psc_scan_one_float, _psc_scan_one_str,
                    _psc_scan_one_bool)
from .parse_types import NamedMaterial, SceneParseState, PSC_NAME_MAX, PSC_FILE_MAX
from .geometry import RGB, MatKind
from .measured_bsdf import load_measured_bsdf_reflectance

def _psc_handle_make_named_material(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                                   s: UnsafePointer[SceneParseState, MutAnyOrigin],
                                   inline_type: Bool = False):
    var mat_name = alloc[UInt8](PSC_NAME_MAX)
    _ = scanner_parse_quoted_string(handle, mat_name, PSC_NAME_MAX)

    var rgb = alloc[Float32](3)
    rgb[0] = Float32(0.5); rgb[1] = Float32(0.5); rgb[2] = Float32(0.5)
    # transmittance for DiffuseTransmission (default 0.25 per PBRT)
    var trans_rgb = alloc[Float32](3)
    trans_rgb[0] = Float32(0.25); trans_rgb[1] = Float32(0.25); trans_rgb[2] = Float32(0.25)
    # named-spectrum conductor optical constants (R/G/B at 630/530/450 nm)
    var metal_eta = alloc[Float32](3)
    metal_eta[0] = Float32(0.5); metal_eta[1] = Float32(0.5); metal_eta[2] = Float32(0.5)
    var metal_k   = alloc[Float32](3)
    metal_k[0] = Float32(0.5); metal_k[1] = Float32(0.5); metal_k[2] = Float32(0.5)
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
    # hair material melanin parameters
    var eumelanin   = Float32(0.0)
    var pheomelanin = Float32(0.0)
    var sigma_a_rgb = alloc[Float32](3)
    sigma_a_rgb[0] = Float32(-1); sigma_a_rgb[1] = Float32(-1); sigma_a_rgb[2] = Float32(-1)
    var has_sigma_a = False
    var mat_type = MatKind.diffuse
    var mat_ior  = Float32(1.5)
    var mat_roughU = Float32(0.0)
    var mat_roughV = Float32(0.0)
    # For inline Material directives, the first quoted string is the TYPE, not the name.
    # Map it to mat_type; the name buffer keeps the type string as a synthetic key.
    if inline_type:
        if _psc_streq(mat_name, "conductor"):       mat_type = MatKind.conductor
        elif _psc_streq(mat_name, "dielectric"):    mat_type = MatKind.dielectric
        elif _psc_streq(mat_name, "coateddiffuse"): mat_type = MatKind.coated_diffuse
        elif _psc_streq(mat_name, "diffusetransmission"): mat_type = MatKind.diffuse_transmit
        elif _psc_streq(mat_name, "coatedconductor"): mat_type = MatKind.coated_conductor
        elif _psc_streq(mat_name, "mix"):           mat_type = MatKind.mix
        elif _psc_streq(mat_name, "thindielectric"): mat_type = MatKind.thin_dielectric
        elif _psc_streq(mat_name, "hair"):          mat_type = MatKind.hair
        elif _psc_streq(mat_name, "interface"):     mat_type = MatKind.interface
        elif _psc_streq(mat_name, "diffuse"):       mat_type = MatKind.diffuse
        elif _psc_streq(mat_name, "measured"):
            mat_type = MatKind.conductor
            is_measured = True
            mat_roughU = Float32(0.1); mat_roughV = Float32(0.1)
        elif _psc_streq(mat_name, "subsurface"):
            # Approximated as a coateddiffuse (specular coat + Lambertian
            # base) — real subsurface needs a volumetric random walk /
            # BSSRDF (lateral light transport under the surface, translucency
            # through thin geometry), which this does NOT reproduce. This
            # just gets the base color roughly right (see the "name" param
            # handler below) so the material isn't flat grey; reuses
            # coateddiffuse's existing reflectance-texture/roughness/eta
            # parsing below for free since subsurface uses the same param
            # names for those.
            mat_type = MatKind.coated_diffuse
            is_subsurface = True
        else:
            # Unrecognized material type — used to fall back to a flat
            # 50%-grey diffuse in total silence, which made scenes using it
            # look wrong with no clue why. mat_type already defaults to
            # MatKind.diffuse above, so this just adds the warning.
            var unsup_name = String(unsafe_from_utf8_ptr=mat_name.as_immutable())
            print("Warning: unsupported material type '" + unsup_name + "' — rendering as flat 50%-grey diffuse. Supported: diffuse, conductor, dielectric, thindielectric, coateddiffuse, coatedconductor, diffusetransmission, mix, hair, interface, measured (approximate), subsurface (approximate).")
    # pbrt default: remaproughness=true means roughU/V are a perceptual
    # roughness remapped to GGX alpha via sqrt(); false means the parsed
    # value already IS alpha and must be used as-is (see RoughnessToAlpha
    # in the pbrt-v4 book, section 9.6.1).
    var mat_remap_roughness = True
    var tex_idx_for_mat = Int32(-1)
    var normal_tex_idx_for_mat = Int32(-1)
    # Procedural checkerboard params, only meaningful when tex_idx_for_mat == -2.
    var checker_tex1 = RGB(Float32(1))
    var checker_tex2 = RGB(Float32(0))
    var checker_uscale = Float32(1)
    var checker_vscale = Float32(1)
    var mix_name1 = alloc[UInt8](PSC_NAME_MAX)
    var mix_name2 = alloc[UInt8](PSC_NAME_MAX)
    var mix_amount = Float32(0.5)
    mix_name1[0] = UInt8(0); mix_name2[0] = UInt8(0)
    # Sized for the normalmap/bumpmap filename branch below (scans up to
    # PSC_FILE_MAX * 2 bytes) — a 64-byte buffer here was a real overflow
    # for any normal-map path longer than 64 bytes.
    var str_val  = alloc[UInt8](PSC_FILE_MAX * 2)
    var ps = ParamScanner()
    while ps.next(handle):
        if ps.name_is("type") and ps.is_str():
            _ = scanner_parse_quoted_string(handle, str_val, 64)
            if ps.is_array:
                _ = scanner_scan_char(handle, UInt8(93))
            if _psc_streq(str_val, "conductor"):
                mat_type = MatKind.conductor
            elif _psc_streq(str_val, "dielectric"):
                mat_type = MatKind.dielectric
            elif _psc_streq(str_val, "coateddiffuse"):
                mat_type = MatKind.coated_diffuse
            elif _psc_streq(str_val, "diffusetransmission"):
                mat_type = MatKind.diffuse_transmit
            elif _psc_streq(str_val, "coatedconductor"):
                mat_type = MatKind.coated_conductor
            elif _psc_streq(str_val, "mix"):
                mat_type = MatKind.mix
            elif _psc_streq(str_val, "thindielectric"):
                mat_type = MatKind.thin_dielectric
            elif _psc_streq(str_val, "hair"):
                mat_type = MatKind.hair
            elif _psc_streq(str_val, "interface"):
                mat_type = MatKind.interface
            elif _psc_streq(str_val, "diffuse"):
                mat_type = MatKind.diffuse
            elif _psc_streq(str_val, "measured"):
                mat_type = MatKind.conductor
                is_measured = True
                mat_roughU = Float32(0.1); mat_roughV = Float32(0.1)
            elif _psc_streq(str_val, "subsurface"):
                # See the matching inline_type comment above.
                mat_type = MatKind.coated_diffuse
                is_subsurface = True
            else:
                # See the matching inline_type warning above.
                var unsup_name2 = String(unsafe_from_utf8_ptr=str_val.as_immutable())
                print("Warning: unsupported material type '" + unsup_name2 + "' — rendering as flat 50%-grey diffuse. Supported: diffuse, conductor, dielectric, thindielectric, coateddiffuse, coatedconductor, diffusetransmission, mix, hair, interface, measured (approximate), subsurface (approximate).")
                mat_type = MatKind.diffuse
        elif (ps.name_is("eta") or ps.name_is("k")) and ps.type_buf[0] == UInt8(114):  # 'r' rgb eta/k for conductor
            if ps.name_is("eta"):
                _psc_scan_rgb(handle, metal_eta, ps.is_array)
            else:
                _psc_scan_rgb(handle, metal_k, ps.is_array)
            has_spectral_conductor = True
        elif (ps.name_is("eta") or ps.name_is("intIOR")) and ps.type_buf[0] == UInt8(102):  # 'f' float IOR for dielectric
            var tmp = alloc[Float32](1)
            _ = scanner_scan_float(handle, tmp)
            mat_ior = tmp[0]
            tmp.free()
            if ps.is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        elif (ps.name_is("uroughness") or ps.name_is("roughness")) and ps.is_float():
            mat_roughU = _psc_scan_one_float(handle, ps.is_array)
            if ps.name_is("roughness"):
                mat_roughV = mat_roughU
        elif ps.name_is("vroughness") and ps.is_float():
            mat_roughV = _psc_scan_one_float(handle, ps.is_array)
        elif ps.name_is("remaproughness") and ps.type_buf[0] == UInt8(98):  # 'b' bool
            mat_remap_roughness = _psc_scan_one_bool(handle, ps.is_array)
        elif ps.name_is("reflectance") and ps.is_float():
            _psc_scan_rgb(handle, rgb, ps.is_array)
        elif ps.name_is("transmittance") and ps.is_float():
            # DiffuseTransmission transmittance — stored in trans_rgb, later -> mat.emission
            _psc_scan_rgb(handle, trans_rgb, ps.is_array)
        elif ps.name_is("eumelanin") and ps.is_float():
            # Hair material: melanin concentration -> stored in rgb[0] (temp)
            eumelanin = _psc_scan_one_float(handle, ps.is_array)
        elif ps.name_is("pheomelanin") and ps.is_float():
            pheomelanin = _psc_scan_one_float(handle, ps.is_array)
        elif ps.name_is("sigma_a") and ps.is_float():
            # Hair material: direct absorption coefficients (R,G,B)
            _psc_scan_rgb(handle, sigma_a_rgb, ps.is_array)
            has_sigma_a = True
        elif (ps.name_is("eta") or ps.name_is("k")) and ps.type_buf[0] == UInt8(115):  # 's' = spectrum
            # Named-spectrum conductor: "spectrum eta" ["metal-Ag-eta"] etc.
            # Read the metal name string, look up precomputed F0 per channel.
            var mname = alloc[UInt8](64)
            _ = scanner_parse_quoted_string(handle, mname, 64)
            if ps.is_array:
                _ = scanner_scan_char(handle, UInt8(93))
            # Precomputed Fresnel F0 = ((eta-1)^2+k^2)/((eta+1)^2+k^2) for common metals
            # Channels: R≈630nm, G≈530nm, B≈450nm  (from NIST/Filament spectral data)
            if ps.name_is("eta"):
                if _psc_strncmp(mname, "metal-Ag", 8) == 0:
                    metal_eta[0] = Float32(0.136); metal_eta[1] = Float32(0.130); metal_eta[2] = Float32(0.144)
                elif _psc_strncmp(mname, "metal-Al", 8) == 0:
                    metal_eta[0] = Float32(1.300); metal_eta[1] = Float32(0.826); metal_eta[2] = Float32(0.644)
                elif _psc_strncmp(mname, "metal-Au", 8) == 0:
                    metal_eta[0] = Float32(0.194); metal_eta[1] = Float32(0.608); metal_eta[2] = Float32(1.426)
                elif _psc_strncmp(mname, "metal-Cu", 8) == 0:
                    metal_eta[0] = Float32(0.272); metal_eta[1] = Float32(1.120); metal_eta[2] = Float32(1.160)
                has_spectral_conductor = True
            else:  # "k"
                if _psc_strncmp(mname, "metal-Ag", 8) == 0:
                    metal_k[0] = Float32(3.880); metal_k[1] = Float32(3.070); metal_k[2] = Float32(2.560)
                elif _psc_strncmp(mname, "metal-Al", 8) == 0:
                    metal_k[0] = Float32(7.480); metal_k[1] = Float32(6.280); metal_k[2] = Float32(5.580)
                elif _psc_strncmp(mname, "metal-Au", 8) == 0:
                    metal_k[0] = Float32(3.060); metal_k[1] = Float32(2.120); metal_k[2] = Float32(1.846)
                elif _psc_strncmp(mname, "metal-Cu", 8) == 0:
                    metal_k[0] = Float32(3.240); metal_k[1] = Float32(2.605); metal_k[2] = Float32(2.433)
                has_spectral_conductor = True
            mname.free()
        elif ps.name_is("reflectance") and ps.type_buf[0] == UInt8(116):  # 't' = texture
            _ = scanner_parse_quoted_string(handle, str_val, 64)
            if ps.is_array:
                _ = scanner_scan_char(handle, UInt8(93))
            # Look up str_val in tex_names (imagemap) first, then constant
            # textures, then procedural checkerboard textures.
            var str_name = String(unsafe_from_utf8_ptr=str_val.as_immutable())
            var matched_tex = False
            for ti in range(len(s[0].tex_names)):
                if s[0].tex_names[ti] == str_name:
                    tex_idx_for_mat = Int32(ti)
                    matched_tex = True
                    break
            if not matched_tex:
                for ci in range(len(s[0].const_tex_names)):
                    if s[0].const_tex_names[ci] == str_name:
                        rgb[0] = s[0].const_tex_rgb[ci*3+0]
                        rgb[1] = s[0].const_tex_rgb[ci*3+1]
                        rgb[2] = s[0].const_tex_rgb[ci*3+2]
                        matched_tex = True
                        break
            if not matched_tex:
                for ki in range(len(s[0].checker_tex_names)):
                    if s[0].checker_tex_names[ki] == str_name:
                        # -2 marks the material as using the embedded procedural
                        # checkerboard fields below (see shading.mojo's _tex_lookup)
                        # rather than the imagemap texture table.
                        tex_idx_for_mat = Int32(-2)
                        checker_tex1 = RGB(s[0].checker_tex1[ki*3+0], s[0].checker_tex1[ki*3+1], s[0].checker_tex1[ki*3+2])
                        checker_tex2 = RGB(s[0].checker_tex2[ki*3+0], s[0].checker_tex2[ki*3+1], s[0].checker_tex2[ki*3+2])
                        checker_uscale = s[0].checker_uscale[ki]
                        checker_vscale = s[0].checker_vscale[ki]
                        break
        elif ps.name_is("L") and ps.is_float():
            _psc_scan_rgb(handle, rgb, ps.is_array)
        elif ps.name_is("filename") and is_measured and ps.type_buf[0] == UInt8(115):  # 's' = string
            _ = scanner_parse_quoted_string(handle, str_val, PSC_FILE_MAX * 2)
            if ps.is_array:
                _ = scanner_scan_char(handle, UInt8(93))
            var bsdf_path = s[0].scene_dir + String(unsafe_from_utf8_ptr=str_val.as_immutable())
            var (bsdf_ok, mean_lum) = load_measured_bsdf_reflectance(bsdf_path)
            if bsdf_ok:
                # Clamp away from the extremes — a raw tensor mean can read
                # near 0 or above 1 at the tabulated grid's edges/highlights,
                # neither of which is a sane Fresnel F0 for the conductor
                # approximation this material is being rendered as.
                var refl = min(Float32(0.95), max(Float32(0.03), mean_lum))
                rgb[0] = refl; rgb[1] = refl; rgb[2] = refl
            else:
                print("Warning: could not read measured BRDF file '" + bsdf_path + "' — using default grey")
        elif ps.name_is("name") and is_subsurface and ps.is_str():
            # Named measured-scattering preset (Jensen/Marschner/Levoy/Hanrahan
            # 2001, "A Practical Model for Subsurface Light Transport" — the
            # same table pbrt's GetMediumScatteringProperties uses). Only the
            # presets actually seen in this scene corpus (sssdragon's "Skin1")
            # plus its common companion "Skin2" are included — add more from
            # pbrt's media.cpp SubsurfaceParameterTable if another shows up.
            # Approximates the base reflectance as each channel's single-
            # scattering albedo sigma_s'/(sigma_s'+sigma_a) — not the true
            # dipole diffuse reflectance, but a reasonable, cheap proxy (and,
            # notably, no substitute for the real lateral subsurface light
            # transport this material is completely missing).
            _ = scanner_parse_quoted_string(handle, str_val, 64)
            if ps.is_array:
                _ = scanner_scan_char(handle, UInt8(93))
            if _psc_streq(str_val, "Skin1"):
                rgb[0] = Float32(0.9585); rgb[1] = Float32(0.8381); rgb[2] = Float32(0.6779)
            elif _psc_streq(str_val, "Skin2"):
                rgb[0] = Float32(0.9882); rgb[1] = Float32(0.9578); rgb[2] = Float32(0.9250)
            else:
                var preset_name = String(unsafe_from_utf8_ptr=str_val.as_immutable())
                print("Warning: unrecognized subsurface preset '" + preset_name + "' — using default grey reflectance")
        elif (ps.name_is("normalmap") or ps.name_is("bumpmap")) and ps.type_buf[0] == UInt8(115):  # 's' = string (pbrt syntax: "string normalmap" "file")
            _ = scanner_parse_quoted_string(handle, str_val, PSC_FILE_MAX * 2)
            if ps.is_array:
                _ = scanner_scan_char(handle, UInt8(93))
            # Register the file as an (unnamed) imagemap texture and point the
            # material's normal_tex_idx at it — same path as a texture reference.
            var nm_file = s[0].scene_dir + String(unsafe_from_utf8_ptr=str_val.as_immutable())
            normal_tex_idx_for_mat = Int32(len(s[0].tex_names))
            s[0].tex_names.append(String("__normalmap"))
            s[0].tex_files.append(nm_file)
        elif ps.name_is("amount") and ps.is_float():
            mix_amount = _psc_scan_one_float(handle, ps.is_array)
        elif ps.name_is("materials") and ps.is_str():
            # Two quoted material names for mix
            var tmp1 = alloc[UInt8](PSC_NAME_MAX)
            var tmp2 = alloc[UInt8](PSC_NAME_MAX)
            _ = scanner_parse_quoted_string(handle, tmp1, PSC_NAME_MAX)
            _ = scanner_parse_quoted_string(handle, tmp2, PSC_NAME_MAX)
            _psc_strncpy(mix_name1, tmp1, PSC_NAME_MAX)
            _psc_strncpy(mix_name2, tmp2, PSC_NAME_MAX)
            tmp1.free(); tmp2.free()
            if ps.is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        else:
            ps.skip(handle)

    # Store into named_materials List
    var nm = NamedMaterial(String(unsafe_from_utf8_ptr=mat_name.as_immutable()))
    # For named-spectrum conductors: compute Fresnel F0 per channel
    if has_spectral_conductor and mat_type == MatKind.conductor:
        var f0r = ((metal_eta[0]-Float32(1.0))*(metal_eta[0]-Float32(1.0)) + metal_k[0]*metal_k[0]) / \
                  ((metal_eta[0]+Float32(1.0))*(metal_eta[0]+Float32(1.0)) + metal_k[0]*metal_k[0])
        var f0g = ((metal_eta[1]-Float32(1.0))*(metal_eta[1]-Float32(1.0)) + metal_k[1]*metal_k[1]) / \
                  ((metal_eta[1]+Float32(1.0))*(metal_eta[1]+Float32(1.0)) + metal_k[1]*metal_k[1])
        var f0b = ((metal_eta[2]-Float32(1.0))*(metal_eta[2]-Float32(1.0)) + metal_k[2]*metal_k[2]) / \
                  ((metal_eta[2]+Float32(1.0))*(metal_eta[2]+Float32(1.0)) + metal_k[2]*metal_k[2])
        nm.albedo = RGB(f0r, f0g, f0b)
    elif mat_type == MatKind.hair:
        var ce = eumelanin; var cp = pheomelanin
        if has_sigma_a:
            nm.albedo = RGB(sigma_a_rgb[0], sigma_a_rgb[1], sigma_a_rgb[2])
        else:
            nm.albedo = RGB(
                ce * Float32(0.419) + cp * Float32(0.187),
                ce * Float32(0.697) + cp * Float32(0.400),
                ce * Float32(1.370) + cp * Float32(1.050),
            )
    else:
        nm.albedo = RGB(rgb[0], rgb[1], rgb[2])
    nm.transmittance  = RGB(trans_rgb[0], trans_rgb[1], trans_rgb[2])
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
    nm.checker_tex1   = checker_tex1
    nm.checker_tex2   = checker_tex2
    nm.checker_uscale = checker_uscale
    nm.checker_vscale = checker_vscale
    nm.mix_name1      = String(unsafe_from_utf8_ptr=mix_name1.as_immutable())
    nm.mix_name2      = String(unsafe_from_utf8_ptr=mix_name2.as_immutable())
    nm.mix_amount     = mix_amount
    s[0].named_materials.append(nm^)

    mat_name.free(); str_val.free(); rgb.free()
    trans_rgb.free(); metal_eta.free(); metal_k.free()
    mix_name1.free(); mix_name2.free()
    sigma_a_rgb.free()

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
