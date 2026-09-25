from std.memory.alloc import unsafe_alloc
from std.ffi import external_call
from std.math import sqrt, exp, max, abs
from .diagnostics import warn_unsupported
from .lexer import (PbrtScanner, scanner_parse_quoted_string, _psc_collect_params, ParameterDictionary)
from .parse_types import NamedMaterial, SceneParseState, PSC_NAME_MAX, scene_path
from .geometry import RGB
from .materials import MatKind
from .measured_bsdf import load_measured_bsdf_reflectance
from .spd import load_spd_rgb, named_metal_rgb, named_glass_ior
from .rng import PCG32
from std.math import log, sqrt, cos


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


@fieldwise_init
struct _AffineTex(Copyable, Movable):
    """A resolved texture reference in gonzales's affine form:

        value(uv) = bias + scale * imagemap[tex_idx](uv)

    `tex_idx == -1` means the graph collapsed to a pure constant (`bias`;
    `scale` is then zero). `ok == False` means the graph is not representable
    this way -- the only such case in practice is a product of two *different*
    textures, which would need a second lookup per shading point.

    Everything pbrt's "scale" and "mix" classes do on a single underlying
    imagemap is affine, and affine functions compose, so arbitrarily nested
    scale/mix chains fold into one (scale, bias) pair at parse time and cost
    nothing at render time."""
    var ok:      Bool
    var tex_idx: Int32
    var scale:   RGB
    var bias:    RGB


@always_inline
def _affine_fail() -> _AffineTex:
    return _AffineTex(False, Int32(-1), RGB(Float32(1)), RGB(Float32(0)))


@always_inline
def _affine_const(c: RGB) -> _AffineTex:
    return _AffineTex(True, Int32(-1), RGB(Float32(0)), c)


def _resolve_affine_rgb(s: Pointer[SceneParseState, MutUntrackedOrigin],
                        name: String, depth: Int) -> _AffineTex:
    """Fold a named texture -- possibly a nested scale/mix graph -- into one
    affine (scale, bias) pair over a single imagemap. See _AffineTex.

    Depth-capped rather than cycle-detected: pbrt texture graphs are declared
    strictly bottom-up (a texture can only name one declared earlier), so a
    cycle is impossible in a valid file, and the cap only guards against a
    malformed one."""
    if depth > 8:
        return _affine_fail()

    for ti in range(len(s[unsafe_offset=0].tex_names)):
        if s[unsafe_offset=0].tex_names[ti] == name:
            return _AffineTex(True, Int32(ti), RGB(Float32(1)), RGB(Float32(0)))

    for ci in range(len(s[unsafe_offset=0].const_tex_names)):
        if s[unsafe_offset=0].const_tex_names[ci] == name:
            return _affine_const(RGB(s[unsafe_offset=0].const_tex_rgb[ci*3+0],
                                     s[unsafe_offset=0].const_tex_rgb[ci*3+1],
                                     s[unsafe_offset=0].const_tex_rgb[ci*3+2]))

    for si in range(len(s[unsafe_offset=0].scale_tex_names)):
        if s[unsafe_offset=0].scale_tex_names[si] == name:
            var bn = s[unsafe_offset=0].scale_tex_base[si]
            var base = _affine_const(RGB(s[unsafe_offset=0].scale_tex_base_rgb[si*3+0],
                                         s[unsafe_offset=0].scale_tex_base_rgb[si*3+1],
                                         s[unsafe_offset=0].scale_tex_base_rgb[si*3+2]))
            if bn != "":
                base = _resolve_affine_rgb(s, bn, depth + 1)
            if not base.ok:
                return _affine_fail()
            var sn = s[unsafe_offset=0].scale_tex_scale_name[si]
            if sn == "":
                var k = s[unsafe_offset=0].scale_tex_scale[si]
                return _AffineTex(True, base.tex_idx, base.scale * k, base.bias * k)
            # Texture-valued multiplier. base * (sM*T + bM) stays affine only
            # if at most one of the two operands actually varies -- otherwise
            # it is a genuine product of two lookups. kroken's book covers are
            # the useful case: a constant base tinted by a texture.
            var mul = _resolve_affine_rgb(s, sn, depth + 1)
            if not mul.ok:
                return _affine_fail()
            if base.tex_idx < 0:
                return _AffineTex(True, mul.tex_idx,
                                  mul.scale * base.bias, mul.bias * base.bias)
            if mul.tex_idx < 0:
                return _AffineTex(True, base.tex_idx,
                                  base.scale * mul.bias, base.bias * mul.bias)
            return _affine_fail()

    for mi in range(len(s[unsafe_offset=0].mix_tex_names)):
        if s[unsafe_offset=0].mix_tex_names[mi] == name:
            var c1 = RGB(s[unsafe_offset=0].mix_tex1_rgb[mi*3+0], s[unsafe_offset=0].mix_tex1_rgb[mi*3+1], s[unsafe_offset=0].mix_tex1_rgb[mi*3+2])
            var c2 = RGB(s[unsafe_offset=0].mix_tex2_rgb[mi*3+0], s[unsafe_offset=0].mix_tex2_rgb[mi*3+1], s[unsafe_offset=0].mix_tex2_rgb[mi*3+2])
            var n1 = s[unsafe_offset=0].mix_tex1_name[mi]
            var n2 = s[unsafe_offset=0].mix_tex2_name[mi]
            var na = s[unsafe_offset=0].mix_amount_name[mi]

            var r1 = _affine_const(c1)
            if n1 != "":
                r1 = _resolve_affine_rgb(s, n1, depth + 1)
            var r2 = _affine_const(c2)
            if n2 != "":
                r2 = _resolve_affine_rgb(s, n2, depth + 1)
            if (not r1.ok) or (not r2.ok):
                return _affine_fail()

            if na == "":
                # Constant blend factor: pbrt's (1-a)*tex1 + a*tex2. Both
                # sides are already affine, so the result is affine unless
                # they ride on two *different* imagemaps.
                var a = s[unsafe_offset=0].mix_amount_val[mi]
                var w1 = Float32(1) - a
                var bias = r1.bias * w1 + r2.bias * a
                if r1.tex_idx < 0 and r2.tex_idx < 0:
                    return _affine_const(bias)
                if r2.tex_idx < 0:
                    return _AffineTex(True, r1.tex_idx, r1.scale * w1, bias)
                if r1.tex_idx < 0:
                    return _AffineTex(True, r2.tex_idx, r2.scale * a, bias)
                if r1.tex_idx == r2.tex_idx:
                    return _AffineTex(True, r1.tex_idx,
                                      r1.scale * w1 + r2.scale * a, bias)
                return _affine_fail()

            # Texture-driven blend factor. Substituting the amount's own
            # affine form A(uv) = sA*T + bA into (1-A)*c1 + A*c2 gives
            #   [sA*(c2-c1)] * T + [c1 + bA*(c2-c1)]
            # -- still affine, but only when both blended sides are constants;
            # otherwise the expansion carries a T*T term.
            if r1.tex_idx >= 0 or r2.tex_idx >= 0:
                return _affine_fail()
            var ra = _resolve_affine_rgb(s, na, depth + 1)
            if not ra.ok:
                return _affine_fail()
            var d = r2.bias - r1.bias
            var bias2 = r1.bias + ra.bias * d
            if ra.tex_idx < 0:
                return _affine_const(bias2)
            return _AffineTex(True, ra.tex_idx, ra.scale * d, bias2)

    return _affine_fail()


def _sss_walk_reflectance(alpha: Float32, eta: Float32, g: Float32, n_walks: Int) -> Float32:
    """Monte-Carlo the diffuse reflectance of a semi-infinite medium of
    single-scattering albedo `alpha` sitting behind a smooth dielectric
    boundary of relative index `eta`, using the SAME transport the renderer
    itself runs (exponential free flight, HG phase, Fresnel/TIR at the
    boundary). Units are sigma_t = 1, which the answer is independent of.

    Deterministic: a fixed seed, so the bisection in _sss_invert_alpha sees a
    smooth monotone function (common random numbers) rather than a noisy one.
    """
    var rng = PCG32(UInt64(0x9E3779B97F4A7C15), UInt64(1))
    var escaped = Float32(0)
    # Relative index crossing OUT of the medium.
    var eta_out = Float32(1.0) / max(eta, Float32(1e-4))
    for _ in range(n_walks):
        # Entry direction. The surfaces this inverts for are lit by an
        # environment, not a collimated beam, so sample a COSINE-weighted
        # incident direction and refract it, rather than assuming normal
        # incidence: refraction into a denser medium concentrates grazing
        # light toward the normal, and how deep a photon starts is exactly
        # what sets how much of it comes back out.
        var ci = sqrt(max(rng.next_float(), Float32(0.0)))   # cosine-weighted
        var si = sqrt(max(Float32(1.0) - ci*ci, Float32(0.0)))
        var st = si / max(eta, Float32(1e-4))                # Snell, into the medium
        var wz = sqrt(max(Float32(1.0) - st*st, Float32(0.0)))
        var z = Float32(0.0)
        var alive = True
        for _step in range(10000):
            var t = -log(max(rng.next_float(), Float32(1e-7)))
            z += wz * t
            if z < Float32(0.0):
                # Reached the boundary. cos of the angle to the normal.
                var ct = min(abs(wz), Float32(1.0))
                # Fresnel for going inside -> outside; below the critical
                # angle sin_t2 > 1 means total internal reflection.
                var sin_t2 = (Float32(1.0) - ct*ct) / (eta_out*eta_out)
                var refl = Float32(1.0)
                if sin_t2 < Float32(1.0):
                    var ct2 = sqrt(max(Float32(1.0) - sin_t2, Float32(0.0)))
                    var rs = (eta_out*ct - ct2) / (eta_out*ct + ct2)
                    var rp = (ct - eta_out*ct2) / (ct + eta_out*ct2)
                    refl = Float32(0.5) * (rs*rs + rp*rp)
                if rng.next_float() > refl:
                    escaped += Float32(1.0)
                    alive = False
                    break
                # Total (or Fresnel) internal reflection: back inside.
                z = -z
                wz = -wz
                continue
            # Real collision: absorb, or scatter.
            if rng.next_float() > alpha:
                alive = False
                break
            # New direction's z component. Isotropic for g = 0, else HG.
            var u = rng.next_float()
            var mu: Float32
            if abs(g) < Float32(1e-3):
                mu = Float32(2.0) * u - Float32(1.0)
            else:
                var sq = (Float32(1.0) - g*g) / (Float32(1.0) + g - Float32(2.0)*g*u)
                mu = (Float32(1.0) + g*g - sq*sq) / (Float32(2.0)*g)
            # Rotate the old direction by mu about a uniformly random azimuth;
            # only the z component is tracked, so this is the standard
            # cos-composition with a uniform azimuth.
            var sz = sqrt(max(Float32(1.0) - wz*wz, Float32(0.0)))
            var smu = sqrt(max(Float32(1.0) - mu*mu, Float32(0.0)))
            var phi = Float32(6.2831853) * rng.next_float()
            var cphi = cos(phi)
            wz = wz * mu + sz * smu * cphi
            if wz > Float32(1.0): wz = Float32(1.0)
            if wz < Float32(-1.0): wz = Float32(-1.0)
        _ = alive
    return escaped / Float32(n_walks)


def _sss_invert_alpha(target: Float32, eta: Float32, g: Float32) -> Float32:
    """The single-scattering albedo whose random walk actually REPRODUCES
    diffuse reflectance `target` under this boundary.

    A closed-form albedo fit (Christensen & Burley 2015, what Cycles uses)
    ignores eta entirely, and the boundary is not a small correction: total internal
    reflection keeps photons inside longer, so with alpha < 1 more of them are
    absorbed and the surface goes DARK. Measured on head.pbrt, the same
    material rendered 0.91x pbrt at eta = 1 but 0.64x at eta = 1.33, while
    pbrt barely moved (0.3577 -> 0.3518) because its tabulated
    SubsurfaceFromDiffuse solves for sigma WITH the boundary in the loop.

    So solve it the same way, but against the walk this renderer actually
    runs -- which is more faithful than porting a closed form fitted to a
    different transport model, and costs one bisection at parse time.
    Common random numbers keep the objective monotone, so plain bisection is
    stable at a few thousand walks."""
    var t = min(max(target, Float32(0.0)), Float32(0.999))
    if t <= Float32(0.0):
        return Float32(0.0)
    # 16 bisection steps resolve alpha to 2^-16, far finer than the Monte
    # Carlo noise floor, so more would only cost parse time.
    comptime N_WALKS = 6000
    var lo = Float32(0.0)
    var hi = Float32(1.0)
    for _ in range(16):
        var mid = Float32(0.5) * (lo + hi)
        if _sss_walk_reflectance(mid, eta, g, N_WALKS) < t:
            lo = mid
        else:
            hi = mid
    return Float32(0.5) * (lo + hi)


def _sss_reflectance(s: Pointer[SceneParseState, MutUntrackedOrigin],
                     params: ParameterDictionary) -> RGB:
    """The subsurface `reflectance`, resolved even when it is a TEXTURE.

    This used to be a bare `params.get_rgb("reflectance", RGB(0.5))`, which
    silently returns the grey 0.5 default for a texture-valued parameter --
    and `head.pbrt` specifies `"texture reflectance" ["albedomap"]`, so all of
    the skin's colour was being discarded. The head rendered NEUTRAL GREY
    (chromaticity .332/.343/.325) where pbrt gives skin (.441/.290/.235); the
    albedomap's own mean is .478/.287/.235, i.e. the colour was entirely in
    the texture that was being thrown away. Exactly the silent-asset-failure
    shape catalogued in project_silent_asset_load_failures: a plausible image,
    never an error.

    The interior is registered as ONE homogeneous medium, so a spatially
    varying reflectance cannot be represented exactly; the image's mean is the
    honest approximation and is warned about. Constant and imagemap textures
    are resolved; anything else keeps pbrt's default and says so."""
    var refl_f = params.get_floats("reflectance")
    if len(refl_f) >= 3:
        return RGB(refl_f[0], refl_f[1], refl_f[2])
    if len(refl_f) == 1:
        return RGB(refl_f[0], refl_f[0], refl_f[0])
    var tname = params.get_string("reflectance", "")
    if tname == "":
        return RGB(Float32(0.5))
    for ci in range(len(s[unsafe_offset=0].const_tex_names)):
        if s[unsafe_offset=0].const_tex_names[ci] == tname:
            return RGB(s[unsafe_offset=0].const_tex_rgb[ci*3+0], s[unsafe_offset=0].const_tex_rgb[ci*3+1], s[unsafe_offset=0].const_tex_rgb[ci*3+2])
    for ti in range(len(s[unsafe_offset=0].tex_names)):
        if s[unsafe_offset=0].tex_names[ti] == tname:
            var fstr = s[unsafe_offset=0].tex_files[ti]
            var flen = fstr.byte_length()
            var fbuf = unsafe_alloc[UInt8](flen + 1)
            for k in range(flen): fbuf[unsafe_offset=k] = fstr.unsafe_ptr()[unsafe_offset=k]
            fbuf[unsafe_offset=flen] = UInt8(0)
            var data_out = unsafe_alloc[Pointer[Float32, MutUntrackedOrigin]](1)
            var w_out = unsafe_alloc[Int32](1)
            var h_out = unsafe_alloc[Int32](1)
            w_out[unsafe_offset=0] = Int32(0); h_out[unsafe_offset=0] = Int32(0)
            var ok = external_call["load_texture_rgb", Int32,
                Pointer[UInt8, MutUntrackedOrigin],
                Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin],
                Pointer[Int32, MutUntrackedOrigin],
                Pointer[Int32, MutUntrackedOrigin],
                Int32](fbuf, data_out, w_out, h_out, Int32(0))
            var out = RGB(Float32(0.5))
            if ok != 0 and Int(w_out[unsafe_offset=0]) > 0 and Int(h_out[unsafe_offset=0]) > 0:
                var n = Int(w_out[unsafe_offset=0]) * Int(h_out[unsafe_offset=0])
                var ptr = data_out[unsafe_offset=0]
                var sr = Float64(0); var sg = Float64(0); var sb = Float64(0)
                for k in range(n):
                    sr += Float64(ptr[unsafe_offset=k*3+0]); sg += Float64(ptr[unsafe_offset=k*3+1]); sb += Float64(ptr[unsafe_offset=k*3+2])
                var inv = Float64(1) / Float64(n)
                out = RGB(Float32(sr*inv), Float32(sg*inv), Float32(sb*inv))
                print("Note: subsurface \"reflectance\" is the texture '" + tname +
                      "'; the interior is one homogeneous medium, so its MEAN colour is used.")
                _ = external_call["free_texture_rgb", Int32,
                    Pointer[Float32, MutUntrackedOrigin]](ptr)
            else:
                print("Warning: subsurface \"reflectance\" texture '" + tname +
                      "' (" + fstr + ") failed to load — falling back to grey 0.5, which will render colourless.")
            fbuf.unsafe_free(); data_out.unsafe_free(); w_out.unsafe_free(); h_out.unsafe_free()
            return out
    print("Warning: subsurface \"reflectance\" names texture '" + tname +
          "', which is not a constant or imagemap — using grey 0.5.")
    return RGB(Float32(0.5))


def _psc_handle_make_named_material(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                                   s: Pointer[SceneParseState, MutUntrackedOrigin],
                                   inline_type: Bool = False):
    """Builds a NamedMaterial from a `Material`/`MakeNamedMaterial` directive.
    Scans every parameter ONCE, generically, into a ParameterDictionary
    (_psc_collect_params -- lexer.mojo), then interprets specific named
    parameters by querying it -- mirrors real pbrt's own ParameterDictionary
    design instead of hand-scanning each (name, type) combination inline.
    See project_parser_architecture memory for why this replaced the old
    single-pass special-cased scan."""
    var mat_name = unsafe_alloc[UInt8](PSC_NAME_MAX)
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
        type_str = String(unsafe_from_utf8_ptr=mat_name.as_imm())
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
        elif type_str == "interface" or type_str == "none" or type_str == "":
            # pbrt: `Material ""` (and `"none"`) creates a NULL material --
            # no surface scattering at all, the shape exists only to bound a
            # participating medium. It is the standard idiom for a medium
            # boundary and clouds.pbrt uses exactly that (`Material ""` on the
            # sphere holding the cloud medium).
            #
            # This used to fall through to the unsupported-type branch below
            # and render as flat 50%-grey diffuse -- an opaque grey ball where
            # the scene wanted an invisible boundary, which is why clouds.pbrt
            # produced a featureless grey disc (std 0.006) rather than a cloud.
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
            # is why Medium.is_sss exists (those events must not be charged
            # to the path's maxdepth budget). Same approach as Cycles.
            mat_type = MatKind.dielectric
            is_subsurface = True
        else:
            # Unrecognized material type — used to fall back to a flat
            # 50%-grey diffuse in total silence, which made scenes using it
            # look wrong with no clue why. mat_type already defaults to
            # MatKind.diffuse above, so this just adds the warning.
            warn_unsupported("material type", type_str, "renders as flat 50%-grey diffuse",
                             "diffuse, conductor, dielectric, thindielectric, coateddiffuse,"
                             + " coatedconductor, diffusetransmission, mix, subsurface,"
                             + " interface/none, hair, measured")
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
                    var (f_eta, f_ok) = load_spd_rgb(scene_path(s[unsafe_offset=0].scene_dir, eta_name, "conductor eta spectrum"))
                    if f_ok:
                        metal_eta = f_eta
                        has_spectral_conductor = True
                    else:
                        print("SPD load FAILED (cannot open/parse), material '"
                              + String(unsafe_from_utf8_ptr=mat_name.as_imm())
                              + "' eta falls back to 0.5:", scene_path(s[unsafe_offset=0].scene_dir, eta_name, "conductor eta spectrum"))
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
                var (f_k, fk_ok) = load_spd_rgb(scene_path(s[unsafe_offset=0].scene_dir, k_name, "conductor k spectrum"))
                if fk_ok:
                    metal_k = f_k
                    has_spectral_conductor = True
                else:
                    print("SPD load FAILED (cannot open/parse), material '"
                          + String(unsafe_from_utf8_ptr=mat_name.as_imm())
                          + "' k falls back to 0.5:", scene_path(s[unsafe_offset=0].scene_dir, k_name, "conductor k spectrum"))
            elif k_name != "":
                print("Warning: unknown named spectrum '" + k_name
                      + "' for k — falling back to 0.5. Supported: metal-{Ag,Al,Au,Cu,CuZn,TiO2,MgO}-*, or a .spd file path.")

    # "reflectance": either an RGB/float value, OR a texture reference --
    # looked up in tex_names (imagemap) first, then constant textures, then
    # procedural checkerboard textures (which need Material's embedded
    # checker_* fields and so can't participate in a texture graph), and
    # finally through _resolve_affine_rgb, which folds arbitrarily nested
    # "scale"/"mix" graphs over a single imagemap into one (scale, bias) pair.
    # That last path covers ~49 scale and ~20 mix spectrum declarations in the
    # corpus. Only a product of two *different* textures is out of reach.
    var tex_idx_for_mat = Int32(-1)
    var sss_mean_refl_for_mat = RGB(Float32(1))
    var tex_scale_for_mat = RGB(Float32(1))
    var tex_bias_for_mat = RGB(Float32(0))
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
                for ti in range(len(s[unsafe_offset=0].tex_names)):
                    if s[unsafe_offset=0].tex_names[ti] == tex_name:
                        tex_idx_for_mat = Int32(ti)
                        matched_tex = True
                        break
                if not matched_tex:
                    for ci in range(len(s[unsafe_offset=0].const_tex_names)):
                        if s[unsafe_offset=0].const_tex_names[ci] == tex_name:
                            rgb = RGB(s[unsafe_offset=0].const_tex_rgb[ci*3+0], s[unsafe_offset=0].const_tex_rgb[ci*3+1], s[unsafe_offset=0].const_tex_rgb[ci*3+2])
                            matched_tex = True
                            break
                if not matched_tex:
                    for ki in range(len(s[unsafe_offset=0].checker_tex_names)):
                        if s[unsafe_offset=0].checker_tex_names[ki] == tex_name:
                            # -2 marks the material as using the embedded
                            # procedural checkerboard fields below (see
                            # shading.mojo's _tex_lookup) rather than the
                            # imagemap texture table.
                            tex_idx_for_mat = Int32(-2)
                            checker_tex1 = RGB(s[unsafe_offset=0].checker_tex1[ki*3+0], s[unsafe_offset=0].checker_tex1[ki*3+1], s[unsafe_offset=0].checker_tex1[ki*3+2])
                            checker_tex2 = RGB(s[unsafe_offset=0].checker_tex2[ki*3+0], s[unsafe_offset=0].checker_tex2[ki*3+1], s[unsafe_offset=0].checker_tex2[ki*3+2])
                            checker_uscale = s[unsafe_offset=0].checker_uscale[ki]
                            checker_vscale = s[unsafe_offset=0].checker_vscale[ki]
                            matched_tex = True
                            break
                if not matched_tex:
                    # Everything else -- "scale", "mix", and any nesting of
                    # them over one imagemap -- folds into a single affine
                    # (scale, bias) pair. See _resolve_affine_rgb.
                    var aff = _resolve_affine_rgb(s, tex_name, 0)
                    if aff.ok:
                        if aff.tex_idx >= 0:
                            tex_idx_for_mat = aff.tex_idx
                            tex_scale_for_mat = aff.scale
                            tex_bias_for_mat = aff.bias
                        else:
                            # Collapsed to a constant (e.g. a mix of two
                            # constants by a constant amount).
                            rgb = aff.bias
                        matched_tex = True
                if not matched_tex:
                    var is_known = False
                    for mi in range(len(s[unsafe_offset=0].mix_tex_names)):
                        if s[unsafe_offset=0].mix_tex_names[mi] == tex_name:
                            is_known = True
                            break
                    if not is_known:
                        for si in range(len(s[unsafe_offset=0].scale_tex_names)):
                            if s[unsafe_offset=0].scale_tex_names[si] == tex_name:
                                is_known = True
                                break
                    if is_known:
                        print("Warning: texture '" + tex_name + "' combines two"
                              + " different textures (a texture-valued scale or"
                              + " mix amount over a textured base) — gonzales"
                              + " folds texture graphs into one lookup, so this"
                              + " falls back to flat albedo.")
                    else:
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
    # pbrt NAMESPACES a coatedconductor's two roughnesses: the dielectric coat
    # is "interface.roughness" and the metal underneath is
    # "conductor.roughness"; plain "roughness" is not what such a material
    # declares. We only ever queried the plain name, so BOTH were silently
    # dropped and the material fell back to roughness 0 -- a PERFECT MIRROR
    # where the scene asked for a slightly rough metal under a slightly rough
    # coat. killeroo-coated-gold ("interface.roughness" 0.02,
    # "conductor.roughness" 0.002) is exactly that scene, and a delta mirror is
    # unshadeable for SPPM: a visible point on it can gather no photons and
    # NEE through no lobe, so 408 of 9216 pixels held a visible point that
    # received neither photons nor light, and rendered BLACK.
    #
    # Same namespacing bug already fixed once for this material's
    # conductor.eta/conductor.k (project_spd_spectrum_files_unsupported); the
    # roughness pair was missed in that pass. Our coated_conductor reuses the
    # conductor GGX lobe (see shade_coated_conductor), so the CONDUCTOR
    # roughness is the one that maps onto roughU/roughV; the coat's own
    # roughness has nowhere to go yet and says so rather than vanishing.
    if params.has("conductor.roughness"):
        var cr = params.get_floats("conductor.roughness")
        if len(cr) > 0:
            mat_roughU = cr[0]; mat_roughV = cr[0]
    if params.has("interface.roughness") and not params.has("conductor.roughness"):
        var ir = params.get_floats("interface.roughness")
        if len(ir) > 0:
            mat_roughU = ir[0]; mat_roughV = ir[0]
    elif params.has("interface.roughness"):
        print("Note: coatedconductor \"interface.roughness\" is not modelled separately —"
              + " the coat reuses the conductor lobe, so only \"conductor.roughness\" is applied.")
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
            for ti in range(len(s[unsafe_offset=0].tex_names)):
                if s[unsafe_offset=0].tex_names[ti] == rough_tex:
                    rough_tex_idx_for_mat = Int32(ti)
                    break
    var urough_f = params.get_floats("uroughness")
    if len(urough_f) > 0:
        mat_roughU = urough_f[0]
    else:
        var urough_tex = params.get_string("uroughness", "")
        if urough_tex != "":
            for ti in range(len(s[unsafe_offset=0].tex_names)):
                if s[unsafe_offset=0].tex_names[ti] == urough_tex:
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
        var bsdf_path = scene_path(s[unsafe_offset=0].scene_dir, params.get_string("filename", ""), "measured BSDF")
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
        # The boundary's relative index, needed by the albedo inversion below:
        # total internal reflection is what makes the same alpha render very
        # differently at eta 1.0 and 1.33 (see _sss_invert_alpha).
        var sss_eta = mat_ior if mat_ior > Float32(0) else Float32(1.33)
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
            var refl = _sss_reflectance(s, params)
            var mfp = _mb_float_or_rgb(params, "mfp", RGB(Float32(1)))
            print("Note: subsurface \"reflectance\"/\"mfp\" inverted with the Christensen-Burley fit, not pbrt's tabulated SubsurfaceFromDiffuse — close in character, not bit-comparable.")
            # Remember what the medium was built from, so the shader can
            # correct each point back to its own texel (Material.sss_mean_refl).
            sss_mean_refl_for_mat = refl
            var ar = _sss_invert_alpha(refl.r, sss_eta, sss_g)
            var ag = _sss_invert_alpha(refl.g, sss_eta, sss_g)
            var ab = _sss_invert_alpha(refl.b, sss_eta, sss_g)
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
        sss_medium_idx = Int32(len(s[unsafe_offset=0].med_names))
        s[unsafe_offset=0].med_names.append(String("__sss_interior"))
        s[unsafe_offset=0].med_sa.append(sig_a.r * sss_scale); s[unsafe_offset=0].med_sa.append(sig_a.g * sss_scale); s[unsafe_offset=0].med_sa.append(sig_a.b * sss_scale)
        s[unsafe_offset=0].med_ss.append(sig_s.r * sss_scale); s[unsafe_offset=0].med_ss.append(sig_s.g * sss_scale); s[unsafe_offset=0].med_ss.append(sig_s.b * sss_scale)
        s[unsafe_offset=0].med_g.append(sss_g)
        s[unsafe_offset=0].med_grid_idx.append(Int32(-1))
        s[unsafe_offset=0].med_nvdb_idx.append(Int32(-1))
        s[unsafe_offset=0].med_nvdb_temp_idx.append(Int32(-1))
        s[unsafe_offset=0].med_le_scale.append(Float32(0)); s[unsafe_offset=0].med_temp_offset.append(Float32(0)); s[unsafe_offset=0].med_temp_scale.append(Float32(1))
        s[unsafe_offset=0].med_is_sss.append(Int32(1))

        # An inline `Material "subsurface"` takes effect immediately, so bind
        # the interior to the live attribute state here -- shapes declared
        # after it pick it up exactly as they would an explicit
        # `MediumInterface "interior" ""`. The MakeNamedMaterial form binds
        # later instead, when `NamedMaterial` activates it.
        if inline_type:
            s[unsafe_offset=0].cur_attr.inside_medium = sss_medium_idx

    # "normalmap"/"bumpmap": register the file as an (unnamed) imagemap
    # texture and point the material's normal_tex_idx at it — same path as a
    # texture reference.
    var normal_tex_idx_for_mat = Int32(-1)
    var normalmap_file = params.get_string("normalmap", "")
    if normalmap_file == "":
        normalmap_file = params.get_string("bumpmap", "")
    if normalmap_file != "":
        var nm_file = scene_path(s[unsafe_offset=0].scene_dir, normalmap_file, "normal map")
        normal_tex_idx_for_mat = Int32(len(s[unsafe_offset=0].tex_names))
        s[unsafe_offset=0].tex_names.append(String("__normalmap"))
        s[unsafe_offset=0].tex_files.append(nm_file)

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
        for ti in range(len(s[unsafe_offset=0].tex_names)):
            if s[unsafe_offset=0].tex_names[ti] == disp_tex:
                bump_tex_idx_for_mat = Int32(ti)
                matched_disp = True
                break
        if not matched_disp:
            # Same affine fold as reflectance, but the bump path carries only
            # a scalar height multiplier with no offset, so a graph resolving
            # to a non-zero bias can't be represented. In practice that never
            # bites: villa's seven `mix` bump maps are all
            # (tex1=0, tex2=h, amount=texture), which is exactly scale=h,
            # bias=0. Channels are equal for a float texture graph, so .r is
            # the whole story.
            var aff = _resolve_affine_rgb(s, disp_tex, 0)
            if aff.ok and aff.tex_idx >= 0:
                if abs(aff.bias.r) > Float32(1e-6):
                    print("Warning: displacement texture '" + disp_tex
                          + "' resolves to an affine graph with a non-zero"
                          + " offset, which the bump path cannot represent —"
                          + " applying the scale only.")
                bump_tex_idx_for_mat = aff.tex_idx
                bump_scale_for_mat = aff.scale.r

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
    var nm = NamedMaterial(String(unsafe_from_utf8_ptr=mat_name.as_imm()))
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
    # same Material fields for unrelated values and must pass through untouched.
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
    nm.sss_mean_refl  = sss_mean_refl_for_mat
    nm.tex_bias       = tex_bias_for_mat
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
    s[unsafe_offset=0].named_materials.append(nm^)

    mat_name.unsafe_free()

def _psc_handle_named_material(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                               s: Pointer[SceneParseState, MutUntrackedOrigin]):
    var mat_name = unsafe_alloc[UInt8](PSC_NAME_MAX)
    _ = scanner_parse_quoted_string(handle, mat_name, PSC_NAME_MAX)
    s[unsafe_offset=0].cur_attr.mat_idx = Int32(-1)
    var name_str = String(unsafe_from_utf8_ptr=mat_name.as_imm())
    for i in range(len(s[unsafe_offset=0].named_materials)):
        if s[unsafe_offset=0].named_materials[i].name == name_str:
            s[unsafe_offset=0].cur_attr.mat_idx = Int32(i)
            # A `subsurface` material carries an interior medium with it (see
            # _psc_handle_make_named_material). Activating the material has to
            # bind that interior to the live attribute state too, or shapes
            # using it would get the dielectric shell with vacuum inside and
            # render as clear glass. The inline `Material` form binds at
            # declaration instead, since it takes effect immediately.
            if s[unsafe_offset=0].named_materials[i].sss_medium_idx >= Int32(0):
                s[unsafe_offset=0].cur_attr.inside_medium = s[unsafe_offset=0].named_materials[i].sss_medium_idx
            break
    mat_name.unsafe_free()
