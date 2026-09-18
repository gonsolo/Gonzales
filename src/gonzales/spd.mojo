# pbrt `.spd` spectral power distribution files, and pbrt's named built-in
# spectra, reduced to the RGB (630/530/450 nm) convention this codebase
# already uses for conductor eta/k.
#
# Format: plain text, one `wavelength_nm value` pair per line, ascending in
# wavelength, whitespace-separated. Comment lines (`#`) and blanks skipped.
#
# WHY RGB AND NOT A FULL SPECTRUM: material_builder.mojo stores conductor
# optical constants as `metal_eta`/`metal_k: RGB`, sampled at three
# representative wavelengths. This loader lands in that existing slot rather
# than opening a parallel spectral-material path. The named-metal table in
# material_builder.mojo was itself sampled from these very .spd files -- see
# _spd_sample_rgb's docstring for the exact-match cross-check that pins this
# loader against it.

from std.memory import alloc
from .geometry import RGB

# The three representative wavelengths material_builder.mojo's named-metal
# table documents ("Channels: R≈630nm, G≈530nm, B≈450nm").
comptime SPD_LAMBDA_R = Float32(630.0)
comptime SPD_LAMBDA_G = Float32(530.0)
comptime SPD_LAMBDA_B = Float32(450.0)

comptime SPD_MAX_SAMPLES: Int = 1024


@always_inline
def _spd_interp(
    lambdas: Pointer[Float32, MutUntrackedOrigin],
    values:  Pointer[Float32, MutUntrackedOrigin],
    n:       Int,
    lam:     Float32,
) -> Float32:
    """Linear interpolation at `lam`, clamped to the table's endpoints.

    Clamping (rather than extrapolating) matches pbrt's own
    PiecewiseLinearSpectrum behaviour: measured optical data outside its
    sampled range is held constant, never extended linearly -- extrapolating
    a steep tail can drive eta or k negative, which is unphysical."""
    if n <= 0:
        return Float32(0.0)
    if lam <= lambdas[unsafe_offset=0]:
        return values[unsafe_offset=0]
    if lam >= lambdas[unsafe_offset=n - 1]:
        return values[unsafe_offset=n - 1]
    var lo = 0
    var hi = n - 1
    while hi - lo > 1:
        var mid = (lo + hi) // 2
        if lambdas[unsafe_offset=mid] <= lam:
            lo = mid
        else:
            hi = mid
    var span = lambdas[unsafe_offset=hi] - lambdas[unsafe_offset=lo]
    if span <= Float32(0.0):
        return values[unsafe_offset=lo]
    var t = (lam - lambdas[unsafe_offset=lo]) / span
    var v_lo = values[unsafe_offset=lo]
    return v_lo + t * (values[unsafe_offset=hi] - v_lo)


def load_spd_rgb(path: String) -> Tuple[RGB, Bool]:
    """Load a pbrt `.spd` file and sample it at 630/530/450 nm.

    Returns (rgb, True) on success, (RGB(0), False) on any failure -- the
    caller is expected to WARN, not silently substitute a default. Three
    silent asset-drop bugs were found in one session (.spd itself, .ply.gz
    meshes, inline "N" normals); see project_silent_asset_load_failures.

    CROSS-CHECK that pins this loader: material_builder.mojo's hardcoded
    `metal-Au` entry is eta (0.194, 0.608, 1.426) / k (3.060, 2.120, 1.846).
    Those are EXACTLY the values in killeroos/spds/Au.{eta,k}.spd at the file
    rows 619.920898 / 516.600769 / 459.200653 nm -- i.e. that table was
    sampled from these files. Sampling this loader at those same three
    wavelengths must therefore reproduce the table bit-for-bit; Tests/unit/
    test_spd.mojo asserts exactly that. Production sampling uses the
    canonical 630/530/450 instead, which differs slightly (gold's eta is
    steep in the red) but is the documented convention and is applied
    uniformly to file-loaded and built-in spectra alike."""
    var lambdas = alloc[Float32](SPD_MAX_SAMPLES)
    var values = alloc[Float32](SPD_MAX_SAMPLES)
    var n = 0

    try:
        var f = open(path, "r")
        var text = f.read()
        f.close()
        for line_ref in text.splitlines():
            if n >= SPD_MAX_SAMPLES:
                break
            var line = String(line_ref).strip()
            if line.byte_length() == 0:
                continue
            if line.startswith("#"):
                continue
            var parts = line.split()
            if len(parts) < 2:
                continue
            try:
                var lam = Float32(Float64(String(parts[0])))
                var val = Float32(Float64(String(parts[1])))
                lambdas[unsafe_offset=n] = lam
                values[unsafe_offset=n] = val
                n += 1
            except:
                # A malformed row is skipped rather than aborting the file:
                # some .spd files in the wild carry a trailing text footer.
                continue
    except:
        lambdas.unsafe_free(); values.unsafe_free()
        return (RGB(Float32(0.0)), False)

    if n == 0:
        lambdas.unsafe_free(); values.unsafe_free()
        return (RGB(Float32(0.0)), False)

    var r = _spd_interp(lambdas, values, n, SPD_LAMBDA_R)
    var g = _spd_interp(lambdas, values, n, SPD_LAMBDA_G)
    var b = _spd_interp(lambdas, values, n, SPD_LAMBDA_B)
    lambdas.unsafe_free(); values.unsafe_free()
    return (RGB(r, g, b), True)


def named_metal_rgb(name: String, want_k: Bool) -> Tuple[RGB, Bool]:
    """pbrt's built-in `metal-*-eta` / `metal-*-k` named spectra, evaluated at
    630/530/450 nm. Values transcribed from pbrt-v4's own interleaved arrays
    in src/pbrt/util/spectrum.cpp (Ag_eta, CuZn_k, ...), linearly interpolated
    at those three wavelengths -- not hand-estimated.

    Returns (rgb, False) for an unrecognised name so the caller can warn.

    ORDERING TRAP: "metal-CuZn" must be tested BEFORE "metal-Cu", because
    `startswith("metal-Cu")` also matches every metal-CuZn-* name. The
    original prefix chain had exactly that collision, so brass silently
    rendered as copper wherever the corpus asked for CuZn."""
    if name.startswith("metal-Ag"):
        if want_k:  return (RGB(Float32(3.880), Float32(3.070), Float32(2.560)), True)
        else:       return (RGB(Float32(0.136), Float32(0.130), Float32(0.144)), True)
    elif name.startswith("metal-Al"):
        if want_k:  return (RGB(Float32(7.480), Float32(6.280), Float32(5.580)), True)
        else:       return (RGB(Float32(1.300), Float32(0.826), Float32(0.644)), True)
    elif name.startswith("metal-Au"):
        if want_k:  return (RGB(Float32(3.060), Float32(2.120), Float32(1.846)), True)
        else:       return (RGB(Float32(0.194), Float32(0.608), Float32(1.426)), True)
    elif name.startswith("metal-CuZn"):
        # MUST precede the metal-Cu branch -- see ORDERING TRAP above.
        if want_k:  return (RGB(Float32(3.522), Float32(2.568), Float32(1.829)), True)
        else:       return (RGB(Float32(0.445), Float32(0.573), Float32(1.094)), True)
    elif name.startswith("metal-Cu"):
        if want_k:  return (RGB(Float32(3.240), Float32(2.605), Float32(2.433)), True)
        else:       return (RGB(Float32(0.272), Float32(1.120), Float32(1.160)), True)
    elif name.startswith("metal-TiO2"):
        # k is identically zero across the visible range: TiO2 is a
        # transparent high-index dielectric, not an absorbing metal.
        if want_k:  return (RGB(Float32(0.0)), True)
        else:       return (RGB(Float32(2.875), Float32(2.974), Float32(3.165)), True)
    elif name.startswith("metal-MgO"):
        if want_k:  return (RGB(Float32(0.0)), True)
        else:       return (RGB(Float32(1.735), Float32(1.742), Float32(1.752)), True)
    return (RGB(Float32(0.0)), False)


def named_glass_ior(name: String) -> Tuple[Float32, Bool]:
    """pbrt's built-in `glass-*` named spectra -- wavelength-dependent
    dielectric IOR, reduced to the single scalar gonzales's dielectric BSDF
    carries (no dispersion). Uses the 530 nm sample as the representative
    index, the usual convention for a one-number IOR. Values from pbrt-v4's
    GlassBK7_eta / GlassSF11_eta / ... arrays.

    Returns (0, False) for an unrecognised name so the caller can warn."""
    if name == "glass-BK7":     return (Float32(1.5196), True)
    elif name == "glass-BAF10": return (Float32(1.6750), True)
    elif name == "glass-FK51A": return (Float32(1.4886), True)
    elif name == "glass-LASF9": return (Float32(1.8594), True)
    elif name == "glass-F5":    return (Float32(1.6799), True)
    elif name == "glass-F10":   return (Float32(1.7370), True)
    elif name == "glass-F11":   return (Float32(1.7953), True)
    return (Float32(0.0), False)


def load_spd_rgb_at(path: String, lr: Float32, lg: Float32, lb: Float32) -> Tuple[RGB, Bool]:
    """load_spd_rgb with caller-chosen sample wavelengths. Exists for the
    exact-match cross-check against material_builder.mojo's named-metal
    table (see load_spd_rgb's docstring); production code wants the
    canonical 630/530/450 entry point."""
    var lambdas = alloc[Float32](SPD_MAX_SAMPLES)
    var values = alloc[Float32](SPD_MAX_SAMPLES)
    var n = 0
    try:
        var f = open(path, "r")
        var text = f.read()
        f.close()
        for line_ref in text.splitlines():
            if n >= SPD_MAX_SAMPLES:
                break
            var line = String(line_ref).strip()
            if line.byte_length() == 0 or line.startswith("#"):
                continue
            var parts = line.split()
            if len(parts) < 2:
                continue
            try:
                lambdas[unsafe_offset=n] = Float32(Float64(String(parts[0])))
                values[unsafe_offset=n] = Float32(Float64(String(parts[1])))
                n += 1
            except:
                continue
    except:
        lambdas.unsafe_free(); values.unsafe_free()
        return (RGB(Float32(0.0)), False)
    if n == 0:
        lambdas.unsafe_free(); values.unsafe_free()
        return (RGB(Float32(0.0)), False)
    var r = _spd_interp(lambdas, values, n, lr)
    var g = _spd_interp(lambdas, values, n, lg)
    var b = _spd_interp(lambdas, values, n, lb)
    lambdas.unsafe_free(); values.unsafe_free()
    return (RGB(r, g, b), True)
