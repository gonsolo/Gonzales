# Ken Perlin's "improved noise" (2002 gradient scheme), ported line-for-line
# from real pbrt-v4 source (~/src/pbrt-v4/src/pbrt/util/noise.{h,cpp}) --
# NOT reconstructed from memory or a generic Perlin implementation. Used
# ONLY to bake pbrt's procedural `MakeNamedMedium "cloud"` density field
# into an ordinary heterogeneous grid at PARSE time (see
# pbrt_parser.mojo's handle_named_medium, "cloud" branch) -- CPU-only, run
# once per cloud medium, never touched by the GPU/render-time path. That
# is a deliberate scope choice, not an oversight: gonzales's delta-tracking
# machinery (grid_sample_density, the local-majorant walk in gpu.mojo) is
# already proven correct for a dense density grid regardless of how that
# grid's values were produced -- baking reuses 100% of it, at the cost of
# swapping pbrt's live infinite-resolution noise evaluation for a fixed
# bake resolution (see CLOUD_BAKE_RES in pbrt_parser.mojo). A visually
# faithful trade for a scene that already ships a fixed-resolution
# reference render.
from std.math import floor

comptime NOISE_PERM_SIZE = 256

def _perlin_perm_table() -> List[Int32]:
    """The exact 512-entry (already self-doubled) permutation table from
    pbrt's noise.cpp -- transcribed via script from the real source, not
    retyped by hand. Doubling lets Grad() index `x+1` (up to 256) straight
    into the table without a second mask, exactly as pbrt does."""
    var t: List[Int32] = [
        151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
        140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
        247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
        57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
        74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
        60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
        65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
        200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
        52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
        207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
        119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
        129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
        218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
        81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
        184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
        222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
        151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
        140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
        247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
        57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
        74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
        60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
        65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
        200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
        52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
        207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
        119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
        129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
        218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
        81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
        184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
        222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
    ]
    return t^

@always_inline
def _perlin_grad(perm: List[Int32], x: Int, y: Int, z: Int, dx: Float32, dy: Float32, dz: Float32) -> Float32:
    var h = Int(perm[Int(perm[Int(perm[x]) + y]) + z])
    h &= 15
    var u = dx if (h < 8 or h == 12 or h == 13) else dy
    var v = dy if (h < 4 or h == 12 or h == 13) else dz
    var su = -u if (h & 1) != 0 else u
    var sv = -v if (h & 2) != 0 else v
    return su + sv

@always_inline
def _perlin_noise_weight(t: Float32) -> Float32:
    var t3 = t * t * t
    var t4 = t3 * t
    var t5 = t4 * t
    return Float32(6.0) * t5 - Float32(15.0) * t4 + Float32(10.0) * t3

@always_inline
def _lerp(t: Float32, a: Float32, b: Float32) -> Float32:
    return (Float32(1.0) - t) * a + t * b

def perlin_noise3(perm: List[Int32], x: Float32, y: Float32, z: Float32) -> Float32:
    """pbrt's Noise(Float,Float,Float): classic Perlin value noise, trilinearly
    blended over the unit lattice cell containing (x,y,z), quintic-smoothed."""
    var ix = Int(floor(x)); var iy = Int(floor(y)); var iz = Int(floor(z))
    var dx = x - Float32(ix); var dy = y - Float32(iy); var dz = z - Float32(iz)

    var mx = ix & (NOISE_PERM_SIZE - 1)
    var my = iy & (NOISE_PERM_SIZE - 1)
    var mz = iz & (NOISE_PERM_SIZE - 1)

    var w000 = _perlin_grad(perm, mx,     my,     mz,     dx,               dy,               dz)
    var w100 = _perlin_grad(perm, mx + 1, my,     mz,     dx - Float32(1.0), dy,               dz)
    var w010 = _perlin_grad(perm, mx,     my + 1, mz,     dx,               dy - Float32(1.0), dz)
    var w110 = _perlin_grad(perm, mx + 1, my + 1, mz,     dx - Float32(1.0), dy - Float32(1.0), dz)
    var w001 = _perlin_grad(perm, mx,     my,     mz + 1, dx,               dy,               dz - Float32(1.0))
    var w101 = _perlin_grad(perm, mx + 1, my,     mz + 1, dx - Float32(1.0), dy,               dz - Float32(1.0))
    var w011 = _perlin_grad(perm, mx,     my + 1, mz + 1, dx,               dy - Float32(1.0), dz - Float32(1.0))
    var w111 = _perlin_grad(perm, mx + 1, my + 1, mz + 1, dx - Float32(1.0), dy - Float32(1.0), dz - Float32(1.0))

    var wx = _perlin_noise_weight(dx); var wy = _perlin_noise_weight(dy); var wz = _perlin_noise_weight(dz)
    var x00 = _lerp(wx, w000, w100)
    var x10 = _lerp(wx, w010, w110)
    var x01 = _lerp(wx, w001, w101)
    var x11 = _lerp(wx, w011, w111)
    var y0 = _lerp(wy, x00, x10)
    var y1 = _lerp(wy, x01, x11)
    return _lerp(wz, y0, y1)

def perlin_dnoise3(perm: List[Int32], x: Float32, y: Float32, z: Float32) -> Tuple[Float32, Float32, Float32]:
    """pbrt's DNoise: finite-difference gradient of Noise, central at +delta,
    step 0.01 -- used only to perturb the cloud lookup point ("wispiness")."""
    var delta = Float32(0.01)
    var n = perlin_noise3(perm, x, y, z)
    var nx = perlin_noise3(perm, x + delta, y, z)
    var ny = perlin_noise3(perm, x, y + delta, z)
    var nz = perlin_noise3(perm, x, y, z + delta)
    return ((nx - n) / delta, (ny - n) / delta, (nz - n) / delta)

def cloud_density(
    perm: List[Int32], px: Float32, py: Float32, pz: Float32,
    frequency: Float32, wispiness: Float32, density: Float32,
) -> Float32:
    """pbrt's CloudMedium::Density(Point3f p), p already in the medium's own
    local space (matching pbrt's `renderFromMedium.ApplyInverse(p)` at the
    call site -- see pbrt_parser.mojo's caller, which applies world_to_medium
    before calling this). Two-octave DNoise perturbation of the lookup point
    ("wispiness"), then a 5-octave noise sum ("density"), then pbrt's altitude
    falloff: fades out above the cloud layer via `(1 - p.y)`, and pads the
    bottom via `2*max(0, 0.5 - p.y)` so the cloud base doesn't look sheared
    off flat. Both magic constants (4.5, 0.5) and the 1.99 octave-frequency
    ratio are pbrt's own, transcribed, not tuned."""
    var pp_x = frequency * px; var pp_y = frequency * py; var pp_z = frequency * pz
    if wispiness > Float32(0.0):
        var vomega = Float32(0.05) * wispiness
        var vlambda = Float32(10.0)
        for _ in range(2):
            var dn = perlin_dnoise3(perm, vlambda * pp_x, vlambda * pp_y, vlambda * pp_z)
            pp_x += vomega * dn[0]; pp_y += vomega * dn[1]; pp_z += vomega * dn[2]
            vomega *= Float32(0.5)
            vlambda *= Float32(1.99)

    var d = Float32(0.0)
    var omega = Float32(0.5)
    var lam = Float32(1.0)
    for _ in range(5):
        d += omega * perlin_noise3(perm, lam * pp_x, lam * pp_y, lam * pp_z)
        omega *= Float32(0.5)
        lam *= Float32(1.99)

    var d1 = (Float32(1.0) - py) * Float32(4.5) * density * d
    if d1 < Float32(0.0): d1 = Float32(0.0)
    if d1 > Float32(1.0): d1 = Float32(1.0)
    var base_pad = Float32(0.5) - py
    if base_pad < Float32(0.0): base_pad = Float32(0.0)
    var d2 = d1 + Float32(2.0) * base_pad
    if d2 < Float32(0.0): d2 = Float32(0.0)
    if d2 > Float32(1.0): d2 = Float32(1.0)
    return d2
