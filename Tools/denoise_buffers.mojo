"""Run gonzales's a-trous denoiser on buffers produced by ANOTHER renderer.

Why this exists: the corpus gallery shows gonzales denoised next to a raw pbrt
reference, so pbrt looks noisier than it is and the comparison is unfair to it.
Denoising the reference with the SAME filter makes the columns comparable.

It takes the guide buffers as data rather than deriving them, so the guides can
come from the renderer being denoised -- pbrt's `Film "gbuffer"` emits
Albedo.*, Ns.* and P.* (camera space by default), which is strictly better than
substituting gonzales's own AOVs for them.

Deliberately NOT an EXR reader: the OIIO bridge loads 3-channel RGB, and
pbrt's gbuffer is 25 named channels. Teaching the bridge arbitrary channel
selection is real work for no gain here, so the caller (Scripts/denoise_ref.py)
does the channel extraction with oiiotool/OIIO in Python and hands over one
flat float32 blob. This tool owns exactly one thing: calling denoise() with the
same parameters pipeline.mojo uses, so both columns get bit-identical
treatment.

Blob layout, little-endian float32 throughout, n = w*h:
    [0]        w  (as float)
    [1]        h
    [2 ..]     beauty   n*3
    [..]       albedo   n*3
    [..]       normals  n*3
    [..]       depth    n      (1e38 on a miss -- gonzales's own sentinel,
                                see render_aux_buffers; a 0 here would make
                                the depth term refuse to blend across sky)
Output blob: denoised RGB, n*3 float32, no header.
"""
from std.sys import argv
from std.memory import alloc
from gonzales.postprocess import denoise


# The exact parameters pipeline.mojo passes on the denoised render path. If
# those ever change, change them here in the same commit or the reference stops
# being treated identically to the renders it is compared against.
comptime N_PASSES = Int32(5)
comptime SIGMA_L  = Float32(4.0)
comptime SIGMA_A  = Float32(0.1)
comptime SIGMA_N  = Float32(0.3)
comptime SIGMA_D  = Float32(0.05)


def main() raises:
    var args = argv()
    if len(args) != 3:
        print("usage: denoise_buffers <in.blob> <out.blob>")
        return

    var fin = open(String(args[1]), "r")
    var raw = fin.read_bytes()
    fin.close()

    var n_floats = len(raw) // 4
    var src = alloc[Float32](n_floats)
    var rp = raw.unsafe_ptr().unsafe_bitcast[Float32]()
    for i in range(n_floats):
        src[i] = rp[unsafe_offset=i]

    var w = Int(src[0])
    var h = Int(src[1])
    var n = w * h
    var want = 2 + n * 3 * 3 + n
    if n_floats < want:
        print("blob too small: have " + String(n_floats) + " floats, need " + String(want))
        src.free()
        return

    var beauty  = src.unsafe_offset(2)
    var albedo  = beauty.unsafe_offset(n * 3)
    var normals = albedo.unsafe_offset(n * 3)
    var depth   = normals.unsafe_offset(n * 3)

    var out = alloc[Float32](n * 3)
    for i in range(n * 3):
        out[i] = Float32(0)

    denoise(beauty, albedo, normals, depth, Int32(w), Int32(h), out,
            N_PASSES, SIGMA_L, SIGMA_A, SIGMA_N, SIGMA_D)

    var blob = List[UInt8](capacity=n * 3 * 4)
    var ob = out.unsafe_bitcast[UInt8]()
    for i in range(n * 3 * 4):
        blob.append(ob[unsafe_offset=i])
    var fout = open(String(args[2]), "w")
    fout.write_bytes(Span(blob))
    fout.close()

    print("denoised " + String(w) + "x" + String(h)
          + " (a-trous, " + String(Int(N_PASSES)) + " passes)")
    src.free()
    out.free()
