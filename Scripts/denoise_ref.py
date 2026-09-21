#!/usr/bin/env python3
"""Denoise a pbrt `Film "gbuffer"` render with gonzales's own a-trous filter.

The corpus gallery puts denoised gonzales renders next to a RAW pbrt
reference, which makes pbrt look noisier than it is. Running the same filter
over the reference makes the columns comparable.

The guide buffers come from pbrt's own gbuffer film rather than from a
gonzales render of the same scene: Albedo.*, Ns.* and P.* describe the very
image being denoised, so there is no risk of guiding with subtly different
geometry.

pbrt's gbuffer P is in CAMERA space by default (film.cpp: the
"coordinatesystem" parameter defaults to "camera"), so depth is simply |P|,
and a miss is P == 0. gonzales writes 1e38 for a miss (render_aux_buffers), and
the depth term is relative -- (d_n - d_0)^2 / d_0^2 -- so a 0 there would give
every sky pixel an enormous relative difference to its neighbours and the sky
would come out completely unfiltered. Map misses to the same sentinel.

ONE guide must come from gonzales, not pbrt: the ALBEDO. Measured on
barcelona, pbrt's Albedo.* carries ~6x less local detail than gonzales's
albedo.exr (local gradient 0.0128 vs 0.0786) and is clamped to [0,1], while
ours keeps the highlight boost added during the ganesha work. sigma_a = 0.1 is
tuned to OUR guide's contrast, so pbrt's flatter one makes the albedo term
stop blending far less often and smears texture that the gonzales columns
keep -- a HARSHER filter on the reference, which is the opposite of the fair
comparison this is for. Normals and depth need no such swap: both renderers
give unit normals, and the depth term is relative, (d_n - d_0)^2 / d_0^2, so
it only needs a true distance from the camera, which pbrt's camera-space P is.

Usage:  denoise_ref.py <gbuffer.exr> <out.exr> [albedo.exr]
"""
import os, struct, subprocess, sys
import numpy as np
import OpenImageIO as oiio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO, "build", "denoise_buffers")
MISS_DEPTH = 1e38          # gonzales's own miss sentinel, see render_aux_buffers


def read_gbuffer(path):
    src = oiio.ImageInput.open(path)
    if src is None:
        raise SystemExit(f"cannot open {path}")
    spec = src.spec()
    px = np.array(src.read_image(format="float")).reshape(
        spec.height, spec.width, spec.nchannels)
    ch = {n: i for i, n in enumerate(spec.channelnames)}
    need = ["R", "G", "B", "Albedo.R", "Albedo.G", "Albedo.B",
            "Ns.X", "Ns.Y", "Ns.Z", "P.X", "P.Y", "P.Z"]
    missing = [c for c in need if c not in ch]
    if missing:
        raise SystemExit(
            f"{path} is not a gbuffer render (missing {missing}). "
            'Render it with Film "gbuffer".')
    grab = lambda *n: np.stack([px[:, :, ch[c]] for c in n], -1)
    beauty = grab("R", "G", "B")
    albedo = grab("Albedo.R", "Albedo.G", "Albedo.B")
    normal = grab("Ns.X", "Ns.Y", "Ns.Z")
    P = grab("P.X", "P.Y", "P.Z")
    depth = np.linalg.norm(P, axis=-1)
    depth[depth == 0.0] = MISS_DEPTH
    return spec, beauty, albedo, normal, depth


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    gbuf, out = sys.argv[1], sys.argv[2]
    alb_override = sys.argv[3] if len(sys.argv) == 4 else None
    if not os.path.exists(TOOL):
        raise SystemExit(f"{TOOL} missing -- run `make denoise_buffers`")

    spec, beauty, albedo, normal, depth = read_gbuffer(gbuf)
    h, w = beauty.shape[:2]
    if alb_override:
        src = oiio.ImageInput.open(alb_override)
        if src is None:
            raise SystemExit(f"cannot open {alb_override}")
        s2 = src.spec()
        a2 = np.array(src.read_image(format="float")).reshape(
            s2.height, s2.width, -1)[:, :, :3]
        if (s2.height, s2.width) != (h, w):
            raise SystemExit(
                f"albedo is {s2.width}x{s2.height} but beauty is {w}x{h} -- "
                "render gonzales at the same resolution as the tile")
        albedo = a2

    blob = (struct.pack("<2f", float(w), float(h))
            + beauty.astype("<f4").tobytes()
            + albedo.astype("<f4").tobytes()
            + normal.astype("<f4").tobytes()
            + depth.astype("<f4").tobytes())
    tmp_in, tmp_out = out + ".in.blob", out + ".out.blob"
    with open(tmp_in, "wb") as f:
        f.write(blob)
    r = subprocess.run([TOOL, tmp_in, tmp_out], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(tmp_out):
        raise SystemExit(f"denoise_buffers failed: {r.stdout}{r.stderr}")

    den = np.frombuffer(open(tmp_out, "rb").read(),
                        dtype="<f4").reshape(h, w, 3)
    o = oiio.ImageOutput.create(out)
    o.open(out, oiio.ImageSpec(w, h, 3, "float"))
    o.write_image(np.ascontiguousarray(den, dtype=np.float32))
    o.close()
    os.remove(tmp_in); os.remove(tmp_out)

    lum = lambda a: float((a @ [0.2126, 0.7152, 0.0722]).mean())
    print(f"{out}  {w}x{h}  mean {lum(beauty):.5f} -> {lum(den):.5f} "
          f"({lum(den)/max(lum(beauty),1e-9):.4f}x)")


if __name__ == "__main__":
    main()
