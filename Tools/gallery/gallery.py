#!/usr/bin/env python3
"""Integrator switcher gallery tooling: render tiles, compare them against the
pbrt reference tile, splice them into the published page.

The published page is the only durable store of the tiles. Its
`const IMAGES = {...};` JSON maps scene name -> {pbrt, pt, sppm, vcm}, each a
base64 JPEG. The page embeds images, so it is never committed; fetch it with
the Artifact tool (`action: read`, `path: index.html`, `out_dir: ...`), which
writes it to disk, then splice and republish with the artifact's url.

    gallery.py render  NAME SCENE MODE [--out DIR] [-- EXTRA_FLAGS...]
    gallery.py compare PAGE TILE.jpg...
    gallery.py splice  PAGE TILE.jpg...

A tile file is named NAME.MODE.jpg, MODE one of pt/sppm/vcm. Tiles are
gallery-identical: GPU, 256 wide at the film's aspect, 64 spp, denoiser on,
tonemapped with ACES 2.0 SDR.
"""
import argparse, base64, io, json, os, re, shutil, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODES = {"pt": [], "sppm": ["--sppm"], "vcm": ["--vcm"]}
# The banner each integrator prints: a tile is only accepted if its log shows
# the integrator its column claims (two PT tiles were once SPPM renders).
BANNERS = {"pt": None, "sppm": "SPPM", "vcm": "VCM"}
WIDTH = 256
IMAGES_RE = re.compile(r"const IMAGES = (\{.*?\});\n", re.S)


def film_aspect(scene):
    """yresolution/xresolution from the scene or anything it includes."""
    seen, todo, res = set(), [scene], {}
    while todo:
        path = os.path.abspath(todo.pop())
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        text = open(path, errors="replace").read()
        for axis in ("x", "y"):
            m = re.search(r'"integer %sresolution"\s*\[\s*(\d+)' % axis, text)
            if m and axis not in res:
                res[axis] = int(m.group(1))
        for inc in re.findall(r'^\s*(?:Include|Import)\s+"([^"]+)"', text, re.M):
            todo.append(os.path.join(os.path.dirname(path), inc))
    x, y = res.get("x", 1280), res.get("y", 720)
    return y / x


def tile_name(path):
    name, mode, _ = os.path.basename(path).rsplit(".", 2)
    if mode not in MODES:
        sys.exit(f"{path}: mode '{mode}' is not one of {sorted(MODES)}")
    return name, mode


def render(args):
    height = max(1, round(WIDTH * film_aspect(args.scene)))
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(os.path.abspath(args.out), f"{args.name}.{args.mode}")
    cmd = [os.path.join(REPO, "build", "gonzales"), "--gpu", "--spp", "64",
           "--resolution", f"{WIDTH}x{height}", *MODES[args.mode], *args.extra,
           os.path.abspath(args.scene)]
    start = time.time()
    with open(stem + ".log", "w") as log:
        rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
    # The film writes into the repo root under the scene's own filename.
    written = [f for f in os.listdir(REPO)
               if f.endswith(".exr") and os.path.getmtime(os.path.join(REPO, f)) >= start]
    beauty = [f for f in written if "albedo" not in f]
    if len(beauty) == 1:
        shutil.move(os.path.join(REPO, beauty[0]), stem + ".exr")
    for f in written:
        if os.path.exists(os.path.join(REPO, f)):
            os.remove(os.path.join(REPO, f))
    log = open(stem + ".log", errors="replace").read()
    if rc != 0:
        sys.exit(f"{args.name}: renderer exited {rc}, see {stem}.log")
    if not re.search(r"^GPU: ", log, re.M):
        sys.exit(f"{args.name}: not a GPU render, see {stem}.log")
    banner = BANNERS[args.mode]
    found = re.search(r"^(SPPM|VCM|BDPT)", log, re.M)
    if (found.group(1) if found else None) != banner:
        sys.exit(f"{args.name}: log shows {found.group(1) if found else 'PT'}, column is {args.mode}")
    if len(beauty) != 1:
        sys.exit(f"{args.name}: expected one new EXR in {REPO}, got {written}")
    subprocess.run(["oiiotool", stem + ".exr", "--ociodisplay", "sRGB - Display",
                    "ACES 2.0 - SDR 100 nits (Rec.709)", "--quality", "88",
                    "-o", stem + ".jpg"], check=True)
    print(stem + ".jpg")


def load_page(page):
    html = open(page, encoding="utf-8").read()
    m = IMAGES_RE.search(html)
    if not m:
        sys.exit(f"{page}: no `const IMAGES = ...;` block")
    return html, m, json.loads(m.group(1))


def compare(args):
    import numpy as np
    from PIL import Image
    _, _, images = load_page(args.page)
    decode = lambda data: np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), float) / 255
    for tile in args.tiles:
        name, mode = tile_name(tile)
        if name not in images or "pbrt" not in images[name]:
            print(f"{name}: no pbrt reference tile")
            continue
        ref = decode(base64.b64decode(images[name]["pbrt"]))
        got = decode(open(tile, "rb").read())
        if ref.shape != got.shape:
            print(f"{name}.{mode}: size {got.shape[:2]} != reference {ref.shape[:2]}")
            continue
        # Lit pixels are chosen on the REFERENCE, so a render that goes black
        # shows up as a low ratio instead of shrinking its own mask.
        lit = ref.mean(axis=2) > 0.15
        ratio = np.median(got.mean(axis=2)[lit]) / np.median(ref.mean(axis=2)[lit])
        channels = np.median(got[lit], axis=0) / np.median(ref[lit], axis=0)
        black = 100 * np.mean(got.mean(axis=2)[lit] < 0.05)
        old = ""
        if mode in images[name]:
            prev = decode(base64.b64decode(images[name][mode]))
            if prev.shape == ref.shape:
                old = f"  (published: {np.median(prev.mean(axis=2)[lit]) / np.median(ref.mean(axis=2)[lit]):.3f})"
        print(f"{name}.{mode}: lit ratio {ratio:.3f}  per-channel {channels.round(3)}  "
              f"black-on-lit {black:.1f}%{old}")


def splice(args):
    html, m, images = load_page(args.page)
    for tile in args.tiles:
        name, mode = tile_name(tile)
        images.setdefault(name, {})[mode] = base64.b64encode(open(tile, "rb").read()).decode()
        print(f"spliced {name}.{mode}")
    html = html[:m.start(1)] + json.dumps(images) + html[m.end(1):]
    open(args.page, "w", encoding="utf-8").write(html)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("name"); r.add_argument("scene"); r.add_argument("mode", choices=MODES)
    r.add_argument("--out", default="gallery-tiles")
    r.add_argument("extra", nargs="*")
    for cmd in ("compare", "splice"):
        c = sub.add_parser(cmd)
        c.add_argument("page"); c.add_argument("tiles", nargs="+")
    args = p.parse_args()
    {"render": render, "compare": compare, "splice": splice}[args.cmd](args)


if __name__ == "__main__":
    main()
