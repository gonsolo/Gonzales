"""Flattens a USD stage into arrays for gonzales.scene.description.usd_reader.

Instancing is kept: the prototypes of instanceable prims and of point instancers become templates, their placements
instances. Matrices are 16 floats in USD's row-vector order, which is gonzales' column-major CTM order."""
import glob, hashlib, os, re
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, UsdLux, Gf

_SKIP_PURPOSES = (UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide)
_ACESCG_TO_REC709 = np.array([[1.70505, -0.62179, -0.08326], [-0.13026, 1.14080, -0.01055], [-0.02400, -0.12897, 1.15297]])
_LIGHT_FILTER = [t for t in os.environ.get("USD_EXTRACT_LIGHTS", "").split(",") if t]   # debugging: only these light types
TEX_CACHE = os.environ.get("GONZALES_USD_TEX_CACHE", "/tmp/gonzales_usd_tex")
TEX_RES = int(os.environ.get("GONZALES_USD_TEX_RES", "64"))   # texels per UDIM tile side of the resident level
CURVE_COLOURS = 16      # materials per curve prim, from its quantized displayColor


def _udim_tiles(pattern):
    """[(udim, file)] of a texture path, one tile for a path without <UDIM>."""
    if "<UDIM>" not in pattern:
        return [(1001, pattern)] if os.path.exists(pattern) else []
    rx = re.compile(re.escape(pattern).replace("<UDIM>", r"(\d{4})") + "$")
    return sorted((int(m.group(1)), f) for f in glob.glob(pattern.replace("<UDIM>", "[0-9]" * 4)) if (m := rx.match(f)))


def _texture_input(surf, name):
    """The file pattern (with <UDIM>) feeding a UsdPreviewSurface input, or ''."""
    inp = surf.GetInput(name)
    if not inp or not inp.HasConnectedSource():
        return ""
    src = inp.GetConnectedSources()[0][0].source
    f = UsdShade.Shader(src.GetPrim()).GetInput("file")
    a = f.Get() if f else None
    if not a:
        return ""
    layer = next((p.layer for p in f.GetAttr().GetPropertyStack() if p.HasDefaultValue()), src.GetPrim().GetPrimStack()[0].layer)
    path = a.resolvedPath or (a.path if os.path.isabs(a.path) else    # relative to the layer that authored it
                              os.path.normpath(os.path.join(os.path.dirname(layer.realPath), a.path)))
    if "<UDIM>" in a.path and "<UDIM>" not in path:
        path = re.sub(r"\.1\d{3}\.", ".<UDIM>.", path)
    return path


def _texture(pattern, cols, rows, kind):
    """A UDIM texture as gonzales loads it: paged per tile at the resolution shading asks for (oiio.cc), or with
    GONZALES_USD_TEX_ATLAS=1 one resident atlas at TEX_RES texels a tile."""
    if os.environ.get("GONZALES_USD_TEX_ATLAS") == "1":
        return _atlas(pattern, cols, rows, kind)
    return f"udim:{kind}:{cols}:{rows}:{pattern}"


def _atlas(pattern, cols, rows, kind):
    """The resident level of a UDIM texture as one image of cols x rows tiles at TEX_RES texels, cached on disk."""
    os.makedirs(TEX_CACHE, exist_ok=True)
    out = os.path.join(TEX_CACHE, hashlib.sha1(f"{pattern}|{kind}|{TEX_RES}|{cols}x{rows}".encode()).hexdigest()[:20] + ".exr")
    if os.path.exists(out):
        return out
    import OpenImageIO as oiio
    r, nc = TEX_RES, 1 if kind == "alpha" else 3
    img = np.full((rows * r, cols * r, nc), {"alpha": 1.0, "rough": 0.0625}.get(kind, 0.0), np.float32)   # tiles without a file: opaque, roughness 0.5
    for udim, f in _udim_tiles(pattern):
        u, v = (udim - 1001) % 10, (udim - 1001) // 10
        if u >= cols or v >= rows:
            continue
        inp = oiio.ImageInput.open(f)
        if not inp:
            continue
        level = 0
        while inp.seek_subimage(0, level + 1) and inp.spec().width >= r:
            level += 1
        inp.seek_subimage(0, level)
        px = inp.read_image(0, level, 0, 1 if kind in ("rough", "metal") else nc, "float")
        inp.close()
        if px is None:
            continue
        px = np.asarray(px, np.float32).reshape(inp.spec().height, inp.spec().width, -1)[:, :, :nc]
        if px.shape[0] != r:
            px = px[:: max(1, px.shape[0] // r), :: max(1, px.shape[1] // r)][:r, :r]
        if kind == "color":
            px = np.maximum(px @ _ACESCG_TO_REC709.T.astype(np.float32), 0)
        elif kind == "rough":
            px = np.repeat(np.clip(px, 0, 1) ** 4, 3, axis=2)  # gonzales takes alpha = sqrt(texel), UsdPreviewSurface roughness^2
        elif kind == "metal":
            px = np.repeat(np.clip(px, 0, 1), 3, axis=2)
        img[(rows - 1 - v) * r:(rows - v) * r, u * r:(u + 1) * r] = px    # v grows upward, image rows downward
    o = oiio.ImageOutput.create(out)
    o.open(out, oiio.ImageSpec(cols * r, rows * r, nc, "half"))
    o.write_image(img)
    o.close()
    return out


def _tile_texels(pattern):
    """{udim: the first channel of the tile's level at TEX_RES texels}."""
    import OpenImageIO as oiio
    out = {}
    for udim, f in _udim_tiles(pattern):
        inp = oiio.ImageInput.open(f)
        if not inp:
            continue
        level = 0
        while inp.seek_subimage(0, level + 1) and inp.spec().width >= TEX_RES:
            level += 1
        inp.seek_subimage(0, level)
        px = inp.read_image(0, level, 0, 1, "float")
        inp.close()
        if px is not None:
            out[udim] = np.asarray(px, np.float32).reshape(-1)
    return out


def _metal_ior(metal, ior):
    """Mean metallic and median IOR over the texels inside UV islands, where the IOR map is not 0."""
    it, mt = (_tile_texels(ior) if ior else {}), (_tile_texels(metal) if metal else {})
    cov = {u: a > 0.5 for u, a in it.items()}
    iv = np.concatenate([a[cov[u]] for u, a in it.items()]) if it else np.zeros(0)
    mv = np.concatenate([a[cov[u]] if u in cov and len(cov[u]) == len(a) else a for u, a in mt.items()]) if mt else np.zeros(0)
    return (float(mv.mean()) if len(mv) else None), (float(np.median(iv)) if len(iv) else None)


def oiio_buf(path):
    import OpenImageIO as oiio
    return oiio.ImageBuf(path).get_pixels(oiio.FLOAT)[:, :, :3].astype(np.float32)


def _sample_atlas(path, uv, cols, rows):
    """Nearest texels of an atlas of cols x rows UDIM tiles (as _atlas writes it) at UDIM uvs."""
    img = oiio_buf(path)
    r = img.shape[0] // rows
    t = np.clip(np.floor(uv).astype(np.int64), 0, [cols - 1, rows - 1])
    f = np.clip(uv - t, 0, 0.9999)
    x = t[:, 0] * r + (f[:, 0] * r).astype(np.int64)
    y = (rows - 1 - t[:, 1]) * r + ((1 - f[:, 1]) * r).astype(np.int64).clip(0, r - 1)
    return img[y, x, :3].astype(np.float32)


def _quantize(col, k, iters=8):
    """k-means of colours: (centres, the label of each colour)."""
    nearest = lambda c: np.argmin((c * c).sum(1)[None] - 2 * col @ c.T, 1)
    uniq = np.unique(col[:100000].round(4), axis=0)
    if len(uniq) <= k and len(np.unique(col.round(4), axis=0)) <= k:
        return uniq, nearest(uniq)
    centres = col[np.random.default_rng(1).choice(len(col), k, replace=False)].copy()
    for _ in range(iters):
        label = nearest(centres)
        for j in range(k):
            if (label == j).any():
                centres[j] = col[label == j].mean(0)
    return centres, nearest(centres)


def _flat(m):
    return np.array(m, dtype=np.float32).reshape(16)


def _triangulate(counts, indices):
    counts = np.asarray(counts, dtype=np.int64)
    idx = np.asarray(indices, dtype=np.int32)
    ntri = np.maximum(counts - 2, 0)
    if ntri.sum() == 0:
        return np.zeros(0, dtype=np.int32)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    face = np.repeat(np.arange(len(counts)), ntri)
    local = np.arange(ntri.sum()) - np.repeat(np.cumsum(ntri) - ntri, ntri)
    s = starts[face]
    return np.stack((idx[s], idx[s + local + 1], idx[s + local + 2]), axis=1).reshape(-1).astype(np.int32)


class _Scene:
    def __init__(self, stage, time):
        self.stage, self.time = stage, time
        self.xc = UsdGeom.XformCache(time)
        self.points, self.indices, self.xforms = [], [], []
        self.mesh_rows = []         # point offset, points, index offset, triangles, material, template (-1: world), has uv
        self.mat_index, self.mat_rgb, self.mat_tex, self.mat_par = {}, [], [], []
        self.uvs = []
        self.templates = {}         # prototype path -> template id
        self.inst_tmpl, self.inst_xf = [], []
        self.curve_cp, self.curve_w, self.curve_mat = [], [], []
        self.curve_mats = {}        # (material, quantized colour) -> material
        self.curve_pending = []     # per curve prim: base material, root points, displayColor per curve, segment curves
        self.skins = {}             # colour texture file name -> [(points, UDIM uvs, xf)] of the world meshes using it
        self.n_points = self.n_indices = 0

    def material(self, prim):
        api = UsdShade.MaterialBindingAPI(prim)
        mat, _ = api.ComputeBoundMaterial(UsdShade.Tokens.full)     # production scenes bind by purpose
        if not mat:
            mat, _ = api.ComputeBoundMaterial()
        key = str(mat.GetPath()) if mat else ""
        if key in self.mat_index:
            return self.mat_index[key]
        rgb, tex, par = (0.5, 0.5, 0.5), {"color": "", "normal": "", "opacity": "", "roughness": "", "metallic": ""}, \
            {"opacity": 1.0, "metallic": 0.0, "ior": 1.5, "roughness": 0.5}
        if mat:
            surf = mat.ComputeSurfaceSource()[0]
            if surf and surf.GetIdAttr().Get() == "UsdPreviewSurface":
                inp = surf.GetInput("diffuseColor")
                if inp and not inp.HasConnectedSource() and inp.Get() is not None:
                    rgb = tuple(float(c) for c in inp.Get())
                tex["color"], tex["normal"] = _texture_input(surf, "diffuseColor"), _texture_input(surf, "normal")
                for name in par:
                    inp = surf.GetInput(name)
                    if inp and not inp.HasConnectedSource() and inp.Get() is not None:
                        par[name] = float(inp.Get())
                    elif inp and inp.HasConnectedSource() and name in tex:
                        tex[name] = _texture_input(surf, name)
                m, i = _metal_ior(tex["metallic"], _texture_input(surf, "ior"))
                par["metallic"] = par["metallic"] if m is None else m
                par["ior"] = par["ior"] if i is None else i
        if not 0.02 < par["metallic"] < 0.98:
            tex["metallic"] = ""        # all dielectric or all metal: no mix
        tiles = [t for p in tex.values() if p for t, _ in _udim_tiles(p)]
        tex["grid"] = (max((t - 1001) % 10 for t in tiles) + 1, max((t - 1001) // 10 for t in tiles) + 1) if tiles else (1, 1)
        self.mat_index[key] = len(self.mat_rgb)
        self.mat_rgb.append(rgb)
        self.mat_tex.append(tex)
        self.mat_par.append(par)
        return self.mat_index[key]

    def mesh(self, prim, xf, template):
        m = UsdGeom.Mesh(prim)
        pts = m.GetPointsAttr().Get(self.time)
        if pts is None or len(pts) == 0:
            return
        pts = np.asarray(pts, dtype=np.float32)
        counts = m.GetFaceVertexCountsAttr().Get(self.time)
        fvi = np.asarray(m.GetFaceVertexIndicesAttr().Get(self.time), dtype=np.int64)
        mat = self.material(prim)
        uv = None
        pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        if pv and pv.HasValue():
            vals = np.asarray(pv.Get(self.time), dtype=np.float32).reshape(-1, 2)
            idx = np.asarray(pv.GetIndices(self.time), dtype=np.int64) if pv.IsIndexed() else None
            interp = pv.GetInterpolation()
            if interp == UsdGeom.Tokens.faceVarying and len(fvi):
                corner = idx if idx is not None else np.arange(len(fvi))
                _, first, inv = np.unique(fvi * (len(vals) + 1) + corner, return_index=True, return_inverse=True)
                pts, uv, fvi = pts[fvi[first]], vals[corner[first]], inv.reshape(-1)    # one vertex per (point, uv)
            elif interp == UsdGeom.Tokens.vertex:
                uv = vals[idx] if idx is not None else vals
        tris = _triangulate(counts, fvi)
        if len(tris) == 0:
            return
        has_uv = uv is not None and len(uv) == len(pts)
        if has_uv and template == -1 and self.mat_tex[mat]["color"]:
            self.skins.setdefault(os.path.basename(self.mat_tex[mat]["color"]), []).append((pts, uv, xf))
        if has_uv:
            cols, rows = self.mat_tex[mat]["grid"]
            uv = uv / np.array((cols, rows), np.float32)        # into the material's atlas of UDIM tiles
        else:
            uv = np.zeros((len(pts), 2), np.float32)
        self.mesh_rows.append((self.n_points, len(pts), self.n_indices, len(tris) // 3, mat, template, int(has_uv)))
        self.points.append(pts.reshape(-1))
        self.uvs.append(uv.reshape(-1))
        self.indices.append(tris)
        self.xforms.append(_flat(xf))
        self.n_points += len(pts)
        self.n_indices += len(tris)

    def visible(self, prim):
        img = UsdGeom.Imageable(prim)
        if not img:
            return True
        return img.ComputeVisibility(self.time) != UsdGeom.Tokens.invisible and img.ComputePurpose() not in _SKIP_PURPOSES

    def template(self, root):
        """The template of a prototype subtree, its meshes relative to root (nested instances expanded)."""
        key = str(root.GetPath())
        if key in self.templates:
            return self.templates[key]
        tid = len(self.templates)
        self.templates[key] = tid
        root_inv = self.xc.GetLocalToWorldTransform(root).GetInverse()
        it = iter(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
        for prim in it:
            if not self.visible(prim):
                it.PruneChildren()
            elif prim.IsA(UsdGeom.Mesh):
                self.mesh(prim, self.xc.GetLocalToWorldTransform(prim) * root_inv, tid)
        return tid

    def point_instancer(self, prim):
        pi = UsdGeom.PointInstancer(prim)
        targets = pi.GetPrototypesRel().GetTargets()
        ids = pi.GetProtoIndicesAttr().Get(self.time)
        if not targets or ids is None:
            return
        tids = [self.template(self.stage.GetPrimAtPath(t)) for t in targets]
        world = self.xc.GetLocalToWorldTransform(prim)
        for k, m in enumerate(pi.ComputeInstanceTransformsAtTime(self.time, self.time)):
            self.inst_tmpl.append(tids[ids[k]])
            self.inst_xf.append(_flat(m * world))

    def traverse(self):
        it = iter(Usd.PrimRange(self.stage.GetPseudoRoot()))
        for prim in it:
            if not self.visible(prim):
                it.PruneChildren()
            elif prim.IsInstance():
                self.inst_tmpl.append(self.template(prim.GetPrototype()))
                self.inst_xf.append(_flat(self.xc.GetLocalToWorldTransform(prim)))
                it.PruneChildren()
            elif prim.IsA(UsdGeom.PointInstancer):
                self.point_instancer(prim)
                it.PruneChildren()
            elif prim.IsA(UsdGeom.Mesh):
                self.mesh(prim, self.xc.GetLocalToWorldTransform(prim), -1)
            elif prim.IsA(UsdGeom.BasisCurves):
                self.curves(prim, self.xc.GetLocalToWorldTransform(prim))

    def curve_material(self, base, rgb):
        """A copy of material base with the constant colour rgb and no textures: curves have no uv."""
        key = (base, tuple(np.round(rgb, 4)))
        if key not in self.curve_mats:
            self.curve_mats[key] = len(self.mat_rgb)
            self.mat_rgb.append(tuple(float(c) for c in rgb))
            self.mat_tex.append({"color": "", "normal": "", "opacity": "", "roughness": "", "metallic": "", "grid": (1, 1)})
            self.mat_par.append(dict(self.mat_par[base]))
        return self.curve_mats[key]

    def curves(self, prim, xf):
        """Cubic B-spline curves as gonzales' curve segments (one per window of 4 points), in world space. Colours come
        from displayColor, quantized to CURVE_COLOURS materials per prim."""
        c = UsdGeom.BasisCurves(prim)
        if c.GetTypeAttr().Get() != "cubic" or c.GetBasisAttr().Get() != "bspline":
            return
        counts = np.asarray(c.GetCurveVertexCountsAttr().Get(self.time), np.int64)
        pts = np.asarray(c.GetPointsAttr().Get(self.time), np.float32)
        if len(counts) == 0 or len(pts) != counts.sum():
            return
        w = c.GetWidthsAttr().Get(self.time)
        w = np.asarray(w, np.float32) if w is not None and len(w) else np.full(1, 0.01, np.float32)
        wv = np.repeat(w, counts) if len(w) == len(counts) else (w if len(w) == len(pts) else np.full(len(pts), w.mean(), np.float32))
        m = np.array(xf, np.float64)
        pts = (pts @ m[:3, :3] + m[3, :3]).astype(np.float32)
        wv = wv * np.float32(np.linalg.norm(m[:3, :3], axis=1).mean())
        nseg = np.maximum(counts - 3, 0)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        first = np.repeat(starts, nseg) + np.arange(nseg.sum()) - np.repeat(np.cumsum(nseg) - nseg, nseg)
        seg_curve = np.repeat(np.arange(len(counts)), nseg)
        self.curve_cp.append(np.stack([pts[first + k] for k in range(4)], 1).reshape(-1))
        self.curve_w.append(np.stack([(wv[first] + 4 * wv[first + 1] + wv[first + 2]) / 6,
                                      (wv[first + 1] + 4 * wv[first + 2] + wv[first + 3]) / 6], 1).reshape(-1))
        base = self.material(prim)
        pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
        col = np.asarray(pv.Get(self.time), np.float32).reshape(-1, 3) if pv and pv.HasValue() else np.zeros((0, 3), np.float32)
        if len(col) == len(pts):
            col = np.add.reduceat(col, starts) / counts[:, None]
        elif len(col) != len(counts):
            col = np.repeat(col[:1] if len(col) else np.array([self.mat_rgb[base]], np.float32), len(counts), 0)
        self.curve_pending.append((base, pts[starts], np.maximum(col @ _ACESCG_TO_REC709.T.astype(np.float32), 0), seg_curve))
        self.curve_mat.append(None)

    def finish_curves(self):
        """Each curve's colour: its material's colour texture at the uv of the closest skin vertex to its root (curves have
        no uv), else displayColor; quantized to CURVE_COLOURS materials per prim."""
        from scipy.spatial import cKDTree
        for k, (base, roots, col, seg_curve) in enumerate(self.curve_pending):
            pattern, (cols, rows) = self.mat_tex[base]["color"], self.mat_tex[base]["grid"]
            skin = self.skins.get(os.path.basename(pattern)) if pattern else None
            if skin:
                m = [np.array(xf, np.float64) for _, _, xf in skin]
                world = np.concatenate([p @ x[:3, :3] + x[3, :3] for (p, _, _), x in zip(skin, m)])
                uv = np.concatenate([u for _, u, _ in skin])[cKDTree(world).query(roots)[1]]
                col = _sample_atlas(_atlas(pattern, cols, rows, "color"), uv, cols, rows)
            elif pattern and _udim_tiles(pattern):      # no skin: the texture's mean colour
                img = oiio_buf(_atlas(pattern, cols, rows, "color"))
                px = img.reshape(-1, 3)
                px = px[px.sum(1) > 0] if (px.sum(1) > 0).any() else px      # texels inside the uv islands
                col = np.repeat(px.mean(0, keepdims=True), len(roots), 0)
            centres, label = _quantize(col, CURVE_COLOURS)
            mats = np.array([self.curve_material(base, cc) for cc in centres], np.int32)
            self.curve_mat[k] = mats[label[seg_curve]]


def _ring(n):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.cos(a), np.sin(a)


def _light_mesh(prim, kind):
    """Object-space emitting triangles of a rect, disk or cylinder light; they face -z, cylinders outward."""
    L = UsdLux
    if kind == "RectLight":
        w, h = L.RectLight(prim).GetWidthAttr().Get() * 0.5, L.RectLight(prim).GetHeightAttr().Get() * 0.5
        return np.array([(-w, -h, 0), (-w, h, 0), (w, h, 0), (w, -h, 0)], np.float32), np.array([0, 1, 2, 0, 2, 3], np.int32), 4 * w * h
    c, s = _ring(24)
    if kind == "DiskLight":
        r = L.DiskLight(prim).GetRadiusAttr().Get()
        pts = np.vstack(([(0, 0, 0)], np.stack((r * c, r * s, 0 * c), 1))).astype(np.float32)
        tri = np.stack((np.zeros(24), (np.arange(24) + 1) % 24 + 1, np.arange(24) + 1), 1).astype(np.int32).reshape(-1)
        return pts, tri, np.pi * r * r
    cyl = L.CylinderLight(prim)
    r, hl = cyl.GetRadiusAttr().Get(), cyl.GetLengthAttr().Get() * 0.5
    pts = np.vstack((np.stack((-hl + 0 * c, r * c, r * s), 1), np.stack((hl + 0 * c, r * c, r * s), 1))).astype(np.float32)
    k = np.arange(24); n = (k + 1) % 24
    tri = np.stack((k, n, 24 + n, k, 24 + n, 24 + k), 1).astype(np.int32).reshape(-1)
    return pts, tri, 2 * np.pi * r * 2 * hl


def _lights(stage, xc, time, visible):
    """World-space lights, instance proxies expanded: the few lights of a scene are not worth instancing."""
    out = {"distant": [], "dome": [], "sphere": [], "emit_pts": [], "emit_idx": [], "emit_rows": [], "emit_rgb": []}
    npts = nidx = 0
    it = iter(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    for prim in it:
        if not visible(prim):
            it.PruneChildren()
            continue
        if not prim.HasAPI(UsdLux.LightAPI):
            continue
        if _LIGHT_FILTER and prim.GetTypeName() not in _LIGHT_FILTER:
            continue
        api = UsdLux.LightAPI(prim)
        rgb = np.array(api.GetColorAttr().Get(time), np.float64) * api.GetIntensityAttr().Get(time) * 2.0 ** (api.GetExposureAttr().Get(time) or 0.0)
        if api.GetEnableColorTemperatureAttr().Get(time):
            rgb *= np.array(UsdLux.BlackbodyTemperatureAsRgb(api.GetColorTemperatureAttr().Get(time)))
        normalize = bool(api.GetNormalizeAttr().Get(time))
        xf = xc.GetLocalToWorldTransform(prim)
        kind = prim.GetTypeName()
        if kind == "DistantLight":
            half = np.radians(UsdLux.DistantLight(prim).GetAngleAttr().Get(time) or 0.0) * 0.5
            omega = 2 * np.pi * (1 - np.cos(half)) if half > 0 else 1.0      # USD gives radiance over the disc, gonzales irradiance
            d = np.array(xf.TransformDir(Gf.Vec3d(0, 0, -1)))
            out["distant"].append(np.concatenate((d / np.linalg.norm(d), rgb * (1.0 if normalize else omega))))
        elif kind == "DomeLight":
            f = UsdLux.DomeLight(prim).GetTextureFileAttr().Get()
            out["dome"].append((f.resolvedPath if f else "", rgb))
        elif kind == "SphereLight":
            r = UsdLux.SphereLight(prim).GetRadiusAttr().Get(time) * np.linalg.norm(np.array(xf.TransformDir(Gf.Vec3d(1, 0, 0))))
            c = xf.ExtractTranslation()
            out["sphere"].append(np.concatenate(((c[0], c[1], c[2], r), rgb / (4 * np.pi * r * r) if normalize else rgb)))
        elif kind in ("RectLight", "DiskLight", "CylinderLight"):
            pts, tri, area = _light_mesh(prim, kind)
            m = np.array(xf, np.float64)
            w = (np.hstack((pts, np.ones((len(pts), 1)))) @ m)[:, :3].astype(np.float32)
            out["emit_rows"].append((npts, len(w), nidx, len(tri) // 3))
            out["emit_pts"].append(w.reshape(-1)); out["emit_idx"].append(tri)
            out["emit_rgb"].append(rgb / area if normalize else rgb)
            npts += len(w); nidx += len(tri)
    return out


def _dome_mean(path):
    try:
        import OpenImageIO as oiio
        b = oiio.ImageBuf(path)
        st = oiio.ImageBufAlgo.computePixelStats(b)
        return np.array(st.avg[:3], np.float64)
    except Exception:
        return np.ones(3)


def _camera(stage, xc, time, path, aspect):
    cam = None
    if path:
        cam = stage.GetPrimAtPath(path) if path.startswith("/") else next((p for p in stage.Traverse() if p.GetName() == path), None)
    if not cam:
        cam = next((p for p in stage.Traverse() if p.IsA(UsdGeom.Camera)), None)
    if not cam:
        return None, 0.0
    c = UsdGeom.Camera(cam).GetCamera(time)
    h = c.horizontalAperture
    v = h / aspect if aspect >= 1 else h        # the shorter axis' aperture
    fov = 2 * np.degrees(np.arctan(0.5 * v / c.focalLength))
    if aspect < 1:
        fov = 2 * np.degrees(np.arctan(0.5 * h / c.focalLength))
    return _flat(xc.GetLocalToWorldTransform(cam)), float(fov)


def extract(path, camera_path="", aspect=16.0 / 9.0, frame=""):
    stage = Usd.Stage.Open(path)
    if frame:
        time = Usd.TimeCode(float(frame))
    else:
        time = Usd.TimeCode(stage.GetStartTimeCode()) if stage.HasAuthoredTimeCodeRange() else Usd.TimeCode.Default()
    sc = _Scene(stage, time)
    sc.traverse()
    sc.finish_curves()
    cam, fov = _camera(stage, sc.xc, time, camera_path, aspect)
    lt = _lights(stage, sc.xc, time, sc.visible)
    sky = np.zeros(3)
    for f, rgb in lt["dome"]:
        sky += rgb * (_dome_mean(f) if f else 1.0)
    cat = lambda xs, dt: np.ascontiguousarray(np.concatenate(xs).astype(dt)) if xs else np.zeros(0, dtype=dt)
    return {
        "points": cat(sc.points, np.float32),
        "indices": cat(sc.indices, np.int32),
        "mesh_xf": cat(sc.xforms, np.float32),
        "meshes": np.ascontiguousarray(np.array(sc.mesh_rows, dtype=np.int64).reshape(-1, 7)),
        "uvs": cat(sc.uvs, np.float32),
        "mat_color": [_texture(t["color"], *t["grid"], "color") if t["color"] else "" for t in sc.mat_tex],
        "mat_normal": [_texture(t["normal"], *t["grid"], "raw") if t["normal"] else "" for t in sc.mat_tex],
        "mat_opacity": [_atlas(t["opacity"], *t["grid"], "alpha") if t["opacity"] else "" for t in sc.mat_tex],
        "mat_rough": [_texture(t["roughness"], *t["grid"], "rough") if t["roughness"] else "" for t in sc.mat_tex],
        "mat_metal": [_texture(t["metallic"], *t["grid"], "metal") if t["metallic"] else "" for t in sc.mat_tex],
        "mat_par": np.ascontiguousarray(np.array([[p["opacity"], p["metallic"], p["ior"], p["roughness"]] for p in sc.mat_par],
                                                 dtype=np.float32).reshape(-1, 4)),   # opacity, metallic, ior, roughness
        "materials": np.ascontiguousarray(np.array(sc.mat_rgb, dtype=np.float32).reshape(-1, 3)),
        "curve_cp": cat(sc.curve_cp, np.float32),
        "curve_w": cat(sc.curve_w, np.float32),
        "curve_mat": cat(sc.curve_mat, np.int32),
        "n_templates": len(sc.templates),
        "inst_tmpl": np.ascontiguousarray(np.array(sc.inst_tmpl, dtype=np.int32)),
        "inst_xf": cat(sc.inst_xf, np.float32),
        "camera": cam if cam is not None else np.zeros(0, dtype=np.float32),
        "fov": fov,
        "distant": np.ascontiguousarray(np.array(lt["distant"], np.float32).reshape(-1, 6)),
        "spheres": np.ascontiguousarray(np.array(lt["sphere"], np.float32).reshape(-1, 7)),
        "emit_pts": cat(lt["emit_pts"], np.float32),
        "emit_idx": cat(lt["emit_idx"], np.int32),
        "emit_rows": np.ascontiguousarray(np.array(lt["emit_rows"], np.int64).reshape(-1, 4)),
        "emit_rgb": np.ascontiguousarray(np.array(lt["emit_rgb"], np.float32).reshape(-1, 3)),
        "sky": np.ascontiguousarray(sky.astype(np.float32)),
        "n_domes": len(lt["dome"]),
    }
