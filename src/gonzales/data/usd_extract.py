"""Flattens a USD stage into arrays for gonzales.scene.description.usd_reader.

Instancing is kept: the prototypes of instanceable prims and of point instancers become templates, their placements
instances. Matrices are 16 floats in USD's row-vector order, which is gonzales' column-major CTM order."""
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Gf

_SKIP_PURPOSES = (UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide)


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
        self.mesh_rows = []         # point offset, points, index offset, triangles, material, template (-1: world)
        self.mat_index, self.mat_rgb = {}, []
        self.templates = {}         # prototype path -> template id
        self.inst_tmpl, self.inst_xf = [], []
        self.n_points = self.n_indices = 0

    def material(self, prim):
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        key = str(mat.GetPath()) if mat else ""
        if key in self.mat_index:
            return self.mat_index[key]
        rgb = (0.5, 0.5, 0.5)
        if mat:
            surf = mat.ComputeSurfaceSource()[0]
            if surf and surf.GetIdAttr().Get() == "UsdPreviewSurface":
                inp = surf.GetInput("diffuseColor")
                if inp and not inp.HasConnectedSource() and inp.Get() is not None:
                    rgb = tuple(float(c) for c in inp.Get())
        self.mat_index[key] = len(self.mat_rgb)
        self.mat_rgb.append(rgb)
        return self.mat_index[key]

    def mesh(self, prim, xf, template):
        m = UsdGeom.Mesh(prim)
        pts = m.GetPointsAttr().Get(self.time)
        if pts is None or len(pts) == 0:
            return
        tris = _triangulate(m.GetFaceVertexCountsAttr().Get(self.time), m.GetFaceVertexIndicesAttr().Get(self.time))
        if len(tris) == 0:
            return
        p = np.asarray(pts, dtype=np.float32).reshape(-1)
        self.mesh_rows.append((self.n_points, len(p) // 3, self.n_indices, len(tris) // 3, self.material(prim), template))
        self.points.append(p)
        self.indices.append(tris)
        self.xforms.append(_flat(xf))
        self.n_points += len(p) // 3
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


def _camera(stage, xc, time, path, aspect):
    cam = stage.GetPrimAtPath(path) if path else None
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


def extract(path, camera_path="", aspect=16.0 / 9.0):
    stage = Usd.Stage.Open(path)
    time = Usd.TimeCode(stage.GetStartTimeCode()) if stage.HasAuthoredTimeCodeRange() else Usd.TimeCode.Default()
    sc = _Scene(stage, time)
    sc.traverse()
    cam, fov = _camera(stage, sc.xc, time, camera_path, aspect)
    cat = lambda xs, dt: np.ascontiguousarray(np.concatenate(xs).astype(dt)) if xs else np.zeros(0, dtype=dt)
    return {
        "points": cat(sc.points, np.float32),
        "indices": cat(sc.indices, np.int32),
        "mesh_xf": cat(sc.xforms, np.float32),
        "meshes": np.ascontiguousarray(np.array(sc.mesh_rows, dtype=np.int64).reshape(-1, 6)),
        "materials": np.ascontiguousarray(np.array(sc.mat_rgb, dtype=np.float32).reshape(-1, 3)),
        "n_templates": len(sc.templates),
        "inst_tmpl": np.ascontiguousarray(np.array(sc.inst_tmpl, dtype=np.int32)),
        "inst_xf": cat(sc.inst_xf, np.float32),
        "camera": cam if cam is not None else np.zeros(0, dtype=np.float32),
        "fov": fov,
    }
