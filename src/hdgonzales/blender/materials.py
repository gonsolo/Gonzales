# Evaluates the Blender materials UsdPreviewSurface cannot describe into lines of the delegate's
# "gonzales:materials" setting:
#   <M> hair sr sg sb beta_m beta_n ior                          hair with one colour
#   <M> hairlut path n_intercept n_random res beta_m beta_n ior  hair colour per (u, v, intercept, random)
#   <M> glass ior alpha
#   <M> principled alpha ior metallic path                       a Principled BSDF; path: baked base colour or -
# Node graphs are evaluated with numpy over many samples at once. Without UVs, images give their mean.
import hashlib
import math
import os
import tempfile

import numpy as np

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_images = {}
_lines = {}
_ramps = {}
LUT_INTERCEPTS, LUT_RANDOMS, LUT_RES = 8, 16, 128
BAKE_MAX = 2048


class Context:
    """Per-sample inputs: uv (n, 2) or None, intercept and random (n,) or scalars. With `filter`, images are
    box-filtered down to that resolution first, so a coarse grid of samples does not alias."""

    def __init__(self, n=1, uv=None, intercept=0.5, random=0.5, filter=None):
        self.n, self.uv, self.intercept, self.random, self.filter = n, uv, intercept, random, filter


def _rgb(v, n):
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 0:
        return np.full((n, 3), v, np.float32)
    if v.ndim == 1 and v.shape[0] == n and n not in (3, 4):
        return np.repeat(v[:, None], 3, axis=1)
    if v.ndim == 1:
        return np.broadcast_to(v[:3], (n, 3))
    return v[:, :3]


def _float(v, n):
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 2 or (v.ndim == 1 and v.shape[0] in (3, 4) and v.shape[0] != n):
        return (_rgb(v, n) * _LUMA).sum(axis=-1)
    return np.broadcast_to(v, (n,))


def _box(a, res):
    """`a` (h, w, ...) averaged down to at most res x res."""
    h, w = a.shape[:2]
    fy, fx = max(1, h // res), max(1, w // res)
    a = a[:h // fy * fy, :w // fx * fx]
    return a.reshape(h // fy, fy, w // fx, fx, *a.shape[2:]).mean(axis=(1, 3))


def _image(image, res=None):
    key = (image.name, tuple(image.size))
    if res is not None:
        if (key, res) not in _images:
            rgb, alpha = _image(image)
            _images[(key, res)] = (_box(rgb, res).astype(np.float32), _box(alpha, res).astype(np.float32))
        return _images[(key, res)]
    if key not in _images:
        px = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(px)
        px = px.reshape(image.size[1], image.size[0], image.channels)
        rgb = px[..., :3] if image.channels >= 3 else np.repeat(px[..., :1], 3, axis=2)
        if image.colorspace_settings.name == 'sRGB':
            rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        alpha = px[..., 3] if image.channels == 4 else np.ones(rgb.shape[:2], np.float32)
        _images[key] = (rgb.astype(np.float32), alpha.astype(np.float32))
    return _images[key]


def _rgb_to_hsv(c):
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    mx, mn = c.max(axis=1), c.min(axis=1)
    d = mx - mn
    s = np.where(mx > 0, d / np.where(mx > 0, mx, 1), 0)
    dd = np.where(d > 0, d, 1)
    h = np.where(mx == r, (g - b) / dd, np.where(mx == g, 2 + (b - r) / dd, 4 + (r - g) / dd))
    h = np.where(d > 0, (h / 6) % 1.0, 0)
    return h, s, mx


def _hsv_to_rgb(h, s, v):
    i = np.floor(h * 6) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    out = np.zeros((len(h), 3), np.float32)
    for k, (a, b, c) in enumerate([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]):
        m = i == k
        out[m] = np.stack([a[m], b[m], c[m]], axis=1)
    return out


def _blend(mode, f, a, b):
    f = f[:, None]
    if mode == 'SCREEN':
        return 1 - ((1 - f) + f * (1 - b)) * (1 - a)
    if mode == 'OVERLAY':
        return np.where(a < 0.5, a * ((1 - f) + 2 * f * b), 1 - ((1 - f) + 2 * f * (1 - b)) * (1 - a))
    if mode == 'SOFT_LIGHT':
        return (1 - f) * a + f * ((1 - a) * b * a + a * (1 - (1 - b) * (1 - a)))
    if mode == 'LINEAR_LIGHT':
        return a + f * (2 * b - 1)
    out = {'ADD': lambda: a + b, 'SUBTRACT': lambda: a - b, 'MULTIPLY': lambda: a * b,
           'DIFFERENCE': lambda: np.abs(a - b), 'DARKEN': lambda: np.minimum(a, b),
           'LIGHTEN': lambda: np.maximum(a, b)}.get(mode)
    if out is not None:
        return (1 - f) * a + f * out()
    if mode in ('HUE', 'SATURATION', 'VALUE', 'COLOR'):
        ha, sa, va = _rgb_to_hsv(np.maximum(a, 0))
        hb, sb, vb = _rgb_to_hsv(np.maximum(b, 0))
        h, s, v = {'HUE': (hb, sa, va), 'SATURATION': (ha, sb, va), 'VALUE': (ha, sa, vb),
                   'COLOR': (hb, sb, va)}[mode]
        return (1 - f) * a + f * _hsv_to_rgb(h, s, v)
    return (1 - f) * a + f * b


def _socket(node, name, kind=None):
    for s in node.inputs:
        if s.enabled and s.name == name and (kind is None or s.type == kind):
            return s
    return None


def _default(socket):
    v = getattr(socket, 'default_value', 0.0)
    return np.float32(v) if isinstance(v, (int, float)) else np.asarray(list(v), np.float32)


def _uv(socket, ctx):
    """The (n, 2) texture coordinates arriving at a texture's Vector input, or None."""
    if ctx.uv is None:
        return None
    if socket is None or not socket.is_linked:
        return ctx.uv
    node = socket.links[0].from_node
    if node.bl_idname == 'ShaderNodeMapping':
        uv = _uv(_socket(node, 'Vector'), ctx)
        if uv is None:
            return None
        loc, scale = _socket(node, 'Location'), _socket(node, 'Scale')
        return uv * (_default(scale)[:2] if scale else 1) + (_default(loc)[:2] if loc else 0)
    if node.bl_idname in ('ShaderNodeUVMap', 'ShaderNodeTexCoord', 'ShaderNodeAttribute'):
        return ctx.uv
    return None


def value(socket, ctx):
    """The value at `socket` for every sample: (n, 3) colours or (n,) floats, or a broadcastable constant."""
    n = ctx.n
    if socket is None:
        return np.float32(0)
    if not socket.is_linked:
        return _default(socket)
    link = socket.links[0]
    node, out = link.from_node, link.from_socket.name
    t = node.bl_idname
    if t == 'ShaderNodeHairInfo':
        v = {'Intercept': ctx.intercept, 'Random': ctx.random}.get(out, 1.0)
        return np.broadcast_to(np.asarray(v, np.float32), (n,))
    if t in ('ShaderNodeRGB', 'ShaderNodeValue'):
        return _default(node.outputs[0])
    if t == 'ShaderNodeTexImage':
        if node.image is None:
            return np.zeros(3, np.float32)
        rgb, alpha = _image(node.image, ctx.filter)
        uv = _uv(_socket(node, 'Vector'), ctx)
        if uv is None:
            return np.float32(alpha.mean()) if out == 'Alpha' else rgb.reshape(-1, 3).mean(axis=0)
        h, w = rgb.shape[:2]
        x = (np.mod(uv[:, 0], 1.0) * w).astype(np.int64).clip(0, w - 1)
        y = (np.mod(uv[:, 1], 1.0) * h).astype(np.int64).clip(0, h - 1)
        return alpha[y, x] if out == 'Alpha' else rgb[y, x]
    if t == 'ShaderNodeValToRGB':
        f = np.clip(_float(value(_socket(node, 'Fac') or node.inputs[0], ctx), n), 0, 1)
        positions = np.linspace(0, 1, 257)
        ramp = node.color_ramp
        key = (node.as_pointer(), ramp.interpolation, tuple((e.position, tuple(e.color)) for e in ramp.elements))
        if key not in _ramps:
            _ramps[key] = np.array([list(ramp.evaluate(p)) for p in positions], np.float32)
        table = _ramps[key]
        c = np.stack([np.interp(f, positions, table[:, k]) for k in range(4)], axis=1).astype(np.float32)
        return c[:, 3] if out == 'Alpha' else c[:, :3]
    if t == 'ShaderNodeMix':
        kind = {'RGBA': 'RGBA', 'FLOAT': 'VALUE', 'VECTOR': 'VECTOR'}.get(node.data_type, 'RGBA')
        f = _float(value(_socket(node, 'Factor', 'VALUE') or _socket(node, 'Factor'), ctx), n)
        if node.clamp_factor:
            f = np.clip(f, 0, 1)
        a, b = value(_socket(node, 'A', kind), ctx), value(_socket(node, 'B', kind), ctx)
        if kind == 'VALUE':
            return (1 - f) * _float(a, n) + f * _float(b, n)
        if kind != 'RGBA':
            return _rgb(a, n)
        c = _blend(node.blend_type, f, _rgb(a, n), _rgb(b, n))
        return np.clip(c, 0, 1) if node.clamp_result else c
    if t == 'ShaderNodeHueSaturation':
        f = _float(value(_socket(node, 'Fac') or _socket(node, 'Factor'), ctx), n)[:, None]
        c = _rgb(value(_socket(node, 'Color'), ctx), n)
        h, s, v = _rgb_to_hsv(np.maximum(c, 0))
        h = (h + _float(value(_socket(node, 'Hue'), ctx), n) - 0.5) % 1.0
        s = np.clip(s * _float(value(_socket(node, 'Saturation'), ctx), n), 0, 1)
        v = v * _float(value(_socket(node, 'Value'), ctx), n)
        return (1 - f) * c + f * _hsv_to_rgb(h, s, v)
    if t == 'ShaderNodeInvert':
        f = _float(value(_socket(node, 'Fac') or _socket(node, 'Factor'), ctx), n)[:, None]
        c = _rgb(value(_socket(node, 'Color'), ctx), n)
        return (1 - f) * c + f * (1 - c)
    if t == 'ShaderNodeMath':
        x, y = _float(value(node.inputs[0], ctx), n), _float(value(node.inputs[1], ctx), n)
        ops = {'ADD': lambda: x + y, 'SUBTRACT': lambda: x - y, 'MULTIPLY': lambda: x * y,
               'DIVIDE': lambda: np.where(y != 0, x / np.where(y != 0, y, 1), 0),
               'MINIMUM': lambda: np.minimum(x, y), 'MAXIMUM': lambda: np.maximum(x, y),
               'POWER': lambda: np.where(x > 0, np.abs(x) ** y, 0)}
        r = ops.get(node.operation, lambda: x)()
        return np.clip(r, 0, 1) if node.use_clamp else r
    # Anything else: the socket's own default, as if it were unlinked.
    return _default(socket)


def _scalar(socket):
    return float(_float(value(socket, Context()), 1)[0])


def _depends(socket, what, seen=None):
    """Whether the value at `socket` depends on images ('uv'), hair info ('hair') or either ('any')."""
    if socket is None or not socket.is_linked:
        return False
    seen = seen if seen is not None else set()
    node = socket.links[0].from_node
    if node.name in seen:
        return False
    seen.add(node.name)
    if node.bl_idname == 'ShaderNodeHairInfo' and what in ('hair', 'any'):
        return True
    if node.bl_idname == 'ShaderNodeTexImage' and what in ('uv', 'any'):
        return True
    return any(_depends(s, what, seen) for s in node.inputs if s.enabled)


def _sigma_from_reflectance(c, beta_n):
    d = (5.969 - 0.215 * beta_n + 2.532 * beta_n ** 2 - 10.73 * beta_n ** 3 + 5.574 * beta_n ** 4
         + 0.245 * beta_n ** 5)
    return [(math.log(min(max(float(x), 1e-4), 1.0)) / d) ** 2 for x in c]


def _write_pfm(path, rgb):
    """rgb (h, w, 3), row 0 at the bottom as in Blender images."""
    h, w = rgb.shape[:2]
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"PF\n%d %d\n-1.0\n" % (w, h))
        f.write(np.ascontiguousarray(rgb, dtype='<f4').tobytes())
    os.replace(tmp, path)


def _signature(material):
    h = hashlib.sha1(material.name.encode())
    for node in sorted(material.node_tree.nodes, key=lambda n: n.name):
        h.update(f"{node.bl_idname} {node.name}".encode())
        for prop in node.bl_rna.properties:
            if prop.type in ('BOOLEAN', 'INT', 'FLOAT', 'STRING', 'ENUM') and not prop.identifier.startswith('bl_'):
                v = getattr(node, prop.identifier, None)
                h.update(f"{prop.identifier}={list(v) if hasattr(v, '__len__') and not isinstance(v, str) else v}".encode())
        if getattr(node, 'image', None):
            h.update(f"{node.image.filepath} {tuple(node.image.size)}".encode())
        if node.bl_idname == 'ShaderNodeValToRGB':
            h.update(str([(e.position, list(e.color)) for e in node.color_ramp.elements]).encode())
        for s in node.inputs:
            v = getattr(s, 'default_value', None)
            v = list(v) if hasattr(v, '__len__') and not isinstance(v, str) else v
            link = f"{s.links[0].from_node.name}.{s.links[0].from_socket.identifier}" if s.is_linked else ""
            h.update(f"{s.identifier}={v}<{link}".encode())
    return h.hexdigest()[:16]


def _hair_colour(hair, ctx):
    """The reflectance colour of a hair shader. A Mix Shader partner is ignored: mixing colours instead of
    radiance darkens the hair far more than the partner does in Cycles."""
    return _rgb(value(_socket(hair, 'Color'), ctx), ctx.n)


def _hair_line(name, material, hair):
    beta_m = _scalar(_socket(hair, 'Roughness'))
    beta_n = _scalar(_socket(hair, 'Radial Roughness'))
    ior = _scalar(_socket(hair, 'IOR'))
    mode = hair.parametrization
    if mode == 'ABSORPTION':
        sigma = _rgb(value(_socket(hair, 'Absorption Coefficient'), Context()), 1)[0]
    elif mode == 'MELANIN':
        # Cycles' melanin model
        q = -math.log(max(1 - _scalar(_socket(hair, 'Melanin')), 1e-4))
        red = _scalar(_socket(hair, 'Melanin Redness'))
        sigma = [q * (1 - red) * e + q * red * p for e, p in zip((0.506, 0.841, 1.653), (0.343, 0.733, 1.924))]
        tint = _sigma_from_reflectance(_rgb(value(_socket(hair, 'Tint'), Context()), 1)[0], beta_n)
        sigma = [s + t for s, t in zip(sigma, tint)]
    else:
        if _depends(_socket(hair, 'Color'), 'any'):
            path = os.path.join(tempfile.gettempdir(), f"gonzales_hairlut_{_signature(material)}.pfm")
            if not os.path.exists(path):
                r = LUT_RES
                u, v = np.meshgrid((np.arange(r) + 0.5) / r, (np.arange(r) + 0.5) / r)
                uv = np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float32)
                rows = []
                for i in range(LUT_INTERCEPTS):
                    tiles = []
                    for k in range(LUT_RANDOMS):
                        ctx = Context(len(uv), uv, (i + 0.5) / LUT_INTERCEPTS, (k + 0.5) / LUT_RANDOMS, r)
                        tiles.append(np.clip(_hair_colour(hair, ctx), 0, 1).reshape(r, r, 3))
                    rows.append(np.concatenate(tiles, axis=1))
                _write_pfm(path, np.concatenate(rows, axis=0))
            return (f"{name} hairlut {path} {LUT_INTERCEPTS} {LUT_RANDOMS} {LUT_RES} "
                    f"{beta_m:.6g} {beta_n:.6g} {ior:.6g}")
        sigma = _sigma_from_reflectance(_hair_colour(hair, Context())[0], beta_n)
    return f"{name} hair {sigma[0]:.6g} {sigma[1]:.6g} {sigma[2]:.6g} {beta_m:.6g} {beta_n:.6g} {ior:.6g}"


def _bake_base_colour(material, socket):
    """A Principled base colour that is more than one image, baked per texel; None if not needed."""
    if socket is None or not socket.is_linked:
        return None
    node = socket.links[0].from_node
    vector = _socket(node, 'Vector') if node.bl_idname == 'ShaderNodeTexImage' else None
    if (vector is not None and not vector.is_linked) or not _depends(socket, 'uv'):
        return None
    path = os.path.join(tempfile.gettempdir(), f"gonzales_base_{_signature(material)}.pfm")
    if not os.path.exists(path):
        sizes = [max(n.image.size) for n in material.node_tree.nodes if n.bl_idname == 'ShaderNodeTexImage' and n.image]
        r = min(max(sizes or [256]), BAKE_MAX)
        u, v = np.meshgrid((np.arange(r) + 0.5) / r, (np.arange(r) + 0.5) / r)
        uv = np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float32)
        c = _rgb(value(socket, Context(len(uv), uv)), len(uv))
        _write_pfm(path, np.clip(c, 0, None).reshape(r, r, 3))
    return path


def _surface(material):
    out = next((n for n in material.node_tree.nodes
                if n.bl_idname == 'ShaderNodeOutputMaterial' and n.is_active_output), None)
    if out is None or not out.inputs['Surface'].is_linked:
        return None
    return out.inputs['Surface'].links[0].from_node


def _line(material):
    node = _surface(material)
    if node is None:
        return None
    name = 'M_' + hex(material.as_pointer())
    if node.bl_idname == 'ShaderNodeBsdfHairPrincipled':
        return _hair_line(name, material, node)
    if node.bl_idname == 'ShaderNodeMixShader':
        shaders = [s.links[0].from_node if s.is_linked else None for s in node.inputs[1:3]]
        hair = next((n for n in shaders if n is not None and n.bl_idname == 'ShaderNodeBsdfHairPrincipled'), None)
        return _hair_line(name, material, hair) if hair else None
    if node.bl_idname == 'ShaderNodeBsdfGlass' or (
            node.bl_idname == 'ShaderNodeBsdfPrincipled' and _scalar(_socket(node, 'Transmission Weight')) >= 0.5):
        r = _scalar(_socket(node, 'Roughness'))
        return f"{name} glass {_scalar(_socket(node, 'IOR')):.6g} {r * r:.6g}"
    if node.bl_idname == 'ShaderNodeBsdfPrincipled':
        # A coated diffuse: Blender's dielectric layer over its diffuse base. Specular IOR Level scales F0.
        r = _scalar(_socket(node, 'Roughness'))
        ior = max(_scalar(_socket(node, 'IOR')), 1.0)
        level = _scalar(_socket(node, 'Specular IOR Level')) if _socket(node, 'Specular IOR Level') else 0.5
        f0 = min(((ior - 1) / (ior + 1)) ** 2 * 2 * level, 0.99)
        eta = (1 + math.sqrt(f0)) / (1 - math.sqrt(f0)) if f0 > 0 else 1.0
        path = _bake_base_colour(material, _socket(node, 'Base Color')) or "-"
        return f"{name} principled {r * r:.6g} {eta:.6g} {_scalar(_socket(node, 'Metallic')):.6g} {path}"
    return None


def overrides(materials):
    lines = []
    for m in materials:
        if m.node_tree:
            key = (m.as_pointer(), _signature(m))
            if key not in _lines:
                _lines[key] = _line(m)
            if _lines[key]:
                lines.append(_lines[key])
    return "\n".join(lines)
