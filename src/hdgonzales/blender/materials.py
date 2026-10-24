# Evaluates the Blender materials UsdPreviewSurface cannot describe (hair, glass) into one line each
# for the delegate's "gonzales:materials" setting. Node inputs are evaluated once, at a representative
# point: hair at mid-length with mid randomness, image textures as their average colour.
import colorsys
import math

import numpy as np

_LUMA = (0.2126, 0.7152, 0.0722)
_image_means = {}


def _rgb(v):
    if isinstance(v, (int, float)):
        return [float(v)] * 3
    v = list(v)
    return (v + [v[-1]] * 3)[:3]


def _float(v):
    if isinstance(v, (int, float)):
        return float(v)
    c = _rgb(v)
    return sum(w * x for w, x in zip(_LUMA, c))


def _image_mean(image):
    key = (image.name, tuple(image.size))
    if key not in _image_means:
        px = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(px)
        mean = px.reshape(-1, image.channels)[:, :3].mean(axis=0) if image.channels >= 3 else np.repeat(px.mean(), 3)
        if image.colorspace_settings.name == 'sRGB':
            mean = np.where(mean <= 0.04045, mean / 12.92, ((mean + 0.055) / 1.055) ** 2.4)
        _image_means[key] = [float(x) for x in mean]
    return _image_means[key]


def _blend(mode, f, a, b):
    hsv = lambda c: colorsys.rgb_to_hsv(*[max(x, 0.0) for x in c])
    out = None
    if mode == 'MIX':
        out = b
    elif mode == 'ADD':
        out = [x + y for x, y in zip(a, b)]
    elif mode == 'SUBTRACT':
        out = [x - y for x, y in zip(a, b)]
    elif mode == 'MULTIPLY':
        out = [x * y for x, y in zip(a, b)]
    elif mode == 'SCREEN':
        return [1 - ((1 - f) + f * (1 - y)) * (1 - x) for x, y in zip(a, b)]
    elif mode == 'DIFFERENCE':
        out = [abs(x - y) for x, y in zip(a, b)]
    elif mode == 'DARKEN':
        out = [min(x, y) for x, y in zip(a, b)]
    elif mode == 'LIGHTEN':
        out = [max(x, y) for x, y in zip(a, b)]
    elif mode == 'OVERLAY':
        return [x * ((1 - f) + 2 * f * y) if x < 0.5 else 1 - ((1 - f) + 2 * f * (1 - y)) * (1 - x)
                for x, y in zip(a, b)]
    elif mode == 'SOFT_LIGHT':
        return [(1 - f) * x + f * ((1 - x) * y * x + x * (1 - (1 - y) * (1 - x))) for x, y in zip(a, b)]
    elif mode == 'LINEAR_LIGHT':
        return [x + f * (2 * y - 1) for x, y in zip(a, b)]
    elif mode in ('HUE', 'SATURATION', 'VALUE', 'COLOR'):
        ha, sa, va = hsv(a)
        hb, sb, vb = hsv(b)
        h, s, v = {'HUE': (hb, sa, va), 'SATURATION': (ha, sb, va), 'VALUE': (ha, sa, vb),
                   'COLOR': (hb, sb, va)}[mode]
        out = list(colorsys.hsv_to_rgb(h, s, v))
    else:
        out = b
    return [(1 - f) * x + f * y for x, y in zip(a, out)]


def _socket(node, name, kind=None):
    for s in node.inputs:
        if s.enabled and s.name == name and (kind is None or s.type == kind):
            return s
    return None


def value(socket):
    """The value at `socket`, evaluating whatever is linked into it."""
    if socket is None:
        return 0.0
    if not socket.is_linked:
        v = getattr(socket, 'default_value', 0.0)
        return float(v) if isinstance(v, (int, float)) else list(v)
    link = socket.links[0]
    node, out = link.from_node, link.from_socket.name
    t = node.bl_idname
    if t == 'ShaderNodeHairInfo':
        return {'Intercept': 0.5, 'Random': 0.5, 'Length': 1.0, 'Is Strand': 1.0}.get(out, 0.0)
    if t in ('ShaderNodeRGB', 'ShaderNodeValue'):
        v = node.outputs[0].default_value
        return float(v) if isinstance(v, (int, float)) else list(v)
    if t == 'ShaderNodeTexImage':
        mean = _image_mean(node.image) if node.image else [0.0] * 3
        return 1.0 if out == 'Alpha' else mean
    if t == 'ShaderNodeValToRGB':
        c = list(node.color_ramp.evaluate(min(max(_float(value(_socket(node, 'Fac') or node.inputs[0])), 0.0), 1.0)))
        return c[3] if out == 'Alpha' else c[:3]
    if t == 'ShaderNodeMix':
        kind = {'RGBA': 'RGBA', 'FLOAT': 'VALUE', 'VECTOR': 'VECTOR'}.get(node.data_type, 'RGBA')
        f = _float(value(_socket(node, 'Factor', 'VALUE') or _socket(node, 'Factor')))
        if node.clamp_factor:
            f = min(max(f, 0.0), 1.0)
        a, b = value(_socket(node, 'A', kind)), value(_socket(node, 'B', kind))
        if kind != 'RGBA':
            return (1 - f) * _float(a) + f * _float(b) if kind == 'VALUE' else _rgb(a)
        c = _blend(node.blend_type, f, _rgb(a), _rgb(b))
        return [min(max(x, 0.0), 1.0) for x in c] if node.clamp_result else c
    if t == 'ShaderNodeHueSaturation':
        f = _float(value(_socket(node, 'Fac') or _socket(node, 'Factor')))
        c = _rgb(value(_socket(node, 'Color')))
        h, s, v = colorsys.rgb_to_hsv(*[max(x, 0.0) for x in c])
        h = (h + _float(value(_socket(node, 'Hue'))) - 0.5) % 1.0
        s = min(max(s * _float(value(_socket(node, 'Saturation'))), 0.0), 1.0)
        v *= _float(value(_socket(node, 'Value')))
        return [(1 - f) * x + f * y for x, y in zip(c, colorsys.hsv_to_rgb(h, s, v))]
    if t == 'ShaderNodeInvert':
        f = _float(value(_socket(node, 'Fac') or _socket(node, 'Factor')))
        c = _rgb(value(_socket(node, 'Color')))
        return [(1 - f) * x + f * (1 - x) for x in c]
    if t == 'ShaderNodeMath':
        x, y = _float(value(node.inputs[0])), _float(value(node.inputs[1]))
        ops = {'ADD': x + y, 'SUBTRACT': x - y, 'MULTIPLY': x * y, 'DIVIDE': x / y if y else 0.0,
               'MINIMUM': min(x, y), 'MAXIMUM': max(x, y), 'POWER': x ** y if x > 0 else 0.0}
        r = ops.get(node.operation, x)
        return min(max(r, 0.0), 1.0) if node.use_clamp else r
    # Anything else: the socket's own default, as if it were unlinked.
    v = getattr(socket, 'default_value', 0.0)
    return float(v) if isinstance(v, (int, float)) else list(v)


def _sigma_from_reflectance(c, beta_n):
    d = (5.969 - 0.215 * beta_n + 2.532 * beta_n ** 2 - 10.73 * beta_n ** 3 + 5.574 * beta_n ** 4
         + 0.245 * beta_n ** 5)
    return [(math.log(min(max(x, 1e-4), 1.0)) / d) ** 2 for x in c]


def _hair(node, tint=None):
    """(sigma_a, beta_m, beta_n, ior) of a Principled Hair BSDF; `tint` replaces its colour."""
    beta_m = _float(value(_socket(node, 'Roughness')))
    beta_n = _float(value(_socket(node, 'Radial Roughness')))
    ior = _float(value(_socket(node, 'IOR')))
    mode = node.parametrization
    if mode == 'ABSORPTION':
        sigma = _rgb(value(_socket(node, 'Absorption Coefficient')))
    elif mode == 'MELANIN':
        # Cycles' melanin model
        q = -math.log(max(1 - _float(value(_socket(node, 'Melanin'))), 1e-4))
        red = _float(value(_socket(node, 'Melanin Redness')))
        eu, pheo = q * (1 - red), q * red
        sigma = [eu * e + pheo * p for e, p in zip((0.506, 0.841, 1.653), (0.343, 0.733, 1.924))]
        tint_sigma = _sigma_from_reflectance(_rgb(value(_socket(node, 'Tint'))), beta_n)
        sigma = [s + t for s, t in zip(sigma, tint_sigma)]
    else:
        sigma = _sigma_from_reflectance(tint or _rgb(value(_socket(node, 'Color'))), beta_n)
    return sigma, beta_m, beta_n, ior


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
    hair = tint = None
    if node.bl_idname == 'ShaderNodeBsdfHairPrincipled':
        hair = node
    elif node.bl_idname == 'ShaderNodeMixShader':
        shaders = [s.links[0].from_node for s in node.inputs[1:3] if s.is_linked]
        hairs = [n for n in shaders if n.bl_idname == 'ShaderNodeBsdfHairPrincipled']
        if hairs:
            hair = hairs[0]
            other = next((n for n in shaders if n is not hair), None)
            if other is not None and other.bl_idname == 'ShaderNodeBsdfPrincipled':
                # Blend the colours by the mix factor: Factor 1 selects the second shader.
                f = min(max(_float(value(node.inputs[0])), 0.0), 1.0)
                w = f if shaders[0] is hair else 1 - f
                tint = [(1 - w) * h + w * o for h, o in
                        zip(_rgb(value(_socket(hair, 'Color'))), _rgb(value(_socket(other, 'Base Color'))))]
    elif node.bl_idname == 'ShaderNodeBsdfGlass' or (
            node.bl_idname == 'ShaderNodeBsdfPrincipled'
            and _float(value(_socket(node, 'Transmission Weight'))) >= 0.5):
        r = _float(value(_socket(node, 'Roughness')))
        return f"{name} glass {_float(value(_socket(node, 'IOR'))):.6g} {r * r:.6g}"
    if hair is None:
        return None
    sigma, beta_m, beta_n, ior = _hair(hair, tint)
    return f"{name} hair {sigma[0]:.6g} {sigma[1]:.6g} {sigma[2]:.6g} {beta_m:.6g} {beta_n:.6g} {ior:.6g}"


def overrides(materials):
    lines = []
    for m in materials:
        if m.node_tree:
            line = _line(m)
            if line:
                lines.append(line)
    return "\n".join(lines)
