# Compiles a Blender node graph feeding one material input into a Gonzales shader program
# (src/gonzales/scene/appearance/shader.mojo): Int32 words, the output register first, then 8-word instructions
# [op, dst, a, b, c, d, i0, i1] on 16 four-float registers. compile() returns None for a graph it cannot run;
# the caller then bakes that input as before.
import os
import struct
import tempfile

import numpy as np

STRIDE, REGS, RAMP_SIZE = 8, 16, 256
OP = dict(const=1, uv=2, image=3, math=4, vmath=5, mix=6, blend=7, clamp=8, map_range=9, ramp=10, hsv=11, invert=12,
          bright=13, gamma=14, split=15, combine=16, hsv_rgb=17, mapping=18, normal_map=19, curves=20, luma=21, average=22)
MATH = {n: i for i, n in enumerate((
    'ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE', 'MULTIPLY_ADD', 'POWER', 'LOGARITHM', 'SQRT', 'INVERSE_SQRT', 'ABSOLUTE',
    'EXPONENT', 'MINIMUM', 'MAXIMUM', 'LESS_THAN', 'GREATER_THAN', 'SIGN', 'COMPARE', 'SMOOTH_MIN', 'SMOOTH_MAX', 'ROUND',
    'FLOOR', 'CEIL', 'TRUNC', 'FRACT', 'MODULO', 'FLOORED_MODULO', 'WRAP', 'SNAP', 'PINGPONG', 'SINE', 'COSINE', 'TANGENT',
    'ARCSINE', 'ARCCOSINE', 'ARCTANGENT', 'ARCTAN2', 'SINH', 'COSH', 'TANH', 'RADIANS', 'DEGREES'))}
VECTOR_LANEWISE = {'ADD': 'ADD', 'SUBTRACT': 'SUBTRACT', 'MULTIPLY': 'MULTIPLY', 'DIVIDE': 'DIVIDE',
                   'MULTIPLY_ADD': 'MULTIPLY_ADD', 'ABSOLUTE': 'ABSOLUTE', 'POWER': 'POWER', 'SIGN': 'SIGN',
                   'MINIMUM': 'MINIMUM', 'MAXIMUM': 'MAXIMUM', 'FLOOR': 'FLOOR', 'CEIL': 'CEIL', 'FRACTION': 'FRACT',
                   'MODULO': 'MODULO', 'WRAP': 'WRAP', 'SNAP': 'SNAP', 'SINE': 'SINE', 'COSINE': 'COSINE',
                   'TANGENT': 'TANGENT', 'ROUND': 'ROUND'}
VECTOR = {n: i for i, n in enumerate(('CROSS_PRODUCT', 'PROJECT', 'REFLECT', 'REFRACT', 'FACEFORWARD', 'DOT_PRODUCT',
                                      'DISTANCE', 'LENGTH', 'SCALE', 'NORMALIZE'))}
BLEND = {n: i for i, n in enumerate(('MIX', 'ADD', 'MULTIPLY', 'SUBTRACT', 'SCREEN', 'DIVIDE', 'DIFFERENCE', 'DARKEN',
                                     'LIGHTEN', 'OVERLAY', 'DODGE', 'BURN', 'HUE', 'SATURATION', 'VALUE', 'COLOR',
                                     'SOFT_LIGHT', 'LINEAR_LIGHT', 'EXCLUSION'))}
MAP_RANGE = {'LINEAR': 0, 'STEPPED': 1, 'SMOOTHSTEP': 2, 'SMOOTHERSTEP': 3}
MAPPING = {'POINT': 0, 'TEXTURE': 1, 'VECTOR': 2, 'NORMAL': 3}
EXTENSION = {'REPEAT': 0, 'EXTEND': 1, 'CLIP': 2, 'MIRROR': 3}
SLOTS = {'base': 0, 'roughness': 1, 'emission': 2, 'normal': 3, 'bump': 4}


class Unsupported(Exception):
    pass


def _bits(f):
    return struct.unpack('<i', struct.pack('<f', float(f)))[0]


class _Instr:
    def __init__(self, op, dst, srcs, i0, i1, data, consts, imm_regs):
        self.op, self.dst, self.srcs, self.i0, self.i1, self.data = op, dst, list(srcs), i0, i1, list(data)
        self.consts = consts
        self.imm_regs = list(imm_regs)   # (virtual register, 'i0' or 'i1', bit shift)

    def uses(self):
        return self.srcs + [v for v, _, _ in self.imm_regs]


class Compiler:
    """Instructions on virtual registers, mapped onto the REGS real ones by words()."""

    def __init__(self):
        self.code, self.images, self.n = [], [], 0
        self.memo = {}

    def emit(self, op, srcs=(), i0=0, i1=0, data=(), consts=None, imm_regs=()):
        self.n += 1
        self.code.append(_Instr(OP[op], self.n, srcs, i0, i1, data, consts, imm_regs))
        return self.n

    def const(self, v):
        v = list(v) if hasattr(v, '__len__') else [v] * 4
        v = [float(x) for x in (v + [0.0, 0.0, 0.0, 1.0][len(v):])[:4]]
        key = ('const', tuple(v))
        if key not in self.memo:
            self.memo[key] = self.emit('const', consts=[_bits(x) for x in v])
        return self.memo[key]

    def image(self, path, raw):
        if (path, raw) not in self.images:
            self.images.append((path, raw))
        return self.images.index((path, raw))

    def words(self, out):
        """The program with registers allocated by last use; None when more than REGS values are live at once."""
        last = {v: k for k, ins in enumerate(self.code) for v in ins.uses()}
        last[out] = len(self.code)
        free, reg, words = list(range(REGS)), {}, [0]
        for k, ins in enumerate(self.code):
            a = ins.consts if ins.consts else ([reg[v] for v in ins.srcs] + [0, 0, 0, 0])[:4]
            i0, i1 = ins.i0, ins.i1
            for v, field, shift in ins.imm_regs:
                if field == 'i0':
                    i0 |= reg[v] << shift
                else:
                    i1 |= reg[v] << shift
            for v in set(ins.uses()):
                if last[v] == k:
                    free.append(reg[v])
            free.sort()
            if not free:
                return None
            dst = free[0]   # a value nobody reads still needs a register nobody else holds
            if ins.dst in last:
                reg[ins.dst] = free.pop(0)
            words += [ins.op, dst] + a + [i0, i1] + [_bits(x) for x in ins.data]
        words[0] = reg[out]
        return words


def _input(node, name, kind=None):
    for s in node.inputs:
        if s.enabled and (s.name == name or s.identifier == name) and (kind is None or s.type == kind):
            return s
    return None


def _image_file(image, raw):
    """A file Gonzales can read for `image`, and whether it holds data rather than sRGB colour."""
    if image.source not in ('FILE', 'GENERATED') or image.size[0] == 0:
        raise Unsupported(f"image {image.name}: source {image.source}")
    import bpy
    path = bpy.path.abspath(image.filepath, library=image.library) if image.filepath else ""
    if image.source == 'FILE' and not image.packed_file and path and os.path.isfile(path):
        return os.path.realpath(path), raw
    # Packed or generated: Blender's pixels, linearised, as a float file.
    px = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(px)
    px = px.reshape(image.size[1], image.size[0], image.channels)
    rgb = px[..., :3] if image.channels >= 3 else np.repeat(px[..., :1], 3, axis=2)
    if not raw and image.colorspace_settings.name == 'sRGB':
        rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    import hashlib
    key = hashlib.sha1(rgb.tobytes()).hexdigest()[:16]
    out = os.path.join(tempfile.gettempdir(), f"gonzales_image_{key}.pfm")
    if not os.path.exists(out):
        h, w = rgb.shape[:2]
        with open(out + ".tmp", "wb") as f:
            f.write(b"PF\n%d %d\n-1.0\n" % (w, h))
            f.write(np.ascontiguousarray(rgb, dtype='<f4').tobytes())
        os.replace(out + ".tmp", out)
    return out, True


class Graph:
    """Compiles sockets of one material's node tree; one instance per program."""

    def __init__(self):
        self.c = Compiler()

    # Values: (register, kind), kind 'f' float, 'c' colour, 'v' vector.
    def as_float(self, rk):
        r, k = rk
        if k == 'f':
            return r
        return self.c.emit('luma' if k == 'c' else 'average', [r])

    def socket(self, sock, stack=()):
        """The value arriving at an input socket."""
        if sock is None:
            raise Unsupported("missing socket")
        if not sock.is_linked:
            return self.default(sock)
        link = sock.links[0]
        if not link.is_valid or link.is_muted:
            return self.default(sock)
        return self.output(link.from_node, link.from_socket, stack)

    def default(self, sock):
        v = getattr(sock, 'default_value', 0.0)
        if sock.type in ('VALUE', 'INT', 'BOOLEAN'):
            return self.c.const(float(v)), 'f'
        if sock.type == 'RGBA':
            return self.c.const(list(v)), 'c'
        if sock.type == 'VECTOR':
            return self.c.const(list(v)[:3] + [0.0]), 'v'
        raise Unsupported(f"socket type {sock.type}")

    def output(self, node, out, stack):
        key = (tuple(n.as_pointer() for n in stack), node.as_pointer(), out.identifier)
        if key not in self.c.memo:
            self.c.memo[key] = self._node(node, out, stack)
        return self.c.memo[key]

    def f(self, node, name, stack, kind='VALUE'):
        return self.as_float(self.socket(_input(node, name, kind), stack))

    def v(self, node, name, stack, kind=None):
        return self.socket(_input(node, name, kind), stack)[0]

    def _node(self, node, out, stack):
        t, c = node.bl_idname, self.c
        if node.mute:
            raise Unsupported(f"muted node {node.name}")
        if t == 'NodeReroute':
            return self.socket(node.inputs[0], stack)
        if t == 'ShaderNodeGroup':
            if node.node_tree is None:
                raise Unsupported("empty group")
            g_out = next((n for n in node.node_tree.nodes if n.bl_idname == 'NodeGroupOutput' and n.is_active_output),
                         None)
            sock = next((s for s in g_out.inputs if s.identifier == out.identifier), None) if g_out else None
            return self.socket(sock, stack + (node,))
        if t == 'NodeGroupInput':
            if not stack:
                raise Unsupported("group input outside a group")
            group = stack[-1]
            sock = next((s for s in group.inputs if s.identifier == out.identifier), None)
            return self.socket(sock, stack[:-1])
        if t in ('ShaderNodeRGB', 'ShaderNodeValue'):
            return self.default(node.outputs[0])
        if t == 'ShaderNodeUVMap' or (t == 'ShaderNodeTexCoord' and out.name == 'UV'):
            return c.emit('uv'), 'v'
        if t == 'ShaderNodeTexImage':
            if out.name != 'Color' or node.image is None:
                raise Unsupported("image alpha or no image")
            if node.projection != 'FLAT' or node.image.source == 'TILED':
                raise Unsupported(f"image projection {node.projection} / {node.image.source}")
            raw = node.image.colorspace_settings.name not in ('sRGB', 'Filmic sRGB', 'AgX Base sRGB')
            path, raw = _image_file(node.image, raw)
            vin = _input(node, 'Vector')
            vec = self.v(node, 'Vector', stack) if vin is not None and vin.is_linked else c.emit('uv')
            flags = (1 if raw else 0) | (EXTENSION.get(node.extension, 0) << 1)
            return c.emit('image', [vec], i0=c.image(path, raw), i1=flags), 'c'
        if t == 'ShaderNodeMapping':
            vec = self.v(node, 'Vector', stack)
            loc = self.v(node, 'Location', stack) if _input(node, 'Location') else c.const([0, 0, 0, 0])
            rot = self.v(node, 'Rotation', stack) if _input(node, 'Rotation') else c.const([0, 0, 0, 0])
            scale = self.v(node, 'Scale', stack)
            return c.emit('mapping', [vec, loc, rot, scale], i0=MAPPING[node.vector_type]), 'v'
        if t == 'ShaderNodeMix':
            kind = {'FLOAT': 'Float', 'VECTOR': 'Vector', 'RGBA': 'Color'}.get(node.data_type)
            if kind is None:
                raise Unsupported(f"mix {node.data_type}")
            if kind == 'Vector' and node.factor_mode == 'NON_UNIFORM':
                fac = self.v(node, 'Factor_Vector', stack)
            else:
                fac = self.f(node, 'Factor_Float', stack)
            if node.clamp_factor:
                fac = c.emit('clamp', [fac, c.const(0.0), c.const(1.0)])
            a, b = self.v(node, 'A_' + kind, stack), self.v(node, 'B_' + kind, stack)
            if kind != 'Color':
                return c.emit('mix', [a, b, fac]), kind[0].lower()
            return c.emit('blend', [fac, a, b], i0=BLEND[node.blend_type], i1=int(node.clamp_result)), 'c'
        if t == 'ShaderNodeMixRGB':
            fac = self.f(node, 'Fac', stack)
            fac = c.emit('clamp', [fac, c.const(0.0), c.const(1.0)])
            a, b = self.v(node, 'Color1', stack), self.v(node, 'Color2', stack)
            return c.emit('blend', [fac, a, b], i0=BLEND[node.blend_type], i1=int(node.use_clamp)), 'c'
        if t == 'ShaderNodeMath':
            args = [self.as_float(self.socket(s, stack)) for s in node.inputs[:3]]
            return c.emit('math', args, i0=MATH[node.operation], i1=int(node.use_clamp)), 'f'
        if t == 'ShaderNodeVectorMath':
            op = node.operation
            args = [self.socket(s, stack)[0] for s in node.inputs[:3]]
            if op in VECTOR_LANEWISE:
                return c.emit('math', args, i0=MATH[VECTOR_LANEWISE[op]]), 'v'
            if op not in VECTOR:
                raise Unsupported(f"vector math {op}")
            scale = self.as_float(self.socket(node.inputs[3], stack))
            return c.emit('vmath', args + [scale], i0=VECTOR[op]), ('f' if out.name == 'Value' else 'v')
        if t == 'ShaderNodeValToRGB':
            ramp = node.color_ramp
            table = [x for i in range(RAMP_SIZE) for x in ramp.evaluate(i / (RAMP_SIZE - 1))]
            r = c.emit('ramp', [self.f(node, 'Fac', stack)], i0=RAMP_SIZE, i1=int(ramp.interpolation != 'CONSTANT'),
                       data=table)
            return (c.emit('split', [r], i0=3), 'f') if out.name == 'Alpha' else (r, 'c')
        if t == 'ShaderNodeHueSaturation':
            args = [self.v(node, 'Color', stack), self.f(node, 'Hue', stack), self.f(node, 'Saturation', stack),
                    self.f(node, 'Value', stack)]
            return c.emit('hsv', args, imm_regs=[(self.f(node, 'Fac', stack), 'i0', 0)]), 'c'
        if t == 'ShaderNodeInvert':
            return c.emit('invert', [self.v(node, 'Color', stack), self.f(node, 'Fac', stack)]), 'c'
        if t == 'ShaderNodeBrightContrast':
            return c.emit('bright', [self.v(node, 'Color', stack), self.f(node, 'Bright', stack),
                                     self.f(node, 'Contrast', stack)]), 'c'
        if t == 'ShaderNodeGamma':
            return c.emit('gamma', [self.v(node, 'Color', stack), self.f(node, 'Gamma', stack)]), 'c'
        if t in ('ShaderNodeSeparateColor', 'ShaderNodeSeparateRGB', 'ShaderNodeSeparateXYZ'):
            src = self.v(node, node.inputs[0].name, stack)
            if t == 'ShaderNodeSeparateColor' and node.mode != 'RGB':
                if node.mode != 'HSV':
                    raise Unsupported(f"separate {node.mode}")
                src = c.emit('hsv_rgb', [src], i0=0)
            lane = [s.identifier for s in node.outputs if s.enabled].index(out.identifier)
            return c.emit('split', [src], i0=lane), 'f'
        if t in ('ShaderNodeCombineColor', 'ShaderNodeCombineRGB', 'ShaderNodeCombineXYZ'):
            args = [self.as_float(self.socket(s, stack)) for s in node.inputs[:3]]
            r = c.emit('combine', args)
            if t == 'ShaderNodeCombineColor' and node.mode != 'RGB':
                if node.mode != 'HSV':
                    raise Unsupported(f"combine {node.mode}")
                r = c.emit('hsv_rgb', [r], i0=1)
            return r, ('v' if t == 'ShaderNodeCombineXYZ' else 'c')
        if t == 'ShaderNodeRGBCurve':
            m = node.mapping
            m.initialize()
            cv = m.curves
            table = []
            for i in range(RAMP_SIZE):
                x = i / (RAMP_SIZE - 1)
                table += [m.evaluate(cv[k], m.evaluate(cv[3], x)) for k in range(3)] + [1.0]
            return c.emit('curves', [self.v(node, 'Color', stack), self.f(node, 'Fac', stack)], i0=RAMP_SIZE,
                          data=table), 'c'
        if t == 'ShaderNodeMapRange':
            vec = node.data_type == 'FLOAT_VECTOR'
            get = (lambda n: self.v(node, n, stack, 'VECTOR')) if vec else (lambda n: self.f(node, n, stack))
            args = [get(n) for n in ('Value', 'From Min', 'From Max', 'To Min')]
            to_max = get('To Max')
            steps = get('Steps') if node.interpolation_type == 'STEPPED' else c.const(4.0)
            r = c.emit('map_range', args, i0=MAP_RANGE[node.interpolation_type], i1=int(node.clamp) << 16,
                       imm_regs=[(to_max, 'i1', 0), (steps, 'i1', 8)])
            return r, ('v' if vec else 'f')
        if t == 'ShaderNodeClamp':
            v, lo, hi = self.f(node, 'Value', stack), self.f(node, 'Min', stack), self.f(node, 'Max', stack)
            if node.clamp_type == 'RANGE':
                lo, hi = c.emit('math', [lo, hi], i0=MATH['MINIMUM']), c.emit('math', [lo, hi], i0=MATH['MAXIMUM'])
            return c.emit('clamp', [v, lo, hi]), 'f'
        raise Unsupported(f"node {t}")


def _program(graph, reg):
    words = graph.c.words(reg)
    if words is None:
        raise Unsupported("too many live values")
    return words, graph.c.images


def compile(socket, slot, strength=None):
    """(words, [(path, raw)]) for the value at `socket` as material input `slot`, or None.
    'roughness' outputs alpha^2 = roughness^4 (Gonzales reads a roughness texture as pbrt's remapped roughness);
    'emission' multiplies the colour by `strength` (a socket); 'normal' takes a Normal Map node."""
    if socket is None or not (socket.is_linked or (slot == 'emission' and strength is not None and strength.is_linked)):
        return None
    g, c = Graph(), None
    try:
        c = g.c
        if slot == 'normal':
            node = socket.links[0].from_node
            if node.bl_idname != 'ShaderNodeNormalMap' or node.space != 'TANGENT':
                raise Unsupported("normal input is not a tangent-space Normal Map")
            col = g.v(node, 'Color', ())
            return _program(g, c.emit('normal_map', [col, g.f(node, 'Strength', ())]))
        if slot == 'bump':
            return _program(g, g.as_float(g.socket(socket)))
        value = g.socket(socket)
        if slot == 'roughness':
            r = c.emit('clamp', [g.as_float(value), c.const(0.0), c.const(1.0)])
            return _program(g, c.emit('math', [r, c.const(4.0)], i0=MATH['POWER']))
        if slot == 'emission':
            s = g.as_float(g.socket(strength)) if strength is not None else c.const(1.0)
            return _program(g, c.emit('math', [value[0], s], i0=MATH['MULTIPLY']))
        return _program(g, value[0])
    except Unsupported as e:
        print(f"Gonzales: {slot} input of {socket.node.name} baked: {e}")
        return None


def token(slot, program, scale=1.0):
    """The material line's 'shader' token group: slot, scale, words in hex, image count, quoted image paths
    each prefixed by its raw flag."""
    from urllib.parse import quote
    words, images = program
    hexwords = "".join(f"{w & 0xffffffff:08x}" for w in words)
    paths = " ".join(quote(p, safe="/") for p, _ in images)
    return f"shader {SLOTS[slot]} {scale:.6g} {hexwords} {len(images)}" + (f" {paths}" if paths else "")
