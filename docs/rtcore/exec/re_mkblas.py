"""Prototype own BLAS builder (no Vulkan): triangles -> bytes the RT core traverses. See docs/rtcore/ACCEL_FORMAT.md."""
import struct, math, sys
sys.path.insert(0, '.')
import codec as C

NODE = 128
TIGHT = False
SCHEME = 'f0'

def sib_words(n, is_root):
    s = n.get('sib')
    if SCHEME == 'f0': return 0xf0000000, 0
    if SCHEME == 'zero': return (0xf0000000 if is_root else 0), 0
    if SCHEME == 'sib': return (0xf0000000, 0) if is_root else (s << 28, 0xffffffff - s)
    if SCHEME == 'parent': return (0xf0000000, 0) if is_root else (s << 28, (0x100000000 - (n['idx'] - n['par'])) & 0xffffffff)
    if SCHEME == 'sibz': return (0xf0000000, 0) if is_root else (s << 28, 0)
    raise ValueError
fb = C.fb


def tri_box(t):
    lo = [min(p[a] for p in t) for a in range(3)]
    hi = [max(p[a] for p in t) for a in range(3)]
    return lo, hi


def union(boxes):
    lo = [min(b[0][a] for b in boxes) for a in range(3)]
    hi = [max(b[1][a] for b in boxes) for a in range(3)]
    return lo, hi


class Tri:
    def __init__(self, v, ident):
        self.v = [tuple(float(c) for c in p) for p in v]
        self.id = ident
        self.box = tri_box(self.v)
        self.c = [(self.box[0][a] + self.box[1][a]) * 0.5 for a in range(3)]


def encodable(tris):
    """Can these triangles share one block? returns the block bytes or None."""
    try:
        blk = C.encode_block([t.v for t in tris], 0)
    except AssertionError:
        return None
    return blk


def make_blocks(tris):
    """Split a list of triangles (<=12) into blocks of <=4 triangles that encode, ids 4k..4k+3."""
    out = []
    i = 0
    while i < len(tris):
        n = min(4, len(tris) - i)
        while n >= 1:
            if encodable(tris[i:i + n]) is not None:
                break
            n -= 1
        out.append(tris[i:i + n])
        i += n
    return out


def quantise_node(boxes):
    """Common frame for a node: exps (3 bytes), origin floats, and slot box -> 6 bytes each (conservative)."""
    lo, hi = union(boxes)
    e = []
    for a in range(3):
        ext = hi[a] - lo[a]
        if ext <= 0:
            e.append(0)
        else:
            ee = math.frexp(ext)[1]       # ext <= 2**ee
            e.append(max(1, min(254, ee + 127)))
    org = [struct.unpack('<f', struct.pack('<f', l))[0] for l in lo]
    org = [o if o <= l else math.nextafter(o, -math.inf) for o, l in zip(org, lo)]
    return e, org


def qbox(e, org, box):
    q = []
    for a in range(3):
        sc = 2.0 ** (e[a] - 127) if e[a] else 1.0
        l = int(math.floor((box[0][a] - org[a]) / sc * 255.0 - 1e-6))
        h = int(math.ceil((box[1][a] - org[a]) / sc * 255.0 + 1e-6))
        q += [max(0, min(255, l)), max(0, min(255, h))]
    return q


class Builder:
    def __init__(self, tris, header_template):
        self.tris = [Tri(t, i) for i, t in enumerate(tris)]
        self.hdr = bytearray(header_template[:0x380])
        self.nodes = []          # list of dict(kind, ...)
        self.blocks = []

    # --- binary tree over triangles; returns nested ('leaf', [tris]) / ('int', [children]) ---
    def split(self, tris, maxleaf):
        if len(tris) <= maxleaf:
            return ('leaf', tris)
        lo, hi = union([t.box for t in tris])
        ax = max(range(3), key=lambda a: hi[a] - lo[a])
        s = sorted(tris, key=lambda t: t.c[ax])
        m = len(s) // 2
        return ('int', [self.split(s[:m], maxleaf), self.split(s[m:], maxleaf)])

    def count(self, n):
        if n[0] == 'leaf':
            return len(n[1])
        return sum(self.count(c) for c in n[1])

    def hw_children(self, node):
        """Expand a binary node into <=12 hardware children (each ('leaf' with <=12 tris) or ('int',...))."""
        kids = [node]
        while True:
            # expand the biggest interior kid while it fits
            cand = [k for k in kids if k[0] == 'int' or (k[0] == 'leaf' and len(k[1]) > 12)]
            if not cand:
                break
            k = max(cand, key=self.count)
            if k[0] == 'leaf':
                s = sorted(k[1], key=lambda t: t.c[max(range(3), key=lambda a: union([x.box for x in k[1]])[1][a] - union([x.box for x in k[1]])[0][a])])
                parts = [('leaf', s[:len(s) // 2]), ('leaf', s[len(s) // 2:])]
            else:
                parts = k[1]
            if len(kids) - 1 + len(parts) > 12:
                break
            kids.remove(k)
            kids += parts
        return kids

    def tri_list(self, n):
        if n[0] == 'leaf':
            return list(n[1])
        return [t for c in n[1] for t in self.tri_list(c)]

    def box_of(self, n):
        return union([t.box for t in self.tri_list(n)])

    def build(self):
        root = self.split(self.tris, 12)
        # layout: nodes list in BFS order; blocks appended later
        order = []             # (nodeinfo)
        # collapse: a hardware node is either leaf-type (<=12 tris) or interior (<=12 hw children)
        def mk(n):
            if self.count(n) <= 12:
                return {'t': 'L', 'tris': self.tri_list(n), 'box': self.box_of(n)}
            kids = self.hw_children(n)
            return {'t': 'I', 'kids': [mk(k) for k in kids], 'box': self.box_of(n)}
        top = mk(root)
        # BFS placement: root at 0, then each node's children contiguous
        q = [top]
        top['idx'] = 0
        nodes = [top]
        head = 0
        while head < len(nodes):
            n = nodes[head]; head += 1
            if n['t'] == 'I':
                n['first'] = len(nodes)
                for si, k in enumerate(n['kids']):
                    k['idx'] = len(nodes); k['sib'] = si; k['par'] = n['idx']; nodes.append(k)
        # blocks, assigned to leaf nodes in node order
        nid = 0
        blocks = []
        for n in nodes:
            if n['t'] == 'L':
                n['blocks'] = []
                for grp in make_blocks(n['tris']):
                    for j, t in enumerate(grp):
                        t.gid = 4 * len(blocks) + j
                    n['blocks'].append((len(blocks), grp))
                    blocks.append(grp)
        nn = len(nodes)
        root_off = 0x380
        blk_base = root_off + nn * NODE
        out = bytearray(self.hdr) + bytearray(nn * NODE) + bytearray(len(blocks) * 128)
        for n in nodes:
            off = root_off + n['idx'] * NODE
            self.write_node(out, off, n, blk_base, nodes, root_off)
        for bi, grp in enumerate(blocks):
            blk = C.encode_block([t.v for t in grp], 4 * bi)
            blk = bytearray(blk)
            # ids: b0 = p, b_j = (p & ~3) + j  (offset fields)
            s = 126 - 14 * (len(grp) - 1)
            idbits = int.from_bytes(blk[0x68:0x78], 'little')
            for j in range(1, len(grp)):
                idbits |= (j & 3) << (s - 2 * j)
            blk[0x68:0x78] = idbits.to_bytes(16, 'little')
            d = struct.unpack_from('<I', blk, 0x7c)[0] | 0x8000
            struct.pack_into('<I', blk, 0x7c, d)
            o = blk_base + bi * 128
            out[o:o + 128] = blk
        struct.pack_into('<I', out, 0x20, root_off)
        self.nodes = nodes; self.nblocks = len(blocks)
        self.gid_to_tri = {}
        for bi, grp in enumerate(blocks):
            for j, t in enumerate(grp):
                self.gid_to_tri[4 * bi + j] = t.id
        return bytes(out)

    def write_node(self, out, off, n, blk_base, nodes, root_off):
        slots = []     # (box, code)
        if n['t'] == 'L':
            # one slot per triangle, groups = blocks; terminal slot of a group gets the suffix box
            groups = [g for _, g in n['blocks']]
            suffix = []
            acc = None
            allboxes = [[t.box for t in g] for g in groups]
            for gi in range(len(groups)):
                later = [b for gg in allboxes[gi + 1:] for b in gg]
                for j, t in enumerate(groups[gi]):
                    last = j == len(groups[gi]) - 1
                    if last:
                        box = t.box if TIGHT else union([t.box] + later)
                        code = 0x01
                    else:
                        box = t.box
                        code = (j + 1) << 3
                    slots.append((box, code))
            first_blk = n['blocks'][0][0] if n['blocks'] else 0
            w10 = blk_base + first_blk * 128 - off
            w1c = 1
        else:
            for k in n['kids']:
                slots.append((k['box'], 0x00))
            w10 = 0
            w1c = (root_off + n['first'] * NODE - off) // NODE
        assert len(slots) <= 12
        e, org = quantise_node([s[0] for s in slots] if slots else [n['box']])
        out[off] = 0x60 if n['t'] == 'L' else 0x40
        out[off + 1:off + 4] = bytes(e)
        struct.pack_into('<3f', out, off + 4, *org)
        struct.pack_into('<I', out, off + 0x10, w10)
        w14, w18 = sib_words(n, off == root_off)
        struct.pack_into('<I', out, off + 0x14, w14)
        struct.pack_into('<I', out, off + 0x18, w18)
        struct.pack_into('<I', out, off + 0x1c, w1c)
        for k in range(12):
            if k < len(slots):
                q = qbox(e, org, slots[k][0])
                out[off + 0x20 + 8 * k:off + 0x28 + 8 * k] = bytes(q + [0xff, slots[k][1]])
            else:
                out[off + 0x20 + 8 * k:off + 0x28 + 8 * k] = bytes([0, 0, 0, 0, 0xff, 0, 0, 0])
