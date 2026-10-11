from std.memory.alloc import unsafe_alloc
from std.math import floor, fma
from .geometry import Point3f
from .primitives import mesh_points_floats
from .bvh import BVH2Node


@no_inline
def quantise_quads(
    n_pts: Pointer[Float32, MutUntrackedOrigin], n_dq: Pointer[Float32, MutUntrackedOrigin], nq: Int,
    q_nodes: Pointer[BVH2Node, MutUntrackedOrigin], q_node_count: Int,
):
    """Pack the positions n_dq (four floats a vertex, leaf order) into n_pts as 16-bit steps in blocks of 256
    vertices, each with its own corner and step (primitives.mesh_pos), and refit the BVH to the values the
    renderer reads back. n_dq ends up holding those values."""
    var n16 = n_pts.unsafe_bitcast[UInt16]()
    var nvq = nq * 4
    for z in range(mesh_points_floats(nvq, Int32(2)) * 2): n16[unsafe_offset=z] = UInt16(0)
    for blk_i in range((nvq + 255) >> 8):
        var v0 = blk_i * 256; var v1 = min(v0 + 256, nvq)
        for a in range(3):
            var lo = Float32(1e38); var hi = Float32(-1e38)
            for v in range(v0, v1):
                lo = min(lo, n_dq[unsafe_offset=v * 4 + a]); hi = max(hi, n_dq[unsafe_offset=v * 4 + a])
            var step = max((hi - lo) / Float32(65535), Float32(1e-30))
            n_pts[unsafe_offset=blk_i * 390 + a] = lo; n_pts[unsafe_offset=blk_i * 390 + 3 + a] = step
            for v in range(v0, v1):
                var qi = floor((n_dq[unsafe_offset=v * 4 + a] - lo) / step + Float32(0.5))
                qi = min(max(qi, Float32(0)), Float32(65535))
                n16[unsafe_offset=blk_i * 780 + 12 + (v - v0) * 3 + a] = UInt16(Int(qi))
                n_dq[unsafe_offset=v * 4 + a] = fma(qi, step, lo)
    refit_quad_bvh(n_dq, q_nodes, q_node_count)
    n_dq.unsafe_free()


@no_inline
def refit_quad_bvh(
    n_dq: Pointer[Float32, MutUntrackedOrigin], q_nodes: Pointer[BVH2Node, MutUntrackedOrigin], q_node_count: Int,
):
    """Refit the nodes to the positions n_dq (four floats a vertex, four vertices a quad, leaf order): children
    always sit after their parent, so one pass from the last node to the first is bottom-up."""
    for back in range(q_node_count):
        var ni = q_node_count - 1 - back
        var nd = q_nodes[unsafe_offset=ni]
        var lx = Float32(1e38); var ly = Float32(1e38); var lz = Float32(1e38)
        var hx = Float32(-1e38); var hy = Float32(-1e38); var hz = Float32(-1e38)
        if nd.count > Int32(0):
            var first = Int(nd.offset) * 4
            var last = (Int(nd.offset) + Int(nd.count)) * 4
            for v in range(first, last):
                lx = min(lx, n_dq[unsafe_offset=v * 4]); hx = max(hx, n_dq[unsafe_offset=v * 4])
                ly = min(ly, n_dq[unsafe_offset=v * 4 + 1]); hy = max(hy, n_dq[unsafe_offset=v * 4 + 1])
                lz = min(lz, n_dq[unsafe_offset=v * 4 + 2]); hz = max(hz, n_dq[unsafe_offset=v * 4 + 2])
        else:
            var lc = q_nodes[unsafe_offset=ni + 1]
            var rc = q_nodes[unsafe_offset=Int(nd.offset)]
            lx = min(lc.min.x, rc.min.x); ly = min(lc.min.y, rc.min.y); lz = min(lc.min.z, rc.min.z)
            hx = max(lc.max.x, rc.max.x); hy = max(lc.max.y, rc.max.y); hz = max(lc.max.z, rc.max.z)
        q_nodes[unsafe_offset=ni] = BVH2Node(Point3f(lx, ly, lz), Point3f(hx, hy, hz), nd.offset, nd.count)
