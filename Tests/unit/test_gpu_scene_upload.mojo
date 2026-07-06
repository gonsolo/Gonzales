from std.math import abs
from std.memory import alloc
from std.sys.info import size_of
from std.sys import has_accelerator
from std.testing import assert_true, TestSuite
from gonzales.lexer import PbrtScanner
from gonzales.parse_types import SceneParseState
from gonzales.pbrt_parser import parse_scene_file, finalize_scene, mojo_parsed_free, ParsedScene_Mojo
from gonzales.pipeline import _gpu_upload_scene
from gonzales.gpu import GpuSceneHandle
from gonzales.geometry import Material_C, MatKind
from gonzales.bvh import BVH2Node

# ── Full SceneDescriptor2_C + GPU scene-upload fixture (task #58) ──────────
# gpu_upload_scene (gpu.mojo) had zero automated coverage before this: every
# one of its ~55 parameters and dozen-plus device-buffer uploads was only
# ever exercised indirectly by manually rendering a real scene with --gpu.
# This drives a real scene through the actual parser (parse_scene_file +
# finalize_scene, the same functions mojo_parse_scene calls) — not a hand-
# faked ParsedScene_Mojo — then uploads it via the same _gpu_upload_scene
# helper pipeline.mojo's real render path uses, and reads several of the
# resulting device buffers back to host to confirm the copies are correct,
# not just crash-free.

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

def _scanner_from_string(s: String) -> UnsafePointer[PbrtScanner, MutAnyOrigin]:
    """Same technique as test_parser_integration.mojo's helper, but here fed
    a full scene (from the first top-level directive) since parse_scene_file
    -- unlike the single-directive handlers that file tests -- expects to
    read directive keywords itself from the very start of the buffer."""
    var n = s.byte_length()
    var buf = alloc[UInt8](n + 1)
    for i in range(n):
        buf[i] = s.as_bytes()[i]
    buf[n] = UInt8(0)
    var handle = alloc[PbrtScanner](1)
    handle[0].buffer = buf
    handle[0].total_bytes = Int32(n)
    handle[0].cursor = Int32(0)
    handle[0].is_at_end = Int32(0)
    return handle

def _parse_minimal_scene() -> UnsafePointer[ParsedScene_Mojo, MutAnyOrigin]:
    """One triangle mesh + one diffuse material, 4x4 pixels -- just enough
    for finalize_scene to produce a real BVH, camera matrices, and a
    non-trivial mesh/material upload, without needing any lights."""
    var scene_text = String('Film "rgb" "string filename" ["test_gpu_upload.exr"] ')
    scene_text += String('"integer xresolution" [4] "integer yresolution" [4]\n')
    scene_text += String("LookAt 0 2 6  0 1 0  0 1 0\n")
    scene_text += String('Camera "perspective" "float fov" [45]\n')
    scene_text += String("WorldBegin\n")
    scene_text += String('MakeNamedMaterial "M" "string type" ["diffuse"] "rgb reflectance" [0.5 0.25 0.75]\n')
    scene_text += String('NamedMaterial "M"\n')
    scene_text += String('Shape "trianglemesh" "point3 P" [0 0 0  1 0 0  0 1 0] "integer indices" [0 1 2]\n')
    scene_text += String("WorldEnd\n")
    var handle = _scanner_from_string(scene_text)
    var s_ptr = alloc[SceneParseState](1)
    s_ptr.init_pointee_move(SceneParseState())
    parse_scene_file(handle, s_ptr)

    var psc = alloc[ParsedScene_Mojo](1)
    finalize_scene(s_ptr, psc, False)
    _ = s_ptr.take_pointee(); s_ptr.free()
    handle[0].buffer.free(); handle.free()
    return psc

def test_gpu_upload_scene_round_trips_mesh_and_material_data() raises:
    comptime if not has_accelerator():
        print("SKIP: no GPU accelerator on this machine")
        return

    var psc = _parse_minimal_scene()
    assert_true(Int(psc[0].mesh_count) == 1)
    assert_true(Int(psc[0].material_count) == 1)

    comptime N_SOBOL_GPU_WORDS = 1024 * 52
    var sobol = alloc[UInt32](N_SOBOL_GPU_WORDS)
    for i in range(N_SOBOL_GPU_WORDS):
        sobol[i] = UInt32(i * 7 + 3)

    var n_pixels = Int(psc[0].film_w) * Int(psc[0].film_h)
    var handle = _gpu_upload_scene(psc, sobol, n_pixels)
    assert_true(Int(handle) != 0)

    assert_true(handle[].mesh_count == 1)
    assert_true(handle[].material_count == 1)
    assert_true(handle[].n_pixels == n_pixels)
    assert_true(handle[].fw == Int(psc[0].film_w))
    assert_true(handle[].fh == Int(psc[0].film_h))

    # Material bytes round-trip: read back the uploaded Material_C and
    # compare against what finalize_scene actually built on the CPU side --
    # not the literal scene text, since e.g. checker/mix fields might differ.
    var cpu_mat = psc[0].materials[0]
    with handle[].materials_buf.map_to_host() as h:
        var gpu_mat = h.unsafe_ptr().bitcast[Material_C]()[0]
        assert_true(gpu_mat.type == cpu_mat.type)
        assert_true(_close(gpu_mat.albedo.r, cpu_mat.albedo.r))
        assert_true(_close(gpu_mat.albedo.g, cpu_mat.albedo.g))
        assert_true(_close(gpu_mat.albedo.b, cpu_mat.albedo.b))

    # Mesh point data round-trip (stride-4 floats per vertex: x,y,z,w=1).
    var n_floats = Int(psc[0].mesh_n_verts[0]) * 4
    with handle[].points_bufs[0].map_to_host() as h:
        var gpu_pts = h.unsafe_ptr().bitcast[Float32]()
        var cpu_pts = psc[0].meshes[0].points
        for i in range(n_floats):
            assert_true(_close(gpu_pts[i], cpu_pts[i]))

    # BVH node data round-trips byte-for-byte (same size_of[BVH2Node]()
    # struct copied verbatim in gpu_upload_scene -- a real, non-trivial
    # tree since finalize_scene built it via the same build_bvh2 the CPU
    # renderer uses).
    var n_bvh_bytes = Int(psc[0].bvh_node_count_cpu) * size_of[BVH2Node]()
    assert_true(n_bvh_bytes > 0)
    with handle[].bvh2Nodes_buf.map_to_host() as h:
        var gpu_bytes = h.unsafe_ptr()
        var cpu_bytes = psc[0].bvh_nodes_cpu.bitcast[UInt8]()
        var mismatch = False
        for i in range(n_bvh_bytes):
            if gpu_bytes[i] != cpu_bytes[i]:
                mismatch = True
                break
        assert_true(not mismatch)

    # Camera matrices and the Sobol table round-trip too.
    with handle[].r2c_buf.map_to_host() as h:
        var gpu_r2c = h.unsafe_ptr().bitcast[Float32]()
        for i in range(16):
            assert_true(_close(gpu_r2c[i], psc[0].raster_to_camera[i]))
    with handle[].sobol_buf.map_to_host() as h:
        var gpu_sobol = h.unsafe_ptr().bitcast[UInt32]()
        assert_true(gpu_sobol[0] == UInt32(3))
        assert_true(gpu_sobol[1] == UInt32(10))
        assert_true(gpu_sobol[N_SOBOL_GPU_WORDS - 1] == sobol[N_SOBOL_GPU_WORDS - 1])

    sobol.free()
    mojo_parsed_free(psc)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
