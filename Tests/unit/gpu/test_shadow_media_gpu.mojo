"""Media shadow rays through an instanced displaced quad: the GPU walk must match the CPU's ray for ray."""
from std.math import abs, sqrt
from std.memory.alloc import unsafe_alloc
from std.os import getenv
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from gonzales.core.geometry import Vec3f
from gonzales.core.rng import PCG32
from gonzales.scene.description.pbrt_parser import mojo_parse_scene, mojo_parsed_scene_descriptor
from gonzales.gpu.scene import gpu_upload_scene
from gonzales.gpu.shadow_media import shadow_media_ray, shadow_media_batch_gpu
from gonzales.core.spectrum import load_spectral_context, spectral_handle

comptime N = 16384
comptime LIGHT = Vec3f(3, -4, 4)


def test_gpu_matches_cpu() raises:
    comptime if not has_accelerator():
        print("SKIP: no GPU accelerator on this machine")
        return
    var path = String("Tests/unit/gpu/shadow_media_scene.pbrt")
    var cpath = unsafe_alloc[UInt8](path.byte_length() + 1)
    for k in range(path.byte_length()):
        cpath[unsafe_offset=k] = path.as_bytes()[k]
    cpath[unsafe_offset=path.byte_length()] = UInt8(0)
    var psc = mojo_parse_scene(cpath)
    cpath.unsafe_free()
    var sp = load_spectral_context(getenv("GONZALES_DATA_DIR", "src/gonzales/data"))
    assert_true(sp[0])
    var spc = sp[1].copy()
    var h = spectral_handle(spc)
    var sd_cpu = mojo_parsed_scene_descriptor(psc, h)
    comptime N_SOBOL_GPU_WORDS = 1024 * 52
    var sobol = unsafe_alloc[UInt32](N_SOBOL_GPU_WORDS)
    for i in range(N_SOBOL_GPU_WORDS):
        sobol[unsafe_offset=i] = UInt32(0)
    var handle = gpu_upload_scene(psc, sobol, 16, h.coeffs, h.res, h.cie_x, h.cie_y, h.cie_z, h.d65, verbose=False)
    var sd_gpu = handle[].scene_descriptor()
    # Points anywhere in the room, towards random points inside the light.
    var rng = PCG32(7, 1)
    var rays = unsafe_alloc[Float32](7 * N)
    for k in range(N):
        var o = Vec3f(rng.next_float() * 4 - 2, rng.next_float() * 4 - 2, rng.next_float() * 2.5)
        var u = Vec3f(rng.next_float() * 2 - 1, rng.next_float() * 2 - 1, rng.next_float() * 2 - 1).normalize()
        var to = LIGHT + u * Float32(0.6) - o
        var dist = sqrt(to.x * to.x + to.y * to.y + to.z * to.z)
        var d = to * (Float32(1) / dist)
        rays[unsafe_offset=7 * k] = o.x; rays[unsafe_offset=7 * k + 1] = o.y; rays[unsafe_offset=7 * k + 2] = o.z
        rays[unsafe_offset=7 * k + 3] = d.x; rays[unsafe_offset=7 * k + 4] = d.y; rays[unsafe_offset=7 * k + 5] = d.z
        var oc = o - LIGHT   # stop short of the light's front surface
        var b = oc.x * d.x + oc.y * d.y + oc.z * d.z
        var c = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z - Float32(0.49)
        rays[unsafe_offset=7 * k + 6] = (-b - sqrt(max(b * b - c, Float32(0)))) * Float32(0.999)
    var ctx = handle[].ctx
    var rd = ctx.enqueue_create_buffer[DType.float32](7 * N)
    ctx.enqueue_copy(rd, rays)
    var out = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.enqueue_function[shadow_media_batch_gpu](sd_gpu, rd.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                                           out.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), Int64(N),
                                           grid_dim=(N + 63) // 64, block_dim=64)
    ctx.synchronize()
    var lit = 0; var bad = 0
    with out.map_to_host() as host:
        var g = host.unsafe_ptr()
        for k in range(N):
            var c = shadow_media_ray(sd_cpu[], rays, k)
            if c > Float32(0): lit += 1
            if abs(g[unsafe_offset=k] - c) > Float32(1e-3):
                if bad < 8:
                    print("ray", k, "cpu", c, "gpu", g[unsafe_offset=k])
                bad += 1
    print("media shadows lit", lit, "of", N, "mismatches", bad)
    assert_true(lit > 100 and lit < N - 100 and bad == 0)
    rays.unsafe_free(); sobol.unsafe_free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
