"""The learned denoiser: our U-Net in Mojo on the GPU (default), or the same weights through the OIDN library (CPU
reference). Both need the trained weights (.tza); without them the a-trous result stays."""
from std.ffi import OwnedDLHandle
from std.os.path import exists
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from gonzales.core.env import env_str
from gonzales.gpu.unet import UNetWeights, unet_denoise


def denoiser_weights() -> String:
    return env_str["GONZALES_DENOISER_WEIGHTS"](env_str["GONZALES_DATA_DIR"]("src/gonzales/data") + "/denoiser.tza")


def oidn_denoise(
    path: String, color: Pointer[Float32, _], albedo: Pointer[Float32, _], normal: Pointer[Float32, _],
    res: Pointer[mut=True, Float32, _], w: Int, h: Int,
) raises:
    var lib = OwnedDLHandle("libOpenImageDenoise.so.2")
    var raw = open(path, "r").read_bytes()
    var dev = lib.call["oidnNewDevice", Int](Int32(1))      # OIDN_DEVICE_TYPE_CPU
    lib.call["oidnCommitDevice"](dev)
    var f = lib.call["oidnNewFilter", Int](dev, "RT\0".unsafe_ptr())
    comptime FLOAT3 = Int32(3)
    lib.call["oidnSetSharedFilterImage"](f, "color\0".unsafe_ptr(), color, FLOAT3, w, h, 0, 0, 0)
    lib.call["oidnSetSharedFilterImage"](f, "albedo\0".unsafe_ptr(), albedo, FLOAT3, w, h, 0, 0, 0)
    lib.call["oidnSetSharedFilterImage"](f, "normal\0".unsafe_ptr(), normal, FLOAT3, w, h, 0, 0, 0)
    lib.call["oidnSetSharedFilterImage"](f, "output\0".unsafe_ptr(), res, FLOAT3, w, h, 0, 0, 0)
    lib.call["oidnSetFilterBool"](f, "hdr\0".unsafe_ptr(), True)
    lib.call["oidnSetFilterBool"](f, "cleanAux\0".unsafe_ptr(), True)
    lib.call["oidnSetSharedFilterData"](f, "weights\0".unsafe_ptr(), raw.unsafe_ptr(), len(raw))
    lib.call["oidnCommitFilter"](f)
    lib.call["oidnExecuteFilter"](f)
    var msg = 0     # const char*, unread
    var err = lib.call["oidnGetDeviceError", Int32](dev, Pointer(to=msg))
    lib.call["oidnReleaseFilter"](f)
    lib.call["oidnReleaseDevice"](dev)
    if err != 0:
        raise Error("OIDN error " + String(err))


def learned_denoise(
    image: Pointer[mut=True, Float32, _], noisy: Pointer[Float32, _], albedo: Pointer[Float32, _],
    normals: Pointer[Float32, _], w: Int, h: Int,
) -> Bool:
    """Overwrites the a-trous `image` with the network's denoising of `noisy`; False if it kept a-trous."""
    var kind = env_str["GONZALES_DENOISER"]("unet")
    if kind == "atrous":
        return False
    var path = denoiser_weights()
    if not exists(path):
        print("Denoiser: no weights at " + path + ", a-trous instead")
        return False
    var t0 = perf_counter_ns()
    try:
        if kind == "oidn":
            oidn_denoise(path, noisy, albedo, normals, image, w, h)
        else:
            unet_denoise(DeviceContext(), UNetWeights(path), noisy, albedo, normals, image, w, h)
    except e:
        print("Denoiser: " + kind + " failed (" + String(e) + "), a-trous instead")
        return False
    print("Denoiser:", kind, Float64(Int((perf_counter_ns() - t0) // 10_000_000)) / 100.0, "s")
    return True
