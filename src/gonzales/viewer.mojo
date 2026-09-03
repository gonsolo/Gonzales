from std.ffi import external_call
from std.memory import alloc
from std.math import sqrt
from .geometry import Point3f, Vec3f

# Mirror of C struct CameraState from viewer.h (40 bytes, pointer-passed).
# Layout: Point3f position (12) + Vec3f direction (12) + Vec3f up (12) + Int32 cameraChanged (4) = 40 B
@fieldwise_init
struct CameraState(TrivialRegisterPassable):
    var position: Point3f
    var direction: Vec3f
    var up: Vec3f
    var cameraChanged: Int32

# Opaque C pointer to the Viewer object.  Treat as UInt8* to stay away from
# the !kgen.pointer<none> representation that Mojo 1.0 rejects.
comptime ViewerHandle = UnsafePointer[UInt8, MutExternalOrigin]

def viewer_create[Ot: Origin[mut=True]](width: Int32, height: Int32,
                 title: UnsafePointer[UInt8, Ot],
                 fullscreen: Int32) -> ViewerHandle:
    return external_call["viewer_create", ViewerHandle,
        Int32, Int32, UnsafePointer[UInt8, MutExternalOrigin], Int32](width, height, title.unsafe_origin_cast[MutExternalOrigin](), fullscreen)

def viewer_update_framebuffer[Opx: Origin[mut=True]](v: ViewerHandle,
                              pixels: UnsafePointer[Float32, Opx],
                              width: Int32, height: Int32):
    external_call["viewer_update_framebuffer", NoneType,
        ViewerHandle, UnsafePointer[Float32, MutExternalOrigin], Int32, Int32](
        v, pixels.unsafe_origin_cast[MutExternalOrigin](), width, height)

def viewer_should_close(v: ViewerHandle) -> Int32:
    return external_call["viewer_should_close", Int32, ViewerHandle](v)

def viewer_poll_events(v: ViewerHandle):
    external_call["viewer_poll_events", NoneType, ViewerHandle](v)

# CameraState is 40 bytes — too large for register return on x86-64.
# The C API uses output pointers; we wrap them here for ergonomics.
def viewer_get_camera_state[Or: Origin[mut=True]](v: ViewerHandle, result: UnsafePointer[CameraState, Or]):
    external_call["viewer_get_camera_state", NoneType, ViewerHandle, UnsafePointer[CameraState, MutExternalOrigin]](v, result.unsafe_origin_cast[MutExternalOrigin]())

def viewer_set_camera_state[Os: Origin[mut=True]](v: ViewerHandle, state: UnsafePointer[CameraState, Os]):
    external_call["viewer_set_camera_state", NoneType, ViewerHandle, UnsafePointer[CameraState, MutExternalOrigin]](v, state.unsafe_origin_cast[MutExternalOrigin]())

def viewer_destroy(v: ViewerHandle):
    external_call["viewer_destroy", NoneType, ViewerHandle](v)

# Build a column-major camera-to-world matrix from position/direction/up.
def build_camera_to_world[Ocs: Origin[mut=True], Oc2w: Origin[mut=True]](cs: UnsafePointer[CameraState, Ocs], c2w: UnsafePointer[Float32, Oc2w]):
    var dx = cs[0].direction.x; var dy = cs[0].direction.y; var dz = cs[0].direction.z
    var ux = cs[0].up.x;        var uy = cs[0].up.y;        var uz = cs[0].up.z

    # right = normalize(cross(dir, up))
    var rx = dy * uz - dz * uy
    var ry = dz * ux - dx * uz
    var rz = dx * uy - dy * ux
    var rlen = sqrt(rx * rx + ry * ry + rz * rz)
    if rlen < Float32(1e-6): rlen = Float32(1)
    rx /= rlen; ry /= rlen; rz /= rlen

    # true_up = cross(right, dir)  — reorthogonalize
    var tux = ry * dz - rz * dy
    var tuy = rz * dx - rx * dz
    var tuz = rx * dy - ry * dx

    # Column 0: right
    c2w[0] = rx;  c2w[1] = ry;  c2w[2] = rz;  c2w[3] = Float32(0)
    # Column 1: true_up
    c2w[4] = tux; c2w[5] = tuy; c2w[6] = tuz; c2w[7] = Float32(0)
    # Column 2: dir (forward = camera +Z)
    c2w[8] = dx;  c2w[9] = dy;  c2w[10] = dz; c2w[11] = Float32(0)
    # Column 3: position
    c2w[12] = cs[0].position.x; c2w[13] = cs[0].position.y
    c2w[14] = cs[0].position.z; c2w[15] = Float32(1)
