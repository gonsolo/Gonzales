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
alias ViewerHandle = UnsafePointer[UInt8, MutAnyOrigin]

fn viewer_create(width: Int32, height: Int32,
                 title: UnsafePointer[UInt8, MutAnyOrigin]) -> ViewerHandle:
    return external_call["viewer_create", ViewerHandle,
        Int32, Int32, UnsafePointer[UInt8, MutAnyOrigin]](width, height, title)

fn viewer_update_framebuffer(v: ViewerHandle,
                              pixels: UnsafePointer[Float32, MutAnyOrigin],
                              width: Int32, height: Int32):
    external_call["viewer_update_framebuffer", NoneType,
        ViewerHandle, UnsafePointer[Float32, MutAnyOrigin], Int32, Int32](
        v, pixels, width, height)

fn viewer_should_close(v: ViewerHandle) -> Int32:
    return external_call["viewer_should_close", Int32, ViewerHandle](v)

fn viewer_poll_events(v: ViewerHandle):
    external_call["viewer_poll_events", NoneType, ViewerHandle](v)

# CameraState is 40 bytes — too large for register return on x86-64.
# The C API uses output pointers; we wrap them here for ergonomics.
fn viewer_get_camera_state(v: ViewerHandle, result: UnsafePointer[CameraState, MutAnyOrigin]):
    external_call["viewer_get_camera_state", NoneType, ViewerHandle, UnsafePointer[CameraState, MutAnyOrigin]](v, result)

fn viewer_set_camera_state(v: ViewerHandle, state: UnsafePointer[CameraState, MutAnyOrigin]):
    external_call["viewer_set_camera_state", NoneType, ViewerHandle, UnsafePointer[CameraState, MutAnyOrigin]](v, state)

fn viewer_destroy(v: ViewerHandle):
    external_call["viewer_destroy", NoneType, ViewerHandle](v)

# Build a column-major camera-to-world matrix from position/direction/up.
fn build_camera_to_world(cs: UnsafePointer[CameraState, MutAnyOrigin], c2w: UnsafePointer[Float32, MutAnyOrigin]):
    var dx = cs[0].direction.x; var dy = cs[0].direction.y; var dz = cs[0].direction.z
    var ux = cs[0].up.x;        var uy = cs[0].up.y;        var uz = cs[0].up.z

    # right = normalize(cross(up, dir))
    var rx = uy * dz - uz * dy
    var ry = uz * dx - ux * dz
    var rz = ux * dy - uy * dx
    var rlen = sqrt(rx * rx + ry * ry + rz * rz)
    if rlen < Float32(1e-6): rlen = Float32(1)
    rx /= rlen; ry /= rlen; rz /= rlen

    # true_up = cross(dir, right)  — reorthogonalize
    var tux = dy * rz - dz * ry
    var tuy = dz * rx - dx * rz
    var tuz = dx * ry - dy * rx

    # Column 0: right
    c2w[0] = rx;  c2w[1] = ry;  c2w[2] = rz;  c2w[3] = Float32(0)
    # Column 1: true_up
    c2w[4] = tux; c2w[5] = tuy; c2w[6] = tuz; c2w[7] = Float32(0)
    # Column 2: dir (forward = camera +Z)
    c2w[8] = dx;  c2w[9] = dy;  c2w[10] = dz; c2w[11] = Float32(0)
    # Column 3: position
    c2w[12] = cs[0].position.x; c2w[13] = cs[0].position.y
    c2w[14] = cs[0].position.z; c2w[15] = Float32(1)
