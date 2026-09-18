from std.ffi import external_call
from std.memory.alloc import unsafe_alloc
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
comptime ViewerHandle = Pointer[UInt8, MutUntrackedOrigin]

def viewer_create[Ot: Origin[mut=True]](width: Int32, height: Int32,
                 title: Pointer[UInt8, Ot],
                 fullscreen: Int32) -> ViewerHandle:
    return external_call["viewer_create", ViewerHandle,
        Int32, Int32, Pointer[UInt8, MutUntrackedOrigin], Int32](width, height, title.unsafe_origin_cast[MutUntrackedOrigin](), fullscreen)

def viewer_update_framebuffer[Opx: Origin[mut=True]](v: ViewerHandle,
                              pixels: Pointer[Float32, Opx],
                              width: Int32, height: Int32):
    external_call["viewer_update_framebuffer", NoneType,
        ViewerHandle, Pointer[Float32, MutUntrackedOrigin], Int32, Int32](
        v, pixels.unsafe_origin_cast[MutUntrackedOrigin](), width, height)

def viewer_should_close(v: ViewerHandle) -> Int32:
    return external_call["viewer_should_close", Int32, ViewerHandle](v)

def viewer_poll_events(v: ViewerHandle):
    external_call["viewer_poll_events", NoneType, ViewerHandle](v)

# CameraState is 40 bytes — too large for register return on x86-64.
# The C API uses output pointers; we wrap them here for ergonomics.
def viewer_get_camera_state[Or: Origin[mut=True]](v: ViewerHandle, result: Pointer[CameraState, Or]):
    external_call["viewer_get_camera_state", NoneType, ViewerHandle, Pointer[CameraState, MutUntrackedOrigin]](v, result.unsafe_origin_cast[MutUntrackedOrigin]())

def viewer_set_camera_state[Os: Origin[mut=True]](v: ViewerHandle, state: Pointer[CameraState, Os]):
    external_call["viewer_set_camera_state", NoneType, ViewerHandle, Pointer[CameraState, MutUntrackedOrigin]](v, state.unsafe_origin_cast[MutUntrackedOrigin]())

def viewer_destroy(v: ViewerHandle):
    external_call["viewer_destroy", NoneType, ViewerHandle](v)

# Build a column-major camera-to-world matrix from position/direction/up.
def build_camera_to_world[Ocs: Origin[mut=True], Oc2w: Origin[mut=True]](cs: Pointer[CameraState, Ocs], c2w: Pointer[Float32, Oc2w]):
    var dx = cs[unsafe_offset=0].direction.x; var dy = cs[unsafe_offset=0].direction.y; var dz = cs[unsafe_offset=0].direction.z
    var ux = cs[unsafe_offset=0].up.x;        var uy = cs[unsafe_offset=0].up.y;        var uz = cs[unsafe_offset=0].up.z

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
    c2w[unsafe_offset=0] = rx;  c2w[unsafe_offset=1] = ry;  c2w[unsafe_offset=2] = rz;  c2w[unsafe_offset=3] = Float32(0)
    # Column 1: true_up
    c2w[unsafe_offset=4] = tux; c2w[unsafe_offset=5] = tuy; c2w[unsafe_offset=6] = tuz; c2w[unsafe_offset=7] = Float32(0)
    # Column 2: dir (forward = camera +Z)
    c2w[unsafe_offset=8] = dx;  c2w[unsafe_offset=9] = dy;  c2w[unsafe_offset=10] = dz; c2w[unsafe_offset=11] = Float32(0)
    # Column 3: position
    c2w[unsafe_offset=12] = cs[unsafe_offset=0].position.x; c2w[unsafe_offset=13] = cs[unsafe_offset=0].position.y
    c2w[unsafe_offset=14] = cs[unsafe_offset=0].position.z; c2w[unsafe_offset=15] = Float32(1)
