"""Helpers for the acceleration structure experiments (see ../ACCEL_FORMAT.md). Run from a directory holding the
built tools as_dump and re_run."""
import struct, subprocess
ENV = {'LD_LIBRARY_PATH': '/home/gonsolo/work/gonzales/build:/opt/cuda/lib64'}
CUBIN = '/home/gonsolo/work/gonzales/build/rt_trace.cubin'

def build(tri, name='tmp'):
    """BLAS bytes the driver builds for triangles given as a flat float list (9 per triangle); also writes name.b(.addr)."""
    open(name + '.t', 'wb').write(struct.pack('%df' % len(tri), *tri))
    subprocess.run(['./as_dump', name + '.t', name + '.b'], capture_output=True, env=ENV)
    return open(name + '.b', 'rb').read()

def rays_file(path, rays):
    """rays: list of (ox,oy,oz,dx,dy,dz); writes the 32-byte records, padded to a multiple of 64 rays."""
    flat = []
    for r in rays:
        flat += [r[0], r[1], r[2], 0.0, r[3], r[4], r[5], 100.0]
    n = len(flat) // 8
    flat += flat[-8:] * (((n + 63) // 64) * 64 - n)
    open(path, 'wb').write(struct.pack('%df' % len(flat), *flat))
    return n

def trace(blob_name, rays_path, n, patches=()):
    """Trace against name.b with patches [(hexoffset, hexbytes)]; returns [(raw, t, u, v)] per ray, or None on a fault."""
    args = ['./re_run', CUBIN, blob_name, rays_path, 'rr.out'] + ['%x=%s' % (o, b) for o, b in patches]
    r = subprocess.run(args, capture_output=True, text=True, env=ENV)
    if 'ok' not in r.stdout:
        return None
    d = open('rr.out', 'rb').read()
    out = []
    for i in range(n):
        t, u, v = struct.unpack_from('<fff', d, 32 * i)
        raw = struct.unpack_from('<I', d, 32 * i + 12)[0]
        out.append((raw, t, u, v))
    return out

def words(d, o, n):
    return [struct.unpack_from('<I', d, o + 4 * i)[0] for i in range(n)]
