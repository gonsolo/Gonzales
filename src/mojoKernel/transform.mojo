from std.ffi import external_call
from std.memory import alloc

# Matrix math. 4x4 matrices are 16 Float32 in column-major order:
# flat[col*4 + row] = matrix[row, col], matching Transform.columnMajorFloats().

def _write_identity(result: UnsafePointer[Float32, MutAnyOrigin]) -> Int32:
    for i in range(16):
        result[i] = Float32(0)
    result[0] = Float32(1)
    result[5] = Float32(1)
    result[10] = Float32(1)
    result[15] = Float32(1)
    return Int32(0)


def matrix_multiply(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    result: UnsafePointer[Float32, MutAnyOrigin],
):
    # result[i, j] = sum_k a[i, k] * b[k, j]   (column-major)
    for j in range(4):
        for i in range(4):
            var s = Float32(0)
            for k in range(4):
                s += a[k * 4 + i] * b[j * 4 + k]
            result[j * 4 + i] = s


def matrix_invert(
    m: UnsafePointer[Float32, MutAnyOrigin],
    result: UnsafePointer[Float32, MutAnyOrigin],
) -> Int32:
    # Gauss-Jordan elimination with full pivoting (mirrors Matrix.invert).
    # On a singular matrix, writes the identity and returns 0.
    var minv = InlineArray[Float32, 16](fill=0)
    for i in range(16):
        minv[i] = m[i]
    var indxc = InlineArray[Int, 4](fill=0)
    var indxr = InlineArray[Int, 4](fill=0)
    var ipiv = InlineArray[Int, 4](fill=0)

    for iteration in range(4):
        var big = Float32(0)
        var irow = 0
        var icol = 0
        for row in range(4):
            if ipiv[row] != 1:
                for col in range(4):
                    if ipiv[col] == 0:
                        var v = minv[col * 4 + row]
                        if v < Float32(0):
                            v = -v
                        if v >= big:
                            big = v
                            irow = row
                            icol = col
                    elif ipiv[col] > 1:
                        return _write_identity(result)
        ipiv[icol] += 1
        if irow != icol:
            for colIndex in range(4):
                var tmp = minv[colIndex * 4 + irow]
                minv[colIndex * 4 + irow] = minv[colIndex * 4 + icol]
                minv[colIndex * 4 + icol] = tmp
        indxr[iteration] = irow
        indxc[iteration] = icol
        if minv[icol * 4 + icol] == Float32(0):
            return _write_identity(result)
        var pivinv = Float32(1) / minv[icol * 4 + icol]
        minv[icol * 4 + icol] = Float32(1)
        for colIndex in range(4):
            minv[colIndex * 4 + icol] *= pivinv
        for rowIndex in range(4):
            if rowIndex != icol:
                var save = minv[icol * 4 + rowIndex]
                minv[icol * 4 + rowIndex] = Float32(0)
                for colIndex in range(4):
                    minv[colIndex * 4 + rowIndex] -= minv[colIndex * 4 + icol] * save

    for ci in range(4):
        var colIndex = 3 - ci
        if indxr[colIndex] != indxc[colIndex]:
            var rs = indxr[colIndex]
            var cs = indxc[colIndex]
            for rowIndex in range(4):
                var tmp = minv[rs * 4 + rowIndex]
                minv[rs * 4 + rowIndex] = minv[cs * 4 + rowIndex]
                minv[cs * 4 + rowIndex] = tmp

    for i in range(16):
        result[i] = minv[i]
    return Int32(1)


# Bulk geometry transform.
# Points are 4 floats each (SIMD4 layout: x,y,z,w=1). Normals are 3 floats each.
# matrix / inv_matrix are 16-float column-major (flat[col*4+row] = m[row,col]).

def transform_points(
    matrix: UnsafePointer[Float32, MutAnyOrigin],
    points_in: UnsafePointer[Float32, MutAnyOrigin],
    count: Int32,
    points_out: UnsafePointer[Float32, MutAnyOrigin],
):
    var m0 = matrix[0];  var m1 = matrix[1];  var m2 = matrix[2];  var m3 = matrix[3]
    var m4 = matrix[4];  var m5 = matrix[5];  var m6 = matrix[6];  var m7 = matrix[7]
    var m8 = matrix[8];  var m9 = matrix[9];  var m10 = matrix[10]; var m11 = matrix[11]
    var m12 = matrix[12]; var m13 = matrix[13]; var m14 = matrix[14]; var m15 = matrix[15]
    for i in range(Int(count)):
        var b = i * 4
        var px = points_in[b];  var py = points_in[b+1];  var pz = points_in[b+2]
        var rx = m0*px + m4*py + m8*pz + m12
        var ry = m1*px + m5*py + m9*pz + m13
        var rz = m2*px + m6*py + m10*pz + m14
        var rw = m3*px + m7*py + m11*pz + m15
        if rw != Float32(1) and rw != Float32(0):
            var inv_rw = Float32(1) / rw
            rx *= inv_rw; ry *= inv_rw; rz *= inv_rw
        points_out[b] = rx;  points_out[b+1] = ry;  points_out[b+2] = rz;  points_out[b+3] = Float32(1)


def transform_normals(
    inv_matrix: UnsafePointer[Float32, MutAnyOrigin],
    normals_in: UnsafePointer[Float32, MutAnyOrigin],
    count: Int32,
    normals_out: UnsafePointer[Float32, MutAnyOrigin],
):
    # Normals transform by the transpose of the inverse 3×3.
    # result[i] = sum_j inv[j*4+i] * n[j]  for i,j in 0..2
    var i0 = inv_matrix[0]; var i1 = inv_matrix[1]; var i2 = inv_matrix[2]
    var i4 = inv_matrix[4]; var i5 = inv_matrix[5]; var i6 = inv_matrix[6]
    var i8 = inv_matrix[8]; var i9 = inv_matrix[9]; var i10 = inv_matrix[10]
    for i in range(Int(count)):
        var b = i * 3
        var nx = normals_in[b];  var ny = normals_in[b+1];  var nz = normals_in[b+2]
        normals_out[b]   = i0*nx + i4*ny + i8*nz
        normals_out[b+1] = i1*nx + i5*ny + i9*nz
        normals_out[b+2] = i2*nx + i6*ny + i10*nz
