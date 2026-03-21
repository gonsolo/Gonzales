import OpenEXR
import Imath
import numpy as np
import sys

def read_exr(filepath):
    f = OpenEXR.InputFile(filepath)
    dw = f.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    c_str = f.channel('R', pt)
    r = np.frombuffer(c_str, dtype=np.float32)
    c_str = f.channel('G', pt)
    g = np.frombuffer(c_str, dtype=np.float32)
    c_str = f.channel('B', pt)
    b = np.frombuffer(c_str, dtype=np.float32)
    
    r.shape = (size[1], size[0])
    g.shape = (size[1], size[0])
    b.shape = (size[1], size[0])
    
    return np.stack([r, g, b], axis=-1)

file1 = sys.argv[1] if len(sys.argv) > 1 else 'gonzales.exr'
file2 = sys.argv[2] if len(sys.argv) > 2 else 'pbrt.exr'
img1 = read_exr(file1)
img2 = read_exr(file2)

diff = np.abs(img1 - img2)
mse = np.mean(diff ** 2)
mae = np.mean(diff)
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")

print(f"{file1} mean: {np.mean(img1):.6f}, max: {np.max(img1):.6f}")
print(f"{file2} mean: {np.mean(img2):.6f}, max: {np.max(img2):.6f}")

max_idx = np.unravel_index(np.argmax(img1), img1.shape)
print(f"{file1} max at (y, x, c): {max_idx}")
# x, y for Gonzales --single
print(f"Trace coordinate: --single {max_idx[1]} {max_idx[0]}")

mean_ratio = np.mean(img1) / np.mean(img2)
print(f"Brightness ratio ({file1}/{file2}): {mean_ratio:.6f}")
