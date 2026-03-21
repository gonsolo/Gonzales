import OpenEXR
import Imath
import numpy as np

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

img1 = read_exr('gonzales.exr')
img2 = read_exr('pbrt.exr')

diff = np.abs(img1 - img2)
mse = np.mean(diff ** 2)
mae = np.mean(diff)
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")

print(f"Gonzales mean: {np.mean(img1):.6f}, max: {np.max(img1):.6f}")
print(f"PBRT mean: {np.mean(img2):.6f}, max: {np.max(img2):.6f}")

mean_ratio = np.mean(img1) / np.mean(img2)
print(f"Brightness ratio (Gonzales/PBRT): {mean_ratio:.6f}")
