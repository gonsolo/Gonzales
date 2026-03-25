import OpenEXR, Imath
import numpy as np

def read_exr(filename):
    exr = OpenEXR.InputFile(filename)
    header = exr.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = exr.channels(['R', 'G', 'B'], FLOAT)
    
    r = np.frombuffer(channels[0], dtype=np.float32).reshape((size[1], size[0]))
    g = np.frombuffer(channels[1], dtype=np.float32).reshape((size[1], size[0]))
    b = np.frombuffer(channels[2], dtype=np.float32).reshape((size[1], size[0]))
    
    return np.stack([r, g, b], axis=-1)

try:
    ref = read_exr('Tests/reference/regression-test.exr')
    out = read_exr('regression-test.exr')
    diff = np.abs(ref - out)
    print("Reference mean:", np.mean(ref), "max:", np.max(ref))
    print("Output    mean:", np.mean(out), "max:", np.max(out))
    print("Diff      mean:", np.mean(diff), "max:", np.max(diff))
    print("Pixels > 0.1 diff:", np.sum(diff > 0.1))
except Exception as e:
    print(e)
