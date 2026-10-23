from std.compile import compile_info
from max.gpu import global_idx
from max.gpu.host.info import get_gpu_target

def scale(p: MutPointer[Float32, MutAnyOrigin], n: Int):
    var i = global_idx.x
    if i < n:
        p[unsafe_offset=i] = Float32(i) * 2.0

def main():
    print(compile_info[scale, emission_kind="llvm-opt", target=get_gpu_target["sm_86"]()]())
