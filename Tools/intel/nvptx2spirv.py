import re, sys
AS = {0: 4, 1: 1, 3: 3, 4: 2, 5: 0}  # NVPTX generic/global/shared/const/local -> SPIR
SREG = {"tid": "LocalInvocationId", "ctaid": "WorkgroupId", "ntid": "WorkgroupSize", "nctaid": "NumWorkgroups"}
src = open(sys.argv[1]).read()
src = re.sub(r'^target datalayout = .*$', 'target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"', src, flags=re.M)
src = src.replace('target triple = "nvptx64-nvidia-cuda"', 'target triple = "spir64-unknown-unknown"')
src = re.sub(r'ptr addrspace\((\d+)\)|addrspace\((\d+)\)|\bptr\b',
             lambda m: f'ptr addrspace({AS[int(m[1])]})' if m[1] else f'addrspace({AS[int(m[2])]})' if m[2] else 'ptr addrspace(4)', src)
src = re.sub(r'^declare .*@llvm\.nvvm\.read\.ptx\.sreg\.\w+\.[xyz]\(\).*$\n', '', src, flags=re.M)
used = set()
def sreg(m):
    used.add((m[1], m[2])); return f'@__sreg_{m[1]}_{m[2]}'
src = re.sub(r'@llvm\.nvvm\.read\.ptx\.sreg\.(\w+)\.([xyz])', sreg, src)
for r, d in sorted(used):
    f = '__spirv_BuiltIn' + SREG[r]; n = f'_Z{len(f)}{f}i'
    src += (f'define internal i32 @__sreg_{r}_{d}() alwaysinline {{\n  %v = call spir_func i64 @{n}(i32 {"xyz".index(d)})\n'
            f'  %t = trunc i64 %v to i32\n  ret i32 %t\n}}\n')
    if f'declare spir_func i64 @{n}' not in src: src += f'declare spir_func i64 @{n}(i32)\n'
def kernel(m):
    head, params, rest = m[1], m[2], m[3]
    casts = []
    def p(q):
        casts.append(f'  {q[2]} = addrspacecast ptr addrspace(1) {q[2]}.g to ptr addrspace(4)\n')
        return f'ptr addrspace(1){q[1]}{q[2]}.g'
    params = re.sub(r'ptr addrspace\(4\)([^,%]*)(%[\w.]+)', p, params)
    return f'{head}({params}){rest}\n' + ''.join(casts)
src = re.sub(r'^(define .*?)ptx_kernel( .*?)$', r'\1spir_kernel\2', src, flags=re.M)
src = re.sub(r'^(define [^(]*spir_kernel [^(]*)\((.*)\)(.*\{\n[\w.]+:)$', kernel, src, flags=re.M)
src = re.sub(r'"target-(cpu|features)"="[^"]*"', '', src)
open(sys.argv[2], 'w').write(src)
