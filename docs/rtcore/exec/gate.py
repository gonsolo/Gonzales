import subprocess,re,sys
cases={
 'baseline':[],
 'root_null':['set $R8=0','set $R9=0'],
 'root_garbage':['set $R8=0x1234','set $R9=0x0'],
 'root_untagged':['set $R9=($R9 & 0x0fffffff)'],
 'tag2':['set $R9=(($R9 & 0x0fffffff)|0x20000000)'],
 'flags_zero':['set $R20=0','set $R21=0','set $R22=0','set $R23=0'],
 'R20_only':['set $R20=0'],
 'R22_only':['set $R22=0'],
 'R21_zero':['set $R21=0'],
 'R21_max':['set $R21=0xffffffff'],
 'R20_ff':['set $R20=0xff030007'],
}
def run(name,sets):
    cmds=["set pagination off","set cuda break_on_launch application","run /tmp/ncu/rt1.cubin /tmp/ncu/asprobe blas","set $K = $pc","tbreak *($K + 0x6e0)","continue"]+sets+[
      'printf "PC_BEFORE %lx\\n", $pc-$K','stepi','printf "PC_AFTER %lx\\n", $pc-$K','info registers R8 R9 R20 R21 R22 R23','detach','quit']
    open('/tmp/ncu/g.cmd','w').write("\n".join(cmds)+"\n")
    out=subprocess.run(['timeout','300','cuda-gdb','-batch','-x','/tmp/ncu/g.cmd','/tmp/run_rt2'],capture_output=True,text=True).stdout
    pc=re.search(r'PC_AFTER ([0-9a-f]+)',out)
    regs={m.group(1):m.group(2) for m in re.finditer(r'^(R\d+)\s+(0x[0-9a-f]+)',out,re.M)}
    print(f"{name:14s} after_pc=+{pc.group(1) if pc else '?'} {regs}")
for n,s in cases.items(): run(n,s)
