import struct,sys
d=sys.argv[1]
res=open('/tmp/ncu/res.bin','rb').read(); ok=bad=0; shown=0
for i,l in enumerate(open(f'{d}/ref.txt')):
    idx,hit,t,u,v,tri=l.split(); rt,ru,rv,r20=struct.unpack_from('<fffI',res,i*32)
    good=(max(abs(rt-float(t)),abs(ru-float(u)),abs(rv-float(v)))<1e-4 and (r20&0x1fffffff)==int(tri)) if int(hit) else r20==0xffffffff
    ok+=good; bad+=not good
    if not good and shown<4: shown+=1; print('bad',i,hit,t,u,v,tri,'got',rt,ru,rv,hex(r20))
print(d,"ok",ok,"bad",bad)
