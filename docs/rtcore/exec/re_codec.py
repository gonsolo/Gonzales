import struct
def fb(x): return struct.unpack('<I',struct.pack('<f',x))[0]
def bf(i): return struct.unpack('<f',struct.pack('<I',i))[0]
def encode_single(v0, v1, v2):
    """Block bytes (36) + descriptor word for one triangle, following the observed coding."""
    vs=[v1,v2]
    allv=[v0,v1,v2]
    # shared mantissa length
    m=0
    for v in allv:
        for c in v:
            man=fb(c)&0x7fffff
            if man:
                tz=(man & -man).bit_length()-1
                m=max(m,23-tz)
    e=[]
    for a in range(3):
        ref=fb(v0[a])>>23
        span=0
        for v in vs: span|=(fb(v[a])>>23)^ref
        e.append(max(1,span.bit_length()))
    bits=0; pos=0
    for v in vs:
        for a in range(3):
            f=fb(v[a]); man=(f&0x7fffff)>>(23-m) if m else 0
            ex=(f>>23)&((1<<e[a])-1)
            fld=man | (ex<<m)
            bits|=fld<<pos; pos+=m+e[a]
    assert pos<=192
    blk=struct.pack('<3f',*v0)+bits.to_bytes(24,'little')
    desc=((55-m)<<24)|((m+e[2]-1)<<10)|((m+e[1]-1)<<5)|(m+e[0]-1)
    return blk,desc
def decode_single(blk, desc):
    v0=struct.unpack_from('<3f',blk,0); bits=int.from_bytes(blk[12:36],'little')
    m=55-(desc>>24); w=[((desc>>(5*a))&31)+1 for a in range(3)]
    pos=0; out=[]
    for v in range(2):
        c=[]
        for a in range(3):
            fld=(bits>>pos)&((1<<w[a])-1); pos+=w[a]
            man=fld&((1<<m)-1); ex=fld>>m; e=w[a]-m
            ref=fb(v0[a])>>23
            ex9=(ref>>e<<e)|ex
            c.append(bf((ex9<<23)|(man<<(23-m) if m else 0)))
        out.append(c)
    return out
if __name__=='__main__':
    import sys,random; sys.path.insert(0,'/home/gonsolo/work/gonzales/docs/rtcore/exec')
    import re_lib as L
    random.seed(3); bad=0; n=0
    for k in range(30):
        mk=random.choice([0,1,3,6,23])
        def c(): 
            v=random.uniform(-4,4)*random.choice([1,1,0.01,10])
            if mk<23: v=round(v*2**mk)/2**mk
            return struct.unpack('<f',struct.pack('<f',v))[0]
        t=[c() for _ in range(9)]
        if random.random()<0.6: t=[abs(x) for x in t]
        b=L.build(t,'tmp')
        blk,desc=encode_single(t[:3],t[3:6],t[6:])
        vb=b[0x400:0x424]; vd=struct.unpack_from('<I',b,0x47c)[0]
        ok=(blk==vb and desc==vd); n+=1; bad+= (not ok)
        if not ok and bad<6: print('MISMATCH',t,'\n  vendor',vb[12:].hex(),'%08x'%vd,'\n  mine  ',blk[12:].hex(),'%08x'%desc)
    print('mismatches',bad,'of',n)

def decode_stream(blk128, nv):
    """Vertices after v0 from a 128-byte block, using its descriptor (single descriptor assumed)."""
    desc=struct.unpack_from('<I',blk128,0x7c)[0]
    v0=struct.unpack_from('<3f',blk128,0); bits=int.from_bytes(blk128[12:0x68],'little')
    m=55-(desc>>24); w=[((desc>>(5*a))&31)+1 for a in range(3)]
    pos=0; out=[]
    for v in range(nv):
        c=[]
        for a in range(3):
            fld=(bits>>pos)&((1<<w[a])-1); pos+=w[a]
            man=fld&((1<<m)-1); ex=fld>>m; e=w[a]-m
            ref=fb(v0[a])>>23
            ex9=(ref>>e<<e)|ex
            c.append(bf((ex9<<23)|((man<<(23-m)) if m else 0)))
        out.append(tuple(round(x,4) for x in c))
    return out

def encode_block(tris, prim_base=0):
    """tris: list of (a,b,c) float triples, 1..5 triangles in block order. Returns 128 block bytes (plus nothing else)."""
    verts=[]; idx=[]
    for t in tris:
        r=[]
        for p in t:
            p=tuple(float(x) for x in p)
            if p not in verts: verts.append(p)
            r.append(verts.index(p))
        idx.append(r)
    assert idx[0]==[0,1,2] and len(verts)<=16
    v0=verts[0]; sv=verts[1:]
    m=0
    for v in verts:
        for c in v:
            man=fb(c)&0x7fffff
            if man: m=max(m,23-((man&-man).bit_length()-1))
    e=[]
    for a in range(3):
        ref=fb(v0[a])>>23; span=0
        for v in sv: span|=(fb(v[a])>>23)^ref
        e.append(max(1,span.bit_length()))
    bits=0; pos=0
    for v in sv:
        for a in range(3):
            f=fb(v[a]); man=(f&0x7fffff)>>(23-m) if m else 0
            ex=(f>>23)&((1<<e[a])-1)
            bits|=(man|(ex<<m))<<pos; pos+=m+e[a]
    assert pos<=8*(0x68-0x0c), pos
    blk=bytearray(128)
    struct.pack_into('<3f',blk,0,*v0)
    blk[12:12+(0x68-12)]=bits.to_bytes(0x68-12,'little')
    idxbits=0
    for j in range(1,len(tris)):
        rec=(idx[j][0]|(idx[j][1]<<4)|(idx[j][2]<<8))<<2
        idxbits|=rec<<(126-14*j)
    blk[0x68:0x78]=idxbits.to_bytes(16,'little')
    # W470 top code: the vendor wrote 0x80000000 for ascending ids
    struct.pack_into('<I',blk,0x78,0x80000000|prim_base)
    desc=((55-m)<<24)|((len(tris)-1)<<20)|((m+e[2]-1)<<10)|((m+e[1]-1)<<5)|(m+e[0]-1)
    struct.pack_into('<I',blk,0x7c,desc)
    return bytes(blk)
