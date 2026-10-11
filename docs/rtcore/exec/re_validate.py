import sys, struct, shutil, random, numpy as np
sys.path.insert(0,'.')
import re_lib as L, mkblas as M
M.TIGHT=True
n=int(sys.argv[1]); nr=int(sys.argv[2]); seed=int(sys.argv[3]) if len(sys.argv)>3 else 1
rng=np.random.default_rng(seed); random.seed(seed)
S=20.0
tris=[]
for i in range(n):
    c=rng.uniform(0,S,3); 
    tris.append(tuple(tuple(float(np.float32(c[a]+rng.uniform(-1.5,1.5))) for a in range(3)) for _ in range(3)))
hdr=L.build([0,0,0,1,0,0,0,1,0],'one')
B=M.Builder(tris,hdr); blob=B.build()
open('own.b','wb').write(blob); shutil.copy('one.b.addr','own.b.addr')
print('built',len(blob),'B',n,'tris',len(B.nodes),'nodes',B.nblocks,'blocks')
# rays: random origins outside, aimed at random points inside
O_=rng.uniform(-5,S+5,(nr,3)); T=rng.uniform(0,S,(nr,3)); D=T-O_; D/=np.linalg.norm(D,axis=1)[:,None]
D+=1e-4*rng.standard_normal(D.shape); D/=np.linalg.norm(D,axis=1)[:,None]
rays=[(*o,*d) for o,d in zip(O_,D)]
cnt=L.rays_file('r.rays',rays)
res=L.trace('own.b','r.rays',cnt)
# brute force
V=np.array(tris,dtype=np.float64)   # n,3,3
e1=V[:,1]-V[:,0]; e2=V[:,2]-V[:,0]
bad=0; hit=0; miss_hw=0; extra_hw=0; tmis=0
for k,(o,d) in enumerate(zip(O_,D)):
    p=np.cross(d,e2); det=(e1*p).sum(1)
    with np.errstate(all='ignore'):
        inv=1.0/det; tv=o-V[:,0]; u=(tv*p).sum(1)*inv; q=np.cross(tv,e1); v=(d*q).sum(1)*inv; t=(e2*q).sum(1)*inv
    ok=(abs(det)>1e-12)&(u>=0)&(v>=0)&(u+v<=1)&(t>1e-6)
    raw,th,uh,vh=res[k]
    if ok.any():
        j=int(np.argmin(np.where(ok,t,np.inf))); tb=t[j]
        if min(u[j],v[j],1-u[j]-v[j])<2e-3: continue
        hit+=1
        if raw==0xffffffff: miss_hw+=1
        else:
            gid=raw&0xffffff; tj=B.gid_to_tri.get(gid,-1)
            if abs(th-tb)>1e-3*max(1,tb) :
                # allow equal-t different triangle (shared edge)
                tmis+=1
                if tmis<5: print('ray',k,'t hw %.5f sw %.5f tri hw %d sw %d'%(th,tb,tj,j))
    else:
        if raw!=0xffffffff and th>1e-4: extra_hw+=1
print('rays',nr,'sw hits',hit,'hw missed',miss_hw,'t mismatch',tmis,'hw hit where sw none',extra_hw)
