import sys, struct, numpy as np
sys.path.insert(0,'/home/gonsolo/work/gonzales/docs/rtcore/exec')
import re_lib as L
# probe rays: three slanted families on a grid
def probe_rays(span=8.0, n=40):
    g=np.linspace(-span,span,n)
    R=[]
    for a in g:
        for b in g:
            R.append((a,b,50.0, 0.013,0.021,-1.0))
            R.append((50.0,a,b, -1.0,0.017,0.011))
            R.append((a,50.0,b, 0.019,-1.0,0.023))
    return R
RAYS=probe_rays(1.5,50)
N=L.rays_file('probe.rays',RAYS)
def set_span(span,n=60):
    global RAYS,N
    RAYS=probe_rays(span,n); N=L.rays_file('probe.rays',RAYS)
def decode_tris(blob_name, patches=()):
    """returns list of (hit points) solved triangle vertices for hits grouped by raw id"""
    res=L.trace(blob_name,'probe.rays',N,patches)
    if res is None: return None
    out={}
    for r,(raw,t,u,v) in zip(RAYS,res):
        if raw==0xffffffff or not np.isfinite(t) or t<=0: continue
        o=np.array(r[:3]);d=np.array(r[3:])
        out.setdefault(raw,[]).append((o+t*d,abs(u),v))
    sol={}
    for raw,h in out.items():
        if len(h)<4: continue
        A=np.array([[1-u-v,u,v] for _,u,v in h]); P=np.array([p for p,_,_ in h])
        X,res_,rk,_=np.linalg.lstsq(A,P,rcond=None)
        sol[raw]=(X, float(np.abs(A@X-P).max()))
    return sol
if __name__=='__main__':
    tri=[0.5,0.25,0.125, 3,0.5,0, 0.25,4,1]
    b=L.build(tri,'tmp')
    s=decode_tris('tmp.b')
    for raw,(X,e) in s.items(): print(hex(raw)); print(X.round(5)); print('maxerr',e)

def idmap(blob_name, tris, patches=()):
    """For every ray hit, find which of the known triangles contains the hit point; returns {tri index: set(raw ids)}."""
    res=L.trace(blob_name,'probe.rays',N,patches)
    if res is None: return None
    T=[np.array(t,dtype=float) for t in tris]
    out={}
    for r,(raw,t,u,v) in zip(RAYS,res):
        if raw==0xffffffff or not np.isfinite(t) or t<=0: continue
        P=np.array(r[:3])+t*np.array(r[3:])
        for j,V in enumerate(T):
            e1=V[1]-V[0]; e2=V[2]-V[0]; n=np.cross(e1,e2)
            if abs(np.dot(P-V[0],n))>1e-3*np.linalg.norm(n)*max(1,abs(P).max()): continue
            # barycentric
            A=np.array([e1,e2]).T; x,_,_,_=np.linalg.lstsq(A,P-V[0],rcond=None)
            if x[0]>=-1e-3 and x[1]>=-1e-3 and x[0]+x[1]<=1+1e-3:
                out.setdefault(j,set()).add(raw&0xffffff); break
    return out
