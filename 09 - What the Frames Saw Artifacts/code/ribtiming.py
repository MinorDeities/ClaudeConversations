import sys; sys.path.insert(0,'/home/claude/session09/work')
from slices import *; from audio import *
import scipy.signal as ss
def run(n, nz=3):
    S,ts=stab(f'clips/10000144{n}.mp4'); bg=np.median(np.stack(S[::6]),axis=0).astype(np.float32)
    H,W=S[0].shape[:2]; zones=[(W*k//nz,W*(k+1)//nz) for k in range(nz)]
    cy=np.full((len(S),nz),np.nan)
    for i,f in enumerate(S):
        m,_=ribbon_mask(f,bg); m[int(H*0.62):]=False
        for z,(a,b) in enumerate(zones):
            ys,xs=np.nonzero(m[:,a:b])
            if len(xs)>30: cy[i,z]=ys.mean()/H
    t,env,fps,x,sr=onset_env(f'audio/{n}.wav')
    best=None
    for bpm in np.arange(97,103,0.05):
        P=60/bpm
        for ph in np.arange(0,P,0.005):
            idx=np.clip((np.arange(ph,t[-1],P)*fps).astype(int),0,len(env)-1); sc=env[idx].mean()
            if best is None or sc>best[0]: best=(sc,bpm,ph)
    sc,bpm,ph=best; P=60/bpm
    print(n,'bpm %.2f'%bpm)
    T=np.arange(0,ts[-1],1/120)
    for z in range(nz):
        ok=~np.isnan(cy[:,z])
        if ok.sum()<50: continue
        y=-np.interp(T,ts[ok],cy[ok,z]); y=np.convolve(y,np.ones(9)/9,'same')
        pk,_=ss.find_peaks(y,distance=int(0.7*120),prominence=0.04)
        pt=T[pk]; phs=((pt-ph)/P)%2
        # circular mean & resultant on 2-beat circle
        ang=phs*np.pi; R=np.abs(np.mean(np.exp(1j*ang))); mu=(np.angle(np.mean(np.exp(1j*ang)))/np.pi)%2
        iv=np.diff(pt)
        print('  zone',z,'n=%d  median interval %.3f s  phase mean %.2f beats, R=%.2f'%(len(pt),np.median(iv) if len(iv) else 0,mu,R))
for n in sys.argv[1:]: run(n)
