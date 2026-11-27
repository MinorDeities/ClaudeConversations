import sys; sys.path.insert(0,'/home/claude/session09/work')
from slices import *
import scipy.io.wavfile as wf, scipy.signal as ss
n=sys.argv[1]
S,ts=stab(f'/home/claude/session09/clips/10000144{n}.mp4'); bg=np.median(np.stack(S[::6]),axis=0).astype(np.float32)
H,W=S[0].shape[:2]
prof=[]  # per frame: vertical profile of ribbon pixels for 3 zones, plus horizontal profile
hprof=[]
for f in S:
    m,_=ribbon_mask(f,bg); m[int(H*0.7):]=False
    prof.append(np.concatenate([m[:,W*k//3:W*(k+1)//3].sum(1) for k in range(3)]))
    hprof.append(m.sum(0))
prof=np.array(prof,float); hprof=np.array(hprof,float)
np.savez(f'/home/claude/session09/work/score{n}.npz',ts=ts,prof=prof,hprof=hprof,H=H,W=W)
print(n,prof.shape)
