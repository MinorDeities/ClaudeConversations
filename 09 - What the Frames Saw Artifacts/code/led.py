import sys; sys.path.insert(0,'/home/claude/session09/work'); from rope import read_portrait; from balance import rim
import cv2, numpy as np, scipy.signal as ss
n=sys.argv[1]
F,T=read_portrait(f'/home/claude/session09/clips/10000144{n}.mp4',640)
R=[rim(f) for f in F]
ry=np.array([r[0] if r else np.nan for r in R],float); ok=~np.isnan(ry); ry=np.interp(np.arange(len(F)),np.nonzero(ok)[0],ry[ok]); ry=ss.medfilt(ry,31).astype(int)
x0=np.array([r[1] if r else np.nan for r in R]); x1=np.array([r[2] if r else np.nan for r in R])
cx=np.interp(np.arange(len(F)),np.nonzero(~np.isnan(x0))[0],((x0+x1)/2)[~np.isnan(x0)])
cols=[]
for i,f in enumerate(F):
    y=ry[i]; band=f[max(0,y-4):y+5, max(0,int(cx[i])-90):int(cx[i])+90].reshape(-1,3).astype(float)
    if len(band)==0: cols.append([np.nan]*3); continue
    v=band.max(1); sel=band[v>=np.percentile(v,80)]
    cols.append(sel.mean(0))
cols=np.array(cols)  # BGR
np.savez(f'/home/claude/session09/work/led{n}.npz',T=T,cols=cols)
d=np.nan_to_num(np.linalg.norm(np.diff(cols,axis=0),axis=1))
ch=T[1:][d>np.percentile(d,97)]
print(n,'frames',len(F),'big colour changes at',np.round(ch,2))
