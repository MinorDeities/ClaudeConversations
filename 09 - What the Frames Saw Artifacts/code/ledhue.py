import sys; sys.path.insert(0,'/home/claude/session09/work'); from rope import read_portrait
import cv2, numpy as np
n=sys.argv[1]; F,T=read_portrait(f'/home/claude/session09/clips/10000144{n}.mp4',640)
ang=[]
for f in F:
    hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV); H=f.shape[0]
    low=hsv[int(H*0.55):]
    m=(low[...,1]>140)&(low[...,2]>170)
    h=low[...,0][m].astype(float)*2*np.pi/180
    if len(h)<30: ang.append(np.nan); continue
    z=np.exp(1j*h).mean(); ang.append(np.angle(z) if abs(z)>0.3 else np.nan)
np.savez(f'/home/claude/session09/work/ledhue{n}.npz',T=T,ang=np.array(ang))
a=np.array(ang); print(n,'valid',np.isfinite(a).mean().round(2))
