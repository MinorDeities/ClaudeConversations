import sys; sys.path.insert(0,'/home/claude/session09/work'); from rope import read_portrait
import cv2, numpy as np, scipy.signal as ss
out=[]
for n in (29,30,31,32,34):
    F,T=read_portrait(f'/home/claude/session09/clips/10000144{n}.mp4',960)
    o=np.load(f'/home/claude/session09/work/rope{n}.npy'); rel=o[:,1]-o[:,2]
    ok=np.abs(o[:,2]-np.median(o[:,2]))<3
    depth=[]; 
    for f in F:
        hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
        # apex: topmost bright purple pixel row in the central half
        pm=(hsv[...,0]>125)&(hsv[...,0]<165)&(hsv[...,2]>90)&(hsv[...,1]>80)
        rows=np.nonzero(pm[:, f.shape[1]//4:3*f.shape[1]//4].any(1))[0]; ay=rows[0] if len(rows) else np.nan
        # performers: saturated blue (hue 100-128), below 20% height
        bm=(hsv[...,0]>=100)&(hsv[...,0]<=128)&(hsv[...,1]>120)&(hsv[...,2]>70); bm[:int(f.shape[0]*0.2)]=False
        ys=np.nonzero(bm)[0]
        depth.append((np.mean(ys)-ay) if len(ys)>300 else np.nan)
    depth=np.array(depth)
    t=o[:,0]; Tu=np.arange(t[0],t[-1],1/60)
    y=np.interp(Tu,t[ok],(rel-np.median(rel[ok]))[ok]); y=np.convolve(y,np.ones(45)/45,'same')
    d=np.interp(Tu,t[~np.isnan(depth)],depth[~np.isnan(depth)])
    zc=np.nonzero(np.diff(np.sign(y))!=0)[0]
    for a,b in zip(zc[:-1],zc[1:]):
        hp=(b-a)/60
        if 1.3<hp<3.2: out.append((n,Tu[a],hp,d[a:b].mean()))
out=np.array(out); np.save('/home/claude/session09/work/hooptest.npy',out)
hp,dep=out[:,2],out[:,3]
r=np.corrcoef(hp,dep)[0,1]
print('half-periods n=%d, mean %.2f s; depth range %.0f-%.0f px; corr(half-period, depth) r=%.2f'%(len(hp),hp.mean(),dep.min(),dep.max(),r))
# within-clip correlations
for n in (29,30,31,32,34):
    k=out[:,0]==n
    if k.sum()>3: print(n,'n=%d r=%.2f'%(k.sum(),np.corrcoef(hp[k],dep[k])[0,1]))
# low vs high split
med=np.median(dep); print('deeper half mean hp %.2f s (n=%d), shallower half %.2f s (n=%d)'%(hp[dep>med].mean(),(dep>med).sum(),hp[dep<=med].mean(),(dep<=med).sum()))
