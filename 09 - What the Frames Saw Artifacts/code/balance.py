import sys; sys.path.insert(0,'/home/claude/session09/work'); from rope import read_portrait
import cv2, numpy as np
def rim(f):
    hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV); H,W=f.shape[:2]
    m=(hsv[...,2]>200)&(hsv[...,1]>60)
    m[:int(H*0.45)]=False
    rows=m.sum(1); y=int(np.argmax(rows))
    band=m[max(0,y-6):y+7]; xs=np.nonzero(band.any(0))[0]
    if len(xs)<20: return None
    # take the main contiguous run
    runs=np.split(xs,np.where(np.diff(xs)>8)[0]+1); r=max(runs,key=len)
    return y, r[0], r[-1]
def body(f, ry):
    hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
    m=(hsv[...,2]>95)&(((hsv[...,0]>=135)&(hsv[...,0]<=179))|(hsv[...,0]<=8))&(hsv[...,1]>50)
    m[ry-4:]=False
    m=cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    md=cv2.dilate(m,np.ones((15,15),np.uint8))
    n,lab,st,cen=cv2.connectedComponentsWithStats(md)
    if n<2: return None
    k=1+np.argmax(st[1:,cv2.CC_STAT_AREA]); bm=(lab==k)&(m>0)
    # fill: close holes in the silhouette (dark costume) via closing then fill
    cl=cv2.morphologyEx(bm.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((21,21),np.uint8))
    cnts,_=cv2.findContours(cl,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    fill=np.zeros_like(cl); cv2.drawContours(fill,cnts,-1,1,-1); bm=fill>0
    ys,xs=np.nonzero(bm)
    hb=(ys>ry-95)&(ys<ry-40)
    hx=xs[hb].mean() if hb.sum()>20 else np.nan
    far=np.argmax((xs-xs.mean())**2+(ys-ys.mean())**2)
    return xs.mean(), ys.mean(), xs.min(), xs.max(), ys.min(), len(xs), bm, hx, xs[far], ys[far]
if __name__=='__main__':
    n=sys.argv[1]
    F,T=read_portrait(f'/home/claude/session09/clips/10000144{n}.mp4',960)
    import scipy.signal as ss
    R=[rim(f) for f in F]
    ry_raw=np.array([r[0] if r else np.nan for r in R],float)
    ok=~np.isnan(ry_raw); ry_raw=np.interp(np.arange(len(F)),np.nonzero(ok)[0],ry_raw[ok])
    ry_s=ss.medfilt(ry_raw,31)
    rows=[]
    for i,(f,t) in enumerate(zip(F,T)):
        r=R[i]
        if r is not None: r=(int(ry_s[i]),r[1],r[2])
        if r is None: rows.append([t]+[np.nan]*12); continue
        ry,x0,x1=r; b=body(f,ry)
        if b is None: rows.append([t,ry,x0,x1]+[np.nan]*9); continue
        cx,cy,bx0,bx1,by0,area,_,hx,fx,fy=b
        rows.append([t,ry,x0,x1,cx,cy,bx0,bx1,by0,area,hx,fx,fy])
    rows=np.array(rows,float); np.save(f'/home/claude/session09/work/bal{n}.npy',rows)
    w=rows[:,3]-rows[:,2]; print(n,len(rows),'rim width px median',np.nanmedian(w),'rim y',np.nanmedian(rows[:,1]))
    # debug overlay of a few frames
    idx=np.linspace(0,len(F)-1,6).astype(int); tiles=[]
    for i in idx:
        f=F[i].copy(); r=rim(f)
        if r:
            ry,x0,x1=r; cv2.line(f,(x0,ry),(x1,ry),(0,255,0),2); b=body(f,ry)
            if b: f[b[6]]=(f[b[6]]*0.4+np.array([0,255,255])*0.6).astype(np.uint8); cv2.circle(f,(int(b[0]),int(b[1])),7,(0,0,255),-1); cv2.line(f,(int(b[0]),0),(int(b[0]),ry),(0,0,255),1)
        tiles.append(cv2.resize(f,(270,480)))
    cv2.imwrite(f'/home/claude/session09/work/bal{n}_dbg.jpg',np.hstack(tiles))
