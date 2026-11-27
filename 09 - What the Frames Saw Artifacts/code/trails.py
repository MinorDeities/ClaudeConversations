import sys; sys.path.insert(0,'/home/claude/session09/work')
from load import *
def trails(path, out, scale=0.5, hue=(95,132), vmin=70, dthr=45, step=1, mode='time', t0=0, t1=None):
    F,ts=frames(path,scale)
    c,r=chain_shifts(F)
    H,W=F[0].shape[:2]
    S=[cv2.warpAffine(f,np.float32([[1,0,-c[i,0]],[0,1,-c[i,1]]]),(W,H)) for i,f in enumerate(F)]
    sel=[i for i in range(len(S)) if ts[i]>=t0 and (t1 is None or ts[i]<=t1)][::step]
    bg=np.median(np.stack([S[i] for i in sel[::3]]),axis=0).astype(np.float32)
    acc=np.zeros((H,W,3),np.float32); cnt=np.zeros((H,W),np.float32)
    n=len(sel)
    for k,i in enumerate(sel):
        f=S[i].astype(np.float32); hsv=cv2.cvtColor(S[i],cv2.COLOR_BGR2HSV)
        d=np.abs(f-bg).sum(-1)
        m=(d>dthr)&(hsv[...,0]>=hue[0])&(hsv[...,0]<=hue[1])&(hsv[...,2]>=vmin)
        m=cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_OPEN,np.ones((2,2),np.uint8))>0
        if mode=='time':
            # time colour: hue from 200deg (blue) through to 30deg (amber)
            tt=k/max(n-1,1); hh=int((110-100*tt)%180)
            col=cv2.cvtColor(np.uint8([[[hh,200,255]]]),cv2.COLOR_HSV2BGR)[0,0].astype(np.float32)
            v=hsv[...,2][m].astype(np.float32)[:,None]/255
            acc[m]=np.maximum(acc[m],col*v)
        cnt[m]+=1
    base=bg*0.28
    img=np.maximum(base,acc)
    cv2.imwrite(out,np.clip(img,0,255).astype(np.uint8))
    return ts[sel[0]],ts[sel[-1]],len(sel)
if __name__=='__main__':
    print(trails(sys.argv[1],sys.argv[2]))
