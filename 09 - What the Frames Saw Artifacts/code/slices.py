import sys; sys.path.insert(0,'/home/claude/session09/work')
from load import *
def stab(path,scale=0.5):
    F,ts=frames(path,scale); c,r=chain_shifts(F); H,W=F[0].shape[:2]
    S=[cv2.warpAffine(f,np.float32([[1,0,-c[i,0]],[0,1,-c[i,1]]]),(W,H)) for i,f in enumerate(F)]
    return S,ts
def ribbon_mask(f,bg,hue=(95,132),vmin=70,dthr=45):
    hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV); d=np.abs(f.astype(np.float32)-bg).sum(-1)
    m=(d>dthr)&(hsv[...,0]>=hue[0])&(hsv[...,0]<=hue[1])&(hsv[...,2]>=vmin)
    return cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_OPEN,np.ones((2,2),np.uint8))>0, hsv[...,2]
def slice_img(S,ts,bg,a,b,colormode='gradient'):
    idx=[i for i in range(len(S)) if a<=ts[i]<b]; H,W=S[0].shape[:2]
    acc=np.zeros((H,W,3),np.float32)
    for k,i in enumerate(idx):
        m,v=ribbon_mask(S[i],bg)
        t=k/max(len(idx)-1,1)
        col=np.array([255*(1-t)+40*t, 120+100*t*(1-t)*4*0.5, 60+195*t],np.float32) # BGR: blue->amber
        acc[m]=np.maximum(acc[m],col*(0.35+0.65*v[m][:,None]/255))
    return np.maximum(bg*0.22,acc)
if __name__=='__main__':
    path,out,win=sys.argv[1],sys.argv[2],float(sys.argv[3])
    S,ts=stab(path); bg=np.median(np.stack(S[::6]),axis=0).astype(np.float32)
    T=ts[-1]; cuts=np.arange(0,T-win*0.5,win)
    tiles=[]
    for a in cuts:
        im=slice_img(S,ts,bg,a,a+win); im=np.clip(im,0,255).astype(np.uint8)
        cv2.putText(im,f'{a:.1f}-{a+win:.1f}s',(8,22),0,0.6,(200,200,200),1); tiles.append(cv2.resize(im,(480,270)))
    while len(tiles)%3: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out,np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)]))
    print(len(tiles))
