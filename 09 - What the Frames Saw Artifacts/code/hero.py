import sys; sys.path.insert(0,'/home/claude/session09/work')
from load import *
from slices import ribbon_mask
def hero(n,a,b,out,sc=0.75):
    F,ts=frames(f'/home/claude/session09/clips/10000144{n}.mp4',0.5); c,_=chain_shifts(F); del F
    cap=cv2.VideoCapture(f'/home/claude/session09/clips/10000144{n}.mp4'); i=0; win=[]; samp=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        if a<=t<b or i%6==0:
            g=cv2.resize(f,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA); H,W=g.shape[:2]
            g=cv2.warpAffine(g,np.float32([[1,0,-c[i,0]*sc/0.5],[0,1,-c[i,1]*sc/0.5]]),(W,H))
            if i%6==0: samp.append(g)
            if a<=t<b: win.append(g)
        i+=1
    bg=np.median(np.stack(samp),axis=0).astype(np.float32)
    acc=np.zeros_like(bg); n_=len(win)
    for k,g in enumerate(win):
        m,v=ribbon_mask(g,bg)
        t=k/max(n_-1,1)
        col=np.array([255*(1-t)+40*t, 120+50*t*(1-t)*4, 60+195*t],np.float32)
        acc[m]=np.maximum(acc[m],col*(0.35+0.65*v[m][:,None]/255))
    img=np.maximum(bg*0.2,acc); cv2.imwrite(out,np.clip(img,0,255).astype(np.uint8),[cv2.IMWRITE_JPEG_QUALITY,88])
n,a,b,o=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]),sys.argv[4]; hero(n,a,b,o)
