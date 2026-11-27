import cv2, numpy as np, sys
def read_portrait(path, h=960):
    cap=cv2.VideoCapture(path); F=[]; T=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        if f.shape[1]>f.shape[0]: f=cv2.rotate(f,cv2.ROTATE_90_CLOCKWISE)
        s=h/f.shape[0]; F.append(cv2.resize(f,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)); T.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
    return F,np.array(T)
def analyse(f, dbg=False):
    hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
    m=((hsv[...,0]>125)&(hsv[...,0]<165)&(hsv[...,2]>90)&(hsv[...,1]>80)).astype(np.uint8)*255
    H,W=m.shape
    top=m[:int(H*0.45)]
    lines=cv2.HoughLinesP(top,1,np.pi/720,60,minLineLength=120,maxLineGap=12)
    if lines is None: return None
    L=[]
    for x1,y1,x2,y2 in lines.reshape(-1,4):
        if y2<y1: x1,y1,x2,y2=x2,y2,x1,y1
        ang=np.degrees(np.arctan2(x2-x1,y2-y1))  # 0 = vertical, + leaning right going down
        L.append((ang,x1,y1,x2,y2,np.hypot(x2-x1,y2-y1)))
    L=np.array(L)
    # apex: intersection region -> use topmost endpoints
    left=L[L[:,0]<-15]; rope=L[np.abs(L[:,0])<8]
    if len(left)==0 or len(rope)==0: return None
    la=np.average(left[:,0],weights=left[:,5])
    # rope: thinnest near-vertical - choose line with x closest to median of near-vertical candidates with smallest |angle - median|
    ra=np.average(rope[:,0],weights=rope[:,5])
    return ra, la, len(rope)
if __name__=='__main__':
    F,T=read_portrait(sys.argv[1])
    out=[]
    for f,t in zip(F,T):
        r=analyse(f); out.append((t,)+(r if r else (np.nan,np.nan,0)))
    out=np.array(out); np.save(sys.argv[2],out)
    ok=~np.isnan(out[:,1]); print(len(out),'ok',ok.sum(),'rope ang range',np.nanmin(out[:,1]).round(2),np.nanmax(out[:,1]).round(2),'leg',np.nanmedian(out[:,2]).round(2))
