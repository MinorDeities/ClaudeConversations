import cv2, numpy as np
def frames(path, scale=0.5, step=1):
    cap=cv2.VideoCapture(path); out=[]; ts=[]; i=0
    while True:
        ok,f=cap.read()
        if not ok: break
        if i%step==0:
            out.append(cv2.resize(f,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)); ts.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        i+=1
    return out, np.array(ts)
def shifts(F, ref=0, mask_top=0):
    g=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float32) for f in F]
    win=cv2.createHanningWindow(g[0].shape[::-1],cv2.CV_32F)
    s=[]
    for x in g:
        (dx,dy),r=cv2.phaseCorrelate(g[ref],x,win); s.append((dx,dy,r))
    return np.array(s)
def chain_shifts(F, maxshift=40):
    # consecutive translation via phase correlation on gradient magnitude, chained
    def prep(f):
        g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float32)
        g=cv2.GaussianBlur(g,(0,0),1.5)
        gx=cv2.Sobel(g,cv2.CV_32F,1,0); gy=cv2.Sobel(g,cv2.CV_32F,0,1)
        return np.sqrt(gx*gx+gy*gy)
    P=[prep(f) for f in F]; win=cv2.createHanningWindow(P[0].shape[::-1],cv2.CV_32F)
    cum=[(0.0,0.0)]; resp=[1]
    for a,b in zip(P[:-1],P[1:]):
        (dx,dy),r=cv2.phaseCorrelate(a,b,win)
        if abs(dx)>maxshift or abs(dy)>maxshift: dx,dy=0,0
        cum.append((cum[-1][0]+dx,cum[-1][1]+dy)); resp.append(r)
    return np.array(cum), np.array(resp)
