import cv2, numpy as np, sys, scipy.signal as ss
def run(n):
    cap=cv2.VideoCapture(f'/home/claude/session09/clips/10000144{n}.mp4'); rows=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        if f.shape[0]>f.shape[1]: f=cv2.rotate(f,cv2.ROTATE_90_CLOCKWISE)
        hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
        m=((hsv[...,0]<=28)|(hsv[...,0]>=172))&(hsv[...,2]>200)&(hsv[...,1]>90)
        m[620:]=False; m=m.astype(np.uint8)
        k,lab,st,cen=cv2.connectedComponentsWithStats(m)
        comps=[(st[i,cv2.CC_STAT_AREA],cen[i][0],cen[i][1],st[i,cv2.CC_STAT_HEIGHT]) for i in range(1,k) if st[i,cv2.CC_STAT_AREA]>150]
        comps=sorted(comps,reverse=True)[:2]; comps=sorted(comps,key=lambda c:c[1])
        rows.append((t,comps))
    return rows
n=sys.argv[1]; rows=run(n)
T=np.array([r[0] for r in rows]); nc=[len(r[1]) for r in rows]
print(n,'frames',len(T),'fps %.2f'%(len(T)/T[-1]),'components per frame', np.bincount(nc))
import pickle; pickle.dump(rows,open(f'/home/claude/session09/work/fire{n}.pkl','wb'))
