import numpy as np, scipy.io.wavfile as wf, scipy.signal as ss
def onset_env(path, hop=256):
    sr,x=wf.read(path); x=x.astype(np.float32); x/=np.abs(x).max()+1e-9
    f,t,Z=ss.stft(x,sr,nperseg=1024,noverlap=1024-hop)
    S=np.log1p(100*np.abs(Z))
    # mel-ish: keep 30Hz-8k
    flux=np.maximum(0,np.diff(S,axis=1)).sum(0)
    flux=np.r_[0,flux]; flux=ss.medfilt(flux,1)
    env=flux-ss.medfilt(flux,31); env=np.maximum(env,0)
    return t, env, sr/hop, x, sr
def tempo(env, fps, lo=50, hi=200):
    e=env-env.mean(); ac=np.correlate(e,e,'full')[len(e)-1:]; ac/=ac[0]
    lags=np.arange(len(ac))/fps; bpm=60/np.maximum(lags,1e-9)
    m=(bpm>=lo)&(bpm<=hi); k=np.argmax(ac*m)
    # top 3 peaks
    pk,_=ss.find_peaks(ac*m); pk=pk[np.argsort(-ac[pk])][:4]
    return [(round(60/lags[p],1),round(ac[p],2)) for p in pk]
