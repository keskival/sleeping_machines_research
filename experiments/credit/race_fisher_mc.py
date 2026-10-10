import numpy as np
rng=np.random.default_rng(0)
# race of K Weibull-shaped clocks h_i(t)=exp(th_i)*k_i*t^(k_i-1), observed until first event or censoring at T
K=4; th=np.array([0.3,-0.5,0.1,-1.2]); kk=np.array([0.7,1.0,1.5,2.5]); T=1.5; N=400000
# simulate each clock's first firing time: H_i(t)=exp(th_i) t^k_i ; t_i = (E/exp(th_i))^(1/k_i)
E=rng.exponential(size=(N,K)); ti=(E/np.exp(th))**(1/kk); tau=ti.min(1); win=ti.argmin(1)
cens=tau>T; tau=np.minimum(tau,T)
H=np.exp(th)[None]*tau[:,None]**kk[None]                     # integrated hazard up to tau
Nw=np.zeros((N,K)); Nw[np.arange(N)[~cens],win[~cens]]=1
score=Nw-H                                                    # d loglik / d th_i
F=score.T@score/N; EN=Nw.mean(0)
print('mean score', score.mean(0).round(4))
print('Fisher\n', F.round(4)); print('E[N_i]', EN.round(4))
print('max |offdiag|', np.abs(F-np.diag(np.diag(F))).max().round(5), ' max |diag-E[N]|', np.abs(np.diag(F)-EN).max().round(5))
# contrast: softmax over which-clock only (no timing): Fisher = diag(p)-pp^T (singular)
p=np.bincount(win[~cens],minlength=K)/ (~cens).sum()
print('softmax-only Fisher eigenvalues', np.linalg.eigvalsh(np.diag(p)-np.outer(p,p)).round(4))
