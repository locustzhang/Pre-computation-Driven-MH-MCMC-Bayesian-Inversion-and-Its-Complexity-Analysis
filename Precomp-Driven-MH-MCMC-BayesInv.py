"""
Precomputation-Driven MH-MCMC: Final Paper Validation  (FIXED v2)
=================================================================
Fix 1 (original): Traditional baseline uses A_cached — no basis_matrix()
  rebuild per iteration. Correct O(NMK) baseline.

Fix 2 (this version): Step sizes now derived from posterior std.
  Original steps=0.06/sqrt(K) was 10-18x larger than posterior std,
  causing near-zero acceptance (K=4: ~0%, K=8: ~1%) and false/true
  non-convergence in Exp-B/D.

  Correct formula (Roberts et al. 1997 RWM optimal):
      steps = (2.38/sqrt(K)) * post_std
  where  post_std = sqrt(diag(sigma^2 * (A^T A)^{-1}))

  This requires steps to be computed AFTER precompute() since Q=A^T A
  is only available then. run_both() now computes and overrides steps
  before sampling.

Model family (paper Eq.2  d = A*theta):
  K=1 : A=[1/cosh(x1)...]^T  (separable/rank-1, paper Eq.4)
  K>=2: A[i,k]=exp(-lk|xi-sk|)  (general linear, paper Eq.2)
"""

import numpy as np, time, os, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

plt.rcParams.update({
    'figure.dpi':300,'font.family':'DejaVu Sans','font.size':9,
    'axes.linewidth':0.8,'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelpad':5,'legend.frameon':True,'legend.framealpha':0.9,
    'legend.fontsize':8,'grid.alpha':0.25,'grid.linewidth':0.4,
    'xtick.direction':'in','ytick.direction':'in',
    'text.usetex':False,'axes.formatter.use_mathtext':True,
})

RESULT_DIR="results_paper"; os.makedirs(RESULT_DIR,exist_ok=True)
SEED=42; N_SAMPLES=80_000; N_BURNIN=16_000; N_CHAINS=3; SIGMA_DEF=0.05


# ─── Model ───────────────────────────────────────────────────────────

class LinearModel:
    """
    d = A*theta,  A in R^{M x K}  (parameter-independent, built once).
    K=1:  A = 1/cosh(x)  (separable rank-1, paper Eq.4)
    K>=2: A[i,k] = exp(-lk|xi-sk|)  (Green's basis, paper Eq.2)

    Precomputed: Q=A^T A, c=2A^T(-d)  -> O(K^2) per iteration.
    Traditional: A cached, A@theta only -> O(MK) per iteration.

    NOTE: self.steps is a placeholder; run_both() overrides it with
    posterior-std-based steps after precompute() is called.
    """
    def __init__(self, K, sigma=SIGMA_DEF):
        self.K = K
        self.sigma = sigma
        rng = np.random.RandomState(0)
        if K == 1:
            self.name = "Separable (K=1)"
            self.theta_true = np.array([np.sqrt((2-1.0+0.5**2/9)/1.0)])
            self.bounds = [(0.5, 2.0)]
            self.steps = np.array([0.012])   # overridden in run_both
            self._fn = "sech"
        else:
            self.name = f"Linear (K={K})"
            self.lambdas = 0.3 + 0.4*rng.rand(K)
            self.sources = np.linspace(-4, 4, K)
            self.theta_true = np.clip(rng.randn(K)*0.4+1.0, 0.2, 2.0)
            self.bounds = [(0.0, 3.0)]*K
            self.steps = np.full(K, 0.06/np.sqrt(K))  # overridden in run_both
            self._fn = "green"

    def basis_matrix(self, x):
        """Build A once — NOT called per MCMC iteration."""
        if self._fn == "sech":
            return (1.0/np.cosh(x)).reshape(-1, 1)
        diff = np.abs(x[:,None] - self.sources[None,:])
        return np.exp(-self.lambdas[None,:] * diff)

    def generate_data(self, x, sigma):
        np.random.seed(SEED)
        return self.basis_matrix(x) @ self.theta_true + sigma*np.random.randn(len(x))

    def precompute(self, x, d_obs):
        """O(MK^2) once. Stores A for traditional path."""
        A = self.basis_matrix(x)
        Q = A.T @ A
        c = 2.0 * (A.T @ (-d_obs))
        c0 = np.dot(d_obs, d_obs)
        return {'Q': Q, 'c': c, 'c0': c0, 'A': A}

    def compute_optimal_steps(self, Q, sigma):
        """
        FIX 2: Compute per-dimension step sizes from posterior std.
        Posterior Cov ~ sigma^2 * (A^T A)^{-1}  (Gaussian likelihood)
        Optimal RWM step (Roberts et al. 1997): 2.38/sqrt(K) * post_std
        This gives acceptance rate ~23% in high dimensions.
        """
        try:
            cov = sigma**2 * np.linalg.inv(Q)
            post_std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        except np.linalg.LinAlgError:
            # Fallback if Q is singular
            post_std = np.full(self.K, sigma / np.sqrt(Q.diagonal().mean()))
        steps = (2.38 / np.sqrt(self.K)) * post_std
        return steps

    def log_like_precomp(self, theta, pc, sigma):
        """O(K^2) per iteration. Constant c0 dropped (cancels in MH ratio)."""
        q = theta @ pc['Q'] @ theta + pc['c'] @ theta
        return -q / (2*sigma**2)

    def log_like_traditional(self, theta, A_cached, d_obs, sigma):
        """O(MK) per iteration. Uses pre-built A_cached — no basis rebuild."""
        r = A_cached @ theta - d_obs
        return -np.dot(r, r) / (2*sigma**2)

    def log_prior(self, theta):
        return 0.0 if all(lo<p<hi for p,(lo,hi) in zip(theta,self.bounds)) else -np.inf


# ─── MCMC engine ─────────────────────────────────────────────────────

def run_mcmc(ll_fn, lp_fn, init, steps,
             n_samples=N_SAMPLES, n_burnin=N_BURNIN):
    d = len(init)
    samples = np.zeros((n_samples, d))
    th = init.copy()
    cur_ll = ll_fn(th); cur_lp = lp_fn(th)
    n_acc = 0
    for i in range(n_samples + n_burnin):
        prop = th + steps * np.random.randn(d)
        lp2 = lp_fn(prop)
        if lp2 == -np.inf:
            if i >= n_burnin: samples[i-n_burnin] = th
            continue
        ll2 = ll_fn(prop)
        if np.log(np.random.rand()) < (lp2+ll2) - (cur_lp+cur_ll):
            th, cur_ll, cur_lp = prop, ll2, lp2
            if i >= n_burnin: n_acc += 1
        if i >= n_burnin: samples[i-n_burnin] = th
    return samples, n_acc/n_samples


def run_both(model, x, d_obs, sigma=SIGMA_DEF):
    """
    FIX 1: A_cached passed to traditional path — no basis_matrix() per iter.
    FIX 2: steps overridden with posterior-std-based optimal steps.
    Both paths use the same steps for a fair comparison.
    """
    pc = model.precompute(x, d_obs)
    A_cached = pc['A']

    # FIX 2: compute optimal steps from Q after precompute
    steps = model.compute_optimal_steps(pc['Q'], sigma)

    # precomputed path
    chains_pre = []; t0 = time.time()
    for ci in range(N_CHAINS):
        np.random.seed(SEED+ci)
        init = model.theta_true + 0.05*np.random.randn(model.K)
        ll_fn = lambda th, _pc=pc, _s=sigma: model.log_like_precomp(th, _pc, _s)
        s, _ = run_mcmc(ll_fn, model.log_prior, init, steps)
        chains_pre.append(s)
    t_pre = time.time()-t0

    # traditional path — A_cached passed, no rebuild
    chains_trad = []; t0 = time.time()
    for ci in range(N_CHAINS):
        np.random.seed(SEED+ci)
        init = model.theta_true + 0.05*np.random.randn(model.K)
        ll_fn = lambda th, _A=A_cached, _d=d_obs, _s=sigma: \
            model.log_like_traditional(th, _A, _d, _s)
        s, _ = run_mcmc(ll_fn, model.log_prior, init, steps)
        chains_trad.append(s)
    t_trad = time.time()-t0

    cp = np.array(chains_pre); ct = np.array(chains_trad)
    mp, sp = _stats(cp); mt, st = _stats(ct)
    return dict(
        M=len(x), K=model.K, name=model.name, sigma=sigma,
        t_pre=t_pre, t_trad=t_trad, speedup=t_trad/t_pre,
        rhat_pre=_rhat(cp), rhat_trad=_rhat(ct),
        mean_pre=mp, std_pre=sp, mean_trad=mt, std_trad=st,
        theta_true=model.theta_true,
        chains_pre=cp, chains_trad=ct,
        steps_used=steps,
    )


def _rhat(chains):
    nc, n, p = chains.shape
    cm = chains.mean(1); cv = chains.var(1, ddof=1)
    W = cv.mean(0); B = n*cm.var(0, ddof=1)
    W = np.maximum(W, 1e-12)
    pv = (n-1)/n*W + B/n
    return np.sqrt(pv/W)

def _stats(chains):
    s = chains.reshape(-1, chains.shape[-1])
    return s.mean(0), s.std(0, ddof=1)


# ─── Experiments ─────────────────────────────────────────────────────

def exp_A():
    print("\n"+"="*60)
    print("Exp-A  K=1 (separable)  multi-scale M")
    print("="*60)
    grid_sizes=[50,100,200,400,800,1600]; time_steps=40
    model = LinearModel(K=1); results = []
    for gs in grid_sizes:
        x = np.linspace(-5,5,gs*time_steps)
        d_obs = model.generate_data(x, SIGMA_DEF)
        print(f"  M={len(x):>6d} ...", end=" ", flush=True)
        r = run_both(model, x, d_obs); results.append(r)
        print(f"speedup={r['speedup']:5.2f}x  theory~{len(x):.0f}x  "
              f"rhat={r['rhat_pre'].max():.3f}  "
              f"{'OK' if r['rhat_pre'].max()<1.2 else 'FAIL'}")
    return results


def exp_B():
    print("\n"+"="*60)
    print("Exp-B  Linear K=2,4,8  M=8000")
    print("  Fix 1: A cached in trad path")
    print("  Fix 2: steps = (2.38/sqrt(K)) * post_std  (RWM optimal)")
    print("="*60)
    K_vals=[2,4,8]; M=8000; x=np.linspace(-5,5,M); results={}
    for K in K_vals:
        model = LinearModel(K=K)
        d_obs = model.generate_data(x, SIGMA_DEF)
        print(f"  K={K:>2d} ...", end=" ", flush=True)
        r = run_both(model, x, d_obs); results[K] = r
        err = np.abs(r['mean_pre'] - r['theta_true']).mean()
        print(f"speedup={r['speedup']:6.2f}x  theory~M/K={M/K:.0f}x  "
              f"rhat={r['rhat_pre'].max():.3f}  err={err:.4f}  "
              f"steps_mean={r['steps_used'].mean():.5f}  "
              f"{'OK' if r['rhat_pre'].max()<1.2 else 'FAIL'}")
    return results


def exp_C(results_A):
    print("\n"+"="*60)
    print("Exp-C  Posterior accuracy  (Theorem 1)")
    print("="*60)
    best = max(results_A, key=lambda r: r['M'])
    K = best['K']; tv = best['theta_true']
    print(f"  M={best['M']}, K={K}")
    print(f"  {'Param':<7} {'True':>9} {'Pre mean':>12} {'Trad mean':>12} {'|Dmean|':>10}")
    print("  "+"-"*54)
    for j in range(K):
        diff = abs(best['mean_pre'][j] - best['mean_trad'][j])
        print(f"  th{j+1:<5} {tv[j]:>9.6f} {best['mean_pre'][j]:>12.8f} "
              f"{best['mean_trad'][j]:>12.8f} {diff:>10.2e}")
    md = max(abs(best['mean_pre'][j]-best['mean_trad'][j]) for j in range(K))
    print(f"\n  Max|Dmean|={md:.2e} -> Theorem 1 "
          f"{'CONFIRMED' if md<1e-4 else 'WARNING'}")
    return best


def exp_D(results_A, results_B):
    print("\n"+"="*60)
    print("Exp-D  R-hat  (Theorem 3)")
    print("="*60)
    print("  [K=1]")
    for r in results_A:
        rp=r['rhat_pre'].max(); rt=r['rhat_trad'].max()
        print(f"  M={r['M']:>6d}  pre={rp:.3f}  trad={rt:.3f}  "
              f"{'OK' if rp<1.2 and rt<1.2 else 'FAIL'}")
    print("  [K=2,4,8]")
    for K,r in results_B.items():
        rp=r['rhat_pre'].max(); rt=r['rhat_trad'].max()
        print(f"  K={K:>2d}  pre={rp:.3f}  trad={rt:.3f}  "
              f"steps_mean={r['steps_used'].mean():.5f}  "
              f"{'OK' if rp<1.2 and rt<1.2 else 'FAIL'}")


def exp_E():
    print("\n"+"="*60)
    print("Exp-E  Noise robustness  K=1, M=8000")
    print("="*60)
    sigmas=[0.02,0.05,0.10,0.20]; M=8000; x=np.linspace(-5,5,M)
    model=LinearModel(K=1); results=[]
    for sigma in sigmas:
        d_obs = model.generate_data(x, sigma)
        print(f"  s={sigma:.2f} ...", end=" ", flush=True)
        r = run_both(model, x, d_obs, sigma=sigma)
        r['sigma'] = sigma; results.append(r)
        err = abs(r['mean_pre'][0]-model.theta_true[0])
        print(f"speedup={r['speedup']:5.2f}x  rhat={r['rhat_pre'].max():.3f}  "
              f"err={err:.5f}  {'OK' if r['rhat_pre'].max()<1.2 else 'FAIL'}")
    return results


def exp_F(results_A, results_B):
    print("\n"+"="*60)
    print("Exp-F  Theory vs Measured  (Remark 1)")
    print("="*60)
    print("  [K=1]")
    print(f"  {'M':>7} {'Theory':>10} {'Measured':>10} {'Ratio':>8}")
    for r in results_A:
        th = r['M']/r['K']
        print(f"  {r['M']:>7d} {th:>10.0f} {r['speedup']:>10.2f} {r['speedup']/th:>8.5f}")
    print("  [K=2,4,8]")
    print(f"  {'K':>5} {'Theory':>10} {'Measured':>10} {'Ratio':>8}")
    for K,r in results_B.items():
        th = r['M']/K
        print(f"  {K:>5d} {th:>10.0f} {r['speedup']:>10.2f} {r['speedup']/th:>8.5f}")
    print("  Ratio < 1: MCMC overhead dominates precomp path (Remark 1).")
    print("  Ratio increases with K as O(K^2) quadratic form cost grows.")


# ─── Figures ─────────────────────────────────────────────────────────

def plot_all(results_A, results_B, best_C, results_E):
    Ms=[r['M'] for r in results_A]
    t_pre=[r['t_pre'] for r in results_A]
    t_trad=[r['t_trad'] for r in results_A]
    speedups=[r['speedup'] for r in results_A]

    # Fig1: time comparison + speedup vs M
    fig, axes = plt.subplots(1,2,figsize=(11,4.2))
    ax = axes[0]
    ax.plot(Ms,t_pre,'o-',color='#1f77b4',lw=1.8,ms=5,label='With precomputation')
    ax.plot(Ms,t_trad,'s-',color='#d62728',lw=1.8,ms=5,label='Without precomputation')
    sc = t_trad[0]/Ms[0]
    ax.plot(Ms,[sc*m for m in Ms],'--',color='#d62728',lw=1,alpha=0.4,label='O(NM) ref.')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('$M$'); ax.set_ylabel('Runtime (s)')
    ax.set_title('(a) Runtime [$K=1$]',fontsize=9)
    ax.legend(); ax.grid(True,which='both',alpha=0.2)
    ax = axes[1]
    th_shape = np.array(Ms,float)/Ms[0]*speedups[0]
    ax.plot(Ms,speedups,'d-',color='#2ca02c',lw=1.8,ms=5,label='Measured')
    ax.plot(Ms,th_shape,'--',color='gray',lw=1.2,label='$M/K$ shape')
    ax.set_xscale('log')
    ax.set_xlabel('$M$'); ax.set_ylabel('Speedup')
    ax.set_title('(b) Speedup [$K=1$]',fontsize=9)
    ax.legend(); ax.grid(True,which='both',alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/time_comparison.png",dpi=300,bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(5.5,4))
    ax.plot(Ms,speedups,'d-',color='#2ca02c',lw=1.8,ms=5,label='Measured speedup')
    ax.plot(Ms,th_shape,'--',color='gray',lw=1.2,label='$M/K$ shape')
    ax.set_xscale('log')
    ax.set_xlabel('$M$'); ax.set_ylabel('Speedup')
    ax.set_title('Speedup vs $M$  [$K_{eff}=1$]',fontsize=10)
    ax.legend(); ax.grid(True,which='both',alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/speedup_vs_M.png",dpi=300,bbox_inches='tight')
    plt.close()
    print("  Saved: time_comparison.png, speedup_vs_M.png")

    # Fig2: speedup vs K + convergence vs K
    K_vals = list(results_B.keys())
    M_fixed = results_B[K_vals[0]]['M']
    sp_meas = [results_B[K]['speedup'] for K in K_vals]
    sp_th   = [M_fixed/K for K in K_vals]
    rp = [results_B[K]['rhat_pre'].max() for K in K_vals]
    rt = [results_B[K]['rhat_trad'].max() for K in K_vals]
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    xs = np.arange(len(K_vals)); bw = 0.35
    ax = axes[0]
    ax.bar(xs-bw/2,sp_th,bw,color='#aec7e8',alpha=0.85,label='Theory $M/K$')
    ax.bar(xs+bw/2,sp_meas,bw,color='#2ca02c',alpha=0.85,label='Measured')
    for i,(th_v,ms_v) in enumerate(zip(sp_th,sp_meas)):
        ax.text(i-bw/2,th_v+2,f'{th_v:.0f}x',ha='center',fontsize=7,color='#1f77b4')
        ax.text(i+bw/2,ms_v+0.5,f'{ms_v:.1f}x',ha='center',fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels([f'$K={k}$' for k in K_vals])
    ax.set_xlabel('$K$'); ax.set_ylabel('Speedup')
    ax.set_title(f'(a) Speedup vs $K$  [$M={M_fixed}$]\nRemark 1: gap=MCMC overhead',fontsize=9)
    ax.legend(); ax.grid(axis='y',alpha=0.25)
    ax = axes[1]
    ax.bar(xs-bw/2,rp,bw,color='#1f77b4',alpha=0.85,label='With precomp.')
    ax.bar(xs+bw/2,rt,bw,color='#d62728',alpha=0.85,label='Without precomp.')
    ax.axhline(1.2,color='#d62728',ls='--',lw=1.2,alpha=0.7,label='Threshold 1.2')
    ax.set_xticks(xs); ax.set_xticklabels([f'$K={k}$' for k in K_vals])
    ax.set_xlabel('$K$'); ax.set_ylabel('Max $\\hat{R}$')
    ax.set_title('(b) Convergence vs $K$  [Theorem 3]',fontsize=9)
    ax.set_ylim(0.95,1.35); ax.legend(); ax.grid(axis='y',alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/speedup_vs_K.png",dpi=300,bbox_inches='tight')
    plt.close()
    print("  Saved: speedup_vs_K.png")

    # Fig3: posterior accuracy
    K = best_C['K']; tv = best_C['theta_true']
    xp = np.arange(K); w = 0.22
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    fig.suptitle(f'Posterior accuracy ($M={best_C["M"]}$, $K={K}$) — Theorem 1',fontsize=10)
    ax = axes[0]
    ax.bar(xp-w,tv,w,color='#7f7f7f',alpha=0.7,label='True $\\theta$')
    ax.bar(xp,best_C['mean_pre'],w,color='#1f77b4',alpha=0.85,
           yerr=2*best_C['std_pre'],capsize=3,error_kw={'lw':1},
           label='With precomp. ($\\pm2\\sigma$)')
    ax.bar(xp+w,best_C['mean_trad'],w,color='#d62728',alpha=0.85,
           yerr=2*best_C['std_trad'],capsize=3,error_kw={'lw':1},
           label='Without precomp. ($\\pm2\\sigma$)')
    ax.set_xticks(xp); ax.set_xticklabels([f'$\\theta_{j+1}$' for j in range(K)])
    ax.set_ylabel('Value'); ax.set_title('(a) Mean vs. true value')
    ax.legend(fontsize=8); ax.grid(axis='y',alpha=0.3)
    ax = axes[1]
    ax.bar(xp-w/2,best_C['std_pre'],w*0.9,color='#1f77b4',alpha=0.85,label='With precomp.')
    ax.bar(xp+w/2,best_C['std_trad'],w*0.9,color='#d62728',alpha=0.85,label='Without precomp.')
    ax.set_xticks(xp); ax.set_xticklabels([f'$\\theta_{j+1}$' for j in range(K)])
    ax.set_ylabel('Std. dev.'); ax.set_title('(b) Uncertainty (must be identical)')
    ax.legend(fontsize=8); ax.grid(axis='y',alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/posterior_accuracy.png",dpi=300,bbox_inches='tight')
    plt.close()
    print("  Saved: posterior_accuracy.png")

    # Fig4: R-hat vs M
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(Ms,[r['rhat_pre'].max() for r in results_A],
            'o-',color='#1f77b4',lw=1.8,ms=5,label='With precomp.')
    ax.plot(Ms,[r['rhat_trad'].max() for r in results_A],
            's--',color='#d62728',lw=1.8,ms=5,alpha=0.7,label='Without precomp.')
    ax.axhline(1.2,color='#d62728',ls='--',lw=1.2,alpha=0.7,label='Threshold 1.2')
    ax.set_xscale('log')
    ax.set_xlabel('$M$'); ax.set_ylabel('Max $\\hat{R}$')
    ax.set_title('Convergence [$K=1$] — Theorem 3',fontsize=10)
    ax.set_ylim(0.97,1.35); ax.legend(); ax.grid(True,which='both',alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/rhat_with_vs_without_precompute.png",dpi=300,bbox_inches='tight')
    plt.close()
    print("  Saved: rhat_with_vs_without_precompute.png")

    # Fig5: noise robustness
    sigmas = [r['sigma'] for r in results_E]
    fig, axes = plt.subplots(1,3,figsize=(13,4))
    fig.suptitle('Noise robustness  ($K=1$, $M=8000$)',fontsize=10)
    for ax,(ys,ttl,yl) in zip(axes,[
        ([r['speedup'] for r in results_E],
         '(a) Speedup vs. $\\sigma$','Speedup'),
        ([r['rhat_pre'].max() for r in results_E],
         '(b) Max $\\hat{R}$ vs. $\\sigma$','Max $\\hat{R}$'),
        ([abs(r['mean_pre'][0]-r['theta_true'][0]) for r in results_E],
         '(c) Error vs. $\\sigma$','$|$mean$-$true$|$'),
    ]):
        ax.plot(sigmas,ys,'o-',color='#2ca02c',lw=1.8,ms=5)
        if 'hat' in yl:
            ax.axhline(1.2,color='#d62728',ls='--',lw=1,alpha=0.7)
            ax.set_ylim(0.97,1.35)
        ax.set_xlabel('$\\sigma$'); ax.set_ylabel(yl)
        ax.set_title(ttl,fontsize=9); ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/noise_robustness.png",dpi=300,bbox_inches='tight')
    plt.close()
    print("  Saved: noise_robustness.png")


# ─── Tables ──────────────────────────────────────────────────────────

def print_tables(results_A, results_B, best_C, results_E):
    print("\n"+"="*72)
    print("TABLE 1  K=1  multi-scale M")
    print("="*72)
    print(f"{'M':>7} {'t_pre':>9} {'t_trad':>10} {'Speedup':>9} "
          f"{'Th.M/K':>8} {'Rhat':>8}")
    print("-"*56)
    for r in results_A:
        print(f"{r['M']:>7d} {r['t_pre']:>9.2f} {r['t_trad']:>10.2f} "
              f"{r['speedup']:>9.2f} {float(r['M']):>8.0f} "
              f"{r['rhat_pre'].max():>8.3f}")

    print("\n"+"="*72)
    print("TABLE 2  K=2,4,8  M=8000  (Fix1: A cached | Fix2: optimal steps)")
    print("="*72)
    print(f"{'K':>5} {'t_pre':>9} {'t_trad':>10} {'Speedup':>9} "
          f"{'Th.M/K':>8} {'Rhat':>8} {'Err':>9}")
    print("-"*65)
    for K,r in results_B.items():
        err = np.abs(r['mean_pre']-r['theta_true']).mean()
        print(f"{K:>5d} {r['t_pre']:>9.2f} {r['t_trad']:>10.2f} "
              f"{r['speedup']:>9.2f} {r['M']/K:>8.0f} "
              f"{r['rhat_pre'].max():>8.3f} {err:>9.4f}")

    print(f"\n"+"="*72)
    print(f"TABLE 3  Posterior equivalence  M={best_C['M']}, K={best_C['K']}")
    print("="*72)
    K = best_C['K']; tv = best_C['theta_true']
    print(f"{'Param':<6} {'True':>10} {'Pre mean':>14} "
          f"{'Trad mean':>14} {'|Dmean|':>11}")
    print("-"*58)
    for j in range(K):
        diff = abs(best_C['mean_pre'][j]-best_C['mean_trad'][j])
        print(f"th{j+1:<4} {tv[j]:>10.6f} {best_C['mean_pre'][j]:>14.9f} "
              f"{best_C['mean_trad'][j]:>14.9f} {diff:>11.3e}")

    print("\n"+"="*72)
    print("TABLE 4  Noise robustness  K=1, M=8000")
    print("="*72)
    print(f"{'sigma':>7} {'Speedup':>9} {'Rhat':>8} {'Err':>12}")
    print("-"*42)
    for r in results_E:
        err = abs(r['mean_pre'][0]-r['theta_true'][0])
        print(f"{r['sigma']:>7.2f} {r['speedup']:>9.2f} "
              f"{r['rhat_pre'].max():>8.3f} {err:>12.5f}")


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED); t0 = time.time()
    print("="*60)
    print("Precomputation-Driven MH-MCMC: Paper Validation (FIXED v2)")
    print(f"  N={N_SAMPLES}  burn-in={N_BURNIN}  chains={N_CHAINS}")
    print("  Fix1: traditional uses A_cached (no A rebuild per iter)")
    print("  Fix2: steps = (2.38/sqrt(K))*post_std  (RWM optimal)")
    print("="*60)

    res_A  = exp_A()
    res_B  = exp_B()
    best_C = exp_C(res_A)
    exp_D(res_A, res_B)
    res_E  = exp_E()
    exp_F(res_A, res_B)

    print("\n-- Figures --"); plot_all(res_A, res_B, best_C, res_E)
    print("\n-- Tables --");  print_tables(res_A, res_B, best_C, res_E)
    print(f"\nTotal: {time.time()-t0:.1f}s  Output: {RESULT_DIR}/")


if __name__ == "__main__":
    main()
