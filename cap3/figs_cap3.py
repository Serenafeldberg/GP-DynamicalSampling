"""Figuras y tabla del Capítulo 3: GP temporal sobre una forzante escalar.
Modelo: w ~ N(0, sw^2 K(rho)), z_n = w_n + eta_n, eta_n ~ N(0, se^2).
Figuras 3.1 y 3.2 en salida/cap3/; la Tabla 3.2 se imprime. El orden de las cuentas importa: la Tabla usa el mismo
generador rng que la Figura 3.2, después de ella."""
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "salida" / "cap3"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
N = 40
t = np.arange(1, N + 1)

def K_ar1(rho):  return rho ** np.abs(t[:, None] - t[None, :])
def K_se(ell):   return np.exp(-0.5 * (t[:, None] - t[None, :]) ** 2 / ell ** 2)

def posterior(z, rho, sw, se):
    Kw = sw**2 * K_ar1(rho)
    c = cho_factor(Kw + se**2 * np.eye(N) + 1e-12*np.eye(N))
    m = Kw @ cho_solve(c, z)
    S = Kw - Kw @ cho_solve(c, Kw)
    return m, np.sqrt(np.clip(np.diag(S), 0, None))

def nlml(th, z):
    rho, sw, se = th[0], np.exp(th[1]), np.exp(th[2])
    Ky = sw**2 * K_ar1(rho) + se**2 * np.eye(N)
    try: c = cho_factor(Ky)
    except (np.linalg.LinAlgError, ValueError): return 1e10
    return 0.5 * z @ cho_solve(c, z) + np.sum(np.log(np.diag(c[0]))) + 0.5*N*np.log(2*np.pi)

def nlml_known(th, z):
    return nlml([th[0], th[1], np.log(se)], z)

def fit_known(z):
    best = None
    for _ in range(8):
        th0 = [rng.uniform(0, 0.99), np.log(rng.uniform(0.02, 0.3))]
        r = minimize(nlml_known, th0, args=(z,), method="L-BFGS-B",
                     bounds=[(0.0, 0.995), (-7, 1)])
        if best is None or r.fun < best.fun: best = r
    return best.x[0], np.exp(best.x[1]), se

def fit(z):
    best = None
    for _ in range(8):
        th0 = [rng.uniform(0, 0.99), np.log(rng.uniform(0.02, 0.3)), np.log(rng.uniform(0.02, 0.3))]
        r = minimize(nlml, th0, args=(z,), method="L-BFGS-B",
                     bounds=[(0.0, 0.995), (-7, 1), (-7, 1)])
        if best is None or r.fun < best.fun: best = r
    th = best.x
    return th[0], np.exp(th[1]), np.exp(th[2])

sw, sn = 0.1, 0.05
se = np.sqrt(2) * sn          # ruido del residuo: 2 sigma_n^2

# ---------- Figura 3.1: muestras del prior ----------
fig, ax = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
for a, (K, tit) in zip(ax, [(np.eye(N), r"Blanco ($\rho=0$)"),
                            (K_ar1(0.9), r"AR(1), $\rho=0{,}9$"),
                            (K_se(4.0), r"SE, $\ell=4$")]):
    L = np.linalg.cholesky(K + 1e-9*np.eye(N))
    for s in range(3):
        a.plot(t, L @ np.random.default_rng(s).standard_normal(N), lw=1.2)
    a.set_title(tit); a.set_xlabel(r"$n$")
ax[0].set_ylabel(r"$w_n/\sigma_w$")
fig.tight_layout(); fig.savefig(OUT / "prior_temporal.pdf")

# ---------- Figura 3.2: posterior sobre una realización ----------
rho = 0.9
w = sw * np.linalg.cholesky(K_ar1(rho)) @ rng.standard_normal(N)
z = w + se * rng.standard_normal(N)
m, s = posterior(z, rho, sw, se)
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.fill_between(t, m - 2*s, m + 2*s, alpha=0.25, label=r"$\hat w_n\pm2\sigma$")
ax.plot(t, w, "k-", lw=1.5, label=r"$w_n$ (verdadera)")
ax.plot(t, z, "o", ms=3.5, color="gray", label=r"$z_n$ (estimación por paso)")
ax.plot(t, m, "C0-", lw=1.5, label=r"$\hat w_n$ (media posterior)")
ax.set_xlabel(r"$n$"); ax.legend(ncol=2, fontsize=8, loc="best")
fig.tight_layout(); fig.savefig(OUT / "posterior_forzante.pdf")

# ---------- Tabla 3.2: estimadores y calibración ----------
R = 300
print("rho_true | per-step | oracle | rho=0.9 fijo | ajustado(sigma_n conocido) | rho_hat [p25,p75] || ajustado(todo)")
for rho in [0.0, 0.5, 0.9]:
    e = {k: [] for k in ["step", "orac", "fix", "fit", "fit3"]}; rh = []; rh2 = []
    for _ in range(R):
        w = sw * np.linalg.cholesky(K_ar1(rho) + 1e-12*np.eye(N)) @ rng.standard_normal(N)
        z = w + se * rng.standard_normal(N)
        rmse = lambda x: np.linalg.norm(x - w) / np.sqrt(N)
        e["step"].append(rmse(z))
        e["orac"].append(rmse(posterior(z, rho, sw, se)[0]))
        e["fix"].append(rmse(posterior(z, 0.9, sw, se)[0]))
        rr, ss, ee = fit_known(z); rh.append(rr)
        e["fit"].append(rmse(posterior(z, rr, ss, ee)[0]))
        r2, s2, e2 = fit(z); rh2.append(r2)
        e["fit3"].append(rmse(posterior(z, r2, s2, e2)[0]))
    med = {k: np.median(v) for k, v in e.items()}
    print(f"{rho:.1f} | {med['step']:.4f} | {med['orac']:.4f} | {med['fix']:.4f} | {med['fit']:.4f} | "
          f"{np.median(rh):.2f} [{np.percentile(rh,25):.2f},{np.percentile(rh,75):.2f}] || "
          f"sin sigma_n: {med['fit3']:.4f} rho {np.median(rh2):.2f} [{np.percentile(rh2,25):.2f},{np.percentile(rh2,75):.2f}]")
