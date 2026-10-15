"""Figuras del Capítulo 2 (salida/cap2/) que no dependen de un ajuste de hiperparámetros:
condicionamiento gaussiano y bloques de la covarianza conjunta (Sección 2.1,
Fig. 2.1-2.2), y muestras del prior con las familias RBF y Matérn
(Sección 2.4, Fig. 2.3-2.4)."""
import os
from pathlib import Path
import numpy as np
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cap2.gp_core import rbf_kernel, matern12_kernel, matern32_kernel, matern52_kernel

OUT = Path(__file__).resolve().parents[1] / "salida" / "cap2"
os.makedirs(OUT, exist_ok=True)

# ---------- Figura 2.1: efecto del condicionamiento (caso escalar) ----------
mu_a, mu_b = 0.0, 0.0
Sigma_aa, Sigma_bb, Sigma_ab = 1.0, 1.0, 0.8
x_b = 1.5

mu_cond = mu_a + Sigma_ab / Sigma_bb * (x_b - mu_b)
var_cond = Sigma_aa - Sigma_ab**2 / Sigma_bb

x = np.linspace(-3, 3, 500)
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(x, norm.pdf(x, mu_a, np.sqrt(Sigma_aa)), lw=2, label=r"Marginal $p(x_a)$")
ax.plot(x, norm.pdf(x, mu_cond, np.sqrt(var_cond)), lw=2,
        label=rf"Condicional $p(x_a\mid x_b={x_b:g})$".replace(".", "{,}"))
ax.axvline(mu_cond, ls="--", lw=1, color="C1", alpha=0.7)
ax.set_xlabel(r"$x_a$"); ax.set_ylabel("densidad"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "marginal_vs_condicional.pdf")

# ---------- Figura 2.2: bloques de la covarianza conjunta ----------
rng = np.random.default_rng(0)
ell, sigma_f = 1.0, 1.0
X = np.sort(rng.uniform(-3, 3, 8))
X_star = np.linspace(-3, 3, 350)

K_y = rbf_kernel(X, X, ell, sigma_f) + 0.05**2 * np.eye(len(X))
K_Xs = rbf_kernel(X, X_star, ell, sigma_f)
K_ss = rbf_kernel(X_star, X_star, ell, sigma_f)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, M, title in zip(axes, [K_y, K_Xs, K_ss],
                         [r"$K_{XX}+\sigma_n^2I$", r"$K_{X*}$", r"$K_{**}$"]):
    im = ax.imshow(M, aspect="auto", cmap="magma")
    ax.set_title(f"{title}\nforma {M.shape}")
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout(); fig.savefig(OUT / "bloques_covarianza.pdf")

# ---------- Figura 2.3 y 2.4: muestras del prior ----------
Xp = np.linspace(-5, 5, 250)


def muestras_prior(kernel, Z, **kw):
    K = kernel(Xp, Xp, **kw) + 1e-9 * np.eye(len(Xp))
    L = np.linalg.cholesky(K)
    return (L @ Z).T

Z = np.random.default_rng(7).standard_normal((len(Xp), 3))
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
for ax, e in zip(axes[0], [0.25, 0.8, 2.0]):
    for s in muestras_prior(rbf_kernel, Z, ell=e, sigma_f=1.0):
        ax.plot(Xp, s, lw=1.5)
    ax.set_title(rf"RBF: $\ell={e}$".replace(".", "{,}"))
for ax, sf in zip(axes[1], [0.5, 1.0, 2.0]):
    for s in muestras_prior(rbf_kernel, Z, ell=0.8, sigma_f=sf):
        ax.plot(Xp, s, lw=1.5)
    ax.set_title(rf"RBF: $\sigma_f={sf}$".replace(".", "{,}"))
for ax in axes[1]:
    ax.set_xlabel(r"$x$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$f(x)$")
fig.tight_layout(); fig.savefig(OUT / "rbf_lengthscale.pdf")

Z2 = np.random.default_rng(3).standard_normal((len(Xp), 2))
specs = [(r"$\nu=1/2$", matern12_kernel), (r"$\nu=3/2$", matern32_kernel),
         (r"$\nu=5/2$", matern52_kernel), (r"$\nu\to\infty$ (RBF)", rbf_kernel)]
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey=True)
for ax, (name, kernel) in zip(axes.ravel(), specs):
    for s in muestras_prior(kernel, Z2, ell=1.0, sigma_f=1.0):
        ax.plot(Xp, s, lw=1.6)
    ax.set_title(name)
for ax in axes[1]:
    ax.set_xlabel(r"$x$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$f(x)$")
fig.tight_layout(); fig.savefig(OUT / "matern_suavidad.pdf")
