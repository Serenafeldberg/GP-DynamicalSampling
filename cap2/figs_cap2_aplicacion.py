"""Caso de estudio de la Sección 2.7: f(x) = sin(2x) + 0.3x, n = 20
observaciones ruidosas en [-3, 1.5]. Ajusta los hiperparámetros una sola
vez y reutiliza ese ajuste para todo lo que sigue: Fig. 2.5 (prior y
posterior) y los números de 2.7.1; Fig. 2.6 y Tabla 2.2 (efecto de ell);
Fig. 2.7 (superficie de verosimilitud marginal); Tabla 2.3 (verificación
contra scikit-learn y GPy); el experimento de 2.7.4 (pocos datos); la
Observación de 2.6 (jitter) y la Fig. 2.8 (costo de Cholesky, 2.7.5). Figuras en salida/cap2/;
los números se imprimen."""
import os
from pathlib import Path
import time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cap2.gp_core import rbf_kernel, gp_posterior, nlml, ajustar_hiperparametros, descomponer_nlml

OUT = Path(__file__).resolve().parents[1] / "salida" / "cap2"
os.makedirs(OUT, exist_ok=True)


def f_true(x):
    return np.sin(2 * x) + 0.3 * x


# ---------- Ajuste sobre el caso de estudio ----------
rng = np.random.default_rng(1)
n, sigma_n_true = 20, 0.1
X = np.sort(rng.uniform(-3, 1.5, n))
y = f_true(X) + sigma_n_true * rng.standard_normal(n)
X_star = np.linspace(-3, 3, 300)
(ell, sf, sn), lml = ajustar_hiperparametros(X, y)
print(f"sigma_f = {sf:.6f}, ell = {ell:.6f}, sigma_n = {sn:.6f}, LML = {lml:.5f}")

mu, cov = gp_posterior(X, y, X_star, rbf_kernel, ell=ell, sigma_f=sf, sigma_n=sn)
std = np.sqrt(np.clip(np.diag(cov), 0, None))

# ---------- Figura 2.5: prior y posterior ----------
Z = np.random.default_rng(1).standard_normal((len(X_star), 3))
L_prior = np.linalg.cholesky(rbf_kernel(X_star, X_star, ell, sf) + 1e-9 * np.eye(len(X_star)))
L_post = np.linalg.cholesky(cov + 1e-8 * np.eye(len(X_star)))

fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
ax[0].fill_between(X_star, -2 * sf, 2 * sf, alpha=0.2, label=r"$\pm2\sigma$")
ax[0].plot(X_star, np.zeros_like(X_star), "k", lw=1.5, label="Media a priori")
for i in range(3):
    ax[0].plot(X_star, L_prior @ Z[:, i], lw=1)
ax[0].set_xlabel(r"$x$"); ax[0].set_ylabel(r"$f(x)$"); ax[0].legend(fontsize=8)

ax[1].fill_between(X_star, mu - 2 * std, mu + 2 * std, alpha=0.2, label=r"$\bar f\pm2\sigma$")
ax[1].plot(X_star, f_true(X_star), "k--", lw=1.2, label="Función real")
ax[1].plot(X_star, mu, "k", lw=1.5, label="Media a posteriori")
for i in range(3):
    ax[1].plot(X_star, mu + L_post @ Z[:, i], lw=1, alpha=0.8)
ax[1].plot(X, y, "o", color="C3", ms=6, label="Observaciones", zorder=5)
ax[1].set_xlabel(r"$x$"); ax[1].set_ylim(-3.2, 2.2); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "prior_posterior.pdf")

err = np.abs(f_true(X_star) - mu)
print(f"sigma(0) = {std[np.argmin(abs(X_star))]:.3f}")
print(f"sigma(3) = {std[np.argmin(abs(X_star - 3))]:.3f}  (sigma_f = {sf:.3f})")
print(f"error máximo en (1,5; 3] = {err[X_star > 1.5].max():.3f}")

# ---------- Figura 2.6 y Tabla 2.2: efecto de la escala característica ----------
ells = [0.15, ell, 2.0]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
print(f"{'ell':>6} {'ajuste':>9} {'complej.':>9} {'const.':>9} {'total':>9}")
for a, e in zip(ax, ells):
    m, C = gp_posterior(X, y, X_star, rbf_kernel, ell=e, sigma_f=sf, sigma_n=sn)
    s = np.sqrt(np.clip(np.diag(C), 0, None))
    a.fill_between(X_star, m - 2 * s, m + 2 * s, alpha=0.2)
    a.plot(X_star, f_true(X_star), "k--", lw=1.2)
    a.plot(X_star, m, "k", lw=1.5)
    a.plot(X, y, "o", color="C3", ms=5, zorder=5)
    a.set_title(rf"$\ell={e:.2f}$".replace(".", "{,}"))
    a.set_xlabel(r"$x$"); a.set_ylim(-3, 3)
    print(f"{e:6.2f} " + " ".join(f"{v:9.2f}" for v in descomponer_nlml(X, y, e, sf, sn)))
ax[0].set_ylabel(r"$f(x)$")
fig.tight_layout(); fig.savefig(OUT / "efecto_lengthscale_lml.pdf")

# ---------- Figura 2.7: superficie de la log-verosimilitud marginal ----------
le = np.linspace(np.log(0.05), np.log(5), 120)
ls = np.linspace(np.log(0.005), np.log(2), 120)
LML = np.array([[-nlml(np.log([np.exp(a), sf, np.exp(b)]), X, y) for a in le] for b in ls])

fig, ax = plt.subplots(figsize=(6, 4.6))
niveles = np.sort(LML.max() - np.array([60, 30, 15, 8, 4, 2, 1, 0.5]))
cs = ax.contour(np.exp(le), np.exp(ls), LML, levels=niveles, cmap="viridis")
ax.clabel(cs, fmt=lambda v: f"{v:.1f}".replace(".", ","), fontsize=7)
ax.plot(ell, sn, "r*", ms=13, label="Óptimo (L-BFGS-B)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"Escala característica $\ell$"); ax.set_ylabel(r"Desvío del ruido $\sigma_n$")
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "superficie_lml.pdf")

# ---------- Tabla 2.3: verificación contra scikit-learn y GPy ----------
var = np.diag(cov)
filas = {"Propia": (sf, ell, sn, lml, None, None)}
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    k_sk = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(0.01, (1e-6, 1))
    gpr = GaussianProcessRegressor(k_sk, n_restarts_optimizer=10, random_state=0).fit(X[:, None], y)
    p = gpr.kernel_.get_params()
    sk = (np.sqrt(p["k1__k1__constant_value"]), p["k1__k2__length_scale"],
          np.sqrt(p["k2__noise_level"]))
    mu_sk, std_sk = gpr.predict(X_star[:, None], return_std=True)
    filas["scikit-learn"] = (*sk, gpr.log_marginal_likelihood_value_,
                              np.max(abs(mu_sk - mu)), np.max(abs((std_sk**2 - sk[2]**2) - var)))
except ImportError:
    print("scikit-learn no disponible; se omite esa fila.")

try:
    import GPy
    np.random.seed(0)                       # GPy reinicia con el generador global de numpy
    mg = GPy.models.GPRegression(X[:, None], y[:, None], GPy.kern.RBF(1))
    mg.optimize_restarts(10, verbose=False, robust=True)
    mu_g, var_g = mg.predict_noiseless(X_star[:, None])
    filas["GPy"] = (np.sqrt(mg.kern.variance.values[0]), mg.kern.lengthscale.values[0],
                     np.sqrt(mg.Gaussian_noise.variance.values[0]), mg.log_likelihood(),
                     np.max(abs(mu_g[:, 0] - mu)), np.max(abs(var_g[:, 0] - var)))
except ImportError:
    print("GPy no disponible; se omite esa fila.")

print(f"{'':14}{'sigma_f':>10}{'ell':>10}{'sigma_n':>10}{'LML':>10}{'dif media':>12}{'dif var':>12}")
for nombre, (a, b, c, d, e, f) in filas.items():
    extra = "" if e is None else f"{e:12.1e}{f:12.1e}"
    print(f"{nombre:14}{a:10.6f}{b:10.6f}{c:10.6f}{d:10.5f}{extra}")


# ---------- Sección 2.7.4: efecto de la cantidad de datos ----------
def experimento_pocos_datos(n_, seed):
    r_ = np.random.default_rng(seed)
    Xa = np.sort(r_.uniform(-3, 1.5, n_))
    ya = f_true(Xa) + sigma_n_true * r_.standard_normal(n_)
    (ell_, sf_, sn_), _ = ajustar_hiperparametros(Xa, ya)
    m, C = gp_posterior(Xa, ya, X_star, rbf_kernel, ell=ell_, sigma_f=sf_, sigma_n=sn_)
    s = np.sqrt(np.clip(np.diag(C), 0, None))
    region = X_star <= 1.5
    razon = np.abs(f_true(X_star) - m)[region] / s[region]
    print(f"n = {n_}: sigma_n estimado = {sn_:.3f} (verdadero {sigma_n_true}), ell = {ell_:.3f}")
    print(f"  cobertura de la banda ±2σ en la región con datos: {np.mean(razon <= 2):.0%}")
    print(f"  beta necesario para cubrir todos los puntos: {razon.max():.1f}")


experimento_pocos_datos(10, 0)

# ---------- Observación 2.7: jitter y número de condición ----------
Xc = np.linspace(-3, 3, 50)
Kc = rbf_kernel(Xc, Xc, 1.0, 1.0)
for eps in [0, 1e-6, 1e-2]:
    print(f"epsilon = {eps:g}: número de condición = {np.linalg.cond(Kc + eps * np.eye(50)):.2e}")

# ---------- Figura 2.8: costo de la factorización de Cholesky ----------
ns = [250, 500, 1000, 2000, 4000]
xr = np.random.default_rng(3).uniform(-3, 3, max(ns))
tiempos = []
for n_ in ns:
    Kn = rbf_kernel(xr[:n_], xr[:n_], 1.0, 1.0) + 0.01 * np.eye(n_)
    medidas = []
    for _ in range(3):
        t0 = time.perf_counter()
        np.linalg.cholesky(Kn)
        medidas.append(time.perf_counter() - t0)
    tiempos.append(np.median(medidas))
pendiente = np.polyfit(np.log(ns[2:]), np.log(tiempos[2:]), 1)[0]

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(ns, tiempos, "o-", label="Tiempo medido")
ax.loglog(ns, tiempos[-1] * (np.array(ns) / ns[-1]) ** 3, "k--", lw=1, label=r"Referencia $\propto n^3$")
ax.set_xlabel(r"Número de datos $n$"); ax.set_ylabel("Tiempo de factorización [s]"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "costo_cholesky.pdf")
print(f"n=250: {tiempos[0]*1e3:.2f} ms | n=4000: {tiempos[-1]:.3f} s | cociente: {tiempos[-1]/tiempos[0]:.0f}")
print(f"pendiente log-log (n >= 1000): {pendiente:.2f}")
