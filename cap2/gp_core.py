"""Núcleo de regresión GP: kernels, posterior y ajuste de hiperparámetros por
verosimilitud marginal. Usado por los scripts figs_cap2_*.py del Capítulo 2."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _col(x):
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def rbf_kernel(X1, X2, ell=1.0, sigma_f=1.0):
    r2 = cdist(_col(X1) / ell, _col(X2) / ell, metric="sqeuclidean")
    return sigma_f**2 * np.exp(-0.5 * r2)


def matern12_kernel(X1, X2, ell=1.0, sigma_f=1.0):
    r = cdist(_col(X1), _col(X2))
    return sigma_f**2 * np.exp(-r / ell)


def matern32_kernel(X1, X2, ell=1.0, sigma_f=1.0):
    r = cdist(_col(X1), _col(X2))
    z = np.sqrt(3) * r / ell
    return sigma_f**2 * (1 + z) * np.exp(-z)


def matern52_kernel(X1, X2, ell=1.0, sigma_f=1.0):
    r = cdist(_col(X1), _col(X2))
    z = np.sqrt(5) * r / ell
    return sigma_f**2 * (1 + z + z**2 / 3) * np.exp(-z)


def gp_posterior(X, y, X_star, kernel=rbf_kernel, ell=1.0, sigma_f=1.0,
                  sigma_n=0.1, jitter=1e-9):
    """Media y covarianza del posterior, ecs. (2.37)-(2.41)."""
    K_XX = kernel(X, X, ell, sigma_f)
    K_Xs = kernel(X, X_star, ell, sigma_f)
    K_ss = kernel(X_star, X_star, ell, sigma_f)
    K_y = K_XX + (sigma_n**2 + jitter) * np.eye(len(X))

    c, lower = cho_factor(K_y, lower=True, check_finite=False)
    alpha = cho_solve((c, lower), y, check_finite=False)
    mu = K_Xs.T @ alpha

    v = solve_triangular(c, K_Xs, lower=True, check_finite=False)
    cov = K_ss - v.T @ v
    return mu, (cov + cov.T) / 2


def nlml(log_params, X, y, kernel=rbf_kernel):
    """Negativa de la log-verosimilitud marginal (2.32), en log-hiperparámetros."""
    ell, sigma_f, sigma_n = np.exp(log_params)
    K_y = kernel(X, X, ell, sigma_f) + (sigma_n**2 + 1e-9) * np.eye(len(X))
    try:
        c, lower = cho_factor(K_y, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e10
    alpha = cho_solve((c, lower), y, check_finite=False)
    log_det = 2 * np.sum(np.log(np.diag(c)))
    return 0.5 * y @ alpha + 0.5 * log_det + 0.5 * len(X) * np.log(2 * np.pi)


def descomponer_nlml(X, y, ell, sigma_f, sigma_n):
    """Términos de ajuste, complejidad y constante de (2.32) (Tabla 2.2)."""
    K_y = rbf_kernel(X, X, ell, sigma_f) + sigma_n**2 * np.eye(len(X))
    c, lower = cho_factor(K_y, lower=True)
    alpha = cho_solve((c, lower), y)
    ajuste = -0.5 * y @ alpha
    complejidad = -np.sum(np.log(np.diag(c)))
    constante = -0.5 * len(X) * np.log(2 * np.pi)
    return ajuste, complejidad, constante, ajuste + complejidad + constante


def ajustar_hiperparametros(X, y, n_reinicios=10, seed=100):
    """Maximiza la verosimilitud marginal con reinicios múltiples (Sección 2.5.2)."""
    mejor = None
    for s in range(n_reinicios):
        x0 = np.random.default_rng(seed + s).uniform(
            np.log([0.1, 0.3, 0.01]), np.log([3, 3, 1]))
        r = minimize(nlml, x0, args=(X, y, rbf_kernel), method="L-BFGS-B")
        if mejor is None or r.fun < mejor.fun:
            mejor = r
    return np.exp(mejor.x), -mejor.fun
