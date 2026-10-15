"""
Reconstrucción de f con forzado: residuos, posterior GP de w, delta_hat y pseudoinversa.

    u_n^(m)  = < y_n , (A^*)^m g_i > + eps ,              m = 0..M
    r_n^(m)  = u_n^(m) - u_{n-1}^(m+1) = <w_n, (A^*)^m g_i> + eta ,   m = 0..M-1
    z = (I_N ⊗ C) w + eta ,  C = [G^T A^m]_{m<M} ,  Cov(eta) = sigma_n^2 (S ⊗ I_p)
    w_hat = (K_t ⊗ Sigma_w C^T) Kz^{-1} z ,  Kz = K_t ⊗ C Sigma_w C^T + sigma_n^2 S ⊗ I_p
    delta_hat_n = A delta_hat_{n-1} + w_hat_n ,   f_hat = O_N^+ (u^(0) - G^* delta_hat)

Todo se trabaja en el espacio de observaciones (tamaño N M p): nunca se arma
K_t ⊗ Sigma_w de (N d)^2. Los estados van aplanados (d = cantidad de píxeles) y
las funciones aceptan un primer eje de realizaciones.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, orth

from . import forzado as fz
from . import sensores as sn


def filas_sensores(G, s, E, subpixel=False):
    """Filas (A^*)^e g_i aplanadas, e = 0..E: arreglo (E+1, p, d)."""
    return sn.bancos_profundidad(G, s, E, subpixel).reshape(E + 1, G.shape[0], -1)


def medir(Y, B, M, sigma_n, rng):
    """u[r, n, m, i] = < y_n , (A^*)^m g_i > + eps para Y de forma (R, N+1, d)."""
    u = np.einsum("rnd,mid->rnmi", Y, B[:M + 1])
    return u + sigma_n * rng.standard_normal(u.shape)


def residuos(u):
    """r_n^(m) = u_n^(m) - u_{n-1}^(m+1), n = 1..N, m = 0..M-1: forma (R, N, M, p)."""
    return u[:, 1:, :-1, :] - u[:, :-1, 1:, :]


def cov_ruido_residuos(N, M):
    """S de (N M) x (N M) con Cov(r) = sigma_n^2 (S ⊗ I_p); índice (n, m), n = 1..N."""
    S = np.zeros((N, M, N, M))
    for n in range(N):
        for m in range(M):
            S[n, m, n, m] = 2.0                     # eps_n^(m) y eps_{n-1}^(m+1)
            if n + 1 < N and m >= 1:                # eps_n^(m) aparece en r_{n+1}^(m-1)
                S[n, m, n + 1, m - 1] = S[n + 1, m - 1, n, m] = -1.0
    return S.reshape(N * M, N * M)


def preparar_gp(C, Sigma, Kt, sigma_n, M):
    """Arma y factoriza Kz = K_t ⊗ C Sigma C^T + sigma_n^2 S ⊗ I_p (tamaño N M p)."""
    N, p = Kt.shape[0], C.shape[0] // M
    CS = C @ Sigma
    Kz = np.kron(Kt, CS @ C.T) + sigma_n ** 2 * np.kron(cov_ruido_residuos(N, M), np.eye(p))
    return dict(cho=cho_factor(Kz, lower=True), CS=CS, Kt=Kt, N=N, pM=C.shape[0])


def media_posterior(gp, z):
    """w_hat (R, N, d) a partir de los residuos z (R, N, M, p)."""
    R = z.shape[0]
    alfa = cho_solve(gp["cho"], z.reshape(R, -1).T).T.reshape(R, gp["N"], gp["pM"])
    return np.einsum("nk,rkj,jd->rnd", gp["Kt"], alfa, gp["CS"])


def acumular(W, aplicar_A, forma):
    """delta_n = A delta_{n-1} + w_n con delta_0 = 0: (R, N, d) -> (R, N+1, d)."""
    R, N, d = W.shape
    delta = fz.acumular_delta(np.moveaxis(W, 1, 0).reshape(N, R, *forma), aplicar_A)
    return np.moveaxis(delta.reshape(N + 1, R, d), 0, 1)


def pinv_O(B, N):
    """O_N^+ con O_N = [G^T A^n]_{n=0..N} de forma ((N+1) p, d)."""
    return np.linalg.pinv(B[:N + 1].reshape(-1, B.shape[-1]))


def reconstruir(Opinv, B0, u0, delta=None):
    """f_hat = O_N^+ (u^(0) - G^T delta_n); delta None = sin corregir. u0 (R, N+1, p)."""
    if delta is not None:
        u0 = u0 - np.einsum("rnd,id->rni", delta, B0)
    return u0.reshape(u0.shape[0], -1) @ Opinv.T


def visibilidad(C, Sigma):
    """Fracción de la varianza del forzado dentro de S_{M-1} = span de las filas de C."""
    Q = orth(C.T)
    return np.trace(Q.T @ Sigma @ Q) / np.trace(Sigma)
