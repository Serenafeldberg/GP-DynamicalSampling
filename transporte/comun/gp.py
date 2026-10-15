"""
Posterior GP del forzado e identidad exacta del error, en los dos espacios reducidos del ejemplo de transporte.

  - Soporte (GPSoporte, exacto_soporte): w_n restringido a los píxeles S de la fuente y solo los sensores cuyas filas
    (A^*)^m g_i (m < M) tocan S; espacio de observaciones (Kz de N M p_a).
  - Pesos (GPPesos, exacto_pesos): K_w = L L^T con L = L_t ⊗ R_S; w_hat = L (I + Q)^{-1} L^T Psi^T R^{-1} z con
    Q = L^T Psi^T R^{-1} Psi L (tamaño N |S|, sin importar p; útil cuando los sensores auxiliares no son puntuales).
Ambas se validan contra reconstruccion.py y error_exacto.py en 16 x 16 (validar_soporte, validar_pesos).
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from . import config as cfg
from . import operadores as op
from . import forzado as fz
from . import sensores as sn
from . import reconstruccion as rc
from . import error_exacto as ee
from .estilo import ESTIMADORES
from .fourier import bancos_ad, DELTA

S = (1, 0)


def medir(Y, B, M, sigma_n, rng):
    """Como rc.medir, pero con un producto matricial: u (R, N+1, M+1, p)."""
    R, N1, d = Y.shape
    u = (Y.reshape(-1, d) @ B[:M + 1].reshape(-1, d).T).reshape(R, N1, M + 1, B.shape[1])
    return u + sigma_n * rng.standard_normal(u.shape)


def reconstruir(Opinv, B0, u0, delta=None):
    """Como rc.reconstruir, con productos matriciales."""
    if delta is not None:
        u0 = u0 - delta @ B0.T
    return u0.reshape(u0.shape[0], -1) @ Opinv.T


class GPSoporte:
    """Posterior de w restringido al soporte S y a los sensores activos (espacio de observaciones)."""

    def __init__(self, B, M, idx, Sigma_S, Kt, sigma_n):
        N = Kt.shape[0]
        filas = B[:M][:, :, idx]                                     # (M, p, |S|)
        self.act = np.any(filas != 0, axis=(0, 2))                   # sensores que tocan S
        self.C = filas[:, self.act].reshape(-1, len(idx))            # (M p_a, |S|), orden (m, i)
        self.idx, self.Kt, self.N, self.M, self.p_a = idx, Kt, N, M, int(self.act.sum())
        self.CS = self.C @ Sigma_S
        Kz = np.kron(Kt, self.CS @ self.C.T)                         # K_t ⊗ C Sigma C^T
        Kz4 = Kz.reshape(N * M, self.p_a, N * M, self.p_a)
        Sr, i = rc.cov_ruido_residuos(N, M), np.arange(self.p_a)
        for a, b in zip(*np.nonzero(Sr)):                            # + sigma_n^2 S ⊗ I_p, en el lugar
            Kz4[a, i, b, i] += sigma_n ** 2 * Sr[a, b]
        self.cho = cho_factor(Kz, lower=True, overwrite_a=True)
        self.dim = Kz.shape[0]

    def media(self, z, d):
        """w_hat (R, N, d) desde los residuos z (R, N, M, p)."""
        R = z.shape[0]
        za = z[..., self.act].reshape(R, -1)
        alfa = cho_solve(self.cho, za.T).T.reshape(R, self.N, -1)
        W = np.zeros((R, self.N, d))
        W[..., self.idx] = np.einsum("nk,rks->rns", self.Kt, alfa @ self.CS)
        return W


def exacto_soporte(B, Opinv, gp, Sigma_S, sigma_n):
    """E||f_hat - f||^2 (sin corregir, GP, oráculo) por la identidad exacta, en el espacio del soporte.

    T_S[:, k] = sum_{n >= k} O^+_n G^T A^{n-k} restringido a S;  T H = (T_S K_w Psi_a^T) Kz^{-1};
    A_w = -T_S + T H Psi_a ;  A_eps = T H D - O^+ S_u ;  K_w = (L_t ⊗ R_S)(L_t ⊗ R_S)^T.
    """
    N, M, idx = gp.N, gp.M, gp.idx
    d, p = Opinv.shape[0], B.shape[1]
    O3 = Opinv.reshape(d, N + 1, p)
    TS = np.zeros((d, N, len(idx)))
    for k in range(1, N + 1):
        for n in range(k, N + 1):
            TS[:, k - 1] += O3[:, n] @ B[n - k][:, idx]
    Lt, Rs = np.linalg.cholesky(gp.Kt), fz.raiz_psd(Sigma_S)
    energia = lambda X: np.sum((np.einsum("dnj,nk->dkj", X, Lt) @ Rs) ** 2)       # tr(X K_w X^T)
    ruido = sigma_n ** 2 * np.sum(Opinv ** 2)

    X = np.einsum("dnq,nk->dkq", TS @ gp.CS.T, gp.Kt).reshape(d, -1)               # T_S K_w Psi_a^T
    Y = cho_solve(gp.cho, X.T).T                                                   # T H
    Aw = (Y.reshape(d, N, -1) @ gp.C) - TS
    Y4 = Y.reshape(d, N, M, gp.p_a)
    YD = np.zeros((d, N + 1, M + 1, gp.p_a))
    YD[:, 1:, :M] += Y4                                                            # +eps_n^(m) en r_n^(m)
    YD[:, :N, 1:] -= Y4                                                            # -eps_n^(m) en r_{n+1}^(m-1)
    YD[:, :, 0] -= O3[:, :, gp.act]                                                # - O^+ S_u (sensores activos)
    ruido_gp = sigma_n ** 2 * (np.sum(YD ** 2) + np.sum(O3[:, :, ~gp.act] ** 2))
    return dict(sin_corregir=energia(TS) + ruido, gp=energia(Aw) + ruido_gp, oraculo=ruido)


def validar_soporte(Lv=16, sep=4, N=4, M=2, ell=2.0, sigma_n=1e-3, sw=0.05):
    """En un caso chico: GPSoporte == reconstruccion.py y exacto_soporte == error_exacto.mse_exacto."""
    forma, d = (Lv, Lv), Lv * Lv
    G = sn.banco(Lv, sep, 1, [("delta",)])
    B = rc.filas_sensores(G, S, max(N, M))
    Opinv = rc.pinv_O(B, N)
    v = fz.ventana_gaussiana(Lv, (5, 8), 1.5)
    Sigma = fz.sigma_espacial(np.kron(fz.kernel_se_1d(Lv, ell), fz.kernel_se_1d(Lv, ell)), v, sw)
    Kt = fz.kernel_temporal(N, cfg.RHO)
    idx = np.arange(d)                                    # soporte completo: la identidad debe ser exacta
    gp = GPSoporte(B, M, idx, Sigma[np.ix_(idx, idx)], Kt, sigma_n)
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((3, N + 1, d))
    u = rc.medir(Y, B, M, sigma_n, np.random.default_rng(1))
    assert np.allclose(u, medir(Y, B, M, sigma_n, np.random.default_rng(1)))
    z = rc.residuos(u)
    ref = rc.media_posterior(rc.preparar_gp(B[:M].reshape(-1, d), Sigma, Kt, sigma_n, M), z)
    e1 = np.max(np.abs(gp.media(z, d) - ref)) / np.max(np.abs(ref))
    delta = rng.standard_normal((3, N + 1, d))
    e2 = np.max(np.abs(reconstruir(Opinv, B[0], u[:, :, 0], delta) - rc.reconstruir(Opinv, B[0], u[:, :, 0], delta)))
    A = ee.matriz_A(lambda w: op.A(w, S), forma)
    ex_d = ee.mse_exacto(A, B[0], Sigma, Kt, M, N, sigma_n)
    ex_s = exacto_soporte(B, Opinv, gp, Sigma, sigma_n)
    e3 = max(abs(ex_s[k] - ex_d[k]) / ex_d[k] for k in ESTIMADORES)
    assert e1 < 1e-8 and e2 < 1e-10 and e3 < 1e-8, (e1, e2, e3)
    return e1, e2, e3


class GPPesos:
    """Posterior de w en el espacio de pesos restringido al soporte S (tamaño N |S|, sin importar p).

    K_w = L L^T con L = L_t ⊗ R_S;  w_hat = L (I + Q)^{-1} L^T Psi^T R^{-1} z,  Q = L^T Psi^T R^{-1} Psi L,
    R = sigma_n^2 (S ⊗ I_p). Exacto (no descarta sensores) y bien condicionado (I + PSD).
    """

    def __init__(self, B, M, idx, Sigma_S, Kt, sigma_n):
        N, nS = Kt.shape[0], len(idx)
        self.N, self.M, self.idx, self.nS, self.Kt = N, M, idx, nS, Kt
        self.C = B[:M][:, :, idx]                                       # (M, p, |S|)
        self.Sinv4 = (np.linalg.inv(rc.cov_ruido_residuos(N, M)) / sigma_n ** 2).reshape(N, M, N, M)
        lam, U = np.linalg.eigh(Sigma_S)
        self.Rs = U * np.sqrt(np.clip(lam, 0, None))
        self.Lt = np.linalg.cholesky(Kt)
        self.Gram = {(m, k): self.C[m].T @ self.C[k] for m in range(M) for k in range(M)}
        Gt = {mk: self.Rs.T @ g @ self.Rs for mk, g in self.Gram.items()}
        coef = np.einsum("na,nmqk,qb->abmk", self.Lt, self.Sinv4, self.Lt)
        Q = np.zeros((N * nS, N * nS))
        for a in range(N):
            for b in range(N):
                Q[a * nS:(a + 1) * nS, b * nS:(b + 1) * nS] = sum(coef[a, b, m, k] * Gt[(m, k)] for (m, k) in Gt)
        Q[np.diag_indices_from(Q)] += 1.0
        self.cho = cho_factor(Q, lower=True, overwrite_a=True)

    def psi_t_rinv(self, z):
        """Psi^T R^{-1} z: (R, N, M, p) -> (R, N, |S|)."""
        y = np.einsum("nmqk,rqki->rnmi", self.Sinv4, z)
        R = z.shape[0]
        return (y.reshape(R, self.N, -1) @ self.C.reshape(-1, self.nS))

    def media(self, z, d):
        R = z.shape[0]
        beta = np.einsum("na,rns->ras", self.Lt, self.psi_t_rinv(z) @ self.Rs).reshape(R, -1)
        gam = cho_solve(self.cho, beta.T).T.reshape(R, self.N, self.nS)
        W = np.zeros((R, self.N, d))
        W[..., self.idx] = np.einsum("na,ras->rns", self.Lt, gam @ self.Rs.T)
        return W

    def W_completa(self):
        """Psi^T R^{-1} Psi (N |S| x N |S|) por bloques: W_{nn'} = sum_{m,m'} Sinv[n,m,n',m'] C_m^T C_m'."""
        N, nS = self.N, self.nS
        Wm = np.zeros((N * nS, N * nS))
        for n in range(N):
            for q in range(N):
                Wm[n * nS:(n + 1) * nS, q * nS:(q + 1) * nS] = sum(self.Sinv4[n, m, q, k] * g
                                                                    for (m, k), g in self.Gram.items())
        return Wm


def T_soporte(B, Opinv, idx, N):
    """T_S[:, k] = sum_{n >= k} O^+_n G^T A^{n-k} restringido a S: (d, N, |S|)."""
    d, p = Opinv.shape[0], B.shape[1]
    O3 = Opinv.reshape(d, N + 1, p)
    TS = np.zeros((d, N, len(idx)))
    for k in range(1, N + 1):
        for n in range(k, N + 1):
            TS[:, k - 1] += O3[:, n] @ B[n - k][:, idx]
    return TS


def exacto_pesos(Opinv, gp, TS, sigma_n):
    """E||f_hat - f||^2 (sin corregir, GP, oráculo) por la identidad exacta, en el espacio de pesos.

    T H = V Psi^T R^{-1} con V = T_S L (I+Q)^{-1} L^T;  A_w = V W - T_S (W = Psi^T R^{-1} Psi);
    ||T H D||_F^2 = sigma_n^{-2} tr(V W V^T) porque D D^T = R / sigma_n^2;  el cruce con O^+ S_u usa
    solo las columnas m = 0 de T H (eps_n^(0) entra únicamente en r_n^(0)).
    """
    N, M, nS, Lt, Rs = gp.N, gp.M, gp.nS, gp.Lt, gp.Rs
    d = Opinv.shape[0]
    energia = lambda X: np.sum((np.einsum("dns,na->das", X, Lt) @ Rs) ** 2)          # tr(X K_w X^T)
    ruido = sigma_n ** 2 * np.sum(Opinv ** 2)
    X = (np.einsum("dns,na->das", TS, Lt) @ Rs).reshape(d, -1)                       # T_S L
    Y = cho_solve(gp.cho, X.T).T.reshape(d, N, nS)
    V = np.einsum("na,das->dns", Lt, Y @ Rs.T)                                        # T_S L (I+Q)^{-1} L^T
    del X, Y
    Wm = gp.W_completa()
    VW = (V.reshape(d, -1) @ Wm).reshape(d, N, nS)
    del Wm
    traza = np.sum(V * VW)
    O3 = Opinv.reshape(d, N + 1, -1)
    cruce = 0.0
    VC = {(q, k): V[:, q] @ gp.C[k].T for q in range(N) for k in range(M)}          # (d, p)
    for n in range(N):
        Zn = sum(gp.Sinv4[n, 0, q, k] * VC[(q, k)] for (q, k) in VC)                 # columnas (n+1, m = 0) de T H
        cruce += np.sum(Zn * O3[:, n + 1])
    ruido_gp = sigma_n ** 2 * (traza / sigma_n ** 2 - 2 * cruce + np.sum(Opinv ** 2))
    return dict(sin_corregir=energia(TS) + ruido, gp=energia(VW - TS) + ruido_gp, oraculo=ruido)


def validar_pesos(Lv=16, N=4, M=2, s=0.75, nu=0.1, sigma_n=1e-3):
    """En 16 x 16: GPPesos == reconstruccion.py y exacto_pesos == error_exacto.mse_exacto (A general)."""
    d = Lv * Lv
    G = sn.banco(Lv, cfg.SEP, 1, DELTA)
    B = bancos_ad(G, s, nu, max(N, M))
    Opinv = rc.pinv_O(B, N)
    v = fz.ventana_gaussiana(Lv, (5, 8), 1.5)
    K1 = fz.kernel_se_1d(Lv, 2.0)
    Sigma = fz.sigma_espacial(np.kron(K1, K1), v, 0.05)
    Kt = fz.kernel_temporal(N, cfg.RHO)
    idx = np.arange(d)
    gp = GPPesos(B, M, idx, Sigma, Kt, sigma_n)
    rng = np.random.default_rng(0)
    z = rc.residuos(rc.medir(rng.standard_normal((3, N + 1, d)), B, M, sigma_n, rng))
    ref = rc.media_posterior(rc.preparar_gp(B[:M].reshape(-1, d), Sigma, Kt, sigma_n, M), z)
    e1 = np.max(np.abs(gp.media(z, d) - ref)) / np.max(np.abs(ref))
    A = op.A_ad(np.eye(d).reshape(d, Lv, Lv), (s, 0), nu).reshape(d, d).T
    ex_d = ee.mse_exacto(A, B[0], Sigma, Kt, M, N, sigma_n)
    ex_p = exacto_pesos(Opinv, gp, T_soporte(B, Opinv, idx, N), sigma_n)
    e2 = max(abs(ex_p[k] - ex_d[k]) / ex_d[k] for k in ESTIMADORES)
    assert e1 < 1e-8 and e2 < 1e-8, (e1, e2)
    return e1, e2
