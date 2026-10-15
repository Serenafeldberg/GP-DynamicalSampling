"""
Esquema 1D (y piezas compartidas con el 2D de 32 x 32) del Cap. 4 aplicado al transporte: simulación, estimadores
(sin corregir, GP, oráculo), barrido en fuente / ell / M y comparación por presupuesto de sensores.
Error relativo ||f_hat - f|| / ||f||; la identidad exacta se contrasta con Monte Carlo (|z| < 4).
"""
from math import gcd

import numpy as np

from . import config as cfg
from . import operadores as op
from . import forzado as fz
from . import sensores as sn
from . import reconstruccion as rc
from . import error_exacto as ee
from .imagen import imagen_f
from .estilo import ESTIMADORES

S = (1, 0)


def simular(rng, R, f, forma, B, M, sigma_n, muestrear_w):
    """R realizaciones: forzado w (R, N, d), delta verdadero (R, N+1, d) y mediciones u (R, N+1, M+1, p)."""
    W = muestrear_w(rng, R)
    N = W.shape[1]
    delta = rc.acumular(W, lambda v: op.A(v, S), forma)
    Anf = np.stack([op.A_pot(f.reshape(forma), S, n).ravel() for n in range(N + 1)])
    u = rc.medir(Anf[None] + delta, B, M, sigma_n, rng)
    return W, delta, u


def estimar(u, delta, B, M, Sigma, Kt, sigma_n, forma, Opinv):
    """f_hat de los estimadores: dict nombre -> (R, d). Con M = 0 solo hay 'sin corregir'."""
    u0 = u[:, :, 0, :]
    salida = dict(sin_corregir=rc.reconstruir(Opinv, B[0], u0))
    if M == 0:
        return salida
    gp = rc.preparar_gp(B[:M].reshape(-1, B.shape[-1]), Sigma, Kt, sigma_n, M)
    delta_hat = rc.acumular(rc.media_posterior(gp, rc.residuos(u)), lambda v: op.A(v, S), forma)
    salida.update(gp=rc.reconstruir(Opinv, B[0], u0, delta_hat), oraculo=rc.reconstruir(Opinv, B[0], u0, delta))
    return salida


def muestreador(Sigma, Kt):
    """w ~ N(0, K_t ⊗ Sigma) a partir de raíces de cada factor."""
    Lt, Rs = np.linalg.cholesky(Kt), fz.raiz_psd(Sigma)
    return lambda rng, R: np.einsum("nk,rkj,dj->rnd", Lt, rng.standard_normal((R, Kt.shape[0], Sigma.shape[0])), Rs)


def sigma_w_usado(Sigma1, Kt, forma, norma_f):
    """sigma_w efectivo y la razón RMS ||delta_N|| / ||f|| que produce.

    Sigma1 es Sigma_w con sigma_w = 1. Si RAZON_FORZADO es None se usa SIGMA_W de config;
    si no, se elige sigma_w para que sqrt(E||delta_N||^2) / ||f|| = RAZON_FORZADO.
    """
    e1 = fz.energia_delta_N(Sigma1, Kt, lambda v, j: op.A_pot(v, S, j), forma)
    sw = cfg.SIGMA_W if cfg.RAZON_FORZADO is None else cfg.RAZON_FORZADO * norma_f / np.sqrt(e1)
    return sw, sw * np.sqrt(e1) / norma_f


def sensores_1D(L, sep):
    """Deltas cada sep píxeles en un renglón: G de forma (p, 1, L)."""
    p = L // sep
    G = np.zeros((p, 1, L))
    G[np.arange(p), 0, np.arange(0, L, sep)] = 1.0
    return G


def fila_f(L):
    """Una fila de la imagen (la de la fuente), interpolada a L puntos si hace falta."""
    f = imagen_f(cfg.L, cfg.IMAGEN)[cfg.FUENTE_CENTRO[1]]
    return f if L == cfg.L else np.interp(np.linspace(0, cfg.L - 1, L), np.arange(cfg.L), f)


def ventana_1D(L, c, ancho):
    x = np.arange(L)
    return np.exp(-(x - c) ** 2 / (2 * ancho ** 2))


def comparar_mc(e2, ex, contexto, chequeos):
    """Assert: la media MC de ||f_hat - f||^2 coincide con la identidad exacta (|z| < 4)."""
    for k, v in e2.items():
        z = (v.mean() - ex[k]) / (v.std(ddof=1) / np.sqrt(len(v)))
        chequeos.append(z)
        assert abs(z) < 4, f"MC != identidad ({contexto}, {k}): z = {z:.1f}"



# ============================================================== 1D: barrido en fuente, ell y M
def barrido_1D(ancho, chequeos):
    L, N, sep, sn_ = cfg.L_1D, cfg.N_REC, cfg.SEP, cfg.SIGMA_N
    forma = (1, L)
    G = sensores_1D(L, sep)
    p = G.shape[0]
    B = rc.filas_sensores(G, S, max(N, max(cfg.M_1D)))
    Opinv = rc.pinv_O(B, N)
    A = ee.matriz_A(lambda v: op.A(v, S), forma)
    f = fila_f(L)
    norma_f = np.linalg.norm(f)
    Kt = fz.kernel_temporal(N, cfg.RHO)

    filas = []
    for nombre_ell, ell in cfg.ELL_BARRIDO.items():
        for nombre_f, c in cfg.FUENTES_1D.items():
            Sigma1 = fz.sigma_espacial(fz.kernel_se_1d(L, ell), ventana_1D(L, c, ancho), 1.0)
            sw, razon = sigma_w_usado(Sigma1, Kt, forma, norma_f)
            Sigma = sw ** 2 * Sigma1
            for M in cfg.M_1D:
                rng = np.random.default_rng(cfg.SEED)                # mismos números para todas las combinaciones
                W, delta, u = simular(rng, cfg.REPS_1D, f, forma, B, M, sn_, muestreador(Sigma, Kt))
                f_hat = estimar(u, delta, B, M, Sigma, Kt, sn_, forma, Opinv)
                e2 = {k: np.sum((v - f) ** 2, axis=1) for k, v in f_hat.items()}
                ex = ee.mse_exacto(A, B[0], Sigma, Kt, M, N, sn_)
                assert np.allclose(sn_ ** 2 * np.kron(rc.cov_ruido_residuos(N, M), np.eye(p)), sn_ ** 2 * ex["DDt"])
                assert np.allclose(ex["C"], B[:M].reshape(-1, L))
                comparar_mc(e2, ex, f"ancho {ancho}, {nombre_f}, {nombre_ell}, M = {M}", chequeos)
                frac = sn.profundidad_gratis(sep, 1, S, M)
                filas.append(dict(fuente=nombre_f, ell_nombre=nombre_ell, ell=ell, M=M, sigma_w=sw,
                                  vis=rc.visibilidad(B[:M].reshape(-1, L), Sigma),
                                  costo=int(round(p * M * (1 - frac))),
                                  razon=razon, razon_mc=np.median(np.linalg.norm(delta[:, N], axis=1)) / norma_f,
                                  med={k: np.median(np.sqrt(e2[k])) / norma_f for k in ESTIMADORES},
                                  q={k: np.percentile(np.sqrt(e2[k]) / norma_f, [25, 75]) for k in ESTIMADORES},
                                  exacto={k: np.sqrt(ex[k]) / norma_f for k in ESTIMADORES},
                                  mc={k: np.sqrt(e2[k].mean()) / norma_f for k in ESTIMADORES}))
    return filas, dict(L=L, p=p, norma_f=norma_f)


# ============================================================== 1D: comparación por presupuesto
def configuraciones(L, N, presupuestos):
    """(presupuesto, m, M): m divide a L, mcd(s, m) = 1, frame para N y p (M + 1) = presupuesto."""
    salida = []
    for P in presupuestos:
        for m in [k for k in range(1, L + 1) if L % k == 0]:
            if (P * m) % L or P * m // L < 1 or gcd(int(S[0]), m) != 1:
                continue
            M = P * m // L - 1
            B = rc.filas_sensores(sensores_1D(L, m), S, max(N, M))
            lam = np.linalg.eigvalsh(B[:N + 1].reshape(-1, L).T @ B[:N + 1].reshape(-1, L))
            if lam[0] > 1e-10 * lam[-1]:
                salida.append((P, m, M))
    return salida


def presupuesto_1D(chequeos):
    L, N, sn_ = cfg.L_1D, cfg.N_REC, cfg.SIGMA_N
    forma = (1, L)
    f = fila_f(L)
    norma_f = np.linalg.norm(f)
    Kt = fz.kernel_temporal(N, cfg.RHO)
    A = ee.matriz_A(lambda v: op.A(v, S), forma)

    # línea de base "denso": todos los píxeles en n = 0, sin dinámica -> f_hat = f + eps
    rng = np.random.default_rng(cfg.SEED)
    e_denso = np.linalg.norm(sn_ * rng.standard_normal((cfg.REPS_1D, L)), axis=1)
    denso = dict(exacto=np.sqrt(L) * sn_ / norma_f, mc=np.sqrt(np.mean(e_denso ** 2)) / norma_f,
                 med=np.median(e_denso) / norma_f)
    z = (np.mean(e_denso ** 2) - L * sn_ ** 2) / (np.std(e_denso ** 2, ddof=1) / np.sqrt(cfg.REPS_1D))
    chequeos.append(z)
    assert abs(z) < 4, f"denso: MC != fórmula (z = {z:.1f})"

    filas = []
    for P, m, M in configuraciones(L, N, cfg.PRESUPUESTOS_1D):
        G = sensores_1D(L, m)
        p = G.shape[0]
        B = rc.filas_sensores(G, S, max(N, M))
        Opinv = rc.pinv_O(B, N)
        degenerado = bool(np.all(np.any(B[:M + 1] != 0, axis=(0, 1))))
        for nombre_ell, ell in cfg.ELL_BARRIDO.items():
            Sigma1 = fz.sigma_espacial(fz.kernel_se_1d(L, ell), ventana_1D(L, cfg.FUENTE_CENTRO[0], cfg.FUENTE_ANCHO), 1.0)
            sw, _ = sigma_w_usado(Sigma1, Kt, forma, norma_f)
            Sigma = sw ** 2 * Sigma1
            rng = np.random.default_rng(cfg.SEED)
            W, delta, u = simular(rng, cfg.REPS_1D, f, forma, B, M, sn_, muestreador(Sigma, Kt))
            f_hat = estimar(u, delta, B, M, Sigma, Kt, sn_, forma, Opinv)
            e2 = {k: np.sum((v - f) ** 2, axis=1) for k, v in f_hat.items()}
            ex = ee.mse_exacto(A, B[0], Sigma, Kt, M, N, sn_)
            comparar_mc(e2, ex, f"presupuesto {P}, m = {m}, M = {M}, {nombre_ell}", chequeos)
            filas.append(dict(P=P, m=m, M=M, p=p, total=p * (M + 1), degenerado=degenerado, ell_nombre=nombre_ell,
                              ell=ell, med={k: np.median(np.sqrt(v)) / norma_f for k, v in e2.items()},
                              q={k: np.percentile(np.sqrt(v) / norma_f, [25, 75]) for k, v in e2.items()},
                              exacto={k: np.sqrt(ex[k]) / norma_f for k in e2}))
    return filas, denso
