"""
Robustez del GP a un prior mal especificado (1D, comparación por presupuesto; Sección 4.7.5): cómputo.

Forzado verdadero con ell_true y razón ||delta_N|| / ||f|| = RAZON. El GP usa un prior supuesto:
ell_gp = c · ell_true (c en FACTORES_ELL) o sigma_w_gp = c · sigma_w (c en FACTORES_SIGMA); rho verdadero.
Configuraciones (m, M): las mejores de cruce_forzado a razón 0,05.

Identidad exacta con prior mal especificado: la ganancia usa el prior supuesto,
    H = K_gp Psi^T (Psi K_gp Psi^T + sigma_n^2 D D^T)^{-1} ,   K_gp = K_t ⊗ Sigma_gp,
y la covarianza del error usa el verdadero:
    E||f_hat - f||^2 = tr(A_w K_w A_w^T) + sigma_n^2 ||A_eps||_F^2 ,  K_w = K_t ⊗ Sigma_w.
Guarda salida/transporte/robustez_prior/resultados.npz; figura y tabla en transporte/figuras/robustez_prior.py.
"""
import os

import numpy as np

from transporte.comun import config as cfg
from transporte.comun import operadores as op
from transporte.comun import forzado as fz
from transporte.comun import reconstruccion as rc
from transporte.comun import error_exacto as ee
from transporte.comun import resultados as res_io
from transporte.comun.rutas import salida
from transporte.comun.uno_d import (S, fila_f, ventana_1D, configuraciones, sigma_w_usado, sensores_1D, simular,
                                    muestreador, estimar, comparar_mc)

OUT = salida("transporte", "robustez_prior")

RAZON = 0.05
FACTORES_ELL = (0.5, 1.0, 2.0)
FACTORES_SIGMA = (0.5, 2.0)
MEJORES = {("chica", 16): (8, 1), ("chica", 32): (8, 3), ("chica", 48): (8, 5),      # cruce_forzado.py,
           ("grande", 16): (8, 1), ("grande", 32): (4, 1), ("grande", 48): (4, 2)}   # razón 0,05


def mse_mal_especificado(A, Gt, Sigma, Sigma_gp, Kt, M, N, sigma_n):
    """E||f_hat_GP - f||^2 con ganancia del prior Sigma_gp y forzado verdadero Sigma (M >= 1)."""
    d, p = A.shape[0], Gt.shape[0]
    pw = [np.eye(d)]
    for _ in range(max(N, M)):
        pw.append(A @ pw[-1])
    C = np.vstack([Gt @ pw[m] for m in range(M)])
    Psi = np.kron(np.eye(N), C)
    ie = lambda n, m, i: (n * (M + 1) + m) * p + i
    D = np.zeros((N * M * p, (N + 1) * (M + 1) * p))
    Su = np.zeros(((N + 1) * p, (N + 1) * (M + 1) * p))
    for n in range(1, N + 1):
        for m in range(M):
            for i in range(p):
                r = ((n - 1) * M + m) * p + i
                D[r, ie(n, m, i)] += 1.0
                D[r, ie(n - 1, m + 1, i)] -= 1.0
    for n in range(N + 1):
        for i in range(p):
            Su[n * p + i, ie(n, 0, i)] = 1.0
    L = np.zeros((N * d, N * d))
    for n in range(N):
        for k in range(n + 1):
            L[n * d:(n + 1) * d, k * d:(k + 1) * d] = pw[n - k]
    Opinv = np.linalg.pinv(np.vstack([Gt @ pw[n] for n in range(N + 1)]))
    iota = np.vstack([np.zeros((p, N * p)), np.eye(N * p)])
    T = Opinv @ iota @ np.kron(np.eye(N), Gt) @ L

    K_gp, K_w = np.kron(Kt, Sigma_gp), np.kron(Kt, Sigma)
    H = np.linalg.solve(Psi @ K_gp @ Psi.T + sigma_n ** 2 * D @ D.T, Psi @ K_gp).T     # ganancia: prior supuesto
    Aw = -T @ (np.eye(N * d) - H @ Psi)
    Ae = T @ H @ D - Opinv @ Su
    return np.trace(Aw @ K_w @ Aw.T) + sigma_n ** 2 * np.sum(Ae ** 2)                  # error: prior verdadero


def casos():
    """(etiqueta, factor de ell, factor de sigma_w) del prior supuesto."""
    return [("ell", c, 1.0) for c in FACTORES_ELL] + [("sigma", 1.0, c) for c in FACTORES_SIGMA]


def main():
    L, N, sn_ = cfg.L_1D, cfg.N_REC, cfg.SIGMA_N
    forma = (1, L)
    f = fila_f(L)
    norma_f = np.linalg.norm(f)
    Kt = fz.kernel_temporal(N, cfg.RHO)
    A = ee.matriz_A(lambda v: op.A(v, S), forma)
    ventana = ventana_1D(L, cfg.FUENTE_CENTRO[0], cfg.FUENTE_ANCHO)
    validas = configuraciones(L, N, cfg.PRESUPUESTOS_1D)
    chequeos = []

    # línea de base densa (igual que en experimento.py)
    rng = np.random.default_rng(cfg.SEED)
    e_denso = np.linalg.norm(sn_ * rng.standard_normal((cfg.REPS_1D, L)), axis=1)
    denso = np.median(e_denso) / norma_f
    z = (np.mean(e_denso ** 2) - L * sn_ ** 2) / (np.std(e_denso ** 2, ddof=1) / np.sqrt(cfg.REPS_1D))
    chequeos.append(z)
    assert abs(z) < 4

    cfg.RAZON_FORZADO = RAZON                                    # solo en esta corrida
    filas = []
    for nombre_ell, ell in cfg.ELL_BARRIDO.items():
        Sigma1 = fz.sigma_espacial(fz.kernel_se_1d(L, ell), ventana, 1.0)
        sw, razon = sigma_w_usado(Sigma1, Kt, forma, norma_f)
        Sigma = sw ** 2 * Sigma1
        for P in cfg.PRESUPUESTOS_1D:
            m, M = MEJORES[(nombre_ell, P)]
            assert (P, m, M) in validas, f"({m}, {M}) no es válida para el presupuesto {P}"
            B = rc.filas_sensores(sensores_1D(L, m), S, max(N, M))
            Opinv = rc.pinv_O(B, N)
            rng = np.random.default_rng(cfg.SEED)                # mismas realizaciones para todos los priors
            W, delta, u = simular(rng, cfg.REPS_1D, f, forma, B, M, sn_, muestreador(Sigma, Kt))
            for tipo, c_ell, c_sw in casos():
                Sigma_gp = (c_sw * sw) ** 2 * fz.sigma_espacial(fz.kernel_se_1d(L, c_ell * ell), ventana, 1.0)
                f_hat = estimar(u, delta, B, M, Sigma_gp, Kt, sn_, forma, Opinv)["gp"]
                e2 = np.sum((f_hat - f) ** 2, axis=1)
                exacto = mse_mal_especificado(A, B[0], Sigma, Sigma_gp, Kt, M, N, sn_)
                if c_ell == 1.0 and c_sw == 1.0:                 # prior correcto: coincide con error_exacto.py
                    ref = ee.mse_exacto(A, B[0], Sigma, Kt, M, N, sn_)["gp"]
                    assert np.isclose(exacto, ref, rtol=1e-8), "identidad mal especificada != error_exacto"
                comparar_mc({"gp": e2}, {"gp": exacto}, f"{nombre_ell}, P = {P}, {tipo} x {c_ell * c_sw}", chequeos)
                med = np.median(np.sqrt(e2)) / norma_f
                filas.append(dict(ell_nombre=nombre_ell, ell=ell, P=P, m=m, M=M, tipo=tipo, c_ell=c_ell, c_sw=c_sw,
                                  med=med, q=np.percentile(np.sqrt(e2) / norma_f, [25, 75]),
                                  exacto=np.sqrt(exacto) / norma_f, gana=med < denso, razon=razon))
    cfg.RAZON_FORZADO = None
    ells = list(cfg.ELL_BARRIDO)
    ruta = os.path.join(OUT, "resultados.npz")
    res_io.guardar(ruta, tablas=dict(filas=filas), denso=denso, chequeos=np.array(chequeos))

    print("=" * 96)
    print(f"asserts OK: identidad mal especificada == error_exacto con prior correcto; MC ({cfg.REPS_1D} "
          f"realizaciones) == identidad en {len(chequeos)} casos, |z| máx = {np.max(np.abs(chequeos)):.2f} (< 4)")
    print(f"razón de forzado {RAZON} (RMS verificado: " + ", ".join(
        f"ell {cfg.ELL_BARRIDO[e]:g} -> {next(r['razon'] for r in filas if r['ell_nombre'] == e):.4f}" for e in ells)
          + f"); denso: {denso:.2e}")
    print(f"{'ell_true':>8s} {'pres.':>5s} {'(m,M)':>6s}  {'prior del GP':16s} {'GP mediana':>11s} {'exacto RMS':>11s}  "
          "¿gana al denso?")
    for r in filas:
        prior = f"ell_gp = {r['c_ell'] * r['ell']:g}" if r["tipo"] == "ell" else f"sigma_w x {r['c_sw']:g}"
        print(f"{r['ell']:8g} {r['P']:5d} ({r['m']},{r['M']})  {prior:16s} {r['med']:11.2e} {r['exacto']:11.2e}  "
              f"{'sí' if r['gana'] else 'no'}")
    print("archivo: " + os.path.relpath(ruta, salida()))


if __name__ == "__main__":
    main()
