"""
Experimento 1D y 2D de 32 x 32 del transporte (Sección 4.7.5): cómputo.

1D: identidad exacta vs Monte Carlo; barridos en M, posición de la fuente (también angosta) y ell;
comparación por presupuesto de sensores. 2D (32 x 32): Monte Carlo para M en M_2D.
Guarda salida/transporte/experimento_1D/resultados.npz; las figuras y tablas las hace
transporte/figuras/experimento_1D.py.
"""
import os

import numpy as np

from transporte.comun import config as cfg
from transporte.comun import operadores as op
from transporte.comun import forzado as fz
from transporte.comun import sensores as sn
from transporte.comun import reconstruccion as rc
from transporte.comun import resultados as res_io
from transporte.comun.estilo import ESTIMADORES
from transporte.comun.imagen import imagen_f
from transporte.comun.rutas import salida
from transporte.comun.uno_d import S, barrido_1D, presupuesto_1D, simular, estimar, sigma_w_usado

OUT = salida("transporte", "experimento_1D")


def caso_2D():
    L, N, sep, sn_ = cfg.L_2D, cfg.N_REC, cfg.SEP, cfg.SIGMA_N
    forma = (L, L)
    G = sn.banco(L, sep, 1, [("delta",)])
    p = G.shape[0]
    B = rc.filas_sensores(G, S, max(N, max(cfg.M_2D)))
    Opinv = rc.pinv_O(B, N)
    f = imagen_f(L, cfg.IMAGEN).ravel()
    norma_f = np.linalg.norm(f)
    ventana = fz.ventana_gaussiana(L, cfg.FUENTE_2D, cfg.FUENTE_ANCHO)
    K1 = fz.kernel_se_1d(L, cfg.ELL)
    Sigma1 = fz.sigma_espacial(np.kron(K1, K1), ventana, 1.0)
    Kt = fz.kernel_temporal(N, cfg.RHO)
    sw, razon = sigma_w_usado(Sigma1, Kt, forma, norma_f)
    Sigma = sw ** 2 * Sigma1
    muestrear = lambda rng, R: np.stack([fz.muestrear_forzado(rng, N, L, cfg.ELL, cfg.RHO, sw, ventana)
                                         for _ in range(R)]).reshape(R, N, -1)
    res = []
    for M in cfg.M_2D:
        rng = np.random.default_rng(cfg.SEED)
        W, delta, u = simular(rng, cfg.REPS_2D, f, forma, B, M, sn_, muestrear)
        f_hat = estimar(u, delta, B, M, Sigma, Kt, sn_, forma, Opinv)
        e = {k: np.linalg.norm(v - f, axis=1) / norma_f for k, v in f_hat.items()}
        Anf = np.stack([op.A_pot(f.reshape(forma), S, n).ravel() for n in range(N + 1)])
        res.append(dict(M=M, med={k: np.median(v) for k, v in e.items()},
                        vis=rc.visibilidad(B[:M].reshape(-1, L * L), Sigma),
                        costo=int(round(p * M * (1 - sn.profundidad_gratis(sep, 1, S, M)))),
                        f_hat={k: v[0] for k, v in f_hat.items()}, y=Anf + delta[0], Kz=N * M * p,
                        razon_mc=np.median(np.linalg.norm(delta[:, N], axis=1)) / norma_f))
    return res, dict(L=L, p=p, f=f, B=B, forma=forma, sigma_w=sw, razon=razon)


def main():
    chequeos = []
    filas, info1 = barrido_1D(cfg.FUENTE_ANCHO, chequeos)
    filas_pos, _ = barrido_1D(cfg.FUENTE_ANCHO_POSICION, chequeos)
    filas_pres, denso = presupuesto_1D(chequeos)
    res2, ctx2 = caso_2D()

    B, L2 = ctx2["B"], ctx2["L"]
    soporte = lambda Bk: np.argwhere(np.any(Bk != 0, axis=0).reshape(L2, L2))
    ruta = os.path.join(OUT, "resultados.npz")
    res_io.guardar(ruta, tablas=dict(filas=filas, filas_pos=filas_pos, filas_pres=filas_pres,
                                     res2=[{k: v for k, v in r.items()} for r in res2]),
                   denso_med=denso["med"], denso_exacto=denso["exacto"], denso_mc=denso["mc"],
                   L2=L2, p2=ctx2["p"], f2=ctx2["f"], forma2=np.array(ctx2["forma"]), sigma_w2=ctx2["sigma_w"],
                   razon2=ctx2["razon"], sensores2=soporte(B[0]), copias2=soporte(B[1:3].reshape(-1, L2 * L2)),
                   chequeos=np.array(chequeos), L1=info1["L"], p1=info1["p"], norma_f1=info1["norma_f"])
    archivos = [ruta]

    print("=" * 100)
    print(f"1D: d = {info1['L']}, p = {info1['p']} sensores (cada {cfg.SEP}), s = 1, N = {cfg.N_REC}, "
          f"sigma_n = {cfg.SIGMA_N:g}, rho = {cfg.RHO}, ||f|| = {info1['norma_f']:.3f}")
    print(f"asserts OK: Cov(residuos) y C == construcción densa; MC ({cfg.REPS_1D} realizaciones) == identidad "
          f"exacta en {len(chequeos)} casos, |z| máx = {np.max(np.abs(chequeos)):.2f} (< 4)")
    modo = "SIGMA_W de config" if cfg.RAZON_FORZADO is None else f"RAZON_FORZADO = {cfg.RAZON_FORZADO}"
    print(f"\nNIVEL DEL FORZADO ({modo}): ||delta_N|| / ||f||, RMS exacto y mediana MC")
    for ancho, fs in ((cfg.FUENTE_ANCHO, filas), (cfg.FUENTE_ANCHO_POSICION, filas_pos)):
        for r in fs:
            if r["M"] == cfg.M_1D[0] and r["fuente"] == list(cfg.FUENTES_1D)[0]:
                print(f"   1D ancho {ancho:g}, ell = {r['ell']:g}: sigma_w = {r['sigma_w']:.4g}  RMS {r['razon']:.4f}  "
                      f"mediana {r['razon_mc']:.4f}")
    print(f"   2D ancho {cfg.FUENTE_ANCHO:g}, ell = {cfg.ELL:g}: sigma_w = {ctx2['sigma_w']:.4g}  RMS {ctx2['razon']:.4f}  "
          f"mediana {res2[0]['razon_mc']:.4f}")

    print(f"\nPOSICIÓN DE LA FUENTE (ancho {cfg.FUENTE_ANCHO_POSICION:g}): visibilidad y mediana del error GP")
    print(f"{'fuente':20s} {'ell':>4s} " + "".join(f"{'M = ' + str(M):>22s}" for M in cfg.M_1D))
    for ell in cfg.ELL_BARRIDO:
        for fuente in cfg.FUENTES_1D:
            sel = [r for r in filas_pos if r["ell_nombre"] == ell and r["fuente"] == fuente]
            print(f"{fuente:20s} {sel[0]['ell']:4g} " + "".join(f"   vis {r['vis']:.3f}  GP {r['med']['gp']:.2e}"
                                                         for r in sel))

    print(f"\nPRESUPUESTO (1D, fuente x = {cfg.FUENTE_CENTRO[0]}, ancho {cfg.FUENTE_ANCHO:g}): medianas; "
          f"denso ({cfg.L_1D} sensores, n = 0): {denso['med']:.2e} (fórmula RMS {denso['exacto']:.2e}, MC {denso['mc']:.2e})")
    print(f"{'pres.':>5s} {'m':>2s} {'M':>2s} {'p':>3s} {'total':>5s} {'degen.':>6s} {'ell':>4s}  "
          f"{'sin corr.':>10s} {'GP':>10s} {'oráculo':>10s}")
    for r in filas_pres:
        v = [f"{r['med'][k]:10.2e}" if k in r["med"] else f"{'---':>10s}" for k in ESTIMADORES]
        print(f"{r['P']:5d} {r['m']:2d} {r['M']:2d} {r['p']:3d} {r['total']:5d} {'sí' if r['degenerado'] else 'no':>6s} "
              f"{r['ell']:4g}  " + " ".join(v))

    print(f"\n2D: L = {ctx2['L']}, p = {ctx2['p']} sensores (columnas cada {cfg.SEP}), fuente {cfg.FUENTE_2D}, "
          f"ell = {cfg.ELL}, {cfg.REPS_2D} realizaciones")
    for r in res2:
        print(f"   M = {r['M']}: visib. = {r['vis']:.3f}, extra = {r['costo']}, Kz de {r['Kz']} x {r['Kz']}; "
              f"medianas: sin corregir {r['med']['sin_corregir']:.2e}, GP {r['med']['gp']:.2e}, "
              f"oráculo {r['med']['oraculo']:.2e}")
    print("archivos:")
    for a in archivos:
        print("   " + os.path.relpath(a, salida()))


if __name__ == "__main__":
    main()
