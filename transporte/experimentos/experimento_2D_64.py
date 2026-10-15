"""
Transporte entero en 64 x 64 (Sección 4.7.5): reconstrucción con forzado (esquema del Cap. 4) y comparación con el
denso. Cómputo; figuras y tabla en transporte/figuras/experimento_2D_64.py.

f = coffee a 64 x 64, s = (1, 0), sensores delta en columnas cada SEP (p = 1024), N = N_REC. Barrido en M, ell del
forzado y razón ||delta_N|| / ||f|| (RAZON_FORZADO en memoria). El posterior se calcula en el soporte de la fuente
(comun/gp.py: GPSoporte) y la identidad exacta (4.29) en ese mismo espacio, sin matrices de (N d)^2.
Guarda salida/transporte/experimento_2D_64/resultados.npz.
"""
import os
import resource
import time

import numpy as np

from transporte.comun import config as cfg
from transporte.comun import operadores as op
from transporte.comun import forzado as fz
from transporte.comun import sensores as sn
from transporte.comun import reconstruccion as rc
from transporte.comun import resultados as res_io
from transporte.comun.estilo import ESTIMADORES
from transporte.comun.gp import medir, reconstruir, GPSoporte, exacto_soporte, validar_soporte
from transporte.comun.imagen import imagen_f
from transporte.comun.rutas import salida
from transporte.comun.uno_d import sigma_w_usado

OUT = salida("transporte", "experimento_2D_64")
L = 64
S = (1, 0)
EJEMPLO = (cfg.ELLS_64[0], 0.20, 2)        # (ell, razón, M) de las figuras de reconstrucción y cuadros


def memoria_densa(N, d, M, p):
    """GB que pediría error_exacto.mse_exacto: L, K_w, I, I - H Psi de (N d)^2 y Psi de (N M p) x (N d)."""
    return (4 * (N * d) ** 2 + N * M * p * N * d) * 8 / 1e9


def preparar():
    """Todo lo que no depende de (ell, razón, M): imagen, sensores, O^+, ventana, soporte, K_t."""
    N, sep = cfg.N_REC, cfg.SEP
    forma, d = (L, L), L * L
    f = imagen_f(L, cfg.IMAGEN).ravel()
    G = sn.banco(L, sep, 1, [("delta",)])
    B = rc.filas_sensores(G, S, max(N, max(cfg.MS_64)))
    ventana = fz.ventana_gaussiana(L, cfg.FUENTE_64, cfg.FUENTE_ANCHO)
    idx = np.flatnonzero(ventana.ravel() > cfg.UMBRAL_SOPORTE * ventana.max())
    masa_fuera = np.sum(np.delete(ventana.ravel(), idx) ** 2) / np.sum(ventana ** 2)   # diag(Sigma_w) = sigma_w^2 v^2
    assert masa_fuera < 1e-8, f"masa de Sigma_w fuera del soporte: {masa_fuera:.2e}"
    return dict(f=f, norma_f=np.linalg.norm(f), p=G.shape[0], B=B, Opinv=rc.pinv_O(B, N), ventana=ventana, idx=idx,
                masa_fuera=masa_fuera, Kt=fz.kernel_temporal(N, cfg.RHO), forma=forma, d=d,
                Anf=np.stack([op.A_pot(f.reshape(forma), S, n).ravel() for n in range(N + 1)]))


def combinacion(ctx, ell, razon, M):
    """Simula las R realizaciones de una combinación (misma semilla que el barrido) y reconstruye."""
    N, sn_, R, d, forma = cfg.N_REC, cfg.SIGMA_N, cfg.REPS_2D, ctx["d"], ctx["forma"]
    K1 = fz.kernel_se_1d(L, ell)
    Sigma1 = fz.sigma_espacial(np.kron(K1, K1), ctx["ventana"], 1.0)
    cfg.RAZON_FORZADO = razon                                            # solo en esta corrida
    sw, razon_rms = sigma_w_usado(Sigma1, ctx["Kt"], forma, ctx["norma_f"])
    cfg.RAZON_FORZADO = None
    Sigma_S = sw ** 2 * Sigma1[np.ix_(ctx["idx"], ctx["idx"])]
    del Sigma1
    rng = np.random.default_rng(cfg.SEED)
    W = np.stack([fz.muestrear_forzado(rng, N, L, ell, cfg.RHO, sw, ctx["ventana"]) for _ in range(R)]).reshape(R, N, d)
    delta = rc.acumular(W, lambda v: op.A(v, S), forma)
    u = medir(ctx["Anf"][None] + delta, ctx["B"], M, sn_, rng)
    gp = GPSoporte(ctx["B"], M, ctx["idx"], Sigma_S, ctx["Kt"], sn_)
    delta_hat = rc.acumular(gp.media(rc.residuos(u), d), lambda v: op.A(v, S), forma)
    u0, B, Opinv = u[:, :, 0], ctx["B"], ctx["Opinv"]
    f_hat = dict(sin_corregir=reconstruir(Opinv, B[0], u0), gp=reconstruir(Opinv, B[0], u0, delta_hat),
                 oraculo=reconstruir(Opinv, B[0], u0, delta))
    return dict(W=W, delta=delta, u=u, gp=gp, f_hat=f_hat, sw=sw, razon_rms=razon_rms, Sigma_S=Sigma_S)


def main():
    t0 = time.perf_counter()
    N, sep, sn_, R = cfg.N_REC, cfg.SEP, cfg.SIGMA_N, cfg.REPS_2D
    val = validar_soporte()
    ctx = preparar()
    f, norma_f, p, B, Opinv, idx, d = (ctx[k] for k in ("f", "norma_f", "p", "B", "Opinv", "idx", "d"))
    masa_fuera = ctx["masa_fuera"]

    # línea de base densa: todos los píxeles en n = 0
    e_d = np.linalg.norm(sn_ * np.random.default_rng(cfg.SEED).standard_normal((R, d)), axis=1)
    z_d = (np.mean(e_d ** 2) - d * sn_ ** 2) / (np.std(e_d ** 2, ddof=1) / np.sqrt(R))
    assert abs(z_d) < 4
    denso = dict(formula=sn_ * np.sqrt(d) / norma_f, med=np.median(e_d) / norma_f, rms=np.sqrt(np.mean(e_d ** 2)) / norma_f)

    filas, ejemplos, chequeos = [], {}, [z_d]
    for ell in cfg.ELLS_64:
        for razon in cfg.RAZONES_64:
            for M in cfg.MS_64:
                t1 = time.perf_counter()
                c = combinacion(ctx, ell, razon, M)
                gp, f_hat, Sigma_S = c["gp"], c["f_hat"], c["Sigma_S"]
                e2 = {k: np.sum((v - f) ** 2, axis=1) for k, v in f_hat.items()}
                ex_s = exacto_soporte(B, Opinv, gp, Sigma_S, sn_)
                zs = {k: (e2[k].mean() - ex_s[k]) / (e2[k].std(ddof=1) / np.sqrt(R)) for k in ESTIMADORES}
                chequeos += list(zs.values())
                for k, zk in zs.items():
                    assert abs(zk) < 4, f"MC != identidad (ell {ell}, razón {razon}, M = {M}, {k}): z = {zk:.1f}"
                filas.append(dict(ell=ell, razon=razon, razon_rms=c["razon_rms"], sw=c["sw"], M=M,
                                  vis=rc.visibilidad(gp.C, Sigma_S),
                                  extra=int(round(p * M * (1 - sn.profundidad_gratis(sep, 1, S, M)))),
                                  degenerado=bool(np.all(np.any(B[:M + 1] != 0, axis=(0, 1)))),
                                  med={k: np.median(np.sqrt(v)) / norma_f for k, v in e2.items()},
                                  q={k: np.percentile(np.sqrt(v) / norma_f, [25, 75]) for k, v in e2.items()},
                                  exacto={k: np.sqrt(ex_s[k]) / norma_f for k in ESTIMADORES},
                                  mc={k: np.sqrt(v.mean()) / norma_f for k, v in e2.items()},
                                  z=max(abs(v) for v in zs.values()), dim=gp.dim, p_a=gp.p_a,
                                  seg=time.perf_counter() - t1))
                if razon == 0.20 and M == 2:
                    ejemplos[ell] = dict({k: v[0] for k, v in f_hat.items()}, delta=c["delta"][0])
                print(f"   ell {ell:g}, razón {razon:g}, M = {M}: {filas[-1]['seg']:.0f} s, Kz {gp.dim}", flush=True)
                del c, gp, f_hat

    B = ctx["B"]
    columnas = lambda Bk: np.flatnonzero(np.any(np.any(Bk != 0, axis=0).reshape(L, L), axis=0))
    ruta = os.path.join(OUT, "resultados.npz")
    res_io.guardar(ruta, tablas=dict(filas=filas), denso_formula=denso["formula"], denso_med=denso["med"],
                   denso_rms=denso["rms"], f=f, Anf=ctx["Anf"], fuente=np.array(cfg.FUENTE_64),
                   columnas_sensores=columnas(B[0]), columnas_copias=columnas(B[1:EJEMPLO[2] + 1].reshape(-1, L * L)),
                   chequeos=np.array(chequeos), validacion=np.array(val), masa_fuera=masa_fuera,
                   **{f"ejemplo_ell{ell:g}_{k}": v for ell, e in ejemplos.items() for k, v in e.items()})
    archivos = [ruta]

    pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9          # bytes en macOS
    print("=" * 110)
    print(f"2D 64 x 64: d = {d}, p = {p} (columnas cada {sep}), s = (1, 0), N = {N}, sigma_n = {sn_:g}, rho = {cfg.RHO}, "
          f"fuente {cfg.FUENTE_64} (ancho {cfg.FUENTE_ANCHO:g}), {R} realizaciones, ||f|| = {norma_f:.3f}")
    print(f"soporte: |S| = {len(idx)} píxeles (ventana > {cfg.UMBRAL_SOPORTE:g} máx); masa de Sigma_w fuera = {masa_fuera:.2e} (< 1e-8) OK")
    print(f"validación en 16 x 16: GP soporte vs reconstruccion.py {val[0]:.1e}, reconstruir {val[1]:.1e}, "
          f"identidad soporte vs error_exacto.py {val[2]:.1e} OK")
    print(f"identidad exacta densa (error_exacto.py) pediría ~{memoria_densa(N, d, max(cfg.MS_64), p):.0f} GB "
          f"(> {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9:.0f} GB de RAM): se evaluó en el espacio del soporte")
    print(f"asserts OK: identidad vs MC en {len(chequeos)} casos, |z| máx = {np.max(np.abs(chequeos)):.2f} (< 4)")
    print(f"denso ({d} sensores, n = 0): fórmula {denso['formula']:.3e}, MC RMS {denso['rms']:.3e}, mediana {denso['med']:.3e}")
    print(f"{'ell':>4s} {'razón':>5s} {'sigma_w':>8s} {'M':>2s} {'1-k^2/tr':>8s} {'extra':>6s} {'degen.':>6s}  "
          f"{'sin corr.':>9s} {'GP':>9s} {'oráculo':>9s} {'denso':>9s}  {'GP exacto/MC (RMS)':>20s} {'|z|':>5s} {'Kz':>6s}")
    for r in filas:
        print(f"{r['ell']:4g} {r['razon']:5g} {r['sw']:8.4f} {r['M']:2d} {r['vis']:8.3f} {r['extra']:6d} "
              f"{'sí' if r['degenerado'] else 'no':>6s}  {r['med']['sin_corregir']:9.2e} {r['med']['gp']:9.2e} "
              f"{r['med']['oraculo']:9.2e} {denso['med']:9.2e}  {r['exacto']['gp']:.3e}/{r['mc']['gp']:.3e} "
              f"{r['z']:5.2f} {r['dim']:6d}")
    print(f"tiempo total {time.perf_counter() - t0:.0f} s; pico de memoria {pico:.2f} GB")
    print("archivos:")
    for a in archivos:
        print("   " + os.path.relpath(a, salida()))


if __name__ == "__main__":
    main()
