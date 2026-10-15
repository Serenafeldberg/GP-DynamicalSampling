"""
Transporte fraccional con difusión en 64 x 64 (Sección 4.7.5): cómputo.

    a_hat(xi) = exp(-i s xi_x) exp(-nu [(2 - 2 cos xi_x) + (2 - 2 cos xi_y)])
    En xi_x = pi (Nyquist, L par) la fase usa la regla de operadores.fase_1d: e^{-i s pi} se reemplaza
    por sign(cos(pi s)), así que a_hat(pi, xi_y) = sign(cos(pi s)) e^{-4 nu} e^{-nu (2 - 2 cos xi_y)} y A es real.

Modo "diagnostico": alpha_N por bloques de alias (con chequeo denso en 16 x 16), N mínimo, separación de nodos,
piso del oráculo, barrido fino en nu y barrido en s; control s = 0 (símbolo par, Observación 4.10).
Modo "experimento": forzado con s = 0,75 y nu en {0; 0,05; 0,1} (más s = 1,3, nu = 0), posterior en el espacio de
pesos, identidad exacta vs MC (|z| < 4).
Guarda salida/transporte/adveccion_difusion/resultados_diagnostico.npz y resultados.npz; figuras y tablas en
transporte/figuras/adveccion_difusion.py.
"""
import os
import resource
import sys
import time

import numpy as np

from transporte.comun import config as cfg
from transporte.comun import operadores as op
from transporte.comun import forzado as fz
from transporte.comun import frames as fr
from transporte.comun import sensores as sn
from transporte.comun import reconstruccion as rc
from transporte.comun import resultados as res_io
from transporte.comun.chequeos import chequear_adveccion_difusion
from transporte.comun.estilo import ESTIMADORES
from transporte.comun.fourier import DELTA, bancos_ad, pinv_gesvd, ancho_efectivo
from transporte.comun.gp import GPPesos, T_soporte, exacto_pesos, validar_pesos, medir, reconstruir
from transporte.comun.imagen import imagen_f
from transporte.comun.rutas import salida

OUT = salida("transporte", "adveccion_difusion")
L, L_DENSO = 64, 16
NUS = (0.0, 0.05, 0.1, 0.2, 0.5)
SS = (1.0, 1.3)
UMBRAL_ALFA = 1e-8                       # debajo de esto la reconstrucción no tiene sentido
S_BARRIDO = [round(v, 2) for v in np.arange(0.05, 2.001, 0.05) if abs(np.cos(np.pi * v)) > 1e-9]
NUS_S = (0.0, 0.05, 0.1)
S_EXP = 0.75
OPERADORES = [(S_EXP, 0.0), (S_EXP, 0.05), (S_EXP, 0.1), (1.3, 0.0)]     # (1,3; 0): solo para la tabla
EJEMPLO = (S_EXP, 0.1, 2.0, 0.20, 2)                                    # (s, nu, ell, razón, M) de las figuras
SEMILLA_EXTRA = (S_EXP, 0.05, 8.0, 0.05, 2)                             # combinación repetida con otra semilla


def diagnostico_caso(Lc, s, nu, N):
    """alpha_N, lambda_max, separación mínima de nodos por clase y lambda_min por clase (en N)."""
    H = op.multiplicador_ad(Lc, Lc, (s, 0), nu)
    return fr.alpha_bloques_simbolo(Lc, cfg.SEP, 1, DELTA, H, N, por_clase=True)


def piso_oraculo(s, nu, norma_f):
    """Error relativo del oráculo (solo ruido): sigma_n sqrt(tr (O^* O)^{-1}) / ||f||, por bloques."""
    IY, IX = fr.clases(L, cfg.SEP, 1)
    phi = (np.fft.fft2(sn.nucleo(L, DELTA[0])) / L)[IY, IX]
    Hc = np.conj(op.multiplicador_ad(L, L, (s, 0), nu)[IY, IX])
    P = L * L / cfg.SEP
    B = sum(P * np.einsum("ck,cl->ckl", Hc ** e * phi, np.conj(Hc ** e * phi)) for e in range(cfg.N_REC + 1))
    lam = np.linalg.eigvalsh(B)
    if lam.min() <= fr.TOL_FRAME * lam.max():                                  # sin frame: piso infinito
        return np.inf
    return cfg.SIGMA_N * np.sqrt(np.sum(1 / lam)) / norma_f


def diagnostico():
    t0 = time.perf_counter()
    N = cfg.N_REC
    casos = [(s, nu) for s in SS for nu in NUS] + [(0.0, nu) for nu in NUS if nu > 0]

    # chequeos del operador y de alpha por bloques contra el cálculo denso (16 x 16)
    err_op, err_blq = 0.0, 0.0
    G16 = sn.banco(L_DENSO, cfg.SEP, 1, DELTA)
    for s, nu in casos:
        err_op = max(err_op, max(chequear_adveccion_difusion(L_DENSO, s, nu).values()))
        a_d, b_d = fr.alpha_denso_operador(G16, lambda g: op.A_ad_adj(g, (s, 0), nu), N)
        a_b, b_b, *_ = diagnostico_caso(L_DENSO, s, nu, N)
        e = max(np.max(np.abs(a_d - a_b)), np.max(np.abs(b_d - b_b))) / np.max(b_d)
        assert e < 1e-9, f"denso != bloques (s = {s}, nu = {nu}): {e:.1e}"
        err_blq = max(err_blq, e)

    norma_f = np.linalg.norm(imagen_f(L, cfg.IMAGEN))
    denso = cfg.SIGMA_N * np.sqrt(L * L) / norma_f
    filas = []
    for s, nu in casos:
        a, b, sep, lam_clase, (IY, IX) = diagnostico_caso(L, s, nu, N)
        hay = a > fr.TOL_FRAME * b
        c = int(np.argmin(lam_clase))
        frec = (int((IY[c, 0] + L // 2) % L - L // 2), int((IX[c, 0] + L // 2) % L - L // 2))
        filas.append(dict(s=s, nu=nu, alfa=a, beta=b, hay=hay, N_min=int(np.argmax(hay)) if hay.any() else None,
                          sep=sep.min(), clase_peor=frec,
                          oraculo=piso_oraculo(s, nu, norma_f) if s > 0 else np.inf))

    # barrido fino en nu para s = 1,3: dónde cruza alpha_N el umbral
    nus_fino = np.round(np.arange(0.0, 0.501, 0.01), 3)
    a_fino = {s: np.array([diagnostico_caso(L, s, nu, N)[0][-1] for nu in nus_fino]) for s in SS}
    o_fino = {s: np.array([piso_oraculo(s, nu, norma_f) for nu in nus_fino]) for s in SS}


    print("=" * 100)
    print(f"DIAGNÓSTICO: L = {L}, sensores delta en columnas cada {cfg.SEP} (p = {L * L // cfg.SEP}), N = {N}")
    print(f"chequeos OK: operador AD (adjunta, nu = 0 == roll, ||A|| = max|a_hat|, A real) error máx {err_op:.1e}; "
          f"alpha por bloques == denso (16 x 16) en {len(casos)} casos, error máx {err_blq:.1e}")
    print(f"{'s':>4s} {'nu':>5s} {'N mín':>6s} {'alpha_N':>10s} {'alpha/lmax':>11s} {'sep. mín nodos':>15s} "
          f"{'clase peor (ky,kx)':>19s} {'oráculo rel.':>13s}  ¿alpha_N >= {UMBRAL_ALFA:g}?")
    for r in filas:
        print(f"{r['s']:4g} {r['nu']:5g} {('nunca' if r['N_min'] is None else r['N_min']):>6} {r['alfa'][-1]:10.2e} "
              f"{r['alfa'][-1] / r['beta'][-1]:11.2e} {r['sep']:15.2e} {str(r['clase_peor']):>19s} "
              f"{r['oraculo']:13.2e}  "
              f"{'sí' if r['alfa'][-1] >= UMBRAL_ALFA else 'NO'}")
    for s in SS:
        ok = nus_fino[a_fino[s] >= UMBRAL_ALFA]
        print(f"s = {s:g}: alpha_N >= {UMBRAL_ALFA:g} para nu <= {ok.max():g} (barrido cada 0,01)")
        for tope in (0.01, 0.05, 0.10):
            ok = nus_fino[o_fino[s] <= tope]
            print(f"        piso del oráculo <= {tope:.0%} para nu <= {ok.max():g}")
    print(f"denso ({L * L} sensores, n = 0): {denso:.2e}")
    filas_s, mejor, mejor_neg = barrido_s(norma_f, denso)
    tabla_d = [dict({k: v for k, v in r.items() if k not in ("N_min", "clase_peor")},
                    N_min=-1 if r["N_min"] is None else r["N_min"], clase_peor=np.array(r["clase_peor"])) for r in filas]
    tabla_s = [dict(r, par=np.array(r["par"])) for r in filas_s]
    ruta = os.path.join(OUT, "resultados_diagnostico.npz")
    res_io.guardar(ruta, tablas=dict(diagnostico=tabla_d, barrido_s=tabla_s), nus_fino=nus_fino,
                   a_fino=np.stack([a_fino[s] for s in SS]), o_fino=np.stack([o_fino[s] for s in SS]),
                   SS=np.array(SS), denso=denso, s_propuesto=mejor_neg["s"])
    print(f"tiempo {time.perf_counter() - t0:.0f} s")
    print("archivo: " + os.path.relpath(ruta, salida()))


def colision(s, nu):
    """Par de nodos a_hat más cercano: (separación, clase k0_x, k_y, frecuencias xi_x del par en unidades de pi)."""
    IY, IX = fr.clases(L, cfg.SEP, 1)
    nodos = op.multiplicador_ad(L, L, (s, 0), nu)[IY, IX]
    dif = np.abs(nodos[:, :, None] - nodos[:, None, :])
    K = dif.shape[1]
    dif[:, np.arange(K), np.arange(K)] = np.inf
    c, i, j = np.unravel_index(np.argmin(dif), dif.shape)
    xi = lambda k: ((k + L // 2) % L - L // 2) * 2 / L                     # xi_x / pi (con signo)
    return dif[c, i, j], int(IX[c, 0] % (L // cfg.SEP)), int((IY[c, 0] + L // 2) % L - L // 2), \
        tuple(sorted((xi(IX[c, i]), xi(IX[c, j]))))


def barrido_s(norma_f, denso):
    N = cfg.N_REC
    filas = []
    for nu in NUS_S:
        for s in S_BARRIDO:
            a, b, *_ = diagnostico_caso(L, s, nu, N)
            sep, q, ky, par = colision(s, nu)
            filas.append(dict(s=s, nu=nu, alfa=a[-1], beta=b[-1], sep=sep, q=q, ky=ky, par=par,
                              oraculo=piso_oraculo(s, nu, norma_f)))

    # hipótesis: con s = 1,3 la cuasi-colisión es entre xi_x = ±3pi/4 de la clase q = 8 (1,5 pi s ~ 2 pi).
    # Con nu = 0 todas las clases q = 1..15 tienen un par separado 1,5 pi (bins q-32 y q+16) y empatan;
    # con nu > 0 solo el par ±3pi/4 (q = 8) tiene módulos iguales, así que el mínimo estricto debería ser ese.
    sep_teo = 2 * abs(np.sin(1.5 * np.pi * 1.3 / 2))                          # |e^{-i s 3pi/4} - e^{i s 3pi/4}|
    hipotesis = {}
    for nu in NUS_S:
        H = op.multiplicador_ad(L, L, (1.3, 0), nu)
        IY, IX = fr.clases(L, cfg.SEP, 1)
        nodos = H[IY, IX]
        dif = np.abs(nodos[:, :, None] - nodos[:, None, :])
        dif[:, np.arange(4), np.arange(4)] = np.inf
        por_clase = dif.min(axis=(1, 2))
        q8 = np.abs(H[:, 24] - H[:, 40]).min()                                # xi_x = 3pi/4 y -3pi/4 (bins 24 y 40)
        empatadas = sorted({int(IX[c, 0] % 16) for c in np.flatnonzero(por_clase <= por_clase.min() * (1 + 1e-9))})
        r = next(r for r in filas if r["s"] == 1.3 and r["nu"] == nu)
        hipotesis[nu] = dict(minimo=por_clase.min(), q8=q8, empatadas=empatadas, q=r["q"], par=r["par"])
        assert np.isclose(q8, por_clase.min(), rtol=1e-12), f"nu = {nu}: el par ±3pi/4 de q = 8 no alcanza el mínimo"
        if nu > 0:
            assert empatadas == [8], f"nu = {nu}: el mínimo no es exclusivo de q = 8 ({empatadas})"
    assert np.isclose(hipotesis[0.0]["minimo"], sep_teo, rtol=1e-12)

    # propuesta: s no entero con mejor alpha_N para nu = 0,05, prefiriendo cos(pi s) < 0
    cand = [r for r in filas if r["nu"] == 0.05 and not float(r["s"]).is_integer()]
    mejor = max(cand, key=lambda r: r["alfa"])
    mejor_neg = max([r for r in cand if np.cos(np.pi * r["s"]) < 0], key=lambda r: r["alfa"])

    print("\nBARRIDO EN s (m = %d, N = %d; excluidos s con cos(pi s) = 0)" % (cfg.SEP, N))
    print(f"{'s':>5s} " + "".join(f"{'nu = ' + f'{nu:g}':>58s}" for nu in NUS_S))
    print(f"{'':5s} " + "".join(f"{'alpha_N':>10s} {'sep':>9s} {'q':>3s} {'par xi_x/pi':>16s} {'oráculo':>9s}  " for _ in NUS_S))
    for s in S_BARRIDO:
        celdas = ""
        for nu in NUS_S:
            r = next(r for r in filas if r["s"] == s and r["nu"] == nu)
            par = f"({r['par'][0]:+.3f},{r['par'][1]:+.3f})"
            celdas += f"{r['alfa']:10.2e} {r['sep']:9.2e} {r['q']:3d} {par:>16s} {r['oraculo']:9.2e}  "
        print(f"{s:5.2f} " + celdas)
    print(f"\nhipótesis s = 1,3: cuasi-colisión entre xi_x = ±3pi/4 de la clase q = 8 (1,5 pi s ~ 2 pi)")
    for nu, h in hipotesis.items():
        print(f"   nu = {nu:g}: separación mínima {h['minimo']:.4f}; el par ±3pi/4 de q = 8 da {h['q8']:.4f}; "
              f"clases que alcanzan el mínimo: {h['empatadas']}")
    print(f"   teórica (nu = 0): 2|sin(0,75 pi s)| = {sep_teo:.4f}. CONFIRMADA en el mecanismo (Delta xi_x = 1,5 pi); "
          f"con nu = 0 la comparten las 15 clases q = 1..15, con nu > 0 es exclusiva de q = 8")
    print(f"propuesta (nu = 0,05, s no entero): mejor alpha_N en s = {mejor['s']:g} (alpha {mejor['alfa']:.2e}, "
          f"cos(pi s) = {np.cos(np.pi * mejor['s']):+.2f}); con cos(pi s) < 0: s = {mejor_neg['s']:g} "
          f"(alpha {mejor_neg['alfa']:.2e}, oráculo {mejor_neg['oraculo']:.2e})")
    return filas, mejor, mejor_neg


def experimento():
    t0 = time.perf_counter()
    N, sn_, R, sep = cfg.N_REC, cfg.SIGMA_N, cfg.REPS_2D, cfg.SEP
    forma, d = (L, L), L * L
    val = validar_pesos()

    f = imagen_f(L, cfg.IMAGEN).ravel()
    norma_f = np.linalg.norm(f)
    G = sn.banco(L, sep, 1, DELTA)
    p = G.shape[0]
    ventana = fz.ventana_gaussiana(L, cfg.FUENTE_64, cfg.FUENTE_ANCHO)
    idx = np.flatnonzero(ventana.ravel() > cfg.UMBRAL_SOPORTE * ventana.max())
    yi, xi = np.divmod(idx, L)
    vS = ventana.ravel()[idx]
    Kt = fz.kernel_temporal(N, cfg.RHO)
    e_d = np.linalg.norm(sn_ * np.random.default_rng(cfg.SEED).standard_normal((R, d)), axis=1)
    denso = np.median(e_d) / norma_f

    filas, anchos, razones_mc, chequeos, ejemplo, extra_semilla = [], [], [], [], None, None
    for s, nu in OPERADORES:
        sv = (s, 0)
        aplicar = lambda v, sv=sv, nu=nu: op.A_ad(v, sv, nu)
        t_op = time.perf_counter()
        B = bancos_ad(G, s, nu, max(N, max(cfg.MS_64)))
        Opinv = pinv_gesvd(B, N)
        TS = T_soporte(B, Opinv, idx, N)
        anchos += [(s, nu, m) + ancho_efectivo(B, L, m) for m in (1, 2, 3)]
        Anf = np.stack([op.A_ad(f.reshape(forma), sv, nu, n=n).ravel() for n in range(N + 1)])
        H = op.multiplicador_ad(L, L, sv, nu)
        for ell in cfg.ELLS_64:
            K1 = fz.kernel_se_1d(L, ell)
            Sigma1_S = vS[:, None] * (K1[np.ix_(yi, yi)] * K1[np.ix_(xi, xi)]) * vS[None, :]
            e1 = fz.energia_delta_N_simbolo(Sigma1_S, Kt, H, idx)                 # con sigma_w = 1
            for razon in cfg.RAZONES_64:
                sw = razon * norma_f / np.sqrt(e1)
                Sigma_S = sw ** 2 * Sigma1_S
                rng = np.random.default_rng(cfg.SEED)
                W = np.stack([fz.muestrear_forzado(rng, N, L, ell, cfg.RHO, sw, ventana) for _ in range(R)]).reshape(R, N, d)
                delta = rc.acumular(W, aplicar, forma)
                en = np.sum(delta[:, N] ** 2, axis=1)
                zr = (en.mean() - sw ** 2 * e1) / (en.std(ddof=1) / np.sqrt(R))
                razones_mc.append((s, nu, ell, razon, np.sqrt(en.mean()) / norma_f, zr))
                assert abs(zr) < 4, f"razón de forzado MC != objetivo (s {s}, nu {nu}, ell {ell}, razón {razon}): z = {zr:.1f}"
                for M in cfg.MS_64:
                    t1 = time.perf_counter()
                    rng_M = np.random.default_rng(cfg.SEED + 1000 * M)
                    u = medir(Anf[None] + delta, B, M, sn_, rng_M)
                    gp = GPPesos(B, M, idx, Sigma_S, Kt, sn_)
                    dh = rc.acumular(gp.media(rc.residuos(u), d), aplicar, forma)
                    u0 = u[:, :, 0]
                    f_hat = dict(sin_corregir=reconstruir(Opinv, B[0], u0), gp=reconstruir(Opinv, B[0], u0, dh),
                                 oraculo=reconstruir(Opinv, B[0], u0, delta))
                    e2 = {k: np.sum((v - f) ** 2, axis=1) for k, v in f_hat.items()}
                    ex_p = exacto_pesos(Opinv, gp, TS, sn_)
                    zs = {k: (e2[k].mean() - ex_p[k]) / (e2[k].std(ddof=1) / np.sqrt(R)) for k in ESTIMADORES}
                    chequeos += list(zs.values())
                    for k, zk in zs.items():
                        assert abs(zk) < 4, f"MC != identidad (s {s}, nu {nu}, ell {ell}, razón {razon}, M {M}, {k}): z = {zk:.1f}"
                    filas.append(dict(s=s, nu=nu, ell=ell, razon=razon, M=M, sw=sw,
                                      vis=rc.visibilidad(gp.C.reshape(-1, len(idx)), Sigma_S), extra=p * M,
                                      igual_denso=p * (M + 1) == d,
                                      med={k: np.median(np.sqrt(v)) / norma_f for k, v in e2.items()},
                                      q={k: np.percentile(np.sqrt(v) / norma_f, [25, 75]) for k, v in e2.items()},
                                      exacto={k: np.sqrt(ex_p[k]) / norma_f for k in ESTIMADORES},
                                      z=max(abs(v) for v in zs.values()), zs=zs, seg=time.perf_counter() - t1))
                    if (s, nu, ell, razon, M) == EJEMPLO:
                        ejemplo = dict({k: v[0] for k, v in f_hat.items()}, delta=delta[0], Anf=Anf)
                    if (s, nu, ell, razon, M) == SEMILLA_EXTRA:                 # misma combinación, otra semilla
                        rng2 = np.random.default_rng(cfg.SEED + 1)
                        W2 = np.stack([fz.muestrear_forzado(rng2, N, L, ell, cfg.RHO, sw, ventana)
                                       for _ in range(R)]).reshape(R, N, d)
                        d2 = rc.acumular(W2, aplicar, forma)
                        u2 = medir(Anf[None] + d2, B, M, sn_, rng2)
                        dh2 = rc.acumular(gp.media(rc.residuos(u2), d), aplicar, forma)
                        f2 = dict(sin_corregir=reconstruir(Opinv, B[0], u2[:, :, 0]),
                                  gp=reconstruir(Opinv, B[0], u2[:, :, 0], dh2),
                                  oraculo=reconstruir(Opinv, B[0], u2[:, :, 0], d2))
                        z2 = {k: (np.sum((v - f) ** 2, axis=1).mean() - ex_p[k]) /
                              (np.sum((v - f) ** 2, axis=1).std(ddof=1) / np.sqrt(R)) for k, v in f2.items()}
                        extra_semilla = (zs, z2)
                        del W2, d2, u2, dh2, f2
                    print(f"   s {s:g}, nu {nu:g}, ell {ell:g}, razón {razon:g}, M = {M}: {filas[-1]['seg']:.0f} s", flush=True)
                    del gp, dh, f_hat, u
                del W, delta
        print(f"   operador (s {s:g}, nu {nu:g}) listo en {time.perf_counter() - t_op:.0f} s", flush=True)
        del B, Opinv, TS

    ruta = os.path.join(OUT, "resultados.npz")
    res_io.guardar(ruta, tablas=dict(filas=[{k: v for k, v in r.items()} for r in filas],
                                     razones_mc=[dict(s=a, nu=b, ell=c, razon=d, razon_mc=e, z=g) for a, b, c, d, e, g in razones_mc],
                                     anchos=[dict(s=a, nu=b, m=c, media_x=d, sigma_x=e, sigma_y=g) for a, b, c, d, e, g in anchos]),
                   **{f"ejemplo_{k}": v for k, v in ejemplo.items()}, f=f, fuente=np.array(cfg.FUENTE_64), denso=denso,
                   chequeos=np.array(chequeos), z_semilla0=np.array(list(extra_semilla[0].values())),
                   z_semilla1=np.array(list(extra_semilla[1].values())), validacion=np.array(val))
    archivos = [ruta]
    pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9

    print("=" * 120)
    print(f"ADVECCIÓN-DIFUSIÓN 64 x 64: s = {S_EXP:g} (más s = 1,3, nu = 0 para la tabla), p = {p}, N = {N}, "
          f"sigma_n = {sn_:g}, rho = {cfg.RHO}, fuente {cfg.FUENTE_64} (ancho {cfg.FUENTE_ANCHO:g}), |S| = {len(idx)}, {R} realizaciones")
    print(f"validación 16 x 16: GP en pesos vs reconstruccion.py {val[0]:.1e}; identidad en pesos vs error_exacto.py {val[1]:.1e} OK")
    print(f"posterior en el espacio de pesos: sistema de {N * len(idx)} x {N * len(idx)} (independiente de p)")
    print(f"asserts OK: identidad vs MC en {len(chequeos)} casos, |z| máx = {np.max(np.abs(chequeos)):.2f} (< 4)")
    zs1, zs2 = extra_semilla
    print(f"otra semilla en {SEMILLA_EXTRA}: |z| semilla {cfg.SEED}: " + ", ".join(f"{k} {abs(v):.2f}" for k, v in zs1.items())
          + f"  |  semilla {cfg.SEED + 1}: " + ", ".join(f"{k} {abs(v):.2f}" for k, v in zs2.items()))
    print("razón de forzado obtenida por MC (RMS ||delta_N|| / ||f||):")
    for s, nu, ell, razon, r_mc, zr in razones_mc:
        print(f"   s {s:g}, nu {nu:g}, ell {ell:g}: objetivo {razon:g} -> MC {r_mc:.4f} (z = {zr:+.2f})")
    print("ancho efectivo de (A^*)^m g (perfil |.|^2): corrimiento medio en x, desvío en x, desvío en y [píxeles]")
    for s, nu, m, mx, sx, sy in anchos:
        print(f"   s {s:g}, nu {nu:g}, m = {m}: media x {mx:+.2f}, sigma_x {sx:.2f}, sigma_y {sy:.2f}")
    print(f"denso ({d} sensores, n = 0): {denso:.3e}")
    print(f"{'s':>4s} {'nu':>5s} {'ell':>4s} {'razón':>5s} {'M':>3s} {'1-k^2/tr':>8s} {'extra':>6s}  {'sin corr.':>9s} "
          f"{'GP':>9s} {'oráculo':>9s} {'denso':>9s} {'|z|':>5s}")
    for r in filas:
        print(f"{r['s']:4g} {r['nu']:5g} {r['ell']:4g} {r['razon']:5g} {r['M']:2d}{'*' if r['igual_denso'] else ' '} "
              f"{r['vis']:8.3f} {r['extra']:6d}  {r['med']['sin_corregir']:9.2e} {r['med']['gp']:9.2e} "
              f"{r['med']['oraculo']:9.2e} {denso:9.2e} {r['z']:5.2f}")
    print("* M = 3: mismo número de sensores que el denso (p (M + 1) = 4096)")
    print(f"tiempo total {time.perf_counter() - t0:.0f} s; pico de memoria {pico:.2f} GB")
    print("archivos:")
    for a in archivos:
        print("   " + os.path.relpath(a, salida()))


if __name__ == "__main__":
    if "experimento" in sys.argv[1:]:
        experimento()
    else:
        diagnostico()
