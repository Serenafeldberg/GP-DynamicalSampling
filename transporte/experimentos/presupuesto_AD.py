"""
Comparación por presupuesto en advección-difusión (Sección 4.7.5): cómputo.

s = 0,75, nu = 0,1, 64 x 64, N = 8, sigma_n = 1e-3, rho = 0,9, fuente FUENTE_64, razón de forzado 0,2, ell en {2, 8}.
Presupuesto total P de sensores en {2048, 3072}. Configuraciones (m, M): sensores delta en columnas cada m (m_y = 1),
p = 4096 / m sensores base y p (M + 1) = P, con m en {2, 4, 8}; las que no dan M entero se descartan. M = 0 no tiene
residuos: solo sin corregir. Para cada configuración se verifica el frame (alpha_8 > TOL_FRAME lambda_max).

Estimadores (medianas de ||f_hat - f|| / ||f|| sobre R = 200 realizaciones):
  sin corregir  f_hat = O_8^+ u^(0) con los p sensores base;
  ingenua       los P sensores delta (mismo presupuesto, sensores base: columnas {0, 2} mod 4 si P = 2048 y
                {0, 1, 2} mod 4 si P = 3072) en n = 0..N, f_hat = O_8^+ u, sin GP (naive dinámico de comparacion_justa);
  GP            residuos con profundidad M y posterior en el espacio de pesos (como adveccion_difusion);
  oráculo       con delta verdadero;
  denso         4096 sensores en n = 0, como referencia.
O_8^+ se arma por bloques de Fourier ((O^* O)^{-1} aplicada a las filas (A^*)^n g_i), validado contra pinv denso en
16 x 16. Mismo forzado (semilla SEED) para todas las configuraciones; ruido con semilla SEED + 1000 M (como
adveccion_difusion), así que m = 4 reproduce tabla_AD y la ingenua reproduce el naive dinámico de comparacion_justa.
Identidad exacta vs MC con |z| < 4 en todos los estimadores.

Uso:  python -m transporte.experimentos.presupuesto_AD            (todo)
      python -m transporte.experimentos.presupuesto_AD estimar    (cronometra los casos más caros y estima el total)
Guarda salida/transporte/adveccion_difusion/presupuesto/resultados.npz; la tabla, en
transporte/figuras/presupuesto_AD.py.
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
from transporte.comun.estilo import ESTIMADORES
from transporte.comun.fourier import DELTA, bancos_ad, energia, OperadorFourier, OperadorColumnas
from transporte.comun.gp import GPPesos, T_soporte, exacto_pesos, medir, reconstruir
from transporte.comun.imagen import imagen_f
from transporte.comun.rutas import salida

L, N, R = 64, cfg.N_REC, cfg.REPS_2D
S_AD, NU, RAZON = 0.75, 0.1, 0.20
ELLS = cfg.ELLS_64                                           # (2, 8)
PRESUPUESTOS = (2048, 3072)
SEPARACIONES = (2, 4, 8)
COLUMNAS_INGENUA = {2048: (0, 2), 3072: (0, 1, 2)}           # columnas x mod 4 de la ingenua (P / 64 columnas)


def configuraciones():
    """[(P, m, p, M)] con p (M + 1) = P, y las descartadas (P, m, p) por M no entero."""
    validas, descartadas = [], []
    for P in PRESUPUESTOS:
        for m in SEPARACIONES:
            p = L * L // m
            (validas.append((P, m, p, P // p - 1)) if P % p == 0 else descartadas.append((P, m, p)))
    return validas, descartadas


def pinv_fourier(opf, B):
    """O_N^+ denso (d, (N+1) p): columna (n, i) = (O^* O)^{-1} (A^*)^n g_i, por bloques de Fourier."""
    Lc, p = opf.L, B.shape[1]
    O3 = np.empty((Lc * Lc, opf.N + 1, p))
    for n in range(opf.N + 1):
        O3[:, n] = opf.oto_inv(B[n].reshape(p, Lc, Lc)).reshape(p, -1).T
    return O3.reshape(Lc * Lc, -1)


def validar(Lv=16):
    """En 16 x 16, para cada m: alpha_N por bloques y O^+ de pinv_fourier contra la versión densa."""
    err = 0.0
    for m in SEPARACIONES:
        opf = OperadorFourier(Lv, S_AD, NU, N, sep=m)
        B = bancos_ad(sn.banco(Lv, m, 1, DELTA), S_AD, NU, N)
        O = B[:N + 1].reshape(-1, Lv * Lv)
        lam = np.linalg.eigvalsh(O.T @ O)
        e = max(abs(lam[0] - opf.alfa) / lam[-1], np.max(np.abs(pinv_fourier(opf, B) - rc.pinv_O(B, N))))
        assert e < 1e-9, f"m = {m}: bloques != denso ({e:.1e})"
        err = max(err, e)
    return err


def z_mc(e2, exacto):
    return (e2.mean() - exacto) / (e2.std(ddof=1) / np.sqrt(len(e2)))


def preparar():
    """Imagen, soporte de la fuente, kernels y, para cada ell, sigma_w (razón 0,2 por el símbolo) y delta (R, N+1, d)."""
    f = imagen_f(L, cfg.IMAGEN).ravel()
    nf, forma, d = np.linalg.norm(f), (L, L), L * L
    ventana = fz.ventana_gaussiana(L, cfg.FUENTE_64, cfg.FUENTE_ANCHO)
    idx = np.flatnonzero(ventana.ravel() > cfg.UMBRAL_SOPORTE * ventana.max())
    yi, xi = np.divmod(idx, L)
    vS = ventana.ravel()[idx]
    Kt = fz.kernel_temporal(N, cfg.RHO)
    H = op.multiplicador_ad(L, L, (S_AD, 0), NU)
    aplicar = lambda v: op.A_ad(v, (S_AD, 0), NU)
    forzados = {}
    for ell in ELLS:
        K1 = fz.kernel_se_1d(L, ell)
        Sigma1_S = vS[:, None] * (K1[np.ix_(yi, yi)] * K1[np.ix_(xi, xi)]) * vS[None, :]
        sw = RAZON * nf / np.sqrt(fz.energia_delta_N_simbolo(Sigma1_S, Kt, H, idx))
        rng = np.random.default_rng(cfg.SEED)
        W = np.stack([fz.muestrear_forzado(rng, N, L, ell, cfg.RHO, sw, ventana) for _ in range(R)]).reshape(R, N, d)
        forzados[ell] = dict(sw=sw, Sigma_S=sw ** 2 * Sigma1_S, delta=rc.acumular(W, aplicar, forma))
    Anf = np.stack([op.A_ad(f.reshape(forma), (S_AD, 0), NU, n=n).ravel() for n in range(N + 1)])
    return dict(f=f, nf=nf, idx=idx, Kt=Kt, Lt=np.linalg.cholesky(Kt), aplicar=aplicar, Anf=Anf, forzados=forzados)


def caso(P, m, p, M, ell, ctx, op_m):
    """Medianas, exactos y z de sin corregir, GP y oráculo para (P, m, M, ell); op_m = (B, Opinv, TS) de m."""
    B, Opinv, TS = op_m
    f, nf, d, sn_ = ctx["f"], ctx["nf"], L * L, cfg.SIGMA_N
    fz_ = ctx["forzados"][ell]
    delta = fz_["delta"]
    u = medir(ctx["Anf"][None] + delta, B, M, sn_, np.random.default_rng(cfg.SEED + 1000 * M))
    u0 = u[:, :, 0]
    f_hat = dict(sin_corregir=reconstruir(Opinv, B[0], u0))
    if M > 0:
        gp = GPPesos(B, M, ctx["idx"], fz_["Sigma_S"], ctx["Kt"], sn_)
        dh = rc.acumular(gp.media(rc.residuos(u), d), ctx["aplicar"], (L, L))
        f_hat.update(gp=reconstruir(Opinv, B[0], u0, dh), oraculo=reconstruir(Opinv, B[0], u0, delta))
        ex = exacto_pesos(Opinv, gp, TS, sn_)
        del gp, dh
    else:
        ex = dict(sin_corregir=energia(TS, ctx["Lt"], fz.raiz_psd(fz_["Sigma_S"])) + sn_ ** 2 * np.sum(Opinv ** 2))
    e2 = {k: np.sum((v - f) ** 2, axis=1) for k, v in f_hat.items()}
    zs = {k: z_mc(e2[k], ex[k]) for k in e2}
    for k, zk in zs.items():
        assert abs(zk) < 4, f"MC != identidad (P {P}, m {m}, M {M}, ell {ell:g}, {k}): z = {zk:.1f}"
    return dict(med={k: np.median(np.sqrt(v)) / nf for k, v in e2.items()},
                exacto={k: np.sqrt(ex[k]) / nf for k in e2}, zs=zs)


def operador_m(m, E):
    """Bancos (A^*)^e g_i (e = 0..max(N, E)), O_8^+ por Fourier, T_S y alpha_8 para sensores en columnas cada m."""
    opf = OperadorFourier(L, S_AD, NU, N, sep=m)
    B = bancos_ad(sn.banco(L, m, 1, DELTA), S_AD, NU, max(N, E))
    Opinv = pinv_fourier(opf, B)
    return opf, (B, Opinv, None)


def ingenua(P, ell, ctx, TS_cache):
    """Naive dinámico con los P sensores delta, n = 0..N; ruido con la semilla de comparacion_justa (M = P/1024 - 1)."""
    nv = OperadorColumnas(L, S_AD, NU, N, COLUMNAS_INGENUA[P])
    assert len(nv.pix) == P
    if P not in TS_cache:
        TS_cache[P] = nv.T_soporte(ctx["idx"])
    fz_ = ctx["forzados"][ell]
    Mc = P // 1024 - 1
    ruido = np.random.default_rng(cfg.SEED + 1000 * Mc).standard_normal((R, N + 1, Mc + 1, 1024)).reshape(R, N + 1, -1)
    u = (ctx["Anf"][None] + fz_["delta"])[:, :, nv.pix] + cfg.SIGMA_N * ruido
    e2 = np.sum((nv.reconstruir(u) - ctx["f"]) ** 2, axis=1)
    ex = energia(TS_cache[P], ctx["Lt"], fz.raiz_psd(fz_["Sigma_S"])) + cfg.SIGMA_N ** 2 * nv.traza_inv
    z = z_mc(e2, ex)
    assert abs(z) < 4, f"ingenua: MC != exacto (P {P}, ell {ell:g}): z = {z:.1f}"
    return dict(med=np.median(np.sqrt(e2)) / ctx["nf"], exacto=np.sqrt(ex) / ctx["nf"], z=z, alfa=nv.alfa,
                beta=nv.beta)


def correr(solo_estimar=False):
    t0 = time.perf_counter()
    out = salida("transporte", "adveccion_difusion", "presupuesto")
    validas, descartadas = configuraciones()
    val = validar()
    ctx = preparar()
    t_prep = time.perf_counter() - t0
    e_d = np.linalg.norm(cfg.SIGMA_N * np.random.default_rng(cfg.SEED).standard_normal((R, L * L)), axis=1)
    denso = np.median(e_d) / ctx["nf"]

    if solo_estimar:                                       # los casos más caros de cada tipo, con ell = 2
        tiempos = {}
        for (P, m, p, M) in [(2048, 2, 2048, 0), (3072, 8, 512, 5)]:
            t1 = time.perf_counter()
            opf, (B, Opinv, _) = operador_m(m, M)
            TS = T_soporte(B, Opinv, ctx["idx"], N)
            t_op = time.perf_counter() - t1
            t1 = time.perf_counter()
            caso(P, m, p, M, ELLS[0], ctx, (B, Opinv, TS))
            tiempos[(m, M)] = (t_op, time.perf_counter() - t1)
            del B, Opinv, TS
        t1 = time.perf_counter()
        ingenua(3072, ELLS[0], ctx, {})
        t_ing = time.perf_counter() - t1
        print(f"preparación (validación, forzados): {t_prep:.0f} s; ingenua (P = 3072): {t_ing:.0f} s")
        for (m, M), (a, b) in tiempos.items():
            print(f"m = {m}, M = {M}: operador + T_S {a:.0f} s, caso (un ell) {b:.0f} s")
        return

    filas, alfas, chequeos, ops_t = [], {}, [], {}
    for m in SEPARACIONES:
        mias = [c for c in validas if c[1] == m]
        if not mias:
            continue
        t1 = time.perf_counter()
        opf, (B, Opinv, _) = operador_m(m, max(c[3] for c in mias))
        alfas[m] = dict(alfa=opf.alfa, beta=opf.beta, hay=opf.alfa > fr.TOL_FRAME * opf.beta)
        assert alfas[m]["hay"], f"m = {m}: sin frame (alpha_8 = {opf.alfa:.1e})"
        TS = T_soporte(B, Opinv, ctx["idx"], N)
        ops_t[m] = time.perf_counter() - t1
        for (P, _, p, M) in mias:
            for ell in ELLS:
                t1 = time.perf_counter()
                r = caso(P, m, p, M, ell, ctx, (B, Opinv, TS))
                chequeos += list(r["zs"].values())
                filas.append(dict(P=P, m=m, p=p, M=M, ell=ell, sw=ctx["forzados"][ell]["sw"], alfa=opf.alfa,
                                  med=r["med"], exacto=r["exacto"], z=max(abs(v) for v in r["zs"].values()),
                                  seg=time.perf_counter() - t1))
                print(f"   P {P}, m {m}, M {M}, ell {ell:g}: {filas[-1]['seg']:.0f} s", flush=True)
        del B, Opinv, TS

    TS_ing, ing = {}, {}
    for P in PRESUPUESTOS:
        for ell in ELLS:
            ing[(P, ell)] = ingenua(P, ell, ctx, TS_ing)
            chequeos.append(ing[(P, ell)]["z"])
    for r in filas:
        g = ing[(r["P"], r["ell"])]
        r.update(ingenua=g["med"], ex_ingenua=g["exacto"], z_ingenua=g["z"], alfa_ingenua=g["alfa"])

    # referencias: m = 4 debe reproducir adveccion_difusion; la ingenua, el naive dinámico de comparacion_justa
    ref = {}
    ruta_ad = os.path.join(salida("transporte", "adveccion_difusion"), "resultados.npz")
    if os.path.exists(ruta_ad):
        for a in res_io.tabla(np.load(ruta_ad), "filas"):
            if a["s"] == S_AD and np.isclose(a["nu"], NU) and np.isclose(a["razon"], RAZON):
                ref[("ad", a["ell"], a["M"])] = a["med"]
    ruta_cj = os.path.join(salida("transporte", "adveccion_difusion", "comparacion_justa"), "resultados.npz")
    if os.path.exists(ruta_cj):
        c = np.load(ruta_cj)
        for i in range(len(c["nu"])):
            if np.isclose(c["nu"][i], NU) and np.isclose(c["razon"][i], RAZON):
                ref[("cj", float(c["ell"][i]), 1024 * (int(c["M"][i]) + 1))] = float(c["dinamico"][i])
    dif_ad = [abs(r["med"][k] / ref[("ad", r["ell"], r["M"])][k] - 1) for r in filas
              if r["m"] == 4 and ("ad", r["ell"], r["M"]) in ref for k in r["med"]]
    dif_cj = [abs(g["med"] / ref[("cj", ell, P)] - 1) for (P, ell), g in ing.items() if ("cj", ell, P) in ref]

    pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    ruta = os.path.join(out, "resultados.npz")
    res_io.guardar(ruta, tablas=dict(filas=filas,
                                     ingenua=[dict(P=P, ell=ell, **g) for (P, ell), g in ing.items()],
                                     frames=[dict(m=m, p=L * L // m, **a) for m, a in alfas.items()],
                                     descartadas=[dict(P=P, m=m, p=p) for P, m, p in descartadas]),
                   denso=denso, chequeos=np.array(chequeos), validacion=val, s=S_AD, nu=NU, razon=RAZON,
                   dif_ad=np.array(dif_ad), dif_cj=np.array(dif_cj))

    print("=" * 120)
    print(f"COMPARACIÓN POR PRESUPUESTO, ADVECCIÓN-DIFUSIÓN: s = {S_AD:g}, nu = {NU:g}, 64 x 64, N = {N}, "
          f"sigma_n = {cfg.SIGMA_N:g}, rho = {cfg.RHO}, fuente {cfg.FUENTE_64}, razón {RAZON:g}, {R} realizaciones")
    print(f"validación 16 x 16 (alpha_8 y O^+ por bloques vs denso, m en {SEPARACIONES}): error {val:.1e} OK")
    print("configuraciones (P, m, p, M): " + ", ".join(str(c) for c in validas))
    print("descartadas (M no entero): " + ", ".join(f"P {P}, m {m} (p = {p}: P/p = {P / p:g})" for P, m, p in descartadas))
    print("frame de los sensores base (columnas cada m, n = 0..8): " + ", ".join(
        f"m {m} (p = {L * L // m}): alpha_8 = {a['alfa']:.3e}, alpha/lmax = {a['alfa'] / a['beta']:.2e} "
        f"{'OK' if a['hay'] else 'SIN FRAME'}" for m, a in alfas.items()))
    print("frame de la ingenua: " + ", ".join(f"P {P}: alpha_8 = {ing[(P, ELLS[0])]['alfa']:.3e}" for P in PRESUPUESTOS))
    print(f"asserts OK: identidad exacta vs MC en {len(chequeos)} casos, |z| máx = {np.max(np.abs(chequeos)):.2f} (< 4)")
    print(f"m = 4 vs adveccion_difusion (mismo forzado y ruido): {len(dif_ad)} medianas, dif. relativa máx "
          f"{max(dif_ad) if dif_ad else float('nan'):.1e}")
    print(f"ingenua vs naive dinámico de comparacion_justa: {len(dif_cj)} medianas, dif. relativa máx "
          f"{max(dif_cj) if dif_cj else float('nan'):.1e}")
    print("sigma_w: " + ", ".join(f"ell {ell:g}: {ctx['forzados'][ell]['sw']:.4g}" for ell in ELLS))
    print(f"denso (4096 sensores, n = 0): {denso:.3e}")
    print(f"{'P':>5s} {'m':>2s} {'p':>5s} {'M':>2s} {'alpha_8':>9s} {'ell':>4s}  {'sin corr.':>9s} {'ingenua':>9s} "
          f"{'GP':>9s} {'oráculo':>9s} {'denso':>9s} {'|z| máx':>7s}")
    for r in filas:
        g = lambda k: f"{r['med'][k]:9.3e}" if k in r["med"] else f"{'—':>9s}"
        print(f"{r['P']:5d} {r['m']:2d} {r['p']:5d} {r['M']:2d} {r['alfa']:9.2e} {r['ell']:4g}  {g('sin_corregir')} "
              f"{r['ingenua']:9.3e} {g('gp')} {g('oraculo')} {denso:9.3e} {max(r['z'], abs(r['z_ingenua'])):7.2f}")
    print("tiempos por operador (bancos, O^+, T_S): " + ", ".join(f"m {m}: {t:.0f} s" for m, t in ops_t.items()))
    print(f"tiempo total {time.perf_counter() - t0:.0f} s; pico de memoria {pico:.2f} GB")
    print("archivo: " + os.path.relpath(ruta, salida()))


if __name__ == "__main__":
    correr(solo_estimar="estimar" in sys.argv[1:])
