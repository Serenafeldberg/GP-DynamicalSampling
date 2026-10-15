"""
Comparación a igual cantidad de sensores (Sección 4.7.5): cómputo.

Para cada M, el esquema con GP usa N2 = 1024 (M + 1) sensores (1024 columnas cada 4 y sus M copias). Líneas de base
con N2 sensores puntuales y SIN GP:
  (a) naive dinámico: columnas {0, 2} mod 4 (M = 1) o {0, 1, 2} mod 4 (M = 2), instantes n = 0..N, mismo forzado y
      mismos números de ruido que el experimento original; f_hat = O_N^+ u. Error exacto (lineal-gaussiano, en el
      espacio del soporte) + MC con |z| < 4. En advección-difusión se agrega M = 3 (todas las columnas) para el resumen.
  (b) naive estático: los mismos N2 sensores en n = 0, sin dinámica ni forzado: (b1) síntesis (columnas no medidas
      = 0) y (b2) interpolación lineal periódica en x.
GP, oráculo y sin corregir (y la realización de la figura) se leen de los resultados.npz de experimento_2D_64 y
adveccion_difusion, que deben correrse antes.

Uso:  python -m transporte.experimentos.comparacion_justa entero    (s = 1, nu = 0)
      python -m transporte.experimentos.comparacion_justa ad        (s = 0,75, nu en {0,05; 0,1})
Guarda salida/transporte/<experimento>/comparacion_justa/resultados.npz; tablas y figuras en
transporte/figuras/comparacion_justa.py.
"""
import os
import resource
import sys
import time

import numpy as np

from transporte.comun import config as cfg
from transporte.comun import operadores as op
from transporte.comun import forzado as fz
from transporte.comun import reconstruccion as rc
from transporte.comun import resultados as res_io
from transporte.comun.fourier import OperadorColumnas, energia
from transporte.comun.imagen import imagen_f
from transporte.comun.rutas import salida
from transporte.comun.uno_d import sigma_w_usado

L, N, R = 64, cfg.N_REC, cfg.REPS_2D
ELLS, RAZONES, MS = (2.0, 8.0), (0.05, 0.20), (1, 2)
COLUMNAS = {0: (0,), 1: (0, 2), 2: (0, 1, 2), 3: (0, 1, 2, 3)}   # columnas medidas (mod 4) con N2 = 1024 (M + 1)
FIGURA = (2.0, 0.20, 2)                                          # (ell, razón, M) de las figuras de presentación
S_AD, NU_RESUMEN = 0.75, 0.1                                     # configuración del resumen de advección-difusión


def validar(Lv=16, N_=4, s=0.75, nu=0.1):
    """En 16 x 16: alpha y O^+ por bloques de OperadorColumnas contra la versión densa."""
    out = []
    for cols in COLUMNAS.values():
        opc = OperadorColumnas(Lv, s, nu, N_, cols)
        d = Lv * Lv
        filas = np.zeros((len(opc.pix), d))
        filas[np.arange(len(opc.pix)), opc.pix] = 1.0
        O = np.concatenate([op.A_ad_adj(filas.reshape(-1, Lv, Lv), (s, 0), nu, n=n).reshape(len(opc.pix), d)
                            for n in range(N_ + 1)])
        lam = np.linalg.eigvalsh(O.T @ O)
        u = np.random.default_rng(0).standard_normal((2, N_ + 1, len(opc.pix)))
        e = np.max(np.abs(opc.reconstruir(u) - u.reshape(2, -1) @ np.linalg.pinv(O).T))
        assert abs(lam[0] - opc.alfa) < 1e-10 and e < 1e-10, (lam[0] - opc.alfa, e)
        out.append(max(abs(lam[0] - opc.alfa), e))
    return max(out)


def originales(operador):
    """Medianas (sin corregir, GP, oráculo) por (nu, ell, razón, M) y la realización de la figura, desde el npz del
    experimento de origen."""
    if operador == "entero":
        z = np.load(os.path.join(salida("transporte", "experimento_2D_64"), "resultados.npz"))
        filas = [dict(r, nu=0.0) for r in res_io.tabla(z, "filas")]
        ejemplo = {k: z[f"ejemplo_ell{FIGURA[0]:g}_{k}"] for k in ("sin_corregir", "gp", "oraculo", "delta")}
    else:
        z = np.load(os.path.join(salida("transporte", "adveccion_difusion"), "resultados.npz"))
        filas = [r for r in res_io.tabla(z, "filas") if r["s"] == S_AD]
        ejemplo = {k: z[f"ejemplo_{k}"] for k in ("sin_corregir", "gp", "oraculo", "delta")}
    orig = {(r["nu"], r["ell"], r["razon"], r["M"]): tuple(r["med"][k] for k in ("sin_corregir", "gp", "oraculo"))
            for r in filas}
    return orig, ejemplo


def naive_M3(f, sw, ventana, idx, vS, yi, xi, Kt, Lt):
    """Naive dinámico con todas las columnas (4096 sensores), realización y ruido del original con M = 3
    (advección-difusión, nu = NU_RESUMEN, ell y razón de FIGURA)."""
    ell, d, forma, M = FIGURA[0], L * L, (L, L), 3
    K1 = fz.kernel_se_1d(L, ell)
    Rs = sw * fz.raiz_psd(vS[:, None] * (K1[np.ix_(yi, yi)] * K1[np.ix_(xi, xi)]) * vS[None, :])
    nv = OperadorColumnas(L, S_AD, NU_RESUMEN, N, COLUMNAS[3])
    rng = np.random.default_rng(cfg.SEED)
    W = np.stack([fz.muestrear_forzado(rng, N, L, ell, cfg.RHO, sw, ventana) for _ in range(R)]).reshape(R, N, d)
    delta = rc.acumular(W, lambda v: op.A_ad(v, (S_AD, 0), NU_RESUMEN), forma)
    Anf = np.stack([op.A_ad(f.reshape(forma), (S_AD, 0), NU_RESUMEN, n=n).ravel() for n in range(N + 1)])
    ruido = np.random.default_rng(cfg.SEED + 1000 * M).standard_normal((R, N + 1, M + 1, 1024)).reshape(R, N + 1, -1)
    u = (Anf[None] + delta)[:, :, nv.pix] + cfg.SIGMA_N * ruido
    e2 = np.sum((nv.reconstruir(u) - f) ** 2, axis=1)
    ex = energia(nv.T_soporte(idx), Lt, Rs) + cfg.SIGMA_N ** 2 * nv.traza_inv
    z = z_mc(e2, ex)
    assert abs(z) < 4, f"naive M = 3: MC != exacto (z = {z:.1f})"
    nf = np.linalg.norm(f)
    return dict(mediana=np.median(np.sqrt(e2)) / nf, exacto=np.sqrt(ex) / nf, z=z, alfa=nv.alfa)


def estatico(f, cols, rng, sigma_n, periodo=cfg.SEP):
    """Naive estático en n = 0: (b1) síntesis con ceros y (b2) interpolación lineal periódica en x.
    Devuelve errores MC (R,), exactos (E||.||^2) y los f_hat de la primera realización."""
    img = f.reshape(L, L)
    medido = np.isin(np.arange(L) % periodo, cols)
    u = img[None] + sigma_n * rng.standard_normal((R, L, L))                       # solo se usan columnas medidas
    b1 = np.where(medido[None, None, :], u, 0.0)
    b2 = b1.copy()
    for x in np.flatnonzero(~medido):                                              # vecinos medidos más cercanos
        izq = next(k for k in range(1, L) if medido[(x - k) % L])
        der = next(k for k in range(1, L) if medido[(x + k) % L])
        wi, wd = der / (izq + der), izq / (izq + der)
        b2[:, :, x] = wi * u[:, :, (x - izq) % L] + wd * u[:, :, (x + der) % L]
    sesgo1 = np.where(medido[None, :], 0.0, -img)
    interp = np.where(medido[None, :], img, 0.0)
    pesos2 = np.zeros(L)
    for x in np.flatnonzero(~medido):
        izq = next(k for k in range(1, L) if medido[(x - k) % L])
        der = next(k for k in range(1, L) if medido[(x + k) % L])
        wi, wd = der / (izq + der), izq / (izq + der)
        interp[:, x] = wi * img[:, (x - izq) % L] + wd * img[:, (x + der) % L]
        pesos2[x] = wi ** 2 + wd ** 2
    pesos2[medido] = 1.0
    sesgo2 = interp - img
    n_med = medido.sum() * L
    ex1 = np.sum(sesgo1 ** 2) + sigma_n ** 2 * n_med
    ex2 = np.sum(sesgo2 ** 2) + sigma_n ** 2 * L * pesos2.sum()
    e1 = np.sum((b1 - img[None]) ** 2, axis=(1, 2))
    e2 = np.sum((b2 - img[None]) ** 2, axis=(1, 2))
    return (e1, ex1, b1[0].ravel()), (e2, ex2, b2[0].ravel())


def z_mc(e2, exacto):
    return (e2.mean() - exacto) / (e2.std(ddof=1) / np.sqrt(len(e2)))


def correr(operador):
    t0 = time.perf_counter()
    if operador == "entero":
        nus, s, carpeta = (0.0,), 1.0, "experimento_2D_64"
    else:
        nus, s, carpeta = (0.05, 0.1), S_AD, "adveccion_difusion"
    out = salida("transporte", carpeta, "comparacion_justa")
    val = validar()
    orig, ejemplo = originales(operador)
    f = imagen_f(L, cfg.IMAGEN).ravel()
    nf = np.linalg.norm(f)
    sn_, forma, d = cfg.SIGMA_N, (L, L), L * L
    ventana = fz.ventana_gaussiana(L, cfg.FUENTE_64, cfg.FUENTE_ANCHO)
    idx = np.flatnonzero(ventana.ravel() > cfg.UMBRAL_SOPORTE * ventana.max())
    yi, xi = np.divmod(idx, L)
    vS = ventana.ravel()[idx]
    Kt = fz.kernel_temporal(N, cfg.RHO)
    Lt = np.linalg.cholesky(Kt)
    e_d = np.linalg.norm(sn_ * np.random.default_rng(cfg.SEED).standard_normal((R, d)), axis=1)
    denso = np.median(e_d) / nf

    # naive estático: no depende del operador ni del forzado
    est = {}
    for M in MS:
        (e1, ex1, f1), (e2, ex2, f2) = estatico(f, COLUMNAS[M], np.random.default_rng(cfg.SEED), sn_)
        est[M] = dict(b1=np.median(np.sqrt(e1)) / nf, b2=np.median(np.sqrt(e2)) / nf, ex_b1=np.sqrt(ex1) / nf,
                      ex_b2=np.sqrt(ex2) / nf, z_b1=z_mc(e1, ex1), z_b2=z_mc(e2, ex2), f_b2=f2, f_b1=f1)

    filas, figura, alfas, chequeos = [], None, {}, []
    for nu in nus:
        H = op.multiplicador_ad(L, L, (s, 0), nu)
        aplicar = (lambda v: op.A(v, (1, 0))) if operador == "entero" else (lambda v, nu=nu: op.A_ad(v, (s, 0), nu))
        Anf = np.stack([op.A_ad(f.reshape(forma), (s, 0), nu, n=n).ravel() for n in range(N + 1)])
        naives = {M: OperadorColumnas(L, s, nu, N, COLUMNAS[M]) for M in MS}
        TS = {M: naives[M].T_soporte(idx) for M in MS}
        for M in MS:
            alfas[(nu, M)] = naives[M].alfa
            assert naives[M].alfa > 1e-8, f"naive dinámico sin frame (nu {nu}, M {M}): alpha = {naives[M].alfa:.1e}"
        for ell in ELLS:
            K1 = fz.kernel_se_1d(L, ell)
            Sigma1_S = vS[:, None] * (K1[np.ix_(yi, yi)] * K1[np.ix_(xi, xi)]) * vS[None, :]
            Rs1 = fz.raiz_psd(Sigma1_S)
            for razon in RAZONES:
                if operador == "entero":                                            # como experimento_2D_64.combinacion
                    Sigma1 = fz.sigma_espacial(np.kron(K1, K1), ventana, 1.0)
                    cfg.RAZON_FORZADO = razon
                    sw, _ = sigma_w_usado(Sigma1, Kt, forma, nf)
                    cfg.RAZON_FORZADO = None
                    del Sigma1
                else:                                                               # como experimento_adveccion_difusion
                    sw = razon * nf / np.sqrt(fz.energia_delta_N_simbolo(Sigma1_S, Kt, H, idx))
                rng = np.random.default_rng(cfg.SEED)
                W = np.stack([fz.muestrear_forzado(rng, N, L, ell, cfg.RHO, sw, ventana) for _ in range(R)]).reshape(R, N, d)
                delta = rc.acumular(W, aplicar, forma)
                del W
                Y = Anf[None] + delta
                for M in MS:
                    # mismos números de ruido que el original: entero = la semilla sigue después de las R muestras
                    # de W (experimento_2D_64 vuelve a muestrear W para cada M); AD = semilla SEED + 1000 M
                    if operador == "entero":
                        rng_r = np.random.default_rng(cfg.SEED)
                        for _ in range(R):
                            fz.muestrear_forzado(rng_r, N, L, ell, cfg.RHO, sw, ventana)
                    else:
                        rng_r = np.random.default_rng(cfg.SEED + 1000 * M)
                    ruido = rng_r.standard_normal((R, N + 1, M + 1, 1024)).reshape(R, N + 1, -1)
                    nv = naives[M]
                    u = Y[:, :, nv.pix] + sn_ * ruido
                    fh = nv.reconstruir(u)
                    e2 = np.sum((fh - f) ** 2, axis=1)
                    ex = energia(TS[M], Lt, sw * Rs1) + sn_ ** 2 * nv.traza_inv
                    z = z_mc(e2, ex)
                    chequeos.append(z)
                    assert abs(z) < 4, f"naive dinámico: MC != exacto (nu {nu}, ell {ell}, razón {razon}, M {M}): z = {z:.1f}"
                    sc, gp, orc = orig[(nu, ell, razon, M)]
                    filas.append(dict(nu=nu, ell=ell, razon=razon, M=M, N2=1024 * (M + 1), sc=sc, gp=gp, oraculo=orc,
                                      dinamico=np.median(np.sqrt(e2)) / nf, ex_dinamico=np.sqrt(ex) / nf, z=z,
                                      b1=est[M]["b1"], b2=est[M]["b2"], denso=denso))
                    if (ell, razon, M) == FIGURA and nu == nus[-1]:
                        figura = dict(nu=nu, f_dinamico=fh[0], delta_N=delta[0, N].copy(), sw=sw, y=Y[0].copy())
                del delta, Y

    # realización de la figura: viene del npz del experimento de origen (misma semilla); se verifica delta_N
    ell, razon, M = FIGURA
    nu = figura["nu"]
    sc, gp, orc = orig[(nu, ell, razon, M)]
    assert np.allclose(np.reshape(ejemplo["delta"], (-1, L * L))[-1], figura["delta_N"], rtol=0, atol=1e-12), "la realización no es la misma"
    figura.update(f_gp=ejemplo["gp"], f_sin_corregir=ejemplo["sin_corregir"], f_b2=est[M]["f_b2"], f=f,
                  fuente=np.array(cfg.FUENTE_64))
    naive3 = naive_M3(f, figura["sw"], ventana, idx, vS, yi, xi, Kt, Lt) if operador == "ad" else None

    pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    total = time.perf_counter() - t0
    claves = ("nu", "ell", "razon", "M", "N2", "sc", "dinamico", "ex_dinamico", "z", "b1", "b2", "gp", "oraculo", "denso")
    extra = {} if naive3 is None else {f"naive_M3_{k}": v for k, v in naive3.items()}
    ruta = os.path.join(out, "resultados.npz")
    np.savez(ruta, operador=operador, s=s, **extra,
             **{k: np.array([r[k] for r in filas]) for k in claves},
             alfa_nu=np.array([k[0] for k in alfas]), alfa_M=np.array([k[1] for k in alfas]),
             alfa=np.array(list(alfas.values())),
             **{f"est_{k}_M{M}": v for M, e in est.items() for k, v in e.items() if not k.startswith("f_")},
             **{f"fig_{k}": v for k, v in figura.items()})

    print("=" * 110)
    print(f"COMPARACIÓN A IGUAL CANTIDAD DE SENSORES: {operador} (s = {s:g}, nu en {nus}), 64 x 64, N = {N}, "
          f"sigma_n = {sn_:g}, {R} realizaciones")
    print(f"validación 16 x 16 (alpha y O^+ por bloques con columnas {list(COLUMNAS.values())}): error {val:.1e} OK")
    print("frame del naive dinámico: " + ", ".join(f"nu {k[0]:g}, M {k[1]} ({1024 * (k[1] + 1)} sensores): alpha_N = {v:.3f}"
                                                   for k, v in alfas.items()))
    print(f"asserts OK: naive dinámico MC == exacto en {len(chequeos)} casos, |z| máx = {np.max(np.abs(chequeos)):.2f}; "
          "naive estático: " + ", ".join(f"M {M}: |z| b1 {abs(e['z_b1']):.2f}, b2 {abs(e['z_b2']):.2f}" for M, e in est.items()))
    print(f"realización de la figura {FIGURA} (nu = {nu:g}): delta_N == experimento de origen OK")
    if naive3 is not None:
        print(f"naive dinámico M = 3 (4096 sensores, nu = {NU_RESUMEN:g}, ell = {FIGURA[0]:g}, razón {FIGURA[1]:g}): "
              f"mediana {naive3['mediana']:.3e}, exacto RMS {naive3['exacto']:.3e}, |z| = {abs(naive3['z']):.2f}")
    print(f"naive estático (exacto RMS): " + ", ".join(f"M {M}: b1 {e['ex_b1']:.3e}, b2 {e['ex_b2']:.3e}" for M, e in est.items()))
    print(f"{'nu':>5s} {'ell':>4s} {'razón':>5s} {'M':>2s} {'N2':>5s}  {'sin corr.(1024)':>15s} {'dinámico':>9s} "
          f"{'est. b1':>9s} {'est. b2':>9s} {'GP':>9s} {'oráculo':>9s} {'denso':>9s}")
    for r in filas:
        print(f"{r['nu']:5g} {r['ell']:4g} {r['razon']:5g} {r['M']:2d} {r['N2']:5d}  {r['sc']:15.2e} {r['dinamico']:9.2e} "
              f"{r['b1']:9.2e} {r['b2']:9.2e} {r['gp']:9.2e} {r['oraculo']:9.2e} {r['denso']:9.2e}")
    print(f"tiempo {total:.0f} s; pico de memoria {pico:.2f} GB")
    print("archivo: " + os.path.relpath(ruta, salida()))


if __name__ == "__main__":
    correr(sys.argv[1] if len(sys.argv) > 1 else "entero")
