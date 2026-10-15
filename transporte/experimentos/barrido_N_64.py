"""
Barrido en el horizonte N con profundidad M = 1, en 64 x 64 (Sección 4.7.5): cómputo.

Operadores: (a) transporte entero s = 1, nu = 0; (b) advección-difusión s = 0,75 con nu en {0,05; 0,1}.
Sensores delta en columnas cada SEP (p = 1024); la razón ||delta_N|| / ||f|| se fija en RAZON para cada N
con forzado.energia_delta_N_simbolo.

Cómputo sin armar O_N: como A es un multiplicador de Fourier y los sensores son deltas, O_N^* O_N es diagonal
por bloques de alias (frames.py) y (O^* O)^{-1} se aplica con FFT + bloques de 4 x 4; O^+ u = (O^* O)^{-1} O^* u.
Con M = 1 los residuos solo usan el banco m = 0 (deltas), así que el posterior (espacio de observaciones,
experimento_2D_64.GPSoporte) solo involucra los sensores que caen en el soporte de la fuente. La identidad
exacta se evalúa en el espacio del soporte con T_S[:, k] = (O^*O)^{-1} A^{*k} U_{N-k},
U_J = sum_{j=0}^{J} A^{*j} G G^T A^j (restringido a S).
Salida: salida/transporte/barrido_N/resultados.npz (figuras en transporte/figuras/barrido_N.py).
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
from transporte.comun import reconstruccion as rc
from transporte.comun import error_exacto as ee
from transporte.comun.fourier import OperadorFourier, exacto_fourier
from transporte.comun.gp import GPSoporte, medir, reconstruir
from transporte.comun.imagen import imagen_f
from transporte.comun.rutas import salida

OUT = salida("transporte", "barrido_N")
L, M, RAZON = 64, 1, 0.05
NS = (4, 8, 12, 16, 24, 32)
ELLS = (2.0, 8.0)
OPERADORES = ((1.0, 0.0), (0.75, 0.05), (0.75, 0.1))          # (s, nu)
ESTIMADORES = ("sin_corregir", "gp", "oraculo")


def validar(Lv=16, N=4, s=0.75, nu=0.1, sigma_n=1e-3):
    """En 16 x 16: alpha, O^+ por bloques, reconstrucción e identidad contra las versiones densas."""
    d = Lv * Lv
    opf = OperadorFourier(Lv, s, nu, N)
    B = opf.bancos(N)
    Opinv = rc.pinv_O(B, N)
    a_ref = fr.alpha_bloques_simbolo(Lv, cfg.SEP, 1, [("delta",)], opf.H, N)[0][-1]
    rng = np.random.default_rng(0)
    u0, delta = rng.standard_normal((3, N + 1, B.shape[1])), rng.standard_normal((3, N + 1, d))
    e_rec = np.max(np.abs(opf.reconstruir(u0, delta) - reconstruir(Opinv, B[0], u0, delta)))
    v = fz.ventana_gaussiana(Lv, (5, 8), 1.5)
    K1 = fz.kernel_se_1d(Lv, 2.0)
    Sigma = fz.sigma_espacial(np.kron(K1, K1), v, 0.05)
    Kt = fz.kernel_temporal(N, cfg.RHO)
    idx = np.arange(d)
    gp = GPSoporte(B, M, idx, Sigma, Kt, sigma_n)
    A = op.A_ad(np.eye(d).reshape(d, Lv, Lv), (s, 0), nu).reshape(d, d).T
    ex_d = ee.mse_exacto(A, B[0], Sigma, Kt, M, N, sigma_n)
    ex_f = exacto_fourier(opf, gp, opf.T_soporte(idx), Sigma, sigma_n)
    e_ex = max(abs(ex_f[k] - ex_d[k]) / ex_d[k] for k in ESTIMADORES)
    assert abs(opf.alfa - a_ref) < 1e-12 and e_rec < 1e-10 and e_ex < 1e-8, (opf.alfa - a_ref, e_rec, e_ex)
    return abs(opf.alfa - a_ref), e_rec, e_ex


def gamma_N(H, N):
    """Factor de acumulación: suma y raíz de suma de cuadrados de ||A^k||, k = 0..N-1 (||A^k|| = max|a_hat|^k)."""
    na = np.abs(H).max()
    pot = na ** np.arange(N)
    return float(pot.sum()), float(np.sqrt(np.sum(pot ** 2)))


def caso(ctx, s, nu, ell, N, R):
    """Una combinación (operador, ell, N): MC + identidad exacta. Devuelve un dict de resultados."""
    t0 = time.perf_counter()
    d, forma, idx, sn_ = L * L, (L, L), ctx["idx"], cfg.SIGMA_N
    opf = OperadorFourier(L, s, nu, N)
    B = opf.bancos(M)                                                           # bancos m = 0, 1 para medir
    Kt = fz.kernel_temporal(N, cfg.RHO)
    K1 = fz.kernel_se_1d(L, ell)
    yi, xi = np.divmod(idx, L)
    vS = ctx["ventana"].ravel()[idx]
    Sigma1_S = vS[:, None] * (K1[np.ix_(yi, yi)] * K1[np.ix_(xi, xi)]) * vS[None, :]
    e1 = fz.energia_delta_N_simbolo(Sigma1_S, Kt, opf.H, idx)
    sw = RAZON * ctx["norma_f"] / np.sqrt(e1)
    Sigma_S = sw ** 2 * Sigma1_S
    aplicar = lambda v: op.A_ad(v, (s, 0), nu)

    rng = np.random.default_rng(cfg.SEED)
    W = np.stack([fz.muestrear_forzado(rng, N, L, ell, cfg.RHO, sw, ctx["ventana"]) for _ in range(R)]).reshape(R, N, d)
    delta = rc.acumular(W, aplicar, forma)
    del W
    en = np.sum(delta[:, N] ** 2, axis=1)
    z_razon = (en.mean() - sw ** 2 * e1) / (en.std(ddof=1) / np.sqrt(R))
    assert abs(z_razon) < 4, f"razón MC != objetivo: z = {z_razon:.1f}"
    Anf = np.stack([op.A_ad(ctx["f"].reshape(forma), (s, 0), nu, n=n).ravel() for n in range(N + 1)])
    u = medir(Anf[None] + delta, B, M, sn_, rng)
    gp = GPSoporte(B, M, idx, Sigma_S, Kt, sn_)
    dh = rc.acumular(gp.media(rc.residuos(u), d), aplicar, forma)
    u0 = u[:, :, 0]
    f_hat = dict(sin_corregir=opf.reconstruir(u0), gp=opf.reconstruir(u0, dh), oraculo=opf.reconstruir(u0, delta))
    del delta, dh, u
    e2 = {k: np.sum((v - ctx["f"]) ** 2, axis=1) for k, v in f_hat.items()}
    TS = opf.T_soporte(idx)
    ex = exacto_fourier(opf, gp, TS, Sigma_S, sn_)
    del TS
    zs = {k: (e2[k].mean() - ex[k]) / (e2[k].std(ddof=1) / np.sqrt(R)) for k in ESTIMADORES}
    for k, zk in zs.items():
        assert abs(zk) < 4, f"MC != identidad (s {s}, nu {nu}, ell {ell}, N {N}, {k}): z = {zk:.1f}"
    nf = ctx["norma_f"]
    return dict(alfa=opf.alfa, beta=opf.beta, sw=sw, razon_mc=np.sqrt(en.mean()) / nf, z_razon=z_razon,
                med=[np.median(np.sqrt(e2[k])) / nf for k in ESTIMADORES],
                q25=[np.percentile(np.sqrt(e2[k]), 25) / nf for k in ESTIMADORES],
                q75=[np.percentile(np.sqrt(e2[k]), 75) / nf for k in ESTIMADORES],
                exacto=[np.sqrt(ex[k]) / nf for k in ESTIMADORES], z=[zs[k] for k in ESTIMADORES],
                normA=float(np.abs(opf.H).max()), gamma=gamma_N(opf.H, N), kz=gp.dim, seg=time.perf_counter() - t0)


def contexto():
    f = imagen_f(L, cfg.IMAGEN).ravel()
    ventana = fz.ventana_gaussiana(L, cfg.FUENTE_64, cfg.FUENTE_ANCHO)
    idx = np.flatnonzero(ventana.ravel() > cfg.UMBRAL_SOPORTE * ventana.max())
    return dict(f=f, norma_f=np.linalg.norm(f), ventana=ventana, idx=idx)


def tendencia(Ns, gp):
    """baja / satura / sube / baja y sube, y el N del mínimo."""
    i = int(np.argmin(gp))
    if i == 0:
        t = "sube" if gp[-1] > gp[0] * 1.05 else "satura"
    elif i == len(gp) - 1:
        t = "baja (sin saturar en el rango)" if gp[-2] > gp[-1] * 1.05 else "baja y satura"
    else:
        t = "baja y satura" if gp[-1] <= gp[i] * 1.05 else "baja y después sube"
    return t, Ns[i]


def estimar():
    """Un solo caso con N = 32 (el más caro) para estimar tiempo y memoria."""
    ctx = contexto()
    r = caso(ctx, 0.75, 0.1, 2.0, 32, cfg.REPS_2D)
    pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print(f"caso N = 32 (s 0,75, nu 0,1, ell 2): {r['seg']:.0f} s, pico {pico:.2f} GB, Kz {r['kz']}")
    return r["seg"], pico


def main(R):
    t0 = time.perf_counter()
    val = validar()
    ctx = contexto()
    d = L * L
    e_d = np.linalg.norm(cfg.SIGMA_N * np.random.default_rng(cfg.SEED).standard_normal((R, d)), axis=1)
    denso = np.median(e_d) / ctx["norma_f"]
    forma = (len(OPERADORES), len(ELLS), len(NS))
    campos = {k: np.zeros(forma + (3,)) for k in ("med", "q25", "q75", "exacto", "z")}
    escalares = {k: np.zeros(forma) for k in ("alfa", "beta", "sw", "razon_mc", "z_razon", "normA", "seg", "kz")}
    gam = np.zeros(forma + (2,))
    for a, (s, nu) in enumerate(OPERADORES):
        for b, ell in enumerate(ELLS):
            for c, N in enumerate(NS):
                r = caso(ctx, s, nu, ell, N, R)
                for k in campos:
                    campos[k][a, b, c] = r[k]
                for k in escalares:
                    escalares[k][a, b, c] = r[k]
                gam[a, b, c] = r["gamma"]
                print(f"   s {s:g}, nu {nu:g}, ell {ell:g}, N {N:2d}: {r['seg']:.0f} s  GP {r['med'][1]:.2e}  "
                      f"|z| máx {np.max(np.abs(r['z'])):.2f}", flush=True)
    pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    total = time.perf_counter() - t0
    np.savez(os.path.join(OUT, "resultados.npz"), operadores=np.array(OPERADORES), ells=np.array(ELLS), Ns=np.array(NS),
             estimadores=np.array(ESTIMADORES), denso=denso, R=R, razon=RAZON, M=M, gamma=gam, tiempo=total, pico=pico,
             **campos, **escalares)

    print("=" * 110)
    print(f"BARRIDO EN N (64 x 64, M = {M}, p = 1024, columnas cada {cfg.SEP}, sigma_n = {cfg.SIGMA_N:g}, rho = {cfg.RHO}, "
          f"razón {RAZON}, fuente {cfg.FUENTE_64}, {R} realizaciones)")
    print(f"validación 16 x 16 (alpha, O^+ por bloques, identidad vs error_exacto.py): {val[0]:.1e}, {val[1]:.1e}, {val[2]:.1e} OK")
    print(f"asserts OK: identidad vs MC en {campos['z'].size} casos, |z| máx = {np.max(np.abs(campos['z'])):.2f}; "
          f"razón MC == objetivo, |z| máx = {np.max(np.abs(escalares['z_razon'])):.2f}")
    print(f"denso ({d} sensores, n = 0): {denso:.3e}")
    for a, (s, nu) in enumerate(OPERADORES):
        print(f"\noperador s = {s:g}, nu = {nu:g}: ||A|| = {escalares['normA'][a, 0, 0]:.4f}; Gamma_N (suma ||A^k||, "
              f"raíz suma ||A^k||^2), k < N: " + ", ".join(f"N {N}: ({gam[a, 0, c, 0]:g}; {gam[a, 0, c, 1]:.2f})"
                                                            for c, N in enumerate(NS)))
        print(f"   alpha_N: " + ", ".join(f"N {N}: {escalares['alfa'][a, 0, c]:.2e}" for c, N in enumerate(NS)))
        for b, ell in enumerate(ELLS):
            med = campos["med"][a, b]
            t, Nmin = tendencia(NS, med[:, 1])
            print(f"   ell {ell:g}: GP {t}; mínimo en N = {Nmin} ({med[NS.index(Nmin), 1]:.2e})")
            print(f"      {'N':>3s} {'sigma_w':>8s} {'sin corr.':>9s} {'GP':>9s} {'oráculo':>9s} {'GP exacto':>10s} {'|z| máx':>8s}")
            for c, N in enumerate(NS):
                print(f"      {N:3d} {escalares['sw'][a, b, c]:8.4f} {med[c, 0]:9.2e} {med[c, 1]:9.2e} {med[c, 2]:9.2e} "
                      f"{campos['exacto'][a, b, c, 1]:10.2e} {np.max(np.abs(campos['z'][a, b, c])):8.2f}")
    print(f"\ntiempo total {total:.0f} s; pico de memoria {pico:.2f} GB")
    print("archivo: " + os.path.relpath(os.path.join(OUT, "resultados.npz"), salida()))


if __name__ == "__main__":
    if "estimar" in sys.argv[1:]:
        estimar()
    else:
        main(int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else cfg.REPS_2D)
