"""
Experimento de la Sección 4.7.4: convolución sobre un subretículo. Genera salida/cap4/tabla_reticulo.tex (Tabla 4.3)
e imprime los números, incluida la Observación de símbolos pares.

MODELO (todo en C^L, mismo esquema que 4.7.1: clase Esquema de experimentos_cap4)
    L = 64, sensores puntuales g_i = e_{i m}, i = 0..p-1, con m = 8 y p = L / m = 8.
    A = convolución circular de símbolo a(xi) = exp(1,3 i xi), xi_k = 2 pi k / L, k = 0..L-1:
        A x = IDFT(a * DFT(x)). Sin el ajuste de Nyquist del ejemplo de transporte: las señales son complejas.
        A es unitario, ||A|| = 1.
    Profundidad M = 2. Forzado de rango 3 construido como el caso "rango_bajo" de 4.7.1 (make_sigma): las tres
        primeras columnas de la base ortonormal de Krylov [G, A^* G, ...]; con p = 8 esas tres direcciones son
        e_0, e_8, e_16, contenidas en S_0 = gen{g_i} ⊂ S_{M-1}. Kernel temporal AR(1) con rho = 0,9, sigma_w = 0,1.
    Estimador: media posterior con hiperparámetros verdaderos (como en 4.7.1).

CONVENCIÓN COMPLEJA
    Forzado y ruido gaussianos complejos circulares: x = (a + i b) / sqrt 2 con a, b ~ N(0, 1) independientes, así que
    E|x|^2 = sigma^2 por componente (sigma_w^2 en las direcciones del forzado, sigma_n^2 = 1e-6 en cada muestra) y
    E[x x^T] = 0. f es uniforme en la esfera unitaria de C^L (gaussiana compleja circular normalizada).
    Las muestras son u_n^(m)[i] = <y_n, (A^*)^m g_i> = g_i^* A^m y_n (traspuesta conjugada).

COLUMNAS (para cada N en {7, 8, 12, 20, 24}; Monte Carlo de 500 realizaciones con semilla fija)
    alpha_N      lambda_min(O_N^* O_N), O_N con la órbita {(A^*)^n g_i : 0 <= n <= N}; se calcula denso (SVD de O_N)
                 y por clases de aliasing: con sensores cada m, O_N^* O_N es diagonal por bloques en Fourier sobre las
                 clases k ≡ c (mod p) y cada bloque es (1/m) V^* V, con V[n, j] = a(xi_{c + p j})^n (Vandermonde).
    identidad    sqrt(E||f - f_hat||^2) por la identidad exacta (4.29).
    observado    RMS de ||f - f_hat|| en Monte Carlo.
    sin corregir RMS de ||f - O_N^+ u^(0)||.
    z            (media MC de ||f - f_hat||^2 - identidad^2) / error estándar.

OBSERVACIÓN DE SÍMBOLOS PARES (solo en consola)
    Separación mínima entre nodos con aliasing y alpha_N, N <= 13, para e^{-0,8 (1 - cos xi)}, 0,6 + 0,4 cos xi y
    e^{2 i cos xi}. Con un símbolo par, a(xi_k) = a(xi_{L-k}) y k, L - k caen en la misma clase cuando k ≡ -k (mod p)
    (clases c = 0 y c = p/2): hay nodos repetidos y alpha_N = 0 para todo N.
    Control: 12 símbolos reales (a(xi) en R) no pares, a(xi) = c_0 + sum_{j=1}^{3} (c_j cos j xi + s_j sin j xi) con
    c_j, s_j ~ N(0, 1) (semilla SEMILLA_SIMBOLOS), escalados para que max_k |a(xi_k)| = 1 (||A|| = 1); los términos en
    seno los hacen no pares (se verifica). Se informa el rango (mín, máx) de la separación mínima y de alpha_7.
"""
import os

import numpy as np

from cap4.experimentos_cap4 import Esquema, make_sigma, kernel_temporal, fmt, RHO, SIGMA_W, RANGO, OUT

L, m = 64, 8
p = L // m
M, SIGMA_N = 2, 1e-3
S_ADV = 1.3
NS = (7, 8, 12, 20, 24)
REPS, SEED_MC = 500, 11
N_PARES = 13
N_SIMBOLOS, SEMILLA_SIMBOLOS = 12, 7
XI = 2 * np.pi * np.arange(L) / L
SEP_ESPERADA, ALFA7_ESPERADO = 0.157, 7.83e-3                  # calculados por separado


def operador(simbolo):
    """Matriz de la convolución circular de símbolo a (valores en xi_k): A = F^* diag(a) F, F la DFT unitaria."""
    F = np.fft.fft(np.eye(L), axis=0, norm="ortho")
    return F.conj().T @ np.diag(simbolo) @ F


def sensores():
    G = np.zeros((L, p))
    G[np.arange(p) * m, np.arange(p)] = 1.0
    return G


def clases():
    """Índices k de cada clase de aliasing: k ≡ c (mod p), c = 0..p-1 (m elementos por clase)."""
    return [c + p * np.arange(m) for c in range(p)]


def separacion_minima(simbolo):
    out = np.inf
    for k in clases():
        a = simbolo[k]
        dif = np.abs(a[:, None] - a[None, :])
        dif[np.arange(m), np.arange(m)] = np.inf
        out = min(out, dif.min())
    return out


def alfa_clases(simbolo, N):
    """lambda_min y lambda_max de O_N^* O_N por bloques (1/m) V^* V."""
    lam = []
    for k in clases():
        V = simbolo[k][None, :] ** np.arange(N + 1)[:, None]
        lam.append(np.linalg.eigvalsh(V.conj().T @ V / m))
    lam = np.concatenate(lam)
    return lam.min(), lam.max()


def alfa_denso(A, G, N):
    O = np.vstack([G.conj().T @ np.linalg.matrix_power(A, n) for n in range(N + 1)])
    sv = np.linalg.svd(O, compute_uv=False)
    return sv[-1] ** 2, sv[0] ** 2


def simbolos_aleatorios():
    rng = np.random.default_rng(SEMILLA_SIMBOLOS)
    out = []
    for _ in range(N_SIMBOLOS):
        c, s = rng.standard_normal(4), rng.standard_normal(3)
        a = c[0] + sum(c[j] * np.cos(j * XI) + s[j - 1] * np.sin(j * XI) for j in range(1, 4))
        a = a / np.abs(a).max()
        assert np.max(np.abs(a - a[(-np.arange(L)) % L])) > 1e-3, "símbolo par"
        out.append(a)
    return out


def experimento():
    simbolo = np.exp(1j * S_ADV * XI)
    A, G = operador(simbolo), sensores()
    assert np.allclose(A.conj().T @ A, np.eye(L), atol=1e-12) and abs(np.linalg.norm(A, 2) - 1) < 1e-12
    x = np.random.default_rng(0).standard_normal(L) + 1j * np.random.default_rng(1).standard_normal(L)
    assert np.allclose(A @ x, np.fft.ifft(simbolo * np.fft.fft(x)), atol=1e-12)
    Sigma, _ = make_sigma("rango_bajo", A, G)
    assert np.linalg.matrix_rank(Sigma, tol=1e-12) == RANGO and np.allclose(Sigma, Sigma.conj().T)

    sep = separacion_minima(simbolo)
    assert abs(sep - SEP_ESPERADA) < 5e-4, f"separación mínima {sep:.4f} != {SEP_ESPERADA}"
    filas = []
    for N in NS:
        a_b, b_b = alfa_clases(simbolo, N)
        esq = Esquema(A, G, Sigma, M, N, SIGMA_N, kernel_temporal("ar1", N, RHO))
        a_d, b_d = alfa_denso(A, G, N)
        assert abs(a_d - a_b) < 1e-10 * b_b and abs(esq.alpha - a_b) < 1e-10 * b_b, (N, a_d, a_b, esq.alpha)
        assert esq.complejo and esq.es_frame
        if N == 7:
            assert abs(a_b / ALFA7_ESPERADO - 1) < 2e-3, f"alpha_7 = {a_b:.4e} != {ALFA7_ESPERADO}"
        H = esq.H(esq.Kt, Sigma)
        ident2 = esq.mse_exacto(H)
        esq.preparar_simulacion()
        rng = np.random.default_rng(SEED_MC)
        e_c, e_sc = [], []
        for _ in range(REPS):
            f, u, z = esq.simular(rng)
            e_c.append(np.sum(np.abs(f - esq.reconstruir(u, H @ z)) ** 2))
            e_sc.append(np.sum(np.abs(f - esq.sin_corregir(u)) ** 2))
        e_c, e_sc = np.array(e_c), np.array(e_sc)
        zs = (e_c.mean() - ident2) / (e_c.std(ddof=1) / np.sqrt(REPS))
        assert abs(zs) < 4, f"N = {N}: identidad vs Monte Carlo, z = {zs:.2f}"
        filas.append(dict(N=N, alfa=a_b, beta=b_b, alfa_denso=a_d, ident=np.sqrt(ident2), obs=np.sqrt(e_c.mean()),
                          sc=np.sqrt(e_sc.mean()), sc_exacto=np.sqrt(esq.mse_sin_corregir()), z=zs))
    return sep, filas


def observacion_pares():
    pares = {"e^{-0,8 (1 - cos xi)}": np.exp(-0.8 * (1 - np.cos(XI))),
             "0,6 + 0,4 cos xi": 0.6 + 0.4 * np.cos(XI),
             "e^{2 i cos xi}": np.exp(2j * np.cos(XI))}
    res_pares = {}
    for nombre, a in pares.items():
        assert np.allclose(a, a[(-np.arange(L)) % L]), nombre
        res_pares[nombre] = (separacion_minima(a), [alfa_clases(a, N)[0] for N in range(1, N_PARES + 1)])
    aleatorios = [(separacion_minima(a), alfa_clases(a, 7)[0]) for a in simbolos_aleatorios()]
    return res_pares, np.array(aleatorios)


def cientifico(v, dec=2):
    mant, e = f"{v:.{dec}e}".split("e")
    if int(e) == 0:
        return f"${mant.replace('.', '{,}')}$"
    return f"${mant.replace('.', '{,}')} \\cdot 10^{{{int(e)}}}$"


if __name__ == "__main__":
    sep, filas = experimento()
    res_pares, aleat = observacion_pares()

    print(f"\n=== 4.7.4  convolución sobre un subretículo: L = {L}, m = {m}, p = {p}, a(xi) = exp({S_ADV} i xi), "
          f"M = {M}, rango {RANGO}, rho = {RHO}, sigma_w = {SIGMA_W}, sigma_n = {SIGMA_N:g}, {REPS} realizaciones")
    print(f"asserts OK: A unitario (||A|| = 1) y == IDFT(a DFT(x)); Sigma_w hermítica de rango {RANGO}; "
          f"alpha_N denso == clases de aliasing (Vandermonde) en todos los N; |z| < 4")
    print(f"separación mínima entre nodos con aliasing: {sep:.4f} (esperada ~{SEP_ESPERADA}); "
          f"alpha_7 = {filas[0]['alfa']:.4e} (esperado ~{ALFA7_ESPERADO:g})")
    print(" N   alpha_N     (denso)     identidad  observado  sin corregir  (exacto)  sc/obs     z")
    for r in filas:
        print(f"{r['N']:2d}  {r['alfa']:.4e}  {r['alfa_denso']:.4e}  {r['ident']:.5f}    {r['obs']:.5f}    "
              f"{r['sc']:.5f}      ({r['sc_exacto']:.5f})  {r['sc'] / r['obs']:6.1f}  {r['z']:+.2f}")

    print(f"\n--- Observación: símbolos pares (alpha_N para N = 1..{N_PARES})")
    for nombre, (s, alfas) in res_pares.items():
        print(f"{nombre:24s} separación mínima {s:.2e}; alpha_N máx en N <= {N_PARES}: {max(alfas):.2e}")
        print("    " + " ".join(f"{a:.1e}" for a in alfas))
    print(f"control: {N_SIMBOLOS} símbolos reales no pares (semilla {SEMILLA_SIMBOLOS}): separación mínima en "
          f"[{aleat[:, 0].min():.3e}, {aleat[:, 0].max():.3e}], alpha_7 en [{aleat[:, 1].min():.3e}, {aleat[:, 1].max():.3e}]")

    lineas = [f"{r['N']:2d} & {cientifico(r['alfa'])} & {fmt(r['ident'], 4)} & {fmt(r['obs'], 4)} & {fmt(r['sc'], 4)}\\\\"
              for r in filas]
    with open(os.path.join(OUT, "tabla_reticulo.tex"), "w") as fh:
        fh.write("% N & alpha_N & identidad & observado (RMS) & sin corregir (RMS)\n")
        fh.write("\n".join(lineas) + "\n")
    print("\narchivo: salida/cap4/tabla_reticulo.tex")
