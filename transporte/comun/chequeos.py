"""
Chequeos numéricos del ejemplo de transporte (todos con assert).

    1. <A f, g> = <f, A^* g>              para f, g aleatorias
    1b. A^* A f = f                        también en subpíxel (L par, regla de Nyquist)
    2. A^L f = f                           si s es entero
    3. desplazamiento FFT == np.roll       si s es entero
    4. y_n - A y_{n-1} = w_n               a precisión de máquina
    + alpha_N denso == por bloques, medir() == <A^{n+k} f, g_i> y el operador de advección-difusión.
"""
import numpy as np

from . import operadores as op

TOL = 1e-12


def correr_chequeos(s, L, w, y, subpixel=False, seed=0):
    """Corre los chequeos; levanta AssertionError si alguno falla. Devuelve los errores."""
    rng = np.random.default_rng(seed)
    f, g = rng.standard_normal((2, L, L))
    err = {}

    # 1. adjunta (con la variante en uso y también con un s subpíxel en FFT)
    for nombre, s_k, sub in [("adjunta", s, subpixel), ("adjunta FFT subpíxel", (0.37, -1.2), True)]:
        izq = np.sum(op.A(f, s_k, sub) * g)
        der = np.sum(f * op.A_adj(g, s_k, sub))
        err[nombre] = abs(izq - der) / (np.linalg.norm(f) * np.linalg.norm(g))
        assert err[nombre] < TOL, f"{nombre}: <Af,g> - <f,A*g> = {err[nombre]:.2e}"

    # 1b. unitaria en subpíxel
    for s_k in [(0.37, -1.2), (0.5, 1 / 3)]:
        e = np.linalg.norm(op.A_adj(op.A(f, s_k, True), s_k, True) - f) / np.linalg.norm(f)
        err[f"A*A = I FFT s = {s_k[0]:.2f}, {s_k[1]:.2f}"] = e
        assert e < TOL, f"A*A != I en subpíxel s = {s_k}: {e:.2e}"
    e = np.linalg.norm(op.A_pot(f, (0.37, -1.2), 5, True) - op.A(op.A_pot(f, (0.37, -1.2), 4, True), (0.37, -1.2), True))
    err["A^5 = A A^4 FFT subpíxel"] = e / np.linalg.norm(f)
    assert err["A^5 = A A^4 FFT subpíxel"] < TOL, "A_pot subpíxel no es potencia de A"

    if op.es_entero(s):
        # 2. A^L f = f, aplicando A paso a paso
        u = f.copy()
        for _ in range(L):
            u = op.A(u, s, subpixel)
        err["A^L f = f"] = np.max(np.abs(u - f))
        assert err["A^L f = f"] < TOL, f"A^L f != f: {err['A^L f = f']:.2e}"

        # 3. FFT coincide con np.roll
        err["FFT = roll"] = np.max(np.abs(op.desplazar_fft(f, s) - op.desplazar_roll(f, s)))
        assert err["FFT = roll"] < TOL, f"FFT != roll: {err['FFT = roll']:.2e}"

    # 4. el video cumple la recursión
    N = w.shape[0]
    res = max(np.max(np.abs(y[n] - op.A(y[n - 1], s, subpixel) - w[n - 1])) for n in range(1, N + 1))
    err["y_n - A y_{n-1} = w_n"] = res / np.max(np.abs(y))
    assert err["y_n - A y_{n-1} = w_n"] < TOL, f"recursión: {res:.2e}"

    return err


def chequear_bloques_vs_denso(L, m_x, m_y, nucleos, s, N_max, M, tol=1e-9):
    """alpha_N denso y por bloques coinciden (relativo a lambda_max) para N = 0..N_max."""
    from . import frames as fr
    from . import sensores as sn
    sub = not op.es_entero(s)
    a_d, b_d = fr.alpha_denso(sn.banco(L, m_x, m_y, nucleos), s, N_max, M, sub)
    a_b, b_b = fr.alpha_bloques(L, m_x, m_y, nucleos, s, N_max, M)
    err = max(np.max(np.abs(a_d - a_b)), np.max(np.abs(b_d - b_b))) / np.max(b_d)
    assert err < tol, f"denso != bloques ({m_x}x{m_y}, s = {s}, {nucleos}): {err:.2e}"
    return err


def chequear_medir(L, m_x, m_y, nucleos, s, N, M, seed=0):
    """medir() sobre y_n = A^n f coincide con < A^{n+k} f, g_i > calculado directamente."""
    from . import sensores as sn
    sub = not op.es_entero(s)
    f = np.random.default_rng(seed).standard_normal((L, L))
    G = sn.banco(L, m_x, m_y, nucleos)
    y = np.stack([op.A_pot(f, s, n, sub) for n in range(N + 1)])
    u = sn.medir(y, G, s, M, sub)
    directo = np.array([[[np.sum(op.A_pot(f, s, n + k, sub) * g) for g in G] for k in range(M + 1)]
                        for n in range(N + 1)])
    err = np.max(np.abs(u - directo)) / np.max(np.abs(directo))
    assert err < TOL, f"medir no coincide: {err:.2e}"
    return err


def chequear_adveccion_difusion(L, s, nu, seed=0):
    """Operador de advección-difusión: adjunta, caso nu = 0 con s entero == np.roll, ||A|| = max|a_hat|."""
    rng = np.random.default_rng(seed)
    f, g = rng.standard_normal((2, L, L))
    sv = (s, 0)
    err = {}
    err["adjunta"] = abs(np.sum(op.A_ad(f, sv, nu) * g) - np.sum(f * op.A_ad_adj(g, sv, nu))) / (
        np.linalg.norm(f) * np.linalg.norm(g))
    assert err["adjunta"] < TOL, f"adjunta AD (s = {s}, nu = {nu}): {err['adjunta']:.2e}"
    d = L * L
    A = op.A_ad(np.eye(d).reshape(d, L, L), sv, nu).reshape(d, d).T
    err["||A|| = max|a|"] = abs(np.linalg.norm(A, 2) - np.abs(op.multiplicador_ad(L, L, sv, nu)).max())
    assert err["||A|| = max|a|"] < 1e-10, f"||A|| != max|a_hat|: {err['||A|| = max|a|']:.2e}"
    err["A real"] = np.max(np.abs(np.imag(np.fft.ifft2(np.fft.fft2(f) * op.multiplicador_ad(L, L, sv, nu)))))
    assert err["A real"] < TOL
    for k in (1, 2, 3):
        e = np.max(np.abs(op.A_ad(f, (k, 0), 0.0) - op.desplazar_roll(f, (k, 0))))
        err[f"nu = 0, s = {k}: == roll"] = e
        assert e < TOL, f"A_ad(nu = 0, s = {k}) != np.roll: {e:.2e}"
    return err
