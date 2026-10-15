"""
Diagnóstico de frame: alpha_N = lambda_min(O^* O) para los sensores del transporte.

    O f = ( < A^{n+k} f , g_i > )_{n = 0..N, k = 0..M, i}       (sin forzado)
    O^* O = sum_e c_e sum_i (A^*)^e g_i g_i^T A^e ,   c_e = #{(n,k): n + k = e}

Denso: arma O^* O de LL x LL (solo L chico).
Por bloques: A es diagonal en Fourier y la grilla acopla solo frecuencias que
difieren en (L/m_x, L/m_y)·Z; O^* O es diagonal por bloques de tamaño m_x m_y,
    B(xi_0) = P sum_e c_e sum_q u_{e,q} u_{e,q}^* ,  u_{e,q}[xi] = conj(H(xi))^e phi_q^(xi),
con P = #posiciones y phi^ la DFT unitaria del núcleo. alpha_N = min de los lambda_min.
"""
import numpy as np

from . import operadores as op
from . import sensores as sn

TOL_FRAME = 1e-10       # frame si alpha_N > TOL_FRAME * lambda_max


def pesos(N, M):
    """c_e = cantidad de pares (n, k), n <= N, k <= M, con n + k = e; e = 0..N+M."""
    e = np.arange(N + M + 1)
    return np.minimum(M, e) - np.maximum(0, e - N) + 1


def alpha_denso(G, s, N_max, M, subpixel=False):
    """alpha_N y lambda_max para N = 0..N_max, armando O^* O explícitamente."""
    R = G.reshape(G.shape[0], -1)
    C = []                                              # C_e = sum_i (A^*)^e g_i ((A^*)^e g_i)^T
    for _ in range(N_max + M + 1):
        C.append(R.T @ R)
        G = op.A_adj(G, s, subpixel)
        R = G.reshape(G.shape[0], -1)
    alfa, beta = [], []
    for N in range(N_max + 1):
        lam = np.linalg.eigvalsh(sum(c * C[e] for e, c in enumerate(pesos(N, M))))
        alfa.append(lam[0])
        beta.append(lam[-1])
    return np.array(alfa), np.array(beta)


def clases(L, m_x, m_y):
    """Índices [y, x] de frecuencia de cada bloque: (C, K), con K = m_x m_y frecuencias que se aliasan."""
    cy, cx, b, a = np.meshgrid(np.arange(L // m_y), np.arange(L // m_x), np.arange(m_y), np.arange(m_x),
                               indexing="ij")
    IY = (cy + b * (L // m_y)).reshape(-1, m_y * m_x)
    IX = (cx + a * (L // m_x)).reshape(-1, m_y * m_x)
    return IY, IX


def bloques(L, m_x, m_y, nucleos, s, N_max, M):
    """Genera (N, B_N) con los bloques B_N de forma (C, K, K), para N = 0..N_max."""
    IY, IX = clases(L, m_x, m_y)
    P = L * L / (m_x * m_y)
    phi = np.stack([np.fft.fft2(sn.nucleo(L, spec)) / L for spec in nucleos])[:, IY, IX]   # (Q, C, K)
    Hc = np.conj(op.multiplicador(L, L, s))[IY, IX]                                        # (C, K)

    def C_e(e):
        U = Hc ** e * phi
        return P * np.einsum("qck,qcl->ckl", U, np.conj(U))

    B = sum(C_e(k) for k in range(M + 1))
    for N in range(N_max + 1):
        if N > 0:
            B = B + sum(C_e(N + k) for k in range(M + 1))
        yield N, B


def alpha_bloques(L, m_x, m_y, nucleos, s, N_max, M):
    """alpha_N y lambda_max para N = 0..N_max por bloques de Fourier."""
    alfa, beta = [], []
    for _, B in bloques(L, m_x, m_y, nucleos, s, N_max, M):
        lam = np.linalg.eigvalsh(B)
        alfa.append(lam[:, 0].min())
        beta.append(lam[:, -1].max())
    return np.array(alfa), np.array(beta)


def invisibles(L, m_x, m_y, nucleos, s, N, M, tol=TOL_FRAME):
    """Componentes que los sensores no ven hasta el horizonte N.

    Devuelve (dim, lista) con dim = dimensión (real) del núcleo de O y la lista de
    (imagen real normalizada, frecuencia (k_y, k_x) representativa del bloque, lambda).
    """
    *_, (_, B) = bloques(L, m_x, m_y, nucleos, s, N, M)
    lam, V = np.linalg.eigh(B)
    IY, IX = clases(L, m_x, m_y)
    nulos = np.argwhere(lam < tol * lam.max())
    salida = []
    for c, j in nulos:
        v = np.zeros((L, L), complex)
        v[IY[c], IX[c]] = V[c, :, j]
        u = np.fft.ifft2(v) * L
        img = u.real if np.linalg.norm(u.real) > np.linalg.norm(u.imag) else u.imag
        frec = ((IY[c, 0] + L // 2) % L - L // 2, (IX[c, 0] + L // 2) % L - L // 2)
        salida.append((img / np.linalg.norm(img), frec, lam[c, j]))
    return len(nulos), salida


# ============================================================== símbolo general (agregado)
def alpha_bloques_simbolo(L, m_x, m_y, nucleos, H, N_max, M=0, por_clase=False):
    """Como alpha_bloques, pero con A dado por su símbolo H (L x L), no necesariamente unitario.

    Devuelve alpha_N, lambda_max_N (N = 0..N_max) y la separación mínima entre nodos H dentro de cada clase;
    con por_clase=True agrega lambda_min de cada clase en N_max y los índices (IY, IX) de las clases.
    """
    IY, IX = clases(L, m_x, m_y)
    P = L * L / (m_x * m_y)
    phi = np.stack([np.fft.fft2(sn.nucleo(L, spec)) / L for spec in nucleos])[:, IY, IX]
    nodos = H[IY, IX]                                                   # (C, K)
    Hc = np.conj(nodos)
    C_e = lambda e: P * np.einsum("qck,qcl->ckl", Hc ** e * phi, np.conj(Hc ** e * phi))
    B = sum(C_e(k) for k in range(M + 1))
    alfa, beta = [], []
    for N in range(N_max + 1):
        if N > 0:
            B = B + sum(C_e(N + k) for k in range(M + 1))
        lam = np.linalg.eigvalsh(B)
        alfa.append(lam[:, 0].min())
        beta.append(lam[:, -1].max())
    dif = np.abs(nodos[:, :, None] - nodos[:, None, :])
    dif[:, np.arange(dif.shape[1]), np.arange(dif.shape[1])] = np.inf
    if por_clase:
        return np.array(alfa), np.array(beta), dif.min(axis=(1, 2)), lam[:, 0], (IY, IX)
    return np.array(alfa), np.array(beta), dif.min(axis=(1, 2))


def alpha_denso_operador(G, aplicar_adj, N_max, M=0):
    """Como alpha_denso, con (A^*) dado por una función: alpha_N y lambda_max para N = 0..N_max."""
    R = G.reshape(G.shape[0], -1)
    C = []
    for _ in range(N_max + M + 1):
        C.append(R.T @ R)
        G = aplicar_adj(G)
        R = G.reshape(G.shape[0], -1)
    alfa, beta = [], []
    for N in range(N_max + 1):
        lam = np.linalg.eigvalsh(sum(c * C[e] for e, c in enumerate(pesos(N, M))))
        alfa.append(lam[0])
        beta.append(lam[-1])
    return np.array(alfa), np.array(beta)
