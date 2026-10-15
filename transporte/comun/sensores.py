"""
Sensores g_i sobre una grilla periódica y muestreo con profundidad.

    posiciones x_j = (a m_x, b m_y)                 grilla de separación (m_x, m_y)
    g_{j,q}(x)   = phi_q(x - x_j)                   un núcleo phi_q por familia
    muestras     u[n, k, i] = < y_n , (A^*)^k g_i >,  k = 0..M  (profundidad)

Los núcleos están centrados en el origen, son periódicos y tienen norma 1.
Un núcleo se especifica como tupla (familia, parámetros...), por ejemplo
("gaussiana", 1.0) o ("bspline", 3, 2). Varios núcleos por posición aumentan p.
"""
from math import comb, factorial

import numpy as np

from . import operadores as op


def _coords(L):
    """Distancias periódicas con signo al origen, como grillas [y, x]."""
    d = (np.arange(L) + L // 2) % L - L // 2
    return np.meshgrid(d, d, indexing="ij")


def _caja_1d(L, w):
    d = (np.arange(L) + L // 2) % L - L // 2
    return ((d >= -(w // 2)) & (d < w - w // 2)).astype(float)


def _bspline_1d(L, grado, escala):
    """B-spline centrado de grado k, a escala dada, muestreado en los píxeles (0,5 en saltos)."""
    t = ((np.arange(L) + L // 2) % L - L // 2) / escala
    k = grado
    tot = np.zeros(L)
    for j in range(k + 2):
        x = t + (k + 1) / 2 - j
        pot = np.where(x > 0, 1.0, np.where(x == 0, 0.5, 0.0)) if k == 0 else np.clip(x, 0, None) ** k
        tot += (-1) ** j * comb(k + 1, j) * pot
    return tot / factorial(k)


FAMILIAS = {
    "delta": lambda L: (lambda Y, X: ((Y == 0) & (X == 0)).astype(float))(*_coords(L)),
    "gaussiana": lambda L, sigma: (lambda Y, X: np.exp(-(X**2 + Y**2) / (2 * sigma**2)))(*_coords(L)),
    "caja": lambda L, w: np.outer(_caja_1d(L, w), _caja_1d(L, w)),
    "bspline": lambda L, grado, escala=1: np.outer(*(2 * [_bspline_1d(L, grado, escala)])),
    "dgauss_x": lambda L, sigma: (lambda Y, X: -X * np.exp(-(X**2 + Y**2) / (2 * sigma**2)))(*_coords(L)),
    "dgauss_y": lambda L, sigma: (lambda Y, X: -Y * np.exp(-(X**2 + Y**2) / (2 * sigma**2)))(*_coords(L)),
}


def nucleo(L, spec):
    """Núcleo phi (L x L) centrado en el origen y de norma 1."""
    phi = FAMILIAS[spec[0]](L, *spec[1:])
    return phi / np.linalg.norm(phi)


def etiqueta(spec):
    """Nombre corto del núcleo para tablas y leyendas."""
    return spec[0] + ("(" + ", ".join(f"{v:g}" for v in spec[1:]) + ")" if len(spec) > 1 else "")


def posiciones(L, m_x, m_y):
    """Posiciones (x, y) de la grilla; requiere que m_x y m_y dividan a L."""
    assert L % m_x == 0 and L % m_y == 0, "la separación debe dividir a L"
    b, a = np.meshgrid(np.arange(0, L, m_y), np.arange(0, L, m_x), indexing="ij")
    return np.stack([a.ravel(), b.ravel()], axis=1)


def banco(L, m_x, m_y, nucleos):
    """Sensores G (p, L, L): cada núcleo trasladado a cada posición; p = #posiciones * #núcleos."""
    pos = posiciones(L, m_x, m_y)
    return np.stack([np.roll(nucleo(L, spec), (y, x), axis=(0, 1)) for spec in nucleos for x, y in pos])


def bancos_profundidad(G, s, M, subpixel=False):
    """(A^*)^k G para k = 0..M, de forma (M+1, p, L, L)."""
    salida = [G]
    for _ in range(M):
        salida.append(op.A_adj(salida[-1], s, subpixel))
    return np.stack(salida)


def medir(y, G, s, M, subpixel=False):
    """Muestras u[n, k, i] = < y_n , (A^*)^k g_i > para y de forma (N+1, L, L)."""
    return np.einsum("nyx,kiyx->nki", y, bancos_profundidad(G, s, M, subpixel))


def profundidad_gratis(m_x, m_y, s, M):
    """Fracción de k = 1..M con (A^*)^k g_i sobre otro sensor: k s en la grilla (vale para todo i)."""
    def en_grilla(k):
        d = np.array([k * s[0], k * s[1]])
        r = np.round(d)
        return np.allclose(d, r, atol=1e-9) and r[0] % m_x == 0 and r[1] % m_y == 0
    return np.mean([en_grilla(k) for k in range(1, M + 1)])
