"""
Operador de transporte A y su adjunta A^*.

    (A u)(x) = u(x - s)        desplazamiento periódico de s = (s_x, s_y)
    (A^* u)(x) = u(x + s)      desplazamiento en sentido opuesto

Las imágenes se guardan como arreglos [fila, columna] = [y, x]; las funciones
actúan sobre los dos últimos ejes, así que también aceptan pilas (n, L, L).
Para s entero se usa np.roll (exacto, sin armar la matriz). La variante
subpíxel multiplica por la fase e^{-i s·xi} en Fourier. Con L par, la frecuencia
de Nyquist es su propia conjugada: para que A sea real y unitaria ahí se usa el
real ±1 más cercano a la fase, sign(cos(pi s)). Para s entero coincide con la fase.
"""
import numpy as np


def es_entero(s):
    """True si ambas componentes de s son enteras."""
    return all(float(c).is_integer() for c in s)


def desplazar_roll(u, s):
    """Desplazamiento periódico entero: u(x - s) vía np.roll."""
    s_x, s_y = (int(round(c)) for c in s)
    return np.roll(u, shift=(s_y, s_x), axis=(-2, -1))


def fase_1d(L, s_c):
    """Multiplicador de un eje: e^{-i s_c xi}, con ±1 real en Nyquist si L es par."""
    xi = 2 * np.pi * np.fft.fftfreq(L)
    h = np.exp(-1j * s_c * xi)
    if L % 2 == 0:
        h[L // 2] = 1.0 if np.cos(np.pi * s_c) >= 0 else -1.0
    return h


def multiplicador(L_y, L_x, s, n=1):
    """Diagonal de A^n en Fourier: fft2(A^n u) = H^n fft2(u)."""
    return np.outer(fase_1d(L_y, s[1]), fase_1d(L_x, s[0])) ** n


def desplazar_fft(u, s, n=1):
    """n pasos de desplazamiento periódico (posiblemente subpíxel) vía Fourier."""
    H = multiplicador(*u.shape[-2:], s, n)
    return np.real(np.fft.ifft2(np.fft.fft2(u, axes=(-2, -1)) * H, axes=(-2, -1)))


def A(u, s, subpixel=False):
    """Un paso de transporte: A u."""
    return desplazar_fft(u, s) if subpixel else desplazar_roll(u, s)


def A_adj(u, s, subpixel=False):
    """Adjunta: A^* u = desplazamiento por -s."""
    return A(u, (-s[0], -s[1]), subpixel)


def A_pot(u, s, n, subpixel=False):
    """A^n u (en subpíxel se eleva el multiplicador, para respetar la regla de Nyquist)."""
    if subpixel:
        return desplazar_fft(u, s, n)
    return desplazar_roll(u, (n * s[0], n * s[1]))


# ============================================================== advección-difusión (agregado)
def simbolo_difusion(L_y, L_x, nu):
    """exp(-nu [(2 - 2 cos xi_x) + (2 - 2 cos xi_y)]): símbolo real y par del laplaciano discreto."""
    cy = 2 - 2 * np.cos(2 * np.pi * np.fft.fftfreq(L_y))
    cx = 2 - 2 * np.cos(2 * np.pi * np.fft.fftfreq(L_x))
    return np.exp(-nu * (cy[:, None] + cx[None, :]))


def multiplicador_ad(L_y, L_x, s, nu, n=1):
    """a_hat^n con a_hat(xi) = e^{-i s xi} exp(-nu [...]); en Nyquist la fase usa la regla de fase_1d
    (real ±1 = sign(cos(pi s_c))), así que a_hat es hermítico y A es real."""
    return (multiplicador(L_y, L_x, s) * simbolo_difusion(L_y, L_x, nu)) ** n


def A_ad(u, s, nu, n=1):
    """A^n u para el operador de advección-difusión (FFT 2D, periódico)."""
    H = multiplicador_ad(*u.shape[-2:], s, nu, n)
    return np.real(np.fft.ifft2(np.fft.fft2(u, axes=(-2, -1)) * H, axes=(-2, -1)))


def A_ad_adj(u, s, nu, n=1):
    """(A^*)^n u: símbolo conjugado."""
    H = np.conj(multiplicador_ad(*u.shape[-2:], s, nu, n))
    return np.real(np.fft.ifft2(np.fft.fft2(u, axes=(-2, -1)) * H, axes=(-2, -1)))
