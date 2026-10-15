"""
Ejemplo de transporte (Sección 4.7.5): genera la imagen f, el video y_n = A^n f + delta_n y el forzado, y corre los
chequeos del operador. Cómputo; las figuras y el video los hace transporte/figuras/video_transporte.py.

    y_0 = f ,   y_n = A y_{n-1} + w_n ,   delta_n = sum_{k=1}^{n} A^{n-k} w_k

Guarda salida/transporte/video/resultados.npz.
"""
import os

import numpy as np

from transporte.comun import config as cfg
from transporte.comun import operadores as op
from transporte.comun import forzado as fz
from transporte.comun.chequeos import correr_chequeos
from transporte.comun.imagen import imagen_f
from transporte.comun.rutas import salida

OUT = salida("transporte", "video")


def main():
    rng = np.random.default_rng(cfg.SEED)
    L, N, s = cfg.L, cfg.N, cfg.S
    aplicar_A = lambda u: op.A(u, s, cfg.SUBPIXEL)

    f = imagen_f(L, cfg.IMAGEN)
    norma_f = np.linalg.norm(f)
    transportada = np.stack([op.A_pot(f, s, n, cfg.SUBPIXEL) for n in range(N + 1)])

    ventana = fz.ventana_gaussiana(L, cfg.FUENTE_CENTRO, cfg.FUENTE_ANCHO)
    w = fz.muestrear_forzado(rng, N, L, cfg.ELL, cfg.RHO, cfg.SIGMA_W, ventana)
    delta = fz.acumular_delta(w, aplicar_A)
    y = transportada + delta                                   # y_n = A^n f + delta_n

    errores = correr_chequeos(s, L, w, y, cfg.SUBPIXEL, seed=cfg.SEED)

    ruta = os.path.join(OUT, "resultados.npz")
    np.savez(ruta, f=f, norma_f=norma_f, imagen=cfg.IMAGEN, s=np.array(s), N=N, L=L, ell=cfg.ELL, rho=cfg.RHO,
             sigma_w=cfg.SIGMA_W, fuente_centro=np.array(cfg.FUENTE_CENTRO), fuente_ancho=cfg.FUENTE_ANCHO,
             ventana=ventana, w=w, delta=delta, y=y, transportada=transportada, seed=cfg.SEED)

    rango_f = f.max() - f.min()
    print("=" * 60)
    print(f"f: {cfg.IMAGEN}, {L} x {L}   N = {N}   s = {s}   subpíxel = {cfg.SUBPIXEL}")
    print(f"w: {w.shape}   delta, y: {y.shape}")
    print(f"ell = {cfg.ELL}   rho = {cfg.RHO}   sigma_w = {cfg.SIGMA_W}   fuente = {cfg.FUENTE_CENTRO}, ancho {cfg.FUENTE_ANCHO}")
    print(f"||f|| = {norma_f:.4f}")
    print(f"||delta_N|| / ||f|| = {np.linalg.norm(delta[N]) / norma_f:.4f}")
    print(f"max|delta_N| / rango(f) = {np.max(np.abs(delta[N])) / rango_f:.4f}")
    print("chequeos OK:")
    for nombre, e in errores.items():
        print(f"   {nombre:28s} error = {e:.1e}")
    print("archivo: " + os.path.relpath(ruta, salida()))


if __name__ == "__main__":
    main()
