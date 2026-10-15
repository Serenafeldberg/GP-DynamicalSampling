"""
Figuras y video del ejemplo de transporte (Sección 4.7.5), leyendo salida/transporte/video/resultados.npz:
cuadros.pdf (A^n f, delta_n, y_n), forzado_fuente.pdf, forzado_w.pdf, video.gif y video.mp4 (si hay ffmpeg).
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from transporte.comun import config as cfg
from transporte.comun.estilo import mostrar, barra
from transporte.comun.rutas import salida

OUT = salida("transporte", "video")


def figura_cuadros(transportada, delta, y, ns, ruta):
    """3 filas (A^n f, delta_n, y_n) x len(ns) columnas, misma escala por fila."""
    a = np.max(np.abs(delta))
    filas = [(r"$A^n f$", transportada, dict(cmap="gray", vmin=0, vmax=1)),
             (r"$\delta_n$", delta, dict(cmap="RdBu_r", vmin=-a, vmax=a)),
             (r"$y_n$", y, dict(cmap="gray", vmin=y.min(), vmax=y.max()))]
    fig, axs = plt.subplots(3, len(ns), figsize=(2.2 * len(ns) + 0.8, 7), constrained_layout=True)
    for i, (nombre, serie, kw) in enumerate(filas):
        for j, n in enumerate(ns):
            im = mostrar(axs[i, j], serie[n], **kw)
            if i == 0:
                axs[i, j].set_title(f"$n = {n}$")
        axs[i, 0].set_ylabel(nombre, fontsize=13)
        barra(fig, im, axs[i, :].tolist())
    fig.savefig(ruta)
    plt.close(fig)


def figura_fuente(ventana, w1, centro, ruta):
    """Ventana espacial D y una realización de w_1."""
    a = np.max(np.abs(w1))
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.6), constrained_layout=True)
    for ax, u, titulo, kw in [(axs[0], ventana, "ventana de la fuente", dict(cmap="viridis", vmin=0, vmax=1)),
                              (axs[1], w1, r"realización de $w_1$", dict(cmap="RdBu_r", vmin=-a, vmax=a))]:
        im = mostrar(ax, u, **kw)
        ax.plot(*centro, "k+", ms=10)
        ax.set_title(titulo)
        barra(fig, im, ax)
    fig.savefig(ruta)
    plt.close(fig)


def figura_forzado(w, centro, ruta, columnas=8):
    """w_1..w_N en una grilla, misma escala simétrica para todos."""
    N = w.shape[0]
    a = np.max(np.abs(w))
    filas = -(-N // columnas)
    fig, axs = plt.subplots(filas, columnas, figsize=(1.8 * columnas + 0.8, 2 * filas), constrained_layout=True)
    axs = np.atleast_2d(axs)
    for ax in axs.flat[N:]:
        ax.axis("off")
    for n, ax in enumerate(axs.flat[:N], start=1):
        im = mostrar(ax, w[n - 1], cmap="RdBu_r", vmin=-a, vmax=a)
        ax.plot(*centro, "k+", ms=6)
        ax.set_title(f"$w_{{{n}}}$")
    barra(fig, im, axs.ravel().tolist())
    fig.savefig(ruta)
    plt.close(fig)


def guardar_video(transportada, delta, y, fps, carpeta):
    """Tres paneles (A^n f | delta_n | y_n) que avanzan con n. Devuelve los archivos escritos."""
    a = np.max(np.abs(delta))
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.4), constrained_layout=True)
    ims = [mostrar(axs[0], transportada[0], cmap="gray", vmin=0, vmax=1),
           mostrar(axs[1], delta[0], cmap="RdBu_r", vmin=-a, vmax=a),
           mostrar(axs[2], y[0], cmap="gray", vmin=y.min(), vmax=y.max())]
    for ax, t in zip(axs, [r"$A^n f$", r"$\delta_n$", r"$y_n$"]):
        ax.set_title(t)
    titulo = fig.suptitle("$n = 0$")

    def cuadro(n):
        for im, serie in zip(ims, (transportada, delta, y)):
            im.set_data(serie[n])
        titulo.set_text(f"$n = {n}$")
        return ims

    anim = animation.FuncAnimation(fig, cuadro, frames=len(y))
    archivos = [os.path.join(carpeta, "video.gif")]
    anim.save(archivos[0], writer=animation.PillowWriter(fps=fps))
    if animation.writers.is_available("ffmpeg"):
        archivos.append(os.path.join(carpeta, "video.mp4"))
        anim.save(archivos[1], writer=animation.FFMpegWriter(fps=fps), dpi=150)
    else:
        print("AVISO: ffmpeg no está disponible; se omite video.mp4")
    plt.close(fig)
    return archivos


def main():
    z = np.load(os.path.join(OUT, "resultados.npz"))
    transportada, delta, y, w = z["transportada"], z["delta"], z["y"], z["w"]
    archivos = [os.path.join(OUT, n) for n in ("cuadros.pdf", "forzado_fuente.pdf", "forzado_w.pdf")]
    figura_cuadros(transportada, delta, y, cfg.CUADROS_N, archivos[0])
    figura_fuente(z["ventana"], w[0], cfg.FUENTE_CENTRO, archivos[1])
    figura_forzado(w, cfg.FUENTE_CENTRO, archivos[2])
    archivos += guardar_video(transportada, delta, y, cfg.FPS, OUT)
    print("archivos: " + ", ".join(os.path.relpath(a, salida()) for a in archivos))


if __name__ == "__main__":
    main()
