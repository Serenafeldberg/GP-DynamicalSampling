"""
Ubicación de los sensores sobre y_8 (vista completa y zoom de 16 x 16 centrado en la fuente) para el transporte
entero (s = 1) y la advección-difusión (s = 0,75, nu = 0,1), en la realización de las figuras de presentación
(ell = 2, razón 0,2, M = 2). Lee y_n de salida/transporte/<experimento>/comparacion_justa/resultados.npz:

    salida/transporte/experimento_2D_64/sensores_2D_64.pdf        transporte entero
    salida/transporte/adveccion_difusion/sensores_AD.pdf          advección-difusión
    salida/transporte/adveccion_difusion/sensores_comparacion.pdf las dos filas juntas
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from transporte.comun import config as cfg
from transporte.comun.estilo import mostrar
from transporte.comun.rutas import salida

L, N = 64, cfg.N_REC
ZOOM = 16
FUENTE = cfg.FUENTE_64
X0, Y0 = FUENTE[0] - ZOOM // 2, FUENTE[1] - ZOOM // 2                    # esquina del zoom (centrado en la fuente)
COL = {"base": "#00b4d8", 1: "#f77f00", 2: "#d62daa", "fuente": "#e5383b"}
MARCA = {1: "o", 2: "^"}
EXPERIMENTOS = {"entero": dict(s=1.0, nu=0.0, carpeta="experimento_2D_64",
                               nombre="transporte entero, $s = 1$"),
                "ad": dict(s=0.75, nu=0.1, carpeta="adveccion_difusion",
                           nombre="advección-difusión, $s = 0{,}75$, $\\nu = 0{,}1$")}


def realizacion(clave):
    """y_n (N+1, L, L) de la realización de las figuras de presentación."""
    r = np.load(os.path.join(salida("transporte", EXPERIMENTOS[clave]["carpeta"], "comparacion_justa"), "resultados.npz"))
    return r["fig_y"].reshape(N + 1, L, L)


def flecha(ax, completa):
    """Dirección del flujo (+x)."""
    if completa:
        ax.annotate("", xy=(60, 60), xytext=(44, 60), arrowprops=dict(arrowstyle="-|>", color="white", lw=1.8))
        ax.text(52, 57.5, "flujo", color="white", ha="center", va="bottom", fontsize=8)
    else:
        ax.annotate("", xy=(X0 + 15, Y0 + 14.6), xytext=(X0 + 10.5, Y0 + 14.6),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=1.8))
        ax.text(X0 + 12.75, Y0 + 13.9, "flujo", color="white", ha="center", va="bottom", fontsize=8)


def vista_completa(ax, img, clave, n):
    e = EXPERIMENTOS[clave]
    mostrar(ax, img, cmap="gray", vmin=0, vmax=1)
    cols = np.arange(0, L, cfg.SEP)
    ax.vlines(cols, -0.5, L - 0.5, color=COL["base"], lw=0.6, alpha=0.55)
    for m in (1, 2):                                           # centro de (A^*)^m g: s m píxeles aguas arriba
        ax.vlines(cols - e["s"] * m, -0.5, L - 0.5, color=COL[m], lw=0.6, alpha=0.55,
                  linestyles="--" if m == 1 else ":")
    ax.add_patch(Rectangle((X0 - 0.5, Y0 - 0.5), ZOOM, ZOOM, fill=False, ec="yellow", lw=1.4))
    ax.plot(*FUENTE, "+", color=COL["fuente"], ms=11, mew=2)
    flecha(ax, True)
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(L - 0.5, -0.5)
    ax.set_title(f"$y_{{{n}}}$ ($n = {n}$): 1024 base + 2048 auxiliares", fontsize=10)


def zoom(ax, img, clave, n):
    e = EXPERIMENTOS[clave]
    sub = img[Y0:Y0 + ZOOM, X0:X0 + ZOOM]
    ax.imshow(np.clip(sub, 0, 1), cmap="gray", vmin=0, vmax=1, extent=(X0 - 0.5, X0 + ZOOM - 0.5, Y0 + ZOOM - 0.5, Y0 - 0.5))
    xs = [x for x in range(0, L, cfg.SEP) if X0 <= x < X0 + ZOOM]
    ys = np.arange(Y0, Y0 + ZOOM)
    for x in xs:
        ax.plot(np.full(len(ys), x), ys, "s", ms=3.2, mfc="none", mec=COL["base"], mew=1.0)
    if clave == "entero":                                       # copias puntuales en x - 1 y x - 2
        for m in (1, 2):
            for x in xs:
                if X0 <= x - m:
                    ax.plot(np.full(len(ys), x - m), ys, MARCA[m], ms=3.0, mfc="none", mec=COL[m], mew=0.9)
    else:                                                       # no puntuales: centro 0,75 m píxeles aguas arriba
        for m in (1, 2):
            for x in xs:
                ax.plot(np.full(len(ys), x - e["s"] * m), ys, MARCA[m], ms=3.0, mfc="none", mec=COL[m], mew=0.9)
    ax.plot(*FUENTE, "+", color=COL["fuente"], ms=14, mew=2.2)
    flecha(ax, False)
    ax.set_xlim(X0 - 0.5, X0 + ZOOM - 0.5)
    ax.set_ylim(Y0 + ZOOM - 0.5, Y0 - 0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor("yellow")
        sp.set_linewidth(1.4)
    ax.tick_params(labelsize=7)
    nb = len(xs) * ZOOM
    ax.set_title(f"zoom {ZOOM}$\\times${ZOOM} de $y_{{{n}}}$ ($n = {n}$): {nb} base + {2 * nb} auxiliares", fontsize=10)


def leyenda(fig, claves):
    h = [Line2D([], [], marker="s", ls="-", lw=0.8, color=COL["base"], mfc="none", label="sensores base $G$ ($x \\equiv 0$ mód 4)")]
    for m in (1, 2):
        donde = []
        if "entero" in claves:
            donde.append(f"entero: columna $x - {m}$")
        if "ad" in claves:
            donde.append(f"AD: centro en $x - {0.75 * m:g}$ (no puntual)".replace(".", "{,}"))
        h.append(Line2D([], [], marker=MARCA[m], ls="--" if m == 1 else ":", lw=0.8, color=COL[m], mfc="none",
                        label=f"auxiliar $m = {m}$: " + ("$(A^*) G$" if m == 1 else "$(A^*)^2 G$") + "; " + "; ".join(donde)))
    h += [Line2D([], [], marker="+", ls="none", ms=10, mew=2, color=COL["fuente"], label="fuente"),
          Line2D([], [], ls="-", lw=1.4, color="yellow", label="zona del zoom")]
    fig.legend(handles=h, loc="upper center", ncol=2, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0))


def figura(claves, ruta, ys):
    fig, axs = plt.subplots(len(claves), 2, figsize=(10.4, 5.0 * len(claves) + 0.8), constrained_layout=True)
    axs = np.atleast_2d(axs)
    for i, clave in enumerate(claves):
        y = ys[clave]
        vista_completa(axs[i, 0], y[N], clave, N)
        zoom(axs[i, 1], y[N], clave, N)
        axs[i, 0].set_ylabel(EXPERIMENTOS[clave]["nombre"], fontsize=11)
    leyenda(fig, claves)
    fig.get_layout_engine().set(h_pad=0.25)
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


def main():
    ys = {c: realizacion(c) for c in EXPERIMENTOS}
    rutas = {("entero",): os.path.join(salida("transporte", "experimento_2D_64"), "sensores_2D_64.pdf"),
             ("ad",): os.path.join(salida("transporte", "adveccion_difusion"), "sensores_AD.pdf"),
             ("entero", "ad"): os.path.join(salida("transporte", "adveccion_difusion"), "sensores_comparacion.pdf")}
    for claves, ruta in rutas.items():
        figura(claves, ruta, ys)
    print("archivos: " + ", ".join(os.path.relpath(p, salida()) for p in rutas.values()))


if __name__ == "__main__":
    main()
