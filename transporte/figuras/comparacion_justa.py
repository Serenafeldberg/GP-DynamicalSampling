"""
Tablas y figuras de la comparación a igual cantidad de sensores (Sección 4.7.5), leyendo
salida/transporte/<experimento>/comparacion_justa/resultados.npz:
    experimento_2D_64/comparacion_justa/   tabla_comparacion_justa.tex, figura_presentacion.pdf
    adveccion_difusion/comparacion_justa/  tabla_comparacion_justa.tex, figura_presentacion_reconstruccion.pdf,
                                           figura_presentacion_error.pdf (nu = 0,1)
El naive estático (b1, b2) queda en el npz pero no se muestra.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from transporte.comun import config as cfg
from transporte.comun.estilo import mostrar, barra
from transporte.comun.rutas import salida
from transporte.experimentos.comparacion_justa import FIGURA

CARPETAS = {"entero": "experimento_2D_64", "ad": "adveccion_difusion"}
L = 64
N_FIG = 8


def c(v, dec):
    """Número con coma decimal para LaTeX / mathtext: 0{,}75."""
    return f"{v:.{dec}f}".replace(".", "{,}")


def g(v):
    """Valor corto con coma decimal (0{,}75, 0{,}1, 2)."""
    return f"{v:g}".replace(".", "{,}")


def tabla(r, ruta):
    ad = str(r["operador"]) == "ad"
    lineas, previo = [], None
    for i in range(len(r["ell"])):
        clave = (float(r["nu"][i]), float(r["ell"][i]), float(r["razon"][i]))
        if previo is not None and clave != previo:
            lineas.append("\\midrule")
        previo = clave
        celdas = ([f"${g(r['nu'][i])}$"] if ad else []) + [
            f"${g(r['ell'][i])}$", f"${g(r['razon'][i])}$", f"{int(r['M'][i])}", f"{int(r['N2'][i])}",
            *[f"${c(float(r[k][i]), 4)}$" for k in ("sc", "dinamico", "gp", "oraculo", "denso")]]
        lineas.append(" & ".join(celdas) + " \\\\")
    cab = (["$\\nu$"] if ad else []) + ["$\\ell$", "razón", "$M$", "sensores", "sin corregir (1024)",
                                        "naive dinámico ($N_2$)", "GP ($N_2$)", "oráculo", "denso (4096)"]
    with open(ruta, "w") as fh:
        fh.write(f"% generado por transporte/figuras/comparacion_justa.py; s = {g(float(r['s']))}; medianas de ||f_hat - f|| / ||f||; "
                 "naive dinámico y GP usan N2 = 1024 (M + 1) sensores; GP, oráculo y sin corregir leídos "
                 "del experimento original; requiere booktabs\n")
        fh.write("\\begin{tabular}{" + "c" * len(cab) + "}\n\\toprule\n" + " & ".join(cab) + " \\\\\n\\midrule\n")
        fh.write("\n".join(lineas) + "\n\\bottomrule\n\\end{tabular}\n")


def figura(r, ruta, operador):
    f = r["fig_f"]
    fuente = r["fig_fuente"]
    sc = r["fig_f_sin_corregir"]
    paneles = [(sc, "sin corregir\n(1024 sensores)"),
               (r["fig_f_dinamico"], "naive dinámico, $O_N^{\\dagger} u$\n(3072 sensores)"),
               (r["fig_f_gp"], "GP ($M = 2$)\n(3072 sensores: 1024 y 2 copias)")]
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.6), constrained_layout=True)
    im = mostrar(axs[0, 0], f.reshape(L, L), cmap="gray", vmin=0, vmax=1)
    axs[0, 0].set_title("$f$ (verdadera)\n(4096 píxeles)", fontsize=10)
    for ax, (v, t) in zip(axs[0, 1:], paneles):
        im = mostrar(ax, v.reshape(L, L), cmap="gray", vmin=0, vmax=1)
        ax.set_title(r"$\hat f$, " + t, fontsize=10)
    barra(fig, im, axs[0].tolist())
    dN = r["fig_delta_N"].reshape(L, L)
    a = np.abs(dN).max()
    imd = mostrar(axs[1, 0], dN, cmap="RdBu_r", vmin=-a, vmax=a)
    axs[1, 0].plot(*fuente, "k+", ms=11, mew=1.5)
    axs[1, 0].set_title(f"perturbación acumulada $\\delta_{{{N_FIG}}}$ (+ fuente)\n(escala propia)", fontsize=10)
    barra(fig, imd, axs[1, 0])
    errores = [np.abs(v - f).reshape(L, L) for v, _ in paneles]
    tope = max(e.max() for e in errores)
    for ax, e, (_, t) in zip(axs[1, 1:], errores, paneles):
        im = mostrar(ax, e, cmap="magma", vmin=0, vmax=tope)
        ax.plot(*fuente, "c+", ms=11, mew=1.5)
        ax.set_title(r"$|\hat f - f|$, " + t.replace("\n", " (+ fuente)\n", 1), fontsize=10)
    barra(fig, im, axs[1, 1:].tolist())
    fig.savefig(ruta)
    plt.close(fig)


def titulo_hiper(r):
    """Hiperparámetros de la realización de la figura, con coma decimal."""
    ell, razon, M = FIGURA
    return (f"$s = {g(float(r['s']))}$, $\\nu = {g(float(r['fig_nu']))}$; forzado: $\\ell = {g(ell)}$, "
            f"$\\rho = {g(cfg.RHO)}$, razón $\\|\\delta_N\\|/\\|f\\| = {g(razon)}$ "
            f"($\\sigma_w = {c(float(r['fig_sw']), 4)}$), fuente $({cfg.FUENTE_64[0]}, {cfg.FUENTE_64[1]})$ de ancho "
            f"${g(cfg.FUENTE_ANCHO)}$; $\\sigma_n = 10^{{{int(np.log10(cfg.SIGMA_N))}}}$, $N = {N_FIG}$, $M = {M}$, "
            f"columnas cada {cfg.SEP}")


def figuras_ad(r, d):
    """Advección-difusión: reconstrucciones (f, naive dinámico, GP) y perturbación + errores, en dos figuras."""
    f, fuente = r["fig_f"], r["fig_fuente"]
    paneles = [(r["fig_f_dinamico"], "naive dinámico, $O_N^{\\dagger} u$\n(3072 sensores)"),
               (r["fig_f_gp"], "GP ($M = 2$)\n(3072 sensores: 1024 y 2 copias)")]
    sup = titulo_hiper(r)
    rutas = [os.path.join(d, "figura_presentacion_reconstruccion.pdf"), os.path.join(d, "figura_presentacion_error.pdf")]

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    im = mostrar(axs[0], f.reshape(L, L), cmap="gray", vmin=0, vmax=1)
    axs[0].set_title("$f$ (verdadera)\n(4096 píxeles)", fontsize=10)
    for ax, (v, t) in zip(axs[1:], paneles):
        im = mostrar(ax, v.reshape(L, L), cmap="gray", vmin=0, vmax=1)
        ax.set_title(r"$\hat f$, " + t, fontsize=10)
    barra(fig, im, axs.tolist())
    fig.suptitle("Reconstrucción con advección-difusión a igual cantidad de sensores\n" + sup
                 + "\n(valores fuera de [0, 1] saturados)", fontsize=9.5)
    fig.savefig(rutas[0])
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    dN = r["fig_delta_N"].reshape(L, L)
    a = np.abs(dN).max()
    imd = mostrar(axs[0], dN, cmap="RdBu_r", vmin=-a, vmax=a)
    axs[0].plot(*fuente, "k+", ms=11, mew=1.5)
    axs[0].set_title(f"perturbación acumulada $\\delta_{{{N_FIG}}}$ (+ fuente)\n(escala propia)", fontsize=10)
    barra(fig, imd, axs[0])
    errores = [np.abs(v - f).reshape(L, L) for v, _ in paneles]
    tope = max(e.max() for e in errores)
    for ax, e, (_, t) in zip(axs[1:], errores, paneles):
        im = mostrar(ax, e, cmap="magma", vmin=0, vmax=tope)
        ax.plot(*fuente, "c+", ms=11, mew=1.5)
        ax.set_title(r"$|\hat f - f|$, " + t.replace("\n", " (+ fuente)\n", 1), fontsize=10)
    barra(fig, im, axs[1:].tolist())
    fig.suptitle("Perturbación y error con advección-difusión a igual cantidad de sensores\n" + sup
                 + "\n(los dos mapas de error con la misma escala)", fontsize=9.5)
    fig.savefig(rutas[1])
    plt.close(fig)
    return rutas


def main():
    for operador, carpeta in CARPETAS.items():
        d = salida("transporte", carpeta, "comparacion_justa")
        r = np.load(os.path.join(d, "resultados.npz"))
        tabla(r, os.path.join(d, "tabla_comparacion_justa.tex"))
        if operador == "ad":                                  # dos figuras: reconstrucciones y perturbación + errores
            rutas = figuras_ad(r, d)
        else:
            rutas = [os.path.join(d, "figura_presentacion.pdf")]
            figura(r, rutas[0], operador)
        print(f"{operador}: " + ", ".join(os.path.relpath(p, salida())
                                          for p in [os.path.join(d, "tabla_comparacion_justa.tex")] + rutas))


if __name__ == "__main__":
    main()
