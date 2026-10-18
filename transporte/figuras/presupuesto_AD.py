"""
Tabla de la comparación por presupuesto en advección-difusión (Sección 4.7.5), a partir de
salida/transporte/adveccion_difusion/presupuesto/resultados.npz:

    salida/transporte/adveccion_difusion/presupuesto/tabla_presupuesto_AD.tex
    salida/transporte/adveccion_difusion/presupuesto/ingenua_1024_vs_3072.pdf   (si existe ejemplo_ingenua.npz)
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from transporte.comun import config as cfg
from transporte.comun import resultados as res_io
from transporte.comun.estilo import mostrar, barra
from transporte.comun.rutas import salida

OUT = salida("transporte", "adveccion_difusion", "presupuesto")


def cientifico(v, dec=2):
    """$4{,}91 \\cdot 10^{-5}$."""
    m, e = f"{v:.{dec}e}".split("e")
    return f"${m.replace('.', '{,}')} \\cdot 10^{{{int(e)}}}$"


def num(v, dec=4):
    return "$" + f"{v:.{dec}f}".replace(".", "{,}") + "$"


def tabla(filas, denso, descartadas, ruta):
    lineas, previo = [], None
    for r in sorted(filas, key=lambda r: (r["P"], r["m"], r["ell"])):
        if previo is not None and r["P"] != previo:
            lineas.append("\\midrule")
        previo = r["P"]
        celda = lambda k: num(r["med"][k]) if k in r["med"] else "---"
        lineas.append(f"{r['P']} & {r['m']} & {r['p']} & {r['M']} & {cientifico(r['alfa'])} & {num(r['ell'], 0)} & "
                      f"{celda('sin_corregir')} & {num(r['ingenua'])} & {celda('gp')} & {celda('oraculo')} \\\\")
    desc = "; ".join(f"P = {d['P']}, m = {d['m']} (p = {d['p']}, P/p no entero)" for d in descartadas)
    with open(ruta, "w") as fh:
        fh.write("% generado por transporte/figuras/presupuesto_AD.py; medianas de ||f_hat - f|| / ||f|| sobre 200 "
                 f"realizaciones; s = 0,75, nu = 0,1, razón 0,2, N = 8; denso (4096 sensores, n = 0) = {f"{denso:.4f}".replace(".", ",")}; "
                 f"descartadas: {desc}; requiere booktabs\n")
        fh.write("\\begin{tabular}{cccccccccc}\n\\toprule\n")
        fh.write("$P$ & $m$ & $p$ & $M$ & $\\alpha_8$ & $\\ell$ & sin corregir & ingenua & GP & oráculo \\\\\n"
                 "\\midrule\n")
        fh.write("\n".join(lineas) + "\n\\midrule\n")
        fh.write(f"\\multicolumn{{6}}{{l}}{{denso (4096 sensores, $n = 0$)}} & \\multicolumn{{4}}{{c}}{{{num(denso)}}} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


def g(v):
    return f"{v:g}".replace(".", "{,}")


def figura_ingenua(z, ruta):
    """Fila 1: f y las reconstrucciones ingenuas O_8^+ u; fila 2: delta_N y |f_hat - f| (misma escala)."""
    L = int(np.sqrt(z["f"].size))
    f, fuente = z["f"], tuple(int(c) for c in z["fuente"])
    Ps = [int(P) for P in z["presupuestos"]]
    col = {1024: "cada 4", 2048: "pares", 3072: r"$x \not\equiv 3$ mod 4"}
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 8.4), constrained_layout=True)
    im = mostrar(axs[0, 0], f.reshape(L, L), cmap="gray", vmin=0, vmax=1)
    axs[0, 0].set_title("$f$ (verdadera)", fontsize=10)
    for ax, P in zip(axs[0, 1:], Ps):
        im = mostrar(ax, z[f"P{P}_f_hat"].reshape(L, L), cmap="gray", vmin=0, vmax=1)
        err0 = np.linalg.norm(z[f"P{P}_f_hat"] - f) / np.linalg.norm(f)
        med = float(z[f"P{P}_med"])
        ax.set_title(f"$\\hat f$ ingenua, {P} sensores (columnas {col[P]})\n"
                     f"error relativo ${g(round(err0, 3))}$ (mediana en 200: ${g(round(med, 3))}$)", fontsize=9.5)
    barra(fig, im, axs[0].tolist())
    dN = z["delta_N"].reshape(L, L)
    a = np.abs(dN).max()
    imd = mostrar(axs[1, 0], dN, cmap="RdBu_r", vmin=-a, vmax=a)
    axs[1, 0].plot(*fuente, "k+", ms=11, mew=1.5)
    axs[1, 0].set_title(f"perturbación acumulada $\\delta_{{{cfg.N_REC}}}$ (+ fuente)", fontsize=10)
    barra(fig, imd, axs[1, 0])
    errores = [np.abs(z[f"P{P}_f_hat"] - f).reshape(L, L) for P in Ps]
    tope = max(e.max() for e in errores)
    for ax, e, P in zip(axs[1, 1:], errores, Ps):
        im = mostrar(ax, e, cmap="magma", vmin=0, vmax=tope)
        ax.plot(*fuente, "c+", ms=11, mew=1.5)
        ax.set_title(f"$|\\hat f - f|$, {P} sensores", fontsize=10)
    barra(fig, im, axs[1, 1:].tolist())
    fig.suptitle(f"$s = 0{{,}}75$, $\\nu = 0{{,}}1$, $\\ell = {g(float(z['ell']))}$, razón $0{{,}}2$, $N = {cfg.N_REC}$, "
                 f"$\\sigma_n = 10^{{-3}}$", fontsize=11)
    fig.savefig(ruta)
    plt.close(fig)


def main():
    z = np.load(os.path.join(OUT, "resultados.npz"))
    ruta = os.path.join(OUT, "tabla_presupuesto_AD.tex")
    tabla(res_io.tabla(z, "filas"), float(z["denso"]), res_io.tabla(z, "descartadas"), ruta)
    print("archivo: " + os.path.relpath(ruta, salida()))
    ruta_ej = os.path.join(OUT, "ejemplo_ingenua.npz")
    if os.path.exists(ruta_ej):
        ruta = os.path.join(OUT, "ingenua_1024_vs_3072.pdf")
        figura_ingenua(np.load(ruta_ej), ruta)
        print("archivo: " + os.path.relpath(ruta, salida()))


if __name__ == "__main__":
    main()
