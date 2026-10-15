"""
Figuras y tablas del experimento 1D y 2D de 32 x 32 (Sección 4.7.5), leyendo
salida/transporte/experimento_1D/resultados.npz: error_1D.pdf, error_posicion_1D.pdf, presupuesto_1D.pdf,
reconstruccion_2D.pdf, video_sensores.gif y las tablas tabla_1D.tex, tabla_posicion_1D.tex,
tabla_presupuesto_1D.tex.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.ticker import FixedLocator

from transporte.comun import config as cfg
from transporte.comun import resultados as res_io
from transporte.comun.estilo import ESTIMADORES, NOMBRES, COLORES, MARCAS, fmt, coma_txt, ejes_log, mostrar, barra
from transporte.comun.rutas import salida

OUT = salida("transporte", "experimento_1D")


def figura_1D(filas, ruta):
    ells, fuentes = list(cfg.ELL_BARRIDO), list(cfg.FUENTES_1D)
    fig, axs = plt.subplots(len(ells), len(fuentes), figsize=(11, 6.5), sharex=True, sharey=True,
                            constrained_layout=True)
    for i, ell in enumerate(ells):
        for j, fuente in enumerate(fuentes):
            ax = axs[i, j]
            sel = [r for r in filas if r["ell_nombre"] == ell and r["fuente"] == fuente]
            Ms = [r["M"] for r in sel]
            for k in ESTIMADORES:
                q = np.array([r["q"][k] for r in sel])
                ax.fill_between(Ms, q[:, 0], q[:, 1], color=COLORES[k], alpha=0.15, lw=0)
                ax.plot(Ms, [r["med"][k] for r in sel], "o-", color=COLORES[k], lw=2, ms=6, label=NOMBRES[k])
            for M, r in zip(Ms, sel):
                ax.annotate("vis. " + f"{r['vis']:.2f}".replace(".", ","), (M, r["med"]["sin_corregir"]),
                            textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7, color="0.35")
            ejes_log(ax)
            ax.xaxis.set_major_locator(FixedLocator(Ms))
            ax.set_title(f"fuente {fuente} ($x_c = {cfg.FUENTES_1D[fuente]}$), "
                         + coma_txt(f"$\\ell = {cfg.ELL_BARRIDO[ell]:g}$"), fontsize=9)
            if i == len(ells) - 1:
                ax.set_xlabel("profundidad $M$")
            if j == 0:
                ax.set_ylabel(r"$\|\hat f - f\| / \|f\|$ (mediana)")
    axs[-1, 0].legend(fontsize=8, frameon=False, loc="center left")
    fig.suptitle(f"1D: $d = {cfg.L_1D}$, sensores cada {cfg.SEP}, $s = 1$, $N = {cfg.N_REC}$, "
                 + coma_txt(f"$\\sigma_n = {cfg.SIGMA_N:g}$, $\\rho = {cfg.RHO:g}$, $\\sigma_w = {cfg.SIGMA_W:g}$")
                 + "; banda: cuartiles", fontsize=10)
    fig.savefig(ruta)
    plt.close(fig)


def tabla_1D(filas, ruta, nota):
    lineas = [f"{r['fuente']} & {fmt(r['ell'], 0)} & {r['M']} & {fmt(r['vis'], 2)} & {r['costo']} & "
              f"{fmt(r['med']['sin_corregir'], 4)} & {fmt(r['med']['gp'], 4)} & {fmt(r['med']['oraculo'], 4)} \\\\"
              for r in filas]
    with open(ruta, "w") as fh:
        fh.write(f"% generado por transporte/figuras/experimento_1D.py; {nota}; medianas de ||f_hat - f|| / ||f|| sobre "
                 f"{cfg.REPS_1D} realizaciones; requiere booktabs\n")
        fh.write("\\begin{tabular}{lcccrccc}\n\\toprule\n")
        fh.write("fuente & $\\ell$ & $M$ & visibilidad & sensores extra & sin corregir & GP & oráculo \\\\\n"
                 "\\midrule\n")
        fh.write("\n".join(lineas) + "\n\\bottomrule\n\\end{tabular}\n")


def figura_posicion(filas, ruta):
    """Error GP contra M: una curva por posición de la fuente, una columna por ell (fuente angosta)."""
    ells, fuentes = list(cfg.ELL_BARRIDO), list(cfg.FUENTES_1D)
    fig, axs = plt.subplots(1, len(ells), figsize=(10, 4), sharey=True, constrained_layout=True)
    for ax, ell in zip(axs, ells):
        for c, fuente in zip(plt.get_cmap("tab10").colors[3:], fuentes):
            sel = [r for r in filas if r["ell_nombre"] == ell and r["fuente"] == fuente]
            Ms = [r["M"] for r in sel]
            q = np.array([r["q"]["gp"] for r in sel])
            ax.fill_between(Ms, q[:, 0], q[:, 1], color=c, alpha=0.15, lw=0)
            vis = "/".join(f"{r['vis']:.2f}".replace(".", ",") for r in sel)
            ax.plot(Ms, [r["med"]["gp"] for r in sel], "o-", color=c, lw=2, ms=6,
                    label=f"{fuente} ($x_c = {cfg.FUENTES_1D[fuente]}$); vis. {vis}")
        ref = [r["med"]["oraculo"] for r in filas if r["ell_nombre"] == ell and r["fuente"] == fuentes[0]]
        ax.plot(Ms, ref, ":", color="0.4", lw=1.2, label="oráculo")
        ejes_log(ax)
        ax.xaxis.set_major_locator(FixedLocator(Ms))
        ax.set_xlim(Ms[0] - 0.2, Ms[-1] + 0.2)
        ax.set_xlabel("profundidad $M$")
        ax.set_title(coma_txt(f"$\\ell = {cfg.ELL_BARRIDO[ell]:g}$"), fontsize=10)
    axs[0].set_ylabel(r"$\|\hat f_{\mathrm{GP}} - f\| / \|f\|$ (mediana)")
    axs[1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(coma_txt(f"Posición de la fuente con ventana angosta (ancho {cfg.FUENTE_ANCHO_POSICION:g}): ")
                 + f"sensores cada {cfg.SEP}, $N = {cfg.N_REC}$; vis.: visibilidad para $M = 1/2/3$; banda: cuartiles", fontsize=10)
    fig.savefig(ruta)
    plt.close(fig)


def figura_presupuesto(filas, denso, ruta):
    ells = list(cfg.ELL_BARRIDO)
    fig, axs = plt.subplots(1, len(ells), figsize=(12, 4.6), sharey=True, constrained_layout=True)
    for ax, ell in zip(axs, ells):
        sel = [r for r in filas if r["ell_nombre"] == ell]
        xs, ya = [], set()
        for i, r in enumerate(sel):
            x = i + 0.6 * sorted({q["P"] for q in sel}).index(r["P"])          # separa los grupos de presupuesto
            xs.append(x)
            for d, k in zip((-0.15, 0, 0.15), ESTIMADORES):
                if k in r["med"]:
                    lo, hi = r["q"][k]
                    ax.errorbar(x + d, r["med"][k], yerr=[[r["med"][k] - lo], [hi - r["med"][k]]], fmt=MARCAS[k],
                                color=COLORES[k], ms=6, lw=1, capsize=0, label=None if k in ya else NOMBRES[k])
                    ya.add(k)
            if r["degenerado"]:
                ax.annotate("degenerado", (x, r["med"]["sin_corregir"]), xytext=(0, 10), textcoords="offset points",
                            ha="center", fontsize=7, color="0.3")
        ax.axhline(denso["med"], color="0.3", ls="--", lw=1, label=f"denso ({cfg.L_1D} sensores, $n = 0$)")
        ax.set_xticks(xs, [f"$m = {r['m']}$\n$M = {r['M']}$" for r in sel], fontsize=8)
        for P in sorted({r["P"] for r in sel}):
            xg = [x for x, r in zip(xs, sel) if r["P"] == P]
            ax.text(np.mean(xg), 1.02, f"{P} sensores", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
            if P != max(r["P"] for r in sel):
                ax.axvline(max(xg) + 0.8, color="0.85", lw=1)
        ejes_log(ax)
        ax.set_title(coma_txt(f"$\\ell = {cfg.ELL_BARRIDO[ell]:g}$") + "\n", fontsize=10)
    axs[0].set_ylabel(r"$\|\hat f - f\| / \|f\|$ (mediana, barras: cuartiles)")
    axs[0].legend(fontsize=8, frameon=False, loc="center left")
    fig.suptitle(f"Mismo presupuesto de sensores $p\\,(M+1)$: 1D, $d = {cfg.L_1D}$, $s = 1$, $N = {cfg.N_REC}$; "
                 "$M = 0$ solo admite 'sin corregir'", fontsize=10)
    fig.savefig(ruta)
    plt.close(fig)


def tabla_presupuesto(filas, denso, ruta):
    guion = "---"
    lineas = []
    for r in filas:
        val = [fmt(r["med"][k], 4) if k in r["med"] else guion for k in ESTIMADORES]
        lineas.append(f"{r['P']} & {r['m']} & {r['M']} & {r['p']} & {r['total']} & "
                      f"{'sí' if r['degenerado'] else 'no'} & {fmt(r['ell'], 0)} & " + " & ".join(val) + " \\\\")
    with open(ruta, "w") as fh:
        fh.write(f"% generado por transporte/figuras/experimento_1D.py; medianas de ||f_hat - f|| / ||f|| sobre {cfg.REPS_1D} realizaciones;"
                 " requiere booktabs\n")
        fh.write("\\begin{tabular}{rrrrrlcccc}\n\\toprule\n")
        fh.write("presupuesto & $m$ & $M$ & $p$ & sensores & degenerado & $\\ell$ & sin corregir & GP & oráculo \\\\\n"
                 "\\midrule\n")
        fh.write("\n".join(lineas) + "\n\\midrule\n")
        fh.write(f"denso ($n = 0$) & 1 & --- & {cfg.L_1D} & {cfg.L_1D} & --- & --- & {fmt(denso['med'], 4)} & "
                 f"{guion} & {guion} \\\\\n\\bottomrule\n\\end{{tabular}}\n")


def figura_2D(res, ctx, ruta):
    r = res[-1]
    L, f = ctx["L"], ctx["f"].reshape(ctx["forma"])
    paneles = [(f, "$f$"), (r["f_hat"]["sin_corregir"].reshape(L, L), r"$\hat f$ sin corregir"),
               (r["f_hat"]["gp"].reshape(L, L), r"$\hat f$ GP")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
    for ax, (img, titulo) in zip(axs, paneles):
        im = mostrar(ax, img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(titulo)
    barra(fig, im, axs[:3].tolist())
    err = np.abs(r["f_hat"]["gp"].reshape(L, L) - f)
    im = mostrar(axs[3], err, cmap="magma", vmin=0)
    axs[3].plot(*cfg.FUENTE_2D, "c+", ms=10)
    axs[3].set_title(r"$|\hat f_{\mathrm{GP}} - f|$ (+ fuente)")
    barra(fig, im, axs[3])
    fig.suptitle(f"2D: $L = {L}$, columnas cada {cfg.SEP}, $s = (1, 0)$, $N = {cfg.N_REC}$, $M = {r['M']}$; "
                 "una realización", fontsize=10)
    fig.savefig(ruta)
    plt.close(fig)


def video_sensores(res, ctx, ruta, M=2):
    """y_n cuadro a cuadro con los sensores G y las copias (A^*)^m G, m = 1..M, marcados."""
    r = next(q for q in res if q["M"] == M)
    y = r["y"].reshape(-1, ctx["L"], ctx["L"])
    sens, copias = ctx["sensores"], ctx["copias"]                               # (fila, columna)
    fig, ax = plt.subplots(figsize=(5.2, 5.6), constrained_layout=True)
    im = mostrar(ax, y[0], cmap="gray", vmin=y.min(), vmax=y.max())
    ax.plot(sens[:, 1], sens[:, 0], "s", ms=3.2, mfc="none", mec="#00b4d8", mew=0.9, label="sensores $G$")
    ax.plot(copias[:, 1], copias[:, 0], "o", ms=2.6, mfc="none", mec="#f77f00", mew=0.9,
            label=f"copias $(A^*)^m G$, $m = 1..{M}$")
    ax.plot(*cfg.FUENTE_2D, "+", color="#e5383b", ms=10, mew=1.5, label="fuente")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=7, frameon=False)
    titulo = ax.set_title("$y_0$")

    def cuadro(n):
        im.set_data(y[n])
        titulo.set_text(f"$y_{{{n}}}$  (transporte $s = (1, 0)$, sensores fijos)")
        return [im]

    animation.FuncAnimation(fig, cuadro, frames=len(y)).save(ruta, writer=animation.PillowWriter(fps=2), dpi=110)
    plt.close(fig)

def main():
    z = np.load(os.path.join(OUT, "resultados.npz"))
    filas, filas_pos = res_io.tabla(z, "filas"), res_io.tabla(z, "filas_pos")
    filas_pres, res2 = res_io.tabla(z, "filas_pres"), res_io.tabla(z, "res2")
    denso = dict(med=float(z["denso_med"]), exacto=float(z["denso_exacto"]), mc=float(z["denso_mc"]))
    ctx2 = dict(L=int(z["L2"]), f=z["f2"], forma=tuple(z["forma2"]), sensores=z["sensores2"], copias=z["copias2"])
    figura_1D(filas, os.path.join(OUT, "error_1D.pdf"))
    tabla_1D(filas, os.path.join(OUT, "tabla_1D.tex"), f"ancho de la fuente {cfg.FUENTE_ANCHO:g}")
    figura_posicion(filas_pos, os.path.join(OUT, "error_posicion_1D.pdf"))
    tabla_1D(filas_pos, os.path.join(OUT, "tabla_posicion_1D.tex"), f"fuente angosta, ancho {cfg.FUENTE_ANCHO_POSICION:g}")
    figura_presupuesto(filas_pres, denso, os.path.join(OUT, "presupuesto_1D.pdf"))
    tabla_presupuesto(filas_pres, denso, os.path.join(OUT, "tabla_presupuesto_1D.tex"))
    figura_2D(res2, ctx2, os.path.join(OUT, "reconstruccion_2D.pdf"))
    video_sensores(res2, ctx2, os.path.join(OUT, "video_sensores.gif"))
    print("figuras y tablas en", os.path.relpath(OUT, salida()))


if __name__ == "__main__":
    main()
