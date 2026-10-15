"""
Figura y tabla del cruce con el muestreo denso (Sección 4.7.5), leyendo
salida/transporte/cruce_forzado/resultados.npz: cruce_forzado_1D.pdf y tabla_cruce_1D.tex.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

from transporte.comun import config as cfg
from transporte.comun import resultados as res_io
from transporte.comun.estilo import fmt, coma_txt
from transporte.comun.rutas import salida

OUT = salida("transporte", "cruce_forzado")


def main():
    z = np.load(os.path.join(OUT, "resultados.npz"))
    RAZONES = tuple(float(r) for r in z["razones"])
    denso = dict(med=float(z["denso_med"]))
    res = {}
    for q in res_io.tabla(z, "mejores"):
        res.setdefault(q["razon"], {})[(q["ell_nombre"], q["P"])] = q
    cruces = {(c["ell_nombre"], c["P"]): c.get("cruce") for c in res_io.tabla(z, "cruces")}

    ells, pres = list(cfg.ELL_BARRIDO), list(cfg.PRESUPUESTOS_1D)
    colores = dict(zip(pres, plt.get_cmap("tab10").colors))
    fig, axs = plt.subplots(1, len(ells), figsize=(11, 4.4), sharey=True, constrained_layout=True)
    for ax, ell in zip(axs, ells):
        for P in pres:
            e = np.array([res[r][(ell, P)]["med"]["gp"] for r in RAZONES])
            ax.plot(RAZONES, e, "o-", color=colores[P], lw=2, ms=6, label=f"{P} sensores (GP)")
            for r in RAZONES:
                q = res[r][(ell, P)]
                ax.annotate(f"({q['m']},{q['M']})", (r, q["med"]["gp"]), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=6.5, color=colores[P])
            c = cruces.get((ell, P))
            if c is not None and c > 0:
                ax.axvline(c, color=colores[P], ls=":", lw=1)
        ax.axhline(denso["med"], color="0.3", ls="--", lw=1.2, label=f"denso ({cfg.L_1D} sensores, $n = 0$)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(RAZONES))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.set_xlabel(r"razón de forzado $\|\delta_N\| / \|f\|$ (RMS)")
        ax.set_title(coma_txt(f"$\\ell = {cfg.ELL_BARRIDO[ell]:g}$"), fontsize=10)
    axs[0].set_ylabel(r"$\|\hat f - f\| / \|f\|$ (mediana)")
    axs[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"Mejor $(m, M)$ por presupuesto según el GP contra el muestreo denso: 1D, $d = {cfg.L_1D}$, "
                 f"$N = {cfg.N_REC}$; punteadas: cruce", fontsize=10)
    fig.savefig(os.path.join(OUT, "cruce_forzado_1D.pdf"))
    plt.close(fig)

    lineas = []
    for razon in RAZONES:
        for ell in ells:
            for P in pres:
                q = res[razon][(ell, P)]
                lineas.append(f"{fmt(razon, 2)} & {fmt(q['ell'], 0)} & {P} & $({q['m']}, {q['M']})$ & "
                              f"{fmt(q['med']['sin_corregir'], 4)} & {fmt(q['med']['gp'], 4)} & "
                              f"{fmt(q['med']['oraculo'], 4)} \\\\")
    with open(os.path.join(OUT, "tabla_cruce_1D.tex"), "w") as fh:
        fh.write(f"% generado por transporte/figuras/cruce_forzado.py; medianas de ||f_hat - f|| / ||f|| sobre "
                 f"{cfg.REPS_1D} realizaciones; requiere booktabs\n")
        fh.write("\\begin{tabular}{ccrcccc}\n\\toprule\n")
        fh.write("razón & $\\ell$ & presupuesto & $(m, M)$ & sin corregir & GP & oráculo \\\\\n\\midrule\n")
        fh.write("\n".join(lineas) + "\n\\midrule\n")
        fh.write(f"\\multicolumn{{3}}{{l}}{{denso ({cfg.L_1D} sensores, $n = 0$)}} & --- & "
                 f"{fmt(denso['med'], 4)} & --- & --- \\\\\n\\bottomrule\n\\end{{tabular}}\n")
    print("figura y tabla en", os.path.relpath(OUT, salida()))


if __name__ == "__main__":
    main()
