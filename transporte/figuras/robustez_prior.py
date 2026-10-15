"""
Figura y tabla de robustez al prior mal especificado (Sección 4.7.5), leyendo
salida/transporte/robustez_prior/resultados.npz: robustez_prior_1D.pdf y tabla_robustez_1D.tex.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

from transporte.comun import config as cfg
from transporte.comun import resultados as res_io
from transporte.comun.estilo import fmt, coma_txt
from transporte.comun.rutas import salida
from transporte.experimentos.robustez_prior import RAZON, FACTORES_ELL

OUT = salida("transporte", "robustez_prior")


def main():
    z = np.load(os.path.join(OUT, "resultados.npz"))
    filas, denso = res_io.tabla(z, "filas"), float(z["denso"])
    L, N = cfg.L_1D, cfg.N_REC
    ells, pres = list(cfg.ELL_BARRIDO), list(cfg.PRESUPUESTOS_1D)
    colores = dict(zip(pres, plt.get_cmap("tab10").colors))
    fig, axs = plt.subplots(1, len(ells), figsize=(11, 4.4), sharey=True, constrained_layout=True)
    for ax, ell in zip(axs, ells):
        for P in pres:
            sel = [r for r in filas if r["ell_nombre"] == ell and r["P"] == P]
            se = [r for r in sel if r["tipo"] == "ell"]
            q = np.array([r["q"] for r in se])
            ax.fill_between(FACTORES_ELL, q[:, 0], q[:, 1], color=colores[P], alpha=0.12, lw=0)
            ax.plot(FACTORES_ELL, [r["med"] for r in se], "o-", color=colores[P], lw=2, ms=6,
                    label=f"{P} sensores, $(m, M) = ({se[0]['m']}, {se[0]['M']})$")
            for r, mk, dx in zip([r for r in sel if r["tipo"] == "sigma"], ("v", "^"), (0.94, 1.06)):
                ax.plot(dx, r["med"], mk, color=colores[P], mfc="white", ms=7, mew=1.5)
        ax.plot([], [], "v", color="0.3", mfc="white", mew=1.5, label=r"$\sigma_w$ del GP $\times 0{,}5$")
        ax.plot([], [], "^", color="0.3", mfc="white", mew=1.5, label=r"$\sigma_w$ del GP $\times 2$")
        ax.axhline(denso, color="0.3", ls="--", lw=1.2, label=f"denso ({L} sensores, $n = 0$)")
        ax.axvline(1.0, color="0.85", lw=1, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(FACTORES_ELL))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.set_xlabel(r"$\ell_{\mathrm{GP}} / \ell_{\mathrm{true}}$")
        ax.set_title(coma_txt(f"$\\ell_{{\\mathrm{{true}}}} = {cfg.ELL_BARRIDO[ell]:g}$"), fontsize=10)
    axs[0].set_ylabel(r"$\|\hat f_{\mathrm{GP}} - f\| / \|f\|$ (mediana)")
    axs[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    axs[1].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle(f"Prior mal especificado: razón de forzado {RAZON:g}".replace(".", ",") + r", $\rho$ verdadero; "
                 + f"1D, $d = {L}$, $N = {N}$; banda: cuartiles", fontsize=10)
    ruta_fig = os.path.join(OUT, "robustez_prior_1D.pdf")
    fig.savefig(ruta_fig)
    plt.close(fig)

    ruta_tex = os.path.join(OUT, "tabla_robustez_1D.tex")
    lineas = [f"{fmt(r['ell'], 0)} & {r['P']} & $({r['m']}, {r['M']})$ & {fmt(r['c_ell'] * r['ell'], 1)} & "
              f"{fmt(r['c_sw'], 1)} & {fmt(r['med'], 4)} & {'sí' if r['gana'] else 'no'} \\\\" for r in filas]
    with open(ruta_tex, "w") as fh:
        fh.write(f"% generado por transporte/figuras/robustez_prior.py; razón de forzado {RAZON}; medianas de ||f_hat - f|| / ||f|| "
                 f"sobre {cfg.REPS_1D} realizaciones; denso = {denso:.4f}; requiere booktabs\n")
        fh.write("\\begin{tabular}{crcccc l}\n\\toprule\n")
        fh.write("$\\ell_{\\mathrm{true}}$ & presupuesto & $(m, M)$ & $\\ell_{\\mathrm{GP}}$ & "
                 "$\\sigma_w^{\\mathrm{GP}} / \\sigma_w$ & GP & gana al denso \\\\\n\\midrule\n")
        fh.write("\n".join(lineas) + "\n\\midrule\n")
        fh.write(f"\\multicolumn{{5}}{{l}}{{denso ({L} sensores, $n = 0$)}} & {fmt(denso, 4)} & --- \\\\\n"
                 "\\bottomrule\n\\end{tabular}\n")

    print("figura y tabla en", os.path.relpath(OUT, salida()))


if __name__ == "__main__":
    main()
