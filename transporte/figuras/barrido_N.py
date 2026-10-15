"""
Figuras y tabla del barrido en N (Sección 4.7.5), leyendo salida/transporte/barrido_N/resultados.npz (no recalcula nada).

    error_vs_N.pdf           error contra N, un panel por (operador, ell), sin título interno
    alpha_vs_N_barrido.pdf   alpha_N contra N para los tres operadores
    tabla_barrido_N.tex      medianas, alpha_N, sigma_w y |z|
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator, LogFormatterMathtext

from transporte.comun import estilo  # noqa: F401  (tipografía común)
from transporte.comun.rutas import salida

OUT = salida("transporte", "barrido_N")
NOMBRES = {"sin_corregir": "sin corregir", "gp": "GP", "oraculo": "oráculo"}
COLORES = dict(zip(("sin_corregir", "gp", "oraculo"), plt.get_cmap("tab10").colors))


def num(v, dec):
    """Número para mathtext/LaTeX con coma decimal: 0{,}75."""
    return f"{v:.{dec}f}".rstrip("0").rstrip(".").replace(".", "{,}") if dec else f"{v:g}"


def etiqueta_operador(s, nu):
    if nu == 0 and float(s).is_integer():
        return f"transporte entero, $s = {num(s, 2)}$"
    return f"advección-difusión, $s = {num(s, 2)}$, $\\nu = {num(nu, 2)}$"


def ejes_log(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(numticks=8))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())


def error_vs_N(r, ruta):
    ops, ells, Ns, est = r["operadores"], r["ells"], r["Ns"], [str(e) for e in r["estimadores"]]
    fig, axs = plt.subplots(len(ops), len(ells), figsize=(10, 3.2 * len(ops)), sharex=True, sharey=True,
                            constrained_layout=True)
    for a, (s, nu) in enumerate(ops):
        for b, ell in enumerate(ells):
            ax = axs[a, b]
            for k, nombre in enumerate(est):
                ax.fill_between(Ns, r["q25"][a, b, :, k], r["q75"][a, b, :, k], color=COLORES[nombre], alpha=0.15, lw=0)
                ax.plot(Ns, r["med"][a, b, :, k], "o-", color=COLORES[nombre], lw=2, ms=5, label=NOMBRES[nombre])
            ax.axhline(float(r["denso"]), color="0.3", ls="--", lw=1.2, label="denso (4096 sensores, $n = 0$)")
            ejes_log(ax)
            ax.xaxis.set_major_locator(FixedLocator(Ns))
            ax.set_title(etiqueta_operador(s, nu) + f", $\\ell = {num(ell, 0)}$", fontsize=9)
            if a == len(ops) - 1:
                ax.set_xlabel("horizonte $N$")
            if b == 0:
                ax.set_ylabel(r"$\|\hat f - f\| / \|f\|$ (mediana)", fontsize=9)
    axs[0, 0].legend(fontsize=7.5, frameon=False, loc="center right")
    fig.savefig(ruta)
    plt.close(fig)


def alpha_vs_N(r, ruta):
    ops, Ns = r["operadores"], r["Ns"]
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for a, (s, nu) in enumerate(ops):
        ax.plot(Ns, r["alfa"][a, 0], "o-", lw=2, ms=5, label=etiqueta_operador(s, nu))
    ejes_log(ax)
    ax.xaxis.set_major_locator(FixedLocator(Ns))
    ax.set_xlabel("horizonte $N$")
    ax.set_ylabel(r"$\alpha_N = \lambda_{\min}(O_N^* O_N)$")
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(ruta)
    plt.close(fig)


def tex_num(v, dec=4):
    return "$" + f"{v:.{dec}f}".replace(".", "{,}") + "$"


def tex_cien(v):
    e = int(np.floor(np.log10(abs(v))))
    return f"${v / 10 ** e:.2f}".replace(".", "{,}") + f" \\cdot 10^{{{e}}}$"


def tabla(r, ruta):
    ops, ells, Ns = r["operadores"], r["ells"], r["Ns"]
    lineas = []
    for a, (s, nu) in enumerate(ops):
        for b, ell in enumerate(ells):
            for c, N in enumerate(Ns):
                m = r["med"][a, b, c]
                lineas.append(f"${num(s, 2)}$ & ${num(nu, 2)}$ & ${num(ell, 0)}$ & {N} & {tex_cien(r['alfa'][a, b, c])} & "
                              f"{tex_num(r['sw'][a, b, c])} & {tex_num(m[0])} & {tex_num(m[1])} & {tex_num(m[2])} & "
                              f"{tex_num(float(r['denso']))} & {tex_num(np.max(np.abs(r['z'][a, b, c])), 2)} \\\\")
            lineas.append("\\midrule")
    lineas.pop()
    with open(ruta, "w") as fh:
        fh.write(f"% generado por transporte/figuras/barrido_N.py desde resultados.npz; M = {int(r['M'])}, razón de forzado "
                 f"{float(r['razon'])}, medianas de ||f_hat - f|| / ||f|| sobre {int(r['R'])} realizaciones; requiere booktabs\n")
        fh.write("\\begin{tabular}{cccrccccccc}\n\\toprule\n")
        fh.write("$s$ & $\\nu$ & $\\ell$ & $N$ & $\\alpha_N$ & $\\sigma_w$ & sin corregir & GP & oráculo & denso & "
                 "$|z|$ \\\\\n\\midrule\n")
        fh.write("\n".join(lineas) + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    r = np.load(os.path.join(OUT, "resultados.npz"))
    rutas = [os.path.join(OUT, n) for n in ("error_vs_N.pdf", "alpha_vs_N_barrido.pdf", "tabla_barrido_N.tex")]
    error_vs_N(r, rutas[0])
    alpha_vs_N(r, rutas[1])
    tabla(r, rutas[2])
    print("archivos: " + ", ".join(os.path.relpath(p, salida()) for p in rutas))


if __name__ == "__main__":
    main()
