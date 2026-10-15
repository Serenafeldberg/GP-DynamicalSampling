"""
Figuras y tablas del diagnóstico de frame (transporte; leen salida/transporte/diagnostico/resultados.npz):
alpha_vs_N.pdf, nucleos.pdf, invisibles.pdf, tabla_frames.csv y tabla_frames.tex.
"""
import ast
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, MaxNLocator

from transporte.comun import sensores as sn
from transporte.comun import resultados as res_io
from transporte.comun.rutas import salida
from transporte.experimentos.diagnostico_frames import DELTA, coma, s_txt

OUT = salida("transporte", "diagnostico")
COLORES = plt.get_cmap("tab10").colors          # orden categórico fijo


def corto(spec):
    """Etiqueta compacta de un núcleo para leyendas."""
    nombre = {"delta": r"$\delta$", "gaussiana": "$G$", "dgauss_x": r"$\partial_x G$", "dgauss_y": r"$\partial_y G$",
              "caja": "caja", "bspline": r"$\beta$"}[spec[0]]
    if spec[0] == "bspline":
        return f"$\\beta^{{{spec[1]}}}$(esc. {spec[2]})"
    return nombre + ("(" + ", ".join(f"{v:g}" for v in spec[1:]) + ")" if len(spec) > 1 else "")


def etiqueta(e, campos):
    partes = {"s": f"$s$ = {s_txt(e['s'])}", "nuc": " + ".join(corto(n) for n in e["nuc"]),
              "M": f"$M$ = {e['M']}", "nota": e["nota"]}
    return coma(", ".join(partes[c] for c in campos if partes[c]))


def dibujar(ax, lista, piso, techo, campos):
    """alpha_N en escala log; los ceros (alpha <= tol·lambda_max) se dibujan sobre la banda del piso.

    El grosor de línea decrece con el orden, para que las curvas superpuestas se distingan.
    """
    for i, (c, e) in enumerate(zip(COLORES, lista)):
        N = np.arange(e["N_max"] + 1)
        y = np.where(e["hay"], e["alfa"], piso)
        ax.plot(N, y, "-", color=c, lw=3 - 0.3 * i, alpha=0.85, label=etiqueta(e, campos))
        ax.plot(N[~e["hay"]], y[~e["hay"]], "x", color=c, ms=5)
        ax.plot(N[e["hay"]], y[e["hay"]], "o", color=c, ms=3)
    ax.set_yscale("log")
    ax.axhspan(piso / 4, piso * 4, color="0.88", zorder=0, label=r"$\alpha_N = 0$ (sin frame)")
    ax.set_ylim(piso / 4, techo)
    ax.yaxis.set_major_locator(LogLocator(numticks=6))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=5.5, loc="center right", framealpha=0.9, edgecolor="none")


def piso_de(lista):
    """Piso (dos décadas debajo del menor alpha > 0) y techo comunes a todos los paneles."""
    pos = np.concatenate([e["alfa"][e["hay"]] for e in lista] + [np.array([1.0])])
    return 10.0 ** (np.floor(np.log10(pos.min())) - 2), 10.0 ** (np.ceil(np.log10(pos.max())) + 1.5)


def figura_alpha(E, ruta):
    filas = [("H1", "grilla $m \\times 1$, delta", ["s"], lambda e: True),
             ("H2", "grilla $m \\times m$, delta (otros núcleos: igual)", ["s"], lambda e: e["nuc"] == [DELTA]),
             ("H3", "grilla $m \\times m$, $s = (1, 1/m)$", ["nuc"], lambda e: True),
             ("H4", "grilla $m \\times m$, $s = (1, 0)$, varios núcleos", ["nuc"], lambda e: True),
             ("H5", "profundidad $M$, delta", ["nota", "s", "M"], lambda e: True)]
    sel = [[e for e in E if e["hip"] == h and filtro(e)] for h, _, _, filtro in filas]
    piso, techo = piso_de([e for fila in sel for e in fila])
    fig, axs = plt.subplots(5, 3, figsize=(12, 17), constrained_layout=True)
    for i, (h, desc, campos, _) in enumerate(filas):
        for j, m in enumerate((2, 3, 4)):
            ax = axs[i, j]
            dibujar(ax, [e for e in sel[i] if e["m"] == m], piso, techo, campos)
            ax.set_title(f"{h}: {desc}, $m = {m}$", fontsize=8)
            ax.set_xlabel("$N$", fontsize=8)
            if j == 0:
                ax.set_ylabel(r"$\alpha_N$", fontsize=9)
    fig.savefig(ruta)
    plt.close(fig)


def figura_nucleos(E, ruta):
    nuc = [e for e in E if e["hip"] == "NUC"]
    grupos = [("delta y gaussianas", ("delta", "gaussiana")), ("cajas ($m = 2$)", ("delta", "caja")),
              ("B-splines a escala $m = 2$", ("delta", "bspline"))]
    piso, techo = piso_de(nuc)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True, sharey=True)
    for ax, (titulo, fams) in zip(axs, grupos):
        dibujar(ax, [e for e in nuc if e["nuc"][0][0] in fams], piso, techo, ["nuc"])
        ax.set_title(titulo, fontsize=9)
        ax.set_xlabel("$N$", fontsize=8)
    axs[0].set_ylabel(r"$\alpha_N$", fontsize=9)
    fig.suptitle(r"Familias de núcleos: grilla $2 \times 1$, $s = (1, 0)$, $L = 64$", fontsize=10)
    fig.savefig(ruta)
    plt.close(fig)


def figura_invisibles(filas, ruta, L=16, cuantas=4):
    """Componentes invisibles (núcleo de O) en casos sin frame, con los sensores marcados."""
    titulos = list(dict.fromkeys(r["titulo"] for r in filas))
    fig, axs = plt.subplots(len(titulos), cuantas, figsize=(2.6 * cuantas, 2.8 * len(titulos)), constrained_layout=True)
    for fila, titulo in enumerate(titulos):
        sel = [r for r in filas if r["titulo"] == titulo]
        pos = sn.posiciones(L, sel[0]["mx"], sel[0]["my"])
        for col, r in enumerate(sel[:cuantas]):
            ax = axs[fila, col]
            a = np.max(np.abs(r["img"]))
            ax.imshow(r["img"], cmap="RdBu_r", vmin=-a, vmax=a)
            ax.plot(pos[:, 0], pos[:, 1], "k.", ms=2.5)
            ax.set_title(f"$(k_y, k_x) = ({r['ky']}, {r['kx']})$", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        axs[fila, 0].set_ylabel(titulo, fontsize=9)
    fig.suptitle(f"Componentes que los sensores (•) nunca ven: $L = {L}$, $m = 2$, $N = 8$", fontsize=10)
    fig.savefig(ruta)
    plt.close(fig)


def num_tex(v):
    if v == 0:
        return "$0$"
    exp = int(np.floor(np.log10(abs(v))))
    return f"${v / 10**exp:.2f}".replace(".", "{,}") + f" \\cdot 10^{{{exp}}}$"


def tablas(E, ruta_csv, ruta_tex):
    with open(ruta_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["hipotesis", "nota", "m", "grilla", "s", "nucleos", "p", "M", "L", "N_max", "N_min_frame",
                    "estado_Nmax", "alpha_Nmax", "lambda_max_Nmax", "frac_prof_gratis_M4"])
        for e in E:
            w.writerow([e["hip"], e["nota"], e["m"], f"{e['mx']}x{e['my']}", s_txt(e["s"]),
                        " + ".join(sn.etiqueta(n) for n in e["nuc"]), e["p"], e["M"], e["L"], e["N_max"],
                        "nunca" if e["N_min"] is None else e["N_min"],
                        e["estado"], f"{e['alfa'][-1] if e['estado'] != '0' else 0.0:.6e}", f"{e['beta'][-1]:.6e}",
                        f"{e['gratis']:.2f}"])
    filas = []
    for e in E:
        nuc = coma(" + ".join(sn.etiqueta(n) for n in e["nuc"])).replace("_", r"\_")
        gratis = coma(f"{e['gratis']:.2f}")
        filas.append(f"{e['hip']} & {e['m']} & ${e['mx']}\\times{e['my']}$ & ${s_txt(e['s'])}$ & {nuc} & {e['M']} & "
                     f"{e['p']} & {'nunca' if e['N_min'] is None else e['N_min']} & "
                     f"{num_tex(e['alfa'][-1] if e['estado'] != '0' else 0.0)} & {gratis} \\\\")
    with open(ruta_tex, "w") as fh:
        fh.write("% generado por transporte/figuras/diagnostico_frames.py; requiere \\usepackage{booktabs}\n")
        fh.write("% alpha = 0: alpha_N <= 1e-14 lambda_max (cero numérico); frame si alpha_N > 1e-10 lambda_max; gratis: fracción de k = 1..4 con k s en la grilla\n")
        fh.write("\\begin{tabular}{llllllrlrr}\n\\toprule\n")
        fh.write("Hip. & $m$ & grilla & $s$ & núcleos & $M$ & $p$ & $N$ mín. & $\\alpha_{N_{\\max}}$ & gratis \\\\\n\\midrule\n")
        fh.write("\n".join(filas) + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    z = np.load(os.path.join(OUT, "resultados.npz"))
    E = []
    for e in res_io.tabla(z, "escenarios"):
        n = e["N_max"] + 1
        e.update(s=tuple(float(c) if not float(c).is_integer() else int(c) for c in e["s"]),
                 nuc=ast.literal_eval(e["nuc"]), N_min=None if e["N_min"] < 0 else e["N_min"],
                 alfa=e["alfa"][:n], beta=e["beta"][:n], hay=e["hay"][:n].astype(bool))
        E.append(e)
    archivos = [os.path.join(OUT, n) for n in ("alpha_vs_N.pdf", "nucleos.pdf", "invisibles.pdf",
                                               "tabla_frames.csv", "tabla_frames.tex")]
    figura_alpha(E, archivos[0])
    figura_nucleos(E, archivos[1])
    figura_invisibles(res_io.tabla(z, "invisibles"), archivos[2])
    tablas(E, archivos[3], archivos[4])
    print("archivos: " + ", ".join(os.path.relpath(a, salida()) for a in archivos))


if __name__ == "__main__":
    main()
