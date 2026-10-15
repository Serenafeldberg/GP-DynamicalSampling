"""
Figuras y tabla del transporte entero en 64 x 64 (Sección 4.7.5), leyendo
salida/transporte/experimento_2D_64/resultados.npz: error_vs_M_2D.pdf, reconstruccion_2D_64.pdf (ell = 2),
reconstruccion_2D_64_ell8.pdf, tabla_2D_64.tex, cuadros_2D_64.pdf, cuadros_2D_64_sensores.pdf, cuadro_n8_2D_64.pdf
y sensores_n0_2D_64.pdf (y_0 con columnas medidas).
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from transporte.comun import config as cfg
from transporte.comun import resultados as res_io
from transporte.comun.estilo import ESTIMADORES, NOMBRES, COLORES, fmt, coma_txt, ejes_log, mostrar, barra
from transporte.comun.rutas import salida
from transporte.experimentos.experimento_2D_64 import EJEMPLO

OUT = salida("transporte", "experimento_2D_64")
L = 64
NS_CUADROS = (0, 2, 4, 6, 8)


def figura_error(filas, denso, ruta):
    fig, axs = plt.subplots(len(cfg.ELLS_64), len(cfg.RAZONES_64), figsize=(10, 7), sharex=True, sharey=True, constrained_layout=True)
    for i, ell in enumerate(cfg.ELLS_64):
        for j, razon in enumerate(cfg.RAZONES_64):
            ax = axs[i, j]
            sel = [r for r in filas if r["ell"] == ell and r["razon"] == razon]
            Ms = [r["M"] for r in sel]
            for k in ESTIMADORES:
                q = np.array([r["q"][k] for r in sel])
                ax.fill_between(Ms, q[:, 0], q[:, 1], color=COLORES[k], alpha=0.15, lw=0)
                ax.plot(Ms, [r["med"][k] for r in sel], "o-", color=COLORES[k], lw=2, ms=6, label=NOMBRES[k])
            ax.axhline(denso["med"], color="0.3", ls="--", lw=1.2, label=f"denso ({L * L} sensores, $n = 0$)")
            for r in sel:
                if r["degenerado"]:
                    ax.axvspan(r["M"] - 0.25, r["M"] + 0.25, color="0.9", zorder=0)
                    ax.text(r["M"], 0.97, "degenerado:\nmide todos\nlos píxeles", transform=ax.get_xaxis_transform(),
                            ha="center", va="top", fontsize=7, color="0.35")
            ejes_log(ax)
            ax.xaxis.set_major_locator(FixedLocator(Ms))
            ax.set_xlim(Ms[0] - 0.3, Ms[-1] + 0.3)
            ax.set_title(coma_txt(f"$\\ell = {ell:g}$") + ", razón de forzado " + f"{razon:g}".replace(".", ","),
                         fontsize=9)
            if i == len(cfg.ELLS_64) - 1:
                ax.set_xlabel("profundidad $M$")
            if j == 0:
                ax.set_ylabel(r"$\|\hat f - f\| / \|f\|$ (mediana)")
    axs[0, 0].legend(fontsize=7.5, frameon=False, loc="center left")
    fig.suptitle(f"2D $64 \\times 64$: columnas cada {cfg.SEP} ($p = {L * L // cfg.SEP}$), $s = (1, 0)$, "
                 f"$N = {cfg.N_REC}$, fuente ${cfg.FUENTE_64}$; banda: cuartiles", fontsize=10)
    fig.savefig(ruta)
    plt.close(fig)


def figura_reconstruccion(f, ejemplo, ruta, ell):
    img = lambda v: v.reshape(L, L)
    fig, axs = plt.subplots(2, 3, figsize=(12, 7.6), constrained_layout=True)
    for ax, (v, t) in zip(axs[0], [(f, "$f$"), (ejemplo["sin_corregir"], r"$\hat f$ sin corregir"),
                                   (ejemplo["gp"], r"$\hat f$ GP")]):
        im = mostrar(ax, img(v), cmap="gray", vmin=0, vmax=1)
        ax.set_title(t)
    barra(fig, im, axs[0].tolist())
    errores = [np.abs(img(ejemplo[k]) - img(f)) for k in ("sin_corregir", "gp")]
    tope = max(e.max() for e in errores)                                  # misma escala para los dos mapas
    for ax, e, t in zip(axs[1, :2], errores, [r"$|\hat f - f|$ sin corregir", r"$|\hat f - f|$ GP"]):
        im = mostrar(ax, e, cmap="magma", vmin=0, vmax=tope)
        ax.plot(*cfg.FUENTE_64, "c+", ms=11, mew=1.5)
        ax.set_title(t + " (+ fuente)")
    barra(fig, im, axs[1, :2].tolist())
    axs[1, 2].axis("off")
    fig.suptitle(coma_txt(f"$\\ell = {ell:g}$") + ", razón de forzado 0,2, $M = 2$; una realización",
                 fontsize=10)
    fig.savefig(ruta)
    plt.close(fig)


def tabla(filas, denso, ruta):
    lineas = [f"{fmt(r['ell'], 0)} & {fmt(r['razon'], 2)} & {r['M']}{'$^*$' if r['degenerado'] else ''} & "
              f"{fmt(r['vis'], 3)} & {r['extra']} & {fmt(r['med']['sin_corregir'], 4)} & "
              f"{fmt(r['med']['gp'], 4)} & {fmt(r['med']['oraculo'], 4)} & {fmt(denso['med'], 4)} & "
              f"{fmt(r['z'], 2)} \\\\" for r in filas]
    with open(ruta, "w") as fh:
        fh.write(f"% generado por transporte/figuras/experimento_2D_64.py; medianas de ||f_hat - f|| / ||f|| sobre {cfg.REPS_2D} "
                 "realizaciones; $^*$: degenerado (mide todos los píxeles); |z|: máx. desvío identidad vs MC; "
                 "requiere booktabs\n")
        fh.write("\\begin{tabular}{cccccccccc}\n\\toprule\n")
        fh.write("$\\ell$ & razón & $M$ & $1 - \\kappa_M^2/\\operatorname{tr}\\Sigma_w$ & sensores extra & "
                 "sin corregir & GP & oráculo & denso & $|z|$ \\\\\n\\midrule\n")
        fh.write("\n".join(lineas) + "\n\\bottomrule\n\\end{tabular}\n")


def figuras_cuadros(ctx, delta0, ell):
    """cuadros_2D_64.pdf (A^n f, delta_n, y_n), su versión con sensores, cuadro_n8_2D_64.pdf (solo n = N)
    y sensores_n0_2D_64.pdf (y_0 con sensores y copias marcados)."""
    Anf = ctx["Anf"].reshape(-1, L, L)
    delta = delta0.reshape(-1, L, L)
    y = Anf + delta
    M = EJEMPLO[2]
    sens, copias = ctx["columnas_sensores"], ctx["columnas_copias"]                 # columnas medidas
    a = np.max(np.abs(delta[list(NS_CUADROS)]))
    filas = [(r"$A^n f$", Anf, dict(cmap="gray", vmin=0, vmax=1)),
             (r"$\delta_n$", delta, dict(cmap="RdBu_r", vmin=-a, vmax=a)),
             (r"$y_n$", y, dict(cmap="gray", vmin=0, vmax=1))]                    # misma escala que A^n f
    rutas = []
    for con_sensores, nombre in ((False, "cuadros_2D_64.pdf"), (True, "cuadros_2D_64_sensores.pdf")):
        fig, axs = plt.subplots(3, len(NS_CUADROS), figsize=(2.3 * len(NS_CUADROS) + 0.8, 7.4), constrained_layout=True)
        for i, (etq, serie, kw) in enumerate(filas):
            for j, n in enumerate(NS_CUADROS):
                ax = axs[i, j]
                im = mostrar(ax, serie[n], **kw)
                if i == 0:
                    ax.set_title(f"$n = {n}$")
                if i == 1:
                    ax.plot(*cfg.FUENTE_64, "+", color="k", ms=9, mew=1.4)
                if i == 2 and con_sensores:                     # columnas completas: marca arriba y línea tenue
                    for cols, color in ((copias, "#f77f00"), (sens, "#00b4d8")):
                        ax.vlines(cols, -0.5, L - 0.5, color=color, lw=0.45, alpha=0.35)
                        ax.plot(cols, np.full(len(cols), -2.2), "v", ms=2.6, color=color, clip_on=False)
                    ax.set_ylim(L - 0.5, -0.5)
            axs[i, 0].set_ylabel(etq, fontsize=13)
            barra(fig, im, axs[i, :].tolist())
        titulo = (coma_txt(f"$\\ell = {ell:g}$") + ", razón de forzado 0,2, $M = 2$, fuente "
                  f"${cfg.FUENTE_64}$ (+), $N = {cfg.N_REC}$; la misma realización de reconstruccion_2D_64.pdf\n"
                  "$y_n$: valores fuera de [0, 1] saturados")
        if con_sensores:
            axs[2, 0].plot([], [], "v", color="#00b4d8", label="columnas de sensores $G$")
            axs[2, 0].plot([], [], "v", color="#f77f00", label=r"columnas de las copias $(A^*)G$, $(A^*)^2 G$")
            fig.legend(loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.035))
        fig.suptitle(titulo, fontsize=10)
        rutas.append(os.path.join(OUT, nombre))
        fig.savefig(rutas[-1], bbox_inches="tight")
        plt.close(fig)

    # un solo instante: A^N f, delta_N, y_N con las mismas escalas que cuadros_2D_64
    n = cfg.N_REC
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), constrained_layout=True)
    for ax, (etq, serie, kw) in zip(axs, filas):
        im = mostrar(ax, serie[n], **kw)
        ax.set_title(etq.replace("n", str(n)) if etq != r"$A^n f$" else f"$A^{{{n}}} f$", fontsize=13)
        barra(fig, im, ax)
    axs[1].plot(*cfg.FUENTE_64, "+", color="k", ms=10, mew=1.5)
    fig.suptitle(coma_txt(f"$\\ell = {ell:g}$") + f", razón de forzado 0,2, $M = 2$, fuente ${cfg.FUENTE_64}$ (+), "
                 f"$n = {n}$; $y_{{{n}}}$: valores fuera de [0, 1] saturados", fontsize=10)
    rutas.append(os.path.join(OUT, "cuadro_n8_2D_64.pdf"))
    fig.savefig(rutas[-1], bbox_inches="tight")
    plt.close(fig)

    # sensores en n = 0: y_0 = f con las columnas de G y de las copias (A^*)^m G, m = 1..M
    fig, ax = plt.subplots(figsize=(6, 6.4), constrained_layout=True)
    im = mostrar(ax, y[0], cmap="gray", vmin=0, vmax=1)
    for cols, color, etq in ((copias, "#f77f00", r"copias $(A^*)G$, $(A^*)^2 G$"), (sens, "#00b4d8", "sensores $G$")):
        ax.vlines(cols, -0.5, L - 0.5, color=color, lw=0.8, alpha=0.45)
        ax.plot(cols, np.full(len(cols), -2.0), "v", ms=4.5, color=color, clip_on=False, label=etq)
    ax.plot(*cfg.FUENTE_64, "+", color="#e5383b", ms=11, mew=1.8, label="fuente")
    ax.set_ylim(L - 0.5, -0.5)
    barra(fig, im, ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=8, frameon=False)
    ax.set_title(f"$y_0 = f$: columnas medidas (sensores cada {cfg.SEP}, $p = {L * L // cfg.SEP}$; copias con $M = {M}$)",
                 fontsize=10, pad=20)
    rutas.append(os.path.join(OUT, "sensores_n0_2D_64.pdf"))
    fig.savefig(rutas[-1], bbox_inches="tight")
    plt.close(fig)
    return rutas


def main():
    z = np.load(os.path.join(OUT, "resultados.npz"))
    filas = res_io.tabla(z, "filas")
    denso = dict(formula=float(z["denso_formula"]), med=float(z["denso_med"]), rms=float(z["denso_rms"]))
    f = z["f"]
    ejemplos = {ell: {k: z[f"ejemplo_ell{ell:g}_{k}"] for k in ("sin_corregir", "gp", "oraculo", "delta")}
                for ell in cfg.ELLS_64}
    archivos = [os.path.join(OUT, n) for n in ("error_vs_M_2D.pdf", "reconstruccion_2D_64.pdf",
                                               "reconstruccion_2D_64_ell8.pdf", "tabla_2D_64.tex")]
    figura_error(filas, denso, archivos[0])
    for ell, ruta in zip(cfg.ELLS_64, archivos[1:3]):
        figura_reconstruccion(f, ejemplos[ell], ruta, ell)
    tabla(filas, denso, archivos[3])
    ctx = dict(Anf=z["Anf"], B=None, columnas_sensores=z["columnas_sensores"], columnas_copias=z["columnas_copias"])
    archivos += figuras_cuadros(ctx, ejemplos[EJEMPLO[0]]["delta"], EJEMPLO[0])
    print("archivos: " + ", ".join(os.path.relpath(a, salida()) for a in archivos))


if __name__ == "__main__":
    main()
