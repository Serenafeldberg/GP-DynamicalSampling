"""
Figuras y tablas de advección-difusión (Sección 4.7.5), leyendo salida/transporte/adveccion_difusion/
resultados_diagnostico.npz y resultados.npz (y, como referencia entera, salida/transporte/experimento_2D_64/
resultados.npz): alpha_vs_nu.pdf, alpha_vs_s.pdf, tabla_diagnostico_AD.tex, tabla_barrido_s_AD.tex,
error_vs_M_AD.pdf, reconstruccion_AD.pdf, cuadros_AD.pdf, cuadros_AD_n0_n8.pdf y tabla_AD.tex.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

from transporte.comun import config as cfg
from transporte.comun import frames as fr
from transporte.comun import resultados as res_io
from transporte.comun.estilo import ESTIMADORES, NOMBRES, COLORES, fmt, coma_txt, ejes_log, mostrar, barra
from transporte.comun.rutas import salida
from transporte.experimentos.adveccion_difusion import (L, NUS, SS, UMBRAL_ALFA, NUS_S, S_EXP, OPERADORES, EJEMPLO)

OUT = salida("transporte", "adveccion_difusion")
num_m = lambda v: f"{v:g}".replace(".", "{,}")              # número dentro de mathtext: 0{,}75
num_t = lambda v: f"{v:g}".replace(".", ",")                # número en texto: 0,75


def figura(filas, nus_fino, a_fino, o_fino, denso, ruta):
    coma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    colores = {1.0: "C0", 1.3: "C1", 0.0: "C3"}
    nombre = {1.0: "$s = 1$", 1.3: "$s = 1{,}3$", 0.0: "$s = 0$ (control: difusión pura)"}
    vals = np.concatenate([r["alfa"][r["hay"]] for r in filas] + [a_fino[s][a_fino[s] > 0] for s in SS])
    piso = 10.0 ** (np.floor(np.log10(vals[vals > 0].min())) - 2)
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.4), constrained_layout=True)

    ax = axs[0]
    for s in (1.0, 1.3, 0.0):
        sel = [r for r in filas if r["s"] == s]
        y = [r["alfa"][-1] if r["hay"][-1] else piso for r in sel]
        ax.plot([r["nu"] for r in sel], y, "o", color=colores[s], ms=7, label=nombre[s], zorder=3)
        if s in a_fino:
            ax.plot(nus_fino, np.maximum(a_fino[s], piso), "-", color=colores[s], lw=1.2, alpha=0.7)
    ax.axhline(UMBRAL_ALFA, color="0.3", ls="--", lw=1, label=r"umbral $10^{-8}$")
    ax.axhspan(piso / 3, piso * 3, color="0.88", zorder=0, label=r"$\alpha_N = 0$")
    ax.set_yscale("log")
    ax.set_xlabel(r"difusión por paso $\nu$")
    ax.set_ylabel(fr"$\alpha_N$, $N = {cfg.N_REC}$")
    ax.set_title(r"$\alpha_N$ contra $\nu$ (línea: barrido cada 0,01)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    ax = axs[1]
    for i, r in enumerate([r for r in filas if r["s"] == 1.3]):
        Ns = np.arange(len(r["alfa"]))
        ax.plot(Ns, np.where(r["hay"], r["alfa"], piso), "o-", color=plt.get_cmap("viridis")(i / 4), lw=1.6, ms=4,
                label=f"$\\nu = {r['nu']:g}$".replace(".", "{,}"))
    ax.axhline(UMBRAL_ALFA, color="0.3", ls="--", lw=1)
    ax.axhspan(piso / 3, piso * 3, color="0.88", zorder=0)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(range(cfg.N_REC + 1)))
    ax.set_xlabel("$N$")
    ax.set_title(r"$\alpha_N$ contra $N$, $s = 1{,}3$", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="center right")

    ax = axs[2]
    for s in (1.0, 1.3, 0.0):
        sel = [r for r in filas if r["s"] == s]
        ax.plot([r["nu"] for r in sel], [max(r["sep"], piso) for r in sel], "o-", color=colores[s], ms=6,
                label=nombre[s])
    ax.set_yscale("log")
    ax.set_xlabel(r"difusión por paso $\nu$")
    ax.text(0.25, piso * 3, "control: nodos repetidos (separación 0)", color=colores[0.0], fontsize=7.5, va="bottom")
    ax.set_title(r"separación mínima entre nodos $\hat a$ de una clase", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)

    ax = axs[3]
    for s in SS:
        ax.plot(nus_fino, o_fino[s], "-", color=colores[s], lw=1.2, alpha=0.7)
        sel = [r for r in filas if r["s"] == s]
        ax.plot([r["nu"] for r in sel], [r["oraculo"] for r in sel], "o", color=colores[s], ms=6, label=nombre[s])
    ax.axhline(denso, color="0.3", ls="--", lw=1, label=f"denso ({L * L} sensores, $n = 0$)")
    ax.set_yscale("log")
    ax.set_xlabel(r"difusión por paso $\nu$")
    ax.set_title(r"piso del oráculo: $\sigma_n \sqrt{\mathrm{tr}(O_N^* O_N)^{-1}} / \|f\|$", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    for a in (axs[0], axs[2], axs[3]):
        a.xaxis.set_major_locator(FixedLocator(NUS))
        a.xaxis.set_major_formatter(coma)
    fig.suptitle(f"Advección-difusión, $64 \\times 64$: sensores delta en columnas cada {cfg.SEP}, "
                 f"$N = {cfg.N_REC}$; $\\alpha_N$ por bloques de alias", fontsize=10)
    fig.savefig(ruta)
    plt.close(fig)


def tabla(filas, ruta):
    lineas = []
    for r in filas:
        nmin = "nunca" if r["N_min"] is None else r["N_min"]
        alfa = r["alfa"][-1] if r["hay"][-1] else 0.0
        lineas.append(f"{fmt(r['s'], 1)} & {fmt(r['nu'], 2)} & {nmin} & {num_tex(alfa)} & "
                      f"{num_tex(alfa / r['beta'][-1])} & {num_tex(r['sep'])} & "
                      f"{num_tex(r['oraculo']) if np.isfinite(r['oraculo']) else '---'} \\\\")
    with open(ruta, "w") as fh:
        fh.write("% generado por transporte/figuras/adveccion_difusion.py (diagnóstico); requiere booktabs\n")
        fh.write("\\begin{tabular}{ccccccc}\n\\toprule\n")
        fh.write(f"$s$ & $\\nu$ & $N$ mín. & $\\alpha_{{{cfg.N_REC}}}$ & $\\alpha_{{{cfg.N_REC}}}/\\lambda_{{\\max}}$ & "
                 "sep. mín. nodos & piso oráculo \\\\\n\\midrule\n")
        fh.write("\n".join(lineas) + "\n\\bottomrule\n\\end{tabular}\n")


def num_tex(v):
    if v == 0:
        return "$0$"
    if not np.isfinite(v):
        return "$\\infty$"
    e = int(np.floor(np.log10(abs(v))))
    return f"${v / 10 ** e:.2f}".replace(".", "{,}") + f" \\cdot 10^{{{e}}}$"


def figura_s(filas, denso, s_prop, ruta):
    coma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    vals = np.array([r["alfa"] for r in filas])
    piso = min(10.0 ** (np.floor(np.log10(vals[vals > fr.TOL_FRAME].min())) - 2), UMBRAL_ALFA / 100)
    for c, nu in zip(("C0", "C2", "C4"), NUS_S):
        sel = [r for r in filas if r["nu"] == nu]
        ss = [r["s"] for r in sel]
        lab = f"$\\nu = {nu:g}$".replace(".", "{,}")
        axs[0].plot(ss, [r["alfa"] if r["alfa"] > fr.TOL_FRAME * r["beta"] else piso for r in sel], "o-", color=c,
                    ms=3.5, lw=1.4, label=lab)
        axs[1].plot(ss, [max(r["sep"], piso) for r in sel], "o-", color=c, ms=3.5, lw=1.4, label=lab)
        axs[2].plot(ss, [r["oraculo"] for r in sel], "o-", color=c, ms=3.5, lw=1.4, label=lab)
    axs[0].axhspan(piso / 3, piso * 3, color="0.88", zorder=0, label=r"$\alpha_N = 0$")
    axs[0].axhline(UMBRAL_ALFA, color="0.3", ls="--", lw=1, label=r"umbral $10^{-8}$")
    axs[2].axhline(denso, color="0.3", ls="--", lw=1, label=f"denso ({L * L} sensores)")
    titulos = [fr"$\alpha_N$, $N = {cfg.N_REC}$", r"separación mínima entre nodos $\hat a$",
               r"piso del oráculo $\sigma_n \sqrt{\mathrm{tr}(O_N^*O_N)^{-1}}/\|f\|$"]
    for ax, t in zip(axs, titulos):
        ax.set_yscale("log")
        for x0, x1 in ((0.5, 1.5),):
            ax.axvspan(x0, x1, color="#fdf1e2", zorder=0)
        marcas = ((1.0, "$s = 1$", 1.01, "center"), (1.3, "$s = 1{,}3$", 1.01, "center"),
                  (s_prop, f"propuesto $s = {s_prop:g}$".replace(".", "{,}"), 1.07, "right"))
        for sv, etq, alto, ha in marcas:
            ax.axvline(sv, color="0.45", ls=":", lw=1)
            ax.text(sv, alto, etq, transform=ax.get_xaxis_transform(), ha=ha, va="bottom", fontsize=7)
        ax.set_xlabel("velocidad $s$ (píxeles por paso)")
        ax.xaxis.set_major_locator(FixedLocator(np.arange(0, 2.01, 0.25)))
        ax.xaxis.set_major_formatter(coma)
        ax.set_title(t + "\n\n", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="upper right" if ax is axs[2] else "lower center")
    fig.suptitle(f"Barrido en $s$: sensores delta en columnas cada {cfg.SEP}, $N = {cfg.N_REC}$; "
                 r"fondo naranja: $\cos(\pi s) < 0$", fontsize=10)
    fig.savefig(ruta)
    plt.close(fig)


def tabla_barrido_s(filas, ruta):
    with open(ruta, "w") as fh:
        fh.write("% generado por transporte/figuras/adveccion_difusion.py (barrido en s); requiere booktabs\n")
        fh.write("\\begin{tabular}{ccccccc}\n\\toprule\n$s$ & $\\nu$ & $\\alpha_8$ & sep. mín. & clase $q$ & "
                 "par $\\xi_x/\\pi$ & piso oráculo \\\\\n\\midrule\n")
        for r in filas:
            fh.write(f"{fmt(r['s'], 2)} & {fmt(r['nu'], 2)} & {num_tex(max(r['alfa'], 0))} & {num_tex(r['sep'])} & "
                     f"{r['q']} & $({r['par'][0]:+.3f}, {r['par'][1]:+.3f})$ & {num_tex(r['oraculo'])} \\\\\n"
                     .replace(".", "{,}").replace("{,}tex", ".tex"))
        fh.write("\\bottomrule\n\\end{tabular}\n")


def figura_error_ad(filas, ref, denso, ruta):
    combos = [(ell, razon) for ell in cfg.ELLS_64 for razon in cfg.RAZONES_64]
    renglones = [("referencia: $s = 1$, $\\nu = 0$ (experimento_2D_64)", None)] + \
                [(f"$s = {S_EXP:g}$, $\\nu = {nu:g}$".replace(".", "{,}"), nu) for _, nu in OPERADORES[:3]]
    fig, axs = plt.subplots(len(renglones), len(combos), figsize=(14, 12), sharex=True, sharey=True, constrained_layout=True)
    for i, (etq, nu) in enumerate(renglones):
        for j, (ell, razon) in enumerate(combos):
            ax = axs[i, j]
            if nu is None:
                sel = [r for r in ref if r["ell"] == ell and r["razon"] == razon]
            else:
                sel = [r for r in filas if r["s"] == S_EXP and r["nu"] == nu and r["ell"] == ell and r["razon"] == razon]
            Ms = [r["M"] for r in sel]
            for k in ESTIMADORES:
                if nu is not None:
                    q = np.array([r["q"][k] for r in sel])
                    ax.fill_between(Ms, q[:, 0], q[:, 1], color=COLORES[k], alpha=0.15, lw=0)
                ax.plot(Ms, [r["med"][k] for r in sel], "o-" if nu is not None else "s--", color=COLORES[k],
                        lw=2 if nu is not None else 1.3, ms=6 if nu is not None else 5, label=NOMBRES[k])
            ax.axhline(denso, color="0.3", ls="--", lw=1.2, label=f"denso ({L * L} sensores, $n = 0$)")
            ax.axvspan(2.75, 3.25, color="0.92", zorder=0)
            if i == 0:
                ax.text(3, 0.97, "mismo número\nde sensores\nque el denso", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=6.5, color="0.35")
            ejes_log(ax)
            ax.xaxis.set_major_locator(FixedLocator([1, 2, 3]))
            ax.set_xlim(0.7, 3.3)
            ax.set_title(etq + ", " + coma_txt(f"$\\ell = {ell:g}$") + ", razón " + num_t(razon), fontsize=7.5)
            if i == len(renglones) - 1:
                ax.set_xlabel("profundidad $M$")
            if j == 0:
                ax.set_ylabel(r"$\|\hat f - f\| / \|f\|$ (mediana)", fontsize=8)
    axs[1, 0].legend(fontsize=7, frameon=False, loc="center left")
    fig.suptitle(f"Advección-difusión $64 \\times 64$: columnas cada {cfg.SEP}, $N = {cfg.N_REC}$; "
                 "fila 1: referencia entera (sin cuartiles); banda: cuartiles", fontsize=10)
    fig.savefig(ruta)
    plt.close(fig)


def figura_reconstruccion_ad(f, ej, fuente, ruta):
    img = lambda v: v.reshape(L, L)
    s, nu, ell, razon, M = EJEMPLO
    fig, axs = plt.subplots(2, 3, figsize=(12, 7.6), constrained_layout=True)
    for ax, (v, t) in zip(axs[0], [(f, "$f$"), (ej["sin_corregir"], r"$\hat f$ sin corregir"), (ej["gp"], r"$\hat f$ GP")]):
        im = mostrar(ax, img(v), cmap="gray", vmin=0, vmax=1)
        ax.set_title(t)
    barra(fig, im, axs[0].tolist())
    errores = [np.abs(img(ej[k]) - img(f)) for k in ("sin_corregir", "gp")]
    tope = max(e.max() for e in errores)
    for ax, e, t in zip(axs[1, :2], errores, [r"$|\hat f - f|$ sin corregir", r"$|\hat f - f|$ GP"]):
        im = mostrar(ax, e, cmap="magma", vmin=0, vmax=tope)
        ax.plot(*fuente, "c+", ms=11, mew=1.5)
        ax.set_title(t + " (+ fuente)")
    barra(fig, im, axs[1, :2].tolist())
    axs[1, 2].axis("off")
    fig.suptitle(f"$s = {num_m(s)}$, $\\nu = {num_m(nu)}$, $\\ell = {num_m(ell)}$, razón de forzado {num_t(razon)}, $M = {M}$; "
                 "una realización; $\\hat f$ con valores fuera de [0, 1] saturados", fontsize=10)
    fig.savefig(ruta)
    plt.close(fig)


def figura_cuadros_ad(ej, fuente, ruta, ns=(0, 2, 4, 6, 8), notas=True):
    Anf, delta = ej["Anf"].reshape(-1, L, L), ej["delta"].reshape(-1, L, L)
    y = Anf + delta
    a = np.max(np.abs(delta[list(ns)]))
    filas = [(r"$A^n f$", Anf, dict(cmap="gray", vmin=0, vmax=1)), (r"$\delta_n$", delta, dict(cmap="RdBu_r", vmin=-a, vmax=a)),
             (r"$y_n$", y, dict(cmap="gray", vmin=0, vmax=1))]
    fig, axs = plt.subplots(3, len(ns), figsize=(2.3 * len(ns) + 0.8, 7.4), constrained_layout=True)
    for i, (etq, serie, kw) in enumerate(filas):
        for j, n in enumerate(ns):
            im = mostrar(axs[i, j], serie[n], **kw)
            if i == 0:
                axs[i, j].set_title(f"$n = {n}$")
            if i == 1:
                axs[i, j].plot(*fuente, "+", color="k", ms=9, mew=1.4)
        axs[i, 0].set_ylabel(etq, fontsize=13)
        barra(fig, im, axs[i, :].tolist())
    s, nu, ell, razon, M = EJEMPLO
    corte = "\n" if len(ns) <= 3 else " "
    titulo = (f"$s = {num_m(s)}$, $\\nu = {num_m(nu)}$, $\\ell = {num_m(ell)}$, razón de forzado {num_t(razon)},{corte}$M = {M}$, "
              f"fuente ${fuente}$ (+)")
    if notas:
        titulo += f";{corte}misma realización que reconstruccion_AD.pdf\nfilas 1 y 3: valores fuera de [0, 1] saturados"
    fig.suptitle(titulo, fontsize=10)
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


def tabla_ad(filas, ref, denso, ruta):
    lineas = [f"$1$ & $0$ & {fmt(r['ell'], 0)} & {fmt(r['razon'], 2)} & {r['M']} & {fmt(r['vis'], 3)} & "
              f"{r['extra']} & {fmt(r['med']['sin_corregir'], 4)} & {fmt(r['med']['gp'], 4)} & "
              f"{fmt(r['med']['oraculo'], 4)} & {fmt(denso, 4)} & {fmt(r['z'], 2)} \\\\" for r in ref]
    lineas.append("\\midrule")
    for s_, nu_ in OPERADORES:
        for r in [r for r in filas if r["s"] == s_ and r["nu"] == nu_]:
            lineas.append(f"{fmt(r['s'], 2)} & {fmt(r['nu'], 2)} & {fmt(r['ell'], 0)} & {fmt(r['razon'], 2)} & "
                          f"{r['M']}{'$^*$' if r['igual_denso'] else ''} & {fmt(r['vis'], 3)} & {r['extra']} & "
                          f"{fmt(r['med']['sin_corregir'], 4)} & {fmt(r['med']['gp'], 4)} & "
                          f"{fmt(r['med']['oraculo'], 4)} & {fmt(denso, 4)} & {fmt(r['z'], 2)} \\\\")
        lineas.append("\\midrule")
    lineas.pop()
    with open(ruta, "w") as fh:
        fh.write(f"% generado por transporte/figuras/adveccion_difusion.py; medianas de ||f_hat - f|| / ||f|| sobre {cfg.REPS_2D} "
                 "realizaciones; primer bloque: experimento_2D_64.py (s = 1, nu = 0; en M = 3 sus bancos cubren todos "
                 "los píxeles); $^*$: mismo número de sensores que el denso; requiere booktabs\n")
        fh.write("\\begin{tabular}{cccccccccccc}\n\\toprule\n")
        fh.write("$s$ & $\\nu$ & $\\ell$ & razón & $M$ & $1 - \\kappa_M^2/\\operatorname{tr}\\Sigma_w$ & sensores extra & "
                 "sin corregir & GP & oráculo & denso & $|z|$ \\\\\n\\midrule\n")
        fh.write("\n".join(lineas) + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    d = np.load(os.path.join(OUT, "resultados_diagnostico.npz"))
    filas_d = []
    for r in res_io.tabla(d, "diagnostico"):
        r["N_min"] = None if r["N_min"] < 0 else r["N_min"]
        r["hay"] = r["hay"].astype(bool)
        filas_d.append(r)
    nus_fino = d["nus_fino"]
    a_fino = {float(s): a for s, a in zip(d["SS"], d["a_fino"])}
    o_fino = {float(s): o for s, o in zip(d["SS"], d["o_fino"])}
    denso_d = float(d["denso"])
    archivos = [os.path.join(OUT, n) for n in ("alpha_vs_nu.pdf", "tabla_diagnostico_AD.tex", "alpha_vs_s.pdf",
                                               "tabla_barrido_s_AD.tex")]
    figura(filas_d, nus_fino, a_fino, o_fino, denso_d, archivos[0])
    tabla(filas_d, archivos[1])
    filas_s = res_io.tabla(d, "barrido_s")
    figura_s(filas_s, denso_d, float(d["s_propuesto"]), archivos[2])
    tabla_barrido_s(filas_s, archivos[3])

    ruta = os.path.join(OUT, "resultados.npz")
    if os.path.exists(ruta):
        z = np.load(ruta)
        filas, denso = res_io.tabla(z, "filas"), float(z["denso"])
        ejemplo = {k: z[f"ejemplo_{k}"] for k in ("sin_corregir", "gp", "oraculo", "delta", "Anf")}
        ref_npz = os.path.join(salida("transporte", "experimento_2D_64"), "resultados.npz")
        ref = [r for r in res_io.tabla(np.load(ref_npz), "filas")] if os.path.exists(ref_npz) else []
        nuevos = [os.path.join(OUT, n) for n in ("error_vs_M_AD.pdf", "reconstruccion_AD.pdf", "cuadros_AD.pdf",
                                                 "tabla_AD.tex")]
        figura_error_ad(filas, ref, denso, nuevos[0])
        figura_reconstruccion_ad(z["f"], ejemplo, tuple(int(c) for c in z["fuente"]), nuevos[1])
        figura_cuadros_ad(ejemplo, tuple(int(c) for c in z["fuente"]), nuevos[2])
        tabla_ad(filas, ref, denso, nuevos[3])
        nuevos.append(os.path.join(OUT, "cuadros_AD_n0_n8.pdf"))
        figura_cuadros_ad(ejemplo, tuple(int(c) for c in z["fuente"]), nuevos[-1], ns=(0, 8), notas=False)
        archivos += nuevos
    print("archivos: " + ", ".join(os.path.relpath(a, salida()) for a in archivos))


if __name__ == "__main__":
    main()
