"""
Resumen de errores de advección-difusión contra la profundidad M para la configuración elegida
(s = 0,75, nu = 0,1, ell = 2, razón 0,2, N = 8): sin corregir, naive dinámico, GP, oráculo y denso.

Lee salida/transporte/adveccion_difusion/resultados.npz (sin corregir, GP, oráculo, denso) y
salida/transporte/adveccion_difusion/comparacion_justa/resultados.npz (naive dinámico, M = 1, 2, 3).
Genera salida/transporte/adveccion_difusion/resumen_errores_AD.pdf.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator, LogFormatterMathtext

from transporte.comun import resultados as res_io
from transporte.comun.rutas import salida

CARPETA = salida("transporte", "adveccion_difusion")
S, NU, ELL, RAZON = 0.75, 0.1, 2.0, 0.2
MS = (1, 2, 3)
C = plt.get_cmap("tab10").colors
ESTILO = {"sc": ("sin corregir (1024 sensores)", C[0], "o", "-"),
          "dinamico": ("naive dinámico ($1024\\,(M+1)$ sensores)", C[4], "D", "-"),
          "gp": ("GP ($1024\\,(M+1)$ sensores)", C[1], "s", "-"),
          "oraculo": ("oráculo (1024 sensores + $\\delta$ verdadero)", C[2], "^", "-")}



def valores():
    z = np.load(os.path.join(CARPETA, "resultados.npz"))
    filas = {r["M"]: r for r in res_io.tabla(z, "filas")
             if (r["s"], r["nu"], r["ell"], r["razon"]) == (S, NU, ELL, RAZON)}
    c = np.load(os.path.join(CARPETA, "comparacion_justa", "resultados.npz"))
    v = {k: [] for k in ESTILO}
    for M in MS:
        for k in ("sin_corregir", "gp", "oraculo"):
            v["sc" if k == "sin_corregir" else k].append(filas[M]["med"][k])
        if M < 3:
            i = [j for j in range(len(c["nu"])) if np.isclose(c["nu"][j], NU) and c["ell"][j] == ELL
                 and np.isclose(c["razon"][j], RAZON) and c["M"][j] == M][0]
            v["dinamico"].append(float(c["dinamico"][i]))
    v["dinamico"].append(float(c["naive_M3_mediana"]))
    return v, float(z["denso"])


def coma(x):
    return f"{x:#.3g}".replace(".", ",")


def figura(v, denso, ruta):
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for k, (nombre, color, marca, ls) in ESTILO.items():
        ax.plot(MS, v[k], marker=marca, ls=ls, color=color, lw=2, ms=7, label=nombre)
        for M, y in zip(MS, v[k]):
            ax.annotate(coma(y), (M, y), textcoords="offset points", xytext=(9, 4 if k != "oraculo" else -11),
                        fontsize=8, color=color)
    ax.axhline(denso, color="0.45", ls="--", lw=1.6, label="denso (4096 sensores, $n = 0$)")
    ax.annotate(coma(denso), (MS[-1], denso), textcoords="offset points", xytext=(9, -11), fontsize=8, color="0.4")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(numticks=6))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_major_locator(FixedLocator(MS))
    ax.set_xticklabels([f"$M = {M}$\n{1024 * (M + 1)} sensores" for M in MS])
    ax.set_xlim(MS[0] - 0.3, MS[-1] + 0.55)
    ax.set_xlabel("profundidad")
    ax.set_ylabel(r"$\|\hat f - f\| / \|f\|$ (mediana)")
    ax.grid(axis="y", which="major", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62))
    fig.suptitle(f"$s = {S:g}$, $\\nu = {NU:g}$, $\\ell = {ELL:g}$, razón {RAZON:g}".replace(".", "{,}")
                 .replace("razón 0{,}2", "razón 0,2"), fontsize=10)
    fig.savefig(ruta)
    plt.close(fig)


def main():
    from transporte.comun import estilo  # noqa: F401  (tipografía común)
    v, denso = valores()
    ruta = os.path.join(CARPETA, "resumen_errores_AD.pdf")
    figura(v, denso, ruta)
    print(f"s = {S}, nu = {NU}, ell = {ELL:g}, razón = {RAZON}, N = 8; denso {denso:.3e}")
    for k, (nombre, *_) in ESTILO.items():
        print(f"   {nombre:50s} " + "  ".join(f"M {M}: {y:.3e}" for M, y in zip(MS, v[k])))
    print("archivo: " + os.path.relpath(ruta, salida()))


if __name__ == "__main__":
    main()
