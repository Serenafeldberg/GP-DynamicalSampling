"""
Tabla de la comparación por presupuesto en advección-difusión (Sección 4.7.5), a partir de
salida/transporte/adveccion_difusion/presupuesto/resultados.npz:

    salida/transporte/adveccion_difusion/presupuesto/tabla_presupuesto_AD.tex
"""
import os

import numpy as np

from transporte.comun import resultados as res_io
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


def main():
    z = np.load(os.path.join(OUT, "resultados.npz"))
    ruta = os.path.join(OUT, "tabla_presupuesto_AD.tex")
    tabla(res_io.tabla(z, "filas"), float(z["denso"]), res_io.tabla(z, "descartadas"), ruta)
    print("archivo: " + os.path.relpath(ruta, salida()))


if __name__ == "__main__":
    main()
