"""
Cruce con el muestreo denso según el nivel de forzado (1D, comparación por presupuesto; Sección 4.7.5): cómputo.

Para cada razón objetivo ||delta_N|| / ||f|| (RAZON_FORZADO, fijado solo en memoria) corre la comparación por
presupuesto, elige la mejor (m, M) de cada presupuesto según el GP y la compara con el denso. Guarda
salida/transporte/cruce_forzado/resultados.npz; figura y tabla en transporte/figuras/cruce_forzado.py.
"""
import os

import numpy as np

from transporte.comun import config as cfg
from transporte.comun import resultados as res_io
from transporte.comun.rutas import salida
from transporte.comun.uno_d import presupuesto_1D

OUT = salida("transporte", "cruce_forzado")
RAZONES = (0.02, 0.05, 0.10, 0.20)


def mejores(filas):
    """Por (ell, presupuesto): la configuración con M >= 1 de menor mediana GP."""
    out = {}
    for r in filas:
        if "gp" in r["med"]:
            k = (r["ell_nombre"], r["P"])
            if k not in out or r["med"]["gp"] < out[k]["med"]["gp"]:
                out[k] = r
    return out


def cruce(razones, errores, nivel):
    """Razón (interpolación log-log) donde el error GP pasa a superar al denso; None si no se cruza."""
    lr, le = np.log(razones), np.log(errores) - np.log(nivel)
    if le[0] > 0:
        return "antes"
    for i in range(len(le) - 1):
        if le[i] <= 0 < le[i + 1]:
            return float(np.exp(lr[i] - le[i] * (lr[i + 1] - lr[i]) / (le[i + 1] - le[i])))
    return None


def main():
    chequeos, res, denso = [], {}, None
    for razon in RAZONES:
        cfg.RAZON_FORZADO = razon                              # solo en esta corrida
        filas, denso = presupuesto_1D(chequeos)
        res[razon] = mejores(filas)
    cfg.RAZON_FORZADO = None

    ells, pres = list(cfg.ELL_BARRIDO), list(cfg.PRESUPUESTOS_1D)
    cruces = {(ell, P): cruce(np.array(RAZONES), np.array([res[r][(ell, P)]["med"]["gp"] for r in RAZONES]),
                              denso["med"]) for ell in ells for P in pres}
    filas = [dict(res[r][(ell, P)], razon=r) for r in RAZONES for ell in ells for P in pres]
    cr = [dict(ell_nombre=ell, P=P, cruce=(-1.0 if c == "antes" else np.nan if c is None else c))
          for (ell, P), c in cruces.items()]
    ruta = os.path.join(OUT, "resultados.npz")
    res_io.guardar(ruta, tablas=dict(mejores=filas, cruces=cr), razones=np.array(RAZONES), denso_med=denso["med"],
                   chequeos=np.array(chequeos))

    print("=" * 92)
    print(f"asserts OK: MC ({cfg.REPS_1D} realizaciones) == identidad exacta en {len(chequeos)} casos, "
          f"|z| máx = {np.max(np.abs(chequeos)):.2f} (< 4)")
    print(f"denso ({cfg.L_1D} sensores, n = 0): mediana {denso['med']:.2e} (no depende del forzado)")
    print(f"{'ell':>4s} {'pres.':>5s}  " + "  ".join(f"{'razón ' + f'{r:g}':>22s}" for r in RAZONES)
          + "   cruce (denso gana desde)")
    for ell in ells:
        for P in pres:
            celdas = "  ".join(f"({res[r][(ell, P)]['m']},{res[r][(ell, P)]['M']}) GP {res[r][(ell, P)]['med']['gp']:.2e}"
                               .rjust(22) for r in RAZONES)
            c = cruces[(ell, P)]
            txt = ("denso ya gana en " + f"{RAZONES[0]:g}") if c == "antes" else \
                  ("no se cruza en el rango" if c is None else f"{c:.3f}")
            print(f"{cfg.ELL_BARRIDO[ell]:4g} {P:5d}  {celdas}   {txt}")
    print("archivo: " + os.path.relpath(ruta, salida()))


if __name__ == "__main__":
    main()
