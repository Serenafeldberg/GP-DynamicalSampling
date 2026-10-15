"""
Diagnóstico de frame para los sensores del transporte (sin forzado, sin ruido).

Barre escenarios de grilla, velocidad s, núcleos y profundidad M, calcula
alpha_N = lambda_min(O^* O) por bloques de Fourier (L = 64; 66 si m = 3), lo
contrasta con el cálculo denso (L = 16; 18 si m = 3) y evalúa las hipótesis H1-H6.
Cómputo: guarda salida/transporte/diagnostico/resultados.npz; figuras y tablas en
transporte/figuras/diagnostico_frames.py.
"""
import os
from math import gcd

import numpy as np

from transporte.comun import operadores as op
from transporte.comun import sensores as sn
from transporte.comun import frames as fr
from transporte.comun import resultados as res_io
from transporte.comun.chequeos import chequear_bloques_vs_denso, chequear_medir
from transporte.comun.rutas import salida

OUT = salida("transporte", "diagnostico")
L_BLOQUES, L_DENSO, M_PROF = 64, 16, 4
DELTA, GAUSS = ("delta",), ("gaussiana", 1.0)


def L_para(m, base):
    """Menor múltiplo de m que sea >= base (la grilla tiene que dividir a L)."""
    return -(-base // m) * m


def coma(t):
    return t.replace(".", ",")


def s_txt(s):
    f = lambda c: f"1/{round(1 / c)}" if 0 < c < 1 else f"{c:g}"
    return f"({f(s[0])}, {f(s[1])})"


def esc(hip, m, grilla, s, nucleos, M=0, nota=""):
    return dict(hip=hip, m=m, mx=grilla[0], my=grilla[1], s=s, nuc=nucleos, M=M, nota=nota)


def escenarios():
    E = []
    for m in (2, 3, 4):
        lista_s = [(1, 0), (2, 0), (3, 0), (1, 1), (1, 1 / m)]
        cajas_ok = next(w for w in range(m + 1, 99) if gcd(w, L_para(m, L_BLOQUES)) == 1 and gcd(w, L_para(m, L_DENSO)) == 1)
        E += [esc("H1", m, (m, 1), s, [DELTA]) for s in lista_s]
        E += [esc("H2", m, (m, m), s, [nuc]) for s in lista_s[:4] for nuc in (DELTA, GAUSS, ("dgauss_x", 1.0))]
        E += [esc("H3", m, (m, m), (1, 1 / m), [nuc]) for nuc in (DELTA, GAUSS)]
        G7, DY7 = ("gaussiana", 0.7), ("dgauss_y", 0.7)
        sets = {2: [([GAUSS, ("dgauss_y", 1.0)], "ejemplo: par + impar"),
                    ([G7, ("gaussiana", 1.5)], "par + par"),
                    ([DELTA, G7, DY7], "par + par + impar"),
                    ([G7], "control m-1")],
                3: [([DELTA, G7, DY7], "par + par + impar"),
                    ([G7, ("gaussiana", 1.5), ("dgauss_y", 1.0)], "par + par + impar"),
                    ([DELTA, G7], "control m-1")],
                4: [([DELTA, G7, ("caja", 3), DY7], "3 pares + 1 impar"),
                    ([DELTA, G7, ("caja", 3), DY7, ("dgauss_y", 1.5)], "3 pares + 2 impares"),
                    ([DELTA, G7, ("caja", 3)], "control m-1")]}[m]
        E += [esc("H4", m, (m, m), (1, 0), nucs, nota=nota) for nucs, nota in sets]
        E += [esc("H5", m, (m, 1), (1, 0), [DELTA], M, "H1") for M in (0, 1, 2)]
        E += [esc("H5", m, (m, m), (1, 0), [DELTA], M, "H2") for M in (0, 2)]
        E += [esc("H5", m, (m, m), (1, 1 / m), [DELTA], M, "H3") for M in (0, 2)]
        E += [esc("H6", m, (m, 1), (1, 0), [("caja", w)], nota=n) for w, n in
              [(m, "w = m"), (cajas_ok, "w != m, mcd(w, L) = 1"), (2 * m, "w = 2m")]]
        if m == 3:
            E += [esc("H6", m, (m, m), (1, 1 / m), [("caja", w)], nota=n) for w, n in [(3, "w = m"), (5, "w != m")]]
    nuc_fam = [DELTA, ("gaussiana", 0.5), GAUSS, ("gaussiana", 2.0), ("caja", 2), ("caja", 3), ("caja", 4)]
    nuc_fam += [("bspline", k, 2) for k in range(4)]
    E += [esc("NUC", 2, (2, 1), (1, 0), [nuc]) for nuc in nuc_fam]
    return E


def evaluar(e):
    """alpha_N, lambda_max por bloques; N mínimo con frame; datos para la tabla."""
    L = L_para(e["m"], L_BLOQUES)
    N_max = 2 * e["m"] ** 2
    a, b = fr.alpha_bloques(L, e["mx"], e["my"], e["nuc"], e["s"], N_max, e["M"])
    hay = a > fr.TOL_FRAME * b
    r = a[-1] / b[-1]
    estado = "frame" if hay[-1] else ("~0 (mal cond.)" if r > 1e-14 else "0")
    e.update(estado=estado, L=L, N_max=N_max, alfa=a, beta=b, hay=hay,
             N_min=int(np.argmax(hay)) if hay.any() else None,
             p=(L // e["mx"]) * (L // e["my"]) * len(e["nuc"]),
             gratis=sn.profundidad_gratis(e["mx"], e["my"], e["s"], M_PROF))
    return e


def invisibles(L=16, cuantas=4):
    """Componentes invisibles (núcleo de O) en casos sin frame: imágenes, frecuencias y max |muestra|."""
    casos = [("H2: delta, $s = (1, 0)$", 2, (2, 2), (1, 0), [DELTA]),
             ("H2: gaussiana(1), $s = (1, 0)$", 2, (2, 2), (1, 0), [GAUSS]),
             ("H3: delta, $s = (1, 1/2)$", 2, (2, 2), (1, 0.5), [DELTA])]
    filas, resumen = [], []
    for titulo, m, (mx, my), s, nuc in casos:
        N, sub = 2 * m * m, not op.es_entero(s)
        dim, comps = fr.invisibles(L, mx, my, nuc, s, N, 0)
        comps.sort(key=lambda c: (abs(c[1][0]) + abs(c[1][1]), c[1]))
        vistas, elegidas = set(), []
        for c in comps:
            if c[1] not in vistas:
                vistas.add(c[1])
                elegidas.append(c)
        G = sn.banco(L, mx, my, nuc)
        visto = 0.0
        for img, frec, _ in elegidas[:cuantas]:
            y = np.stack([op.A_pot(img, s, n, sub) for n in range(N + 1)])
            visto = max(visto, np.max(np.abs(sn.medir(y, G, s, 0, sub))))
            filas.append(dict(titulo=titulo, img=img, ky=frec[0], kx=frec[1], mx=mx, my=my))
        resumen.append((titulo, dim, L * L, visto))
    return filas, resumen


def veredictos(E):
    """Imprime si cada hipótesis se confirma, con el número que lo respalda."""
    def nmin(e):
        return "nunca" if e["N_min"] is None else str(e["N_min"])

    def rel(e):
        return e["alfa"][-1] / e["beta"][-1]

    print("\n" + "=" * 78 + "\nHIPÓTESIS\n" + "=" * 78)
    h1 = [e for e in E if e["hip"] == "H1"]
    malos = [e for e in h1 if (gcd(int(e["s"][0]), e["m"]) == 1) != e["hay"].any()]
    print(f"H1 {'CONFIRMADA' if not malos else 'REFUTADA'}: {len(h1) - len(malos)}/{len(h1)} escenarios cumplen "
          f"'frame <=> mcd(s_x, m) = 1'.")
    for e in h1:
        print(f"     m = {e['m']}  s = {s_txt(e['s']):10s} mcd = {gcd(int(e['s'][0]), e['m'])}  N mín = {nmin(e):5s}"
              f"  alpha/lambda_max(N_max) = {max(rel(e), 0):.2e}")

    h2 = [e for e in E if e["hip"] == "H2"]
    con = [e for e in h2 if e["hay"].any()]
    print(f"H2 {'CONFIRMADA' if not con else 'REFUTADA'}: {len(h2) - len(con)}/{len(h2)} escenarios sin frame para "
          f"ningún N <= 2m^2 (3 núcleos x 4 velocidades enteras x 3 m); max alpha/lambda_max = "
          f"{max(rel(e) for e in h2):.1e}.")

    h3 = [e for e in E if e["hip"] == "H3"]
    con = sorted({e["m"] for e in h3 if e["hay"].any()})
    sin = sorted({e["m"] for e in h3 if not e["hay"].any()})
    print(f"H3 {'CONFIRMADA' if not sin else 'REFUTADA para m = ' + str(sin)}: s = (1, 1/m), grilla m x m; "
          f"hay frame para m = {con}.")
    for e in h3:
        print(f"     m = {e['m']}  L = {e['L']}  {sn.etiqueta(e['nuc'][0]):14s} N mín = {nmin(e):5s}"
              f"  alpha/lambda_max(N_max) = {max(rel(e), 0):.2e}")

    h4 = [e for e in E if e["hip"] == "H4"]
    fallan = [e for e in h4 if len(e["nuc"]) >= e["m"] and not e["hay"].any()]
    controles = all(not e["hay"].any() for e in h4 if len(e["nuc"]) < e["m"])
    print(f"H4 {'CONFIRMADA' if not fallan else 'REFUTADA'}: {len(fallan)} conjuntos con >= m núcleos sin frame; "
          f"con m-1 núcleos nunca hay frame: {'sí' if controles else 'no'}. (s = (1, 0), grilla m x m)")
    for e in h4:
        print(f"     m = {e['m']}  {e['nota']:12s} {' + '.join(sn.etiqueta(n) for n in e['nuc']):55s}"
              f" N mín = {nmin(e):5s} alpha/lambda_max = {max(rel(e), 0):.2e}  [{e['estado']}]")

    h5 = [e for e in E if e["hip"] == "H5"]
    print("H5 (profundidad M, sin forzado):")
    ok_exist = True
    for m in (2, 3, 4):
        for base in ("H1", "H2", "H3"):
            g = [e for e in h5 if e["m"] == m and e["nota"] == base]
            ok_exist &= len({e["hay"].any() for e in g}) == 1
            print(f"     m = {m}  base {base}: " + "   ".join(f"M = {e['M']}: N mín = {nmin(e)}" for e in g))
    print(f"   existencia de frame para algún N: {'no cambia (CONFIRMADA)' if ok_exist else 'CAMBIA (REFUTADA)'};"
          f" a N fijo sí cambia: con profundidad M el N mínimo baja en M (equivale a horizonte N + M).")

    h6 = [e for e in E if e["hip"] == "H6"]
    w_m = all(not e["hay"].any() for e in h6 if e["nota"] == "w = m")
    otros = [e for e in h6 if e["nota"] != "w = m" and not e["hay"].any()]
    print(f"H6 {'CONFIRMADA' if w_m else 'REFUTADA'}: caja w = m sin frame en todos los casos; además "
          f"{len(otros)} casos con w != m tampoco tienen frame (w = 2m, mcd(w, L) > 1).")
    for e in h6:
        print(f"     m = {e['m']}  grilla {e['mx']}x{e['my']}  s = {s_txt(e['s']):10s} {sn.etiqueta(e['nuc'][0]):8s}"
              f" ({e['nota']:22s}) L = {e['L']}  N mín = {nmin(e):5s} alpha/lambda_max = {max(rel(e), 0):.2e}")


def main():
    E = [evaluar(e) for e in escenarios()]

    # chequeos: denso == bloques en todos los escenarios (L chico), y medir() == O f
    errores = [chequear_bloques_vs_denso(L_para(e["m"], L_DENSO), e["mx"], e["my"], e["nuc"], e["s"],
                                         e["N_max"], e["M"]) for e in E]
    err_medir = chequear_medir(16, 2, 2, [GAUSS, ("dgauss_y", 1.0)], (1, 0.5), 3, 2)

    filas_inv, inv = invisibles()
    largo = max(e["N_max"] for e in E) + 1
    rellenar = lambda v, x: np.concatenate([v, np.full(largo - len(v), x)])
    tabla_E = [dict({k: v for k, v in e.items() if k not in ("alfa", "beta", "hay", "s", "nuc", "N_min")},
                    s=np.array(e["s"], float), nuc=repr(e["nuc"]), N_min=-1 if e["N_min"] is None else e["N_min"],
                    alfa=rellenar(e["alfa"], np.nan), beta=rellenar(e["beta"], np.nan), hay=rellenar(e["hay"], False))
               for e in E]
    ruta = os.path.join(OUT, "resultados.npz")
    res_io.guardar(ruta, tablas=dict(escenarios=tabla_E, invisibles=filas_inv), errores=np.array(errores),
                   err_medir=err_medir)
    archivos = [ruta]

    print("=" * 78)
    print(f"{len(E)} escenarios; alpha por bloques con L = 64 (66 si m = 3); frame si alpha_N > "
          f"{fr.TOL_FRAME:g} lambda_max")
    print(f"chequeo denso == bloques (L = 16; 18 si m = 3): OK en {len(errores)} escenarios, "
          f"error relativo máx = {max(errores):.1e}")
    print(f"chequeo medir() == < A^(n+k) f, g_i >: OK, error = {err_medir:.1e}")
    print("componentes invisibles (L = 16, N = 8):")
    for titulo, dim, total, visto in inv:
        print(f"   {coma(titulo).replace('$', ''):32s} dim núcleo = {dim:3d} de {total}   "
              f"max |muestra| de una componente invisible = {visto:.1e}")
    print("profundidad gratis (fracción de k = 1..4 con (A^*)^k g sobre otro sensor):")
    for m in (2, 3, 4):
        for grilla in ((m, 1), (m, m)):
            print(f"   m = {m}  grilla {grilla[0]}x{grilla[1]}: " + "  ".join(
                f"s = {s_txt(s)}: {sn.profundidad_gratis(*grilla, s, M_PROF):.2f}"
                for s in [(1, 0), (2, 0), (3, 0), (1, 1), (1, 1 / m)]))
    veredictos(E)
    print("\narchivos:")
    for a in archivos:
        print("   " + os.path.relpath(a, salida()))


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
