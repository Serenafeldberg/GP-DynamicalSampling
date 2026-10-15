"""
Guardado y lectura de resultados.npz sin pickle.

Las tablas de resultados son listas de diccionarios con, a lo sumo, un nivel de anidamiento
(por ejemplo {"M": 2, "med": {"gp": 0,01, ...}}). Se guardan aplanadas como columnas "filas/<clave>" o
"filas/<clave>.<subclave>"; un valor ausente se guarda como NaN (o AUSENTE si es texto) y se omite al leer.
"""
import numpy as np

PREFIJO = "filas/"
AUSENTE = "<ausente>"


def _aplanar(fila):
    out = {}
    for k, v in fila.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out[f"{k}.{k2}"] = v2
        else:
            out[k] = v
    return out


def guardar(ruta, tablas=None, **arrays):
    """tablas: {nombre: lista de dicts}; arrays: arreglos o escalares sueltos."""
    datos = dict(arrays)
    for nombre, filas in (tablas or {}).items():
        planas = [_aplanar(f) for f in filas]
        claves = list(dict.fromkeys(k for f in planas for k in f))
        for k in claves:
            ejemplo = next(f[k] for f in planas if k in f)
            relleno = np.full(np.shape(ejemplo), np.nan) if not isinstance(ejemplo, str) else AUSENTE
            datos[f"{PREFIJO}{nombre}/{k}"] = np.array([f.get(k, relleno) for f in planas])
        datos[f"{PREFIJO}{nombre}/__n"] = len(planas)
    np.savez(ruta, **datos)


def _vacio(v):
    return isinstance(v, str) and v == AUSENTE or (np.ndim(v) == 0 and isinstance(v, float) and np.isnan(v)) or \
        (np.ndim(v) > 0 and np.issubdtype(np.asarray(v).dtype, np.floating) and np.all(np.isnan(v)))


def _python(v):
    v = v.item() if np.ndim(v) == 0 else v
    return v


def tabla(npz, nombre):
    """Reconstruye la lista de dicts guardada con guardar(..., tablas={nombre: ...})."""
    pre = f"{PREFIJO}{nombre}/"
    n = int(npz[pre + "__n"])
    claves = [k[len(pre):] for k in npz.files if k.startswith(pre) and not k.endswith("__n")]
    filas = []
    for i in range(n):
        fila = {}
        for k in claves:
            v = _python(npz[pre + k][i])
            if _vacio(v):
                continue
            if "." in k:
                a, b = k.split(".", 1)
                fila.setdefault(a, {})[b] = v
            else:
                fila[k] = v
        filas.append(fila)
    return filas
