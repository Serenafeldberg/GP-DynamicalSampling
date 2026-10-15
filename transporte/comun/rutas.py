"""Rutas de salida relativas a la raíz del repositorio: salida/<capítulo>/<experimento>/."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]


def salida(*partes):
    """Carpeta salida/<partes...> (la crea si no existe) como str."""
    ruta = RAIZ.joinpath("salida", *partes)
    ruta.mkdir(parents=True, exist_ok=True)
    return str(ruta)
