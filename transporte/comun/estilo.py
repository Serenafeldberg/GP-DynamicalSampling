"""
Estilo común de las figuras: tipografía, paleta (azul sin corregir, naranja GP, verde oráculo, gris denso)
y formato con coma decimal (en mathtext: 0{,}75).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, LogFormatterMathtext

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

ESTIMADORES = ("sin_corregir", "gp", "oraculo")
NOMBRES = {"sin_corregir": "sin corregir", "gp": "GP", "oraculo": "oráculo"}
COLORES = dict(zip(ESTIMADORES, plt.get_cmap("tab10").colors))
COLOR_DENSO = "0.45"
MARCAS = {"sin_corregir": "o", "gp": "s", "oraculo": "^"}
COMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


def fmt(v, dec=3):
    """Número con coma decimal para LaTeX: $0{,}123$."""
    return "$" + f"{v:.{dec}f}".replace(".", "{,}") + "$"


def coma_txt(t):
    """Coma decimal dentro de mathtext."""
    return t.replace(".", "{,}")


def ejes_log(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(numticks=8))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())


def mostrar(ax, u, **kw):
    """imshow con ejes en coma decimal."""
    im = ax.imshow(u, origin="upper", **kw)
    ax.xaxis.set_major_formatter(COMA)
    ax.yaxis.set_major_formatter(COMA)
    ax.tick_params(labelsize=7)
    return im


def barra(fig, im, ax):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_major_formatter(COMA)
    cb.ax.tick_params(labelsize=7)
