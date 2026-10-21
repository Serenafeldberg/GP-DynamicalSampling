"""
Experimentos numéricos del Capítulo 4 (Secciones 4.7.1 a 4.7.3): genera tabla_orbita.tex (Tabla 4.1),
tabla_estimador.tex (Tabla 4.2) y reconstruccion_vs_M.pdf (Figura 4.1) en salida/cap4/, e imprime los números.

MODELO
    y_n = A y_{n-1} + w_n ,  y_0 = f ,  n = 1..N
    muestras  u_n^(m)[i] = < y_n , (A^T)^m g_i > + ruido ,  m = 0..M
ESQUEMA
    1. residuos  u_n^(m) - u_{n-1}^(m+1) = g^T A^m w_n + ruido   (no dependen de f)
    2. estimar w a partir de los residuos
    3. acumular delta_n = sum_k A^(n-k) w_k y restarlo de las muestras m = 0
    4. invertir el frame (pseudoinversa) para obtener f_hat
"""
import os
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

# ============================================================== parametros comunes
d, p = 12, 2            # dimension del estado, cantidad de sensores
SEED_SISTEMA = 0        # el MISMO sistema (A, G) para los tres experimentos
SIGMA_W = 0.1           # escala del forzado
RHO = 0.9               # correlacion temporal AR(1)
RANGO = 3               # rango del forzado (dentro de S_1)
DECAY = 0.75            # decaimiento del espectro (escenarios de rango completo del barrido)
ELL_SE = 4.0            # escala del proceso SE en 4.7.3

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "salida", "cap4")
os.makedirs(OUT, exist_ok=True)


# ============================================================== sistema y forzado
def make_system(seed=SEED_SISTEMA):
    """A = rotaciones por bloques (angulos distintos) conjugadas por una ortogonal; G = p sensores ortonormales."""
    rng = np.random.default_rng(seed)
    thetas = np.sort(rng.uniform(0.15, 2.9, d // 2))
    R = np.zeros((d, d))
    for k, t in enumerate(thetas):
        R[2*k:2*k+2, 2*k:2*k+2] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
    Qo, _ = np.linalg.qr(rng.standard_normal((d, d)))
    A = Qo @ R @ Qo.T
    G, _ = np.linalg.qr(rng.standard_normal((d, p)))
    return A, G


def krylov_basis(A, G):
    """Base ortonormal ordenada por profundidad: las primeras p*M columnas generan S_{M-1}.
    Vale también en C^d: (A^m)^* g_i con la traspuesta conjugada."""
    d_, p_ = A.shape[0], G.shape[1]
    cols, Ak = [], np.eye(d_)
    for _ in range(d_ // p_):
        cols.append(Ak.conj().T @ G)
        Ak = A @ Ak
    B, _ = np.linalg.qr(np.hstack(cols))
    return B


def make_sigma(kind, A, G):
    """Covarianza espacial Sigma_w del forzado."""
    B = krylov_basis(A, G)
    d_ = A.shape[0]
    if kind == "rango_bajo":          # rango RANGO dentro de S_1 (como en 4.7.1)
        lam = np.zeros(d_); lam[:RANGO] = SIGMA_W**2
    elif kind == "alineado":          # rango completo, varianza decreciente segun la profundidad
        lam = SIGMA_W**2 * DECAY ** np.arange(d_)
    elif kind == "generico":          # mismo espectro, direcciones aleatorias
        B, _ = np.linalg.qr(np.random.default_rng(1).standard_normal((d_, d_)))
        lam = SIGMA_W**2 * DECAY ** np.arange(d_)
    S = B @ np.diag(lam) @ B.conj().T
    return (S + S.conj().T) / 2, B


def kernel_temporal(tipo, N, param=None):
    t = np.arange(1, N + 1)
    D = np.abs(t[:, None] - t[None, :])
    if tipo == "ar1":
        return param ** D
    if tipo == "se":
        return np.exp(-D**2 / (2 * param**2))
    if tipo == "blanco":
        return np.eye(N)
    raise ValueError(tipo)


def kappa(Sigma, B, M):
    """Masa del forzado fuera de S_{M-1} (invisible a profundidad M)."""
    k = min(p * M, d)
    Q = np.eye(d) - B[:, :k] @ B[:, :k].T
    return np.sqrt(max(np.trace(Q @ Sigma @ Q), 0.0))


# ============================================================== esquema para (N, M) dados
def adj(X):
    """Traspuesta conjugada (en el caso real, la traspuesta)."""
    return X.conj().T


class Esquema:
    """Vale en R^d y en C^d: los productos internos usan la traspuesta conjugada, y si A, G o Sigma son complejos
    el forzado, el ruido y f se simulan gaussianos complejos circulares (E|x|^2 = sigma^2 por componente)."""

    def __init__(self, A, G, Sigma, M, N, sigma_n, Kt):
        """Kt: kernel temporal VERDADERO del forzado (N x N)."""
        self.A, self.G, self.Sigma, self.M, self.N, self.sn, self.Kt = A, G, Sigma, M, N, sigma_n, Kt
        self.d, self.p = d, p = A.shape[0], G.shape[1]
        self.complejo = any(np.iscomplexobj(X) for X in (A, G, Sigma))
        pw = [np.eye(d)]
        for _ in range(max(N, M) + 1):
            pw.append(A @ pw[-1])
        self.pw = pw
        self.C = np.vstack([adj(G) @ pw[m] for m in range(M)])       # residuos <- w_n
        self.Cpinv = np.linalg.pinv(self.C)
        self.Psi = np.kron(np.eye(N), self.C)
        self.Kw = np.kron(Kt, Sigma)                                  # covarianza verdadera de w
        # ruido: eps indexado (n=0..N, m=0..M, i); eta = D eps ;  Su selecciona m = 0
        ne, nr = (N + 1) * (M + 1) * p, N * M * p
        D, Su = np.zeros((nr, ne)), np.zeros(((N + 1) * p, ne))
        ie = lambda n, m, i: (n * (M + 1) + m) * p + i
        for n in range(1, N + 1):
            for m in range(M):
                for i in range(p):
                    r = ((n - 1) * M + m) * p + i
                    D[r, ie(n, m, i)] += 1.0
                    D[r, ie(n - 1, m + 1, i)] -= 1.0
        for n in range(N + 1):
            for i in range(p):
                Su[n * p + i, ie(n, 0, i)] = 1.0
        self.D, self.Su = D, Su
        self.Seta = sigma_n**2 * D @ D.T
        L = np.zeros((N * d, N * d), dtype=A.dtype)                    # complejo si A lo es
        for n in range(N):
            for k in range(n + 1):
                L[n*d:(n+1)*d, k*d:(k+1)*d] = pw[n - k]
        self.L = L
        self.O = np.vstack([adj(G) @ pw[n] for n in range(N + 1)])
        self.alpha = np.linalg.svd(self.O, compute_uv=False)[-1] ** 2
        self.es_frame = np.linalg.matrix_rank(self.O) == d
        self.Opinv = np.linalg.pinv(self.O)
        iota = np.vstack([np.zeros((p, N * p)), np.eye(N * p)])
        self.T = self.Opinv @ iota @ np.kron(np.eye(N), adj(G)) @ L

    # ---- estimador lineal w_hat = H z (media posterior con un prior dado) ----
    def H(self, Kt_mod, Sigma_mod):
        Km = np.kron(Kt_mod, Sigma_mod)
        Kz = self.Psi @ Km @ adj(self.Psi) + self.Seta
        return adj(np.linalg.solve(Kz, self.Psi @ Km))

    # ---- identidad exacta (vale si G_N es frame) ----
    def mse_exacto(self, H):
        Aw = -self.T @ (np.eye(self.N * self.d) - H @ self.Psi)
        Ae = self.T @ H @ self.D - self.Opinv @ self.Su
        return np.real(np.trace(Aw @ self.Kw @ adj(Aw))) + self.sn**2 * np.sum(np.abs(Ae)**2)

    def mse_sin_corregir(self):
        return np.real(np.trace(self.T @ self.Kw @ adj(self.T))) + self.sn**2 * np.sum(np.abs(self.Opinv @ self.Su) ** 2)

    def mse_solo_ruido(self):
        return self.sn**2 * np.sum(np.abs(self.Opinv @ self.Su) ** 2)

    # ---- simulacion de una realizacion ----
    def preparar_simulacion(self):
        self._Lt = np.linalg.cholesky(self.Kt + 1e-12 * np.eye(self.N))
        ev, V = np.linalg.eigh(self.Sigma)
        self._Sh = V @ np.diag(np.sqrt(np.clip(ev, 0, None))) @ adj(V)
        self._Gm = [adj(self.G) @ self.pw[m] for m in range(self.M + 1)]

    def _normal(self, rng, forma):
        """Normal estándar real o compleja circular ((a + i b) / sqrt 2: E|x|^2 = 1)."""
        if not self.complejo:
            return rng.standard_normal(forma)
        return (rng.standard_normal(forma) + 1j * rng.standard_normal(forma)) / np.sqrt(2)

    def simular(self, rng):
        N, M, d, p = self.N, self.M, self.d, self.p
        f = self._normal(rng, d); f /= np.linalg.norm(f)          # uniforme en la esfera unitaria (real o compleja)
        W = self._Lt @ self._normal(rng, (N, d)) @ self._Sh.T     # filas w_n^T con w_n = Sh x_n, E[w w^*] = Sigma
        y = [f]
        for n in range(1, N + 1):
            y.append(self.A @ y[-1] + W[n - 1])
        eps = self.sn * self._normal(rng, (N + 1, M + 1, p))
        u = np.array([[self._Gm[m] @ y[n] for m in range(M + 1)] for n in range(N + 1)]) + eps
        z = np.array([[u[n, m] - u[n - 1, m + 1] for m in range(M)] for n in range(1, N + 1)]).ravel()
        return f, u, z

    def reconstruir(self, u, w_hat):
        """w_hat: vector (N*d,). Devuelve f_hat."""
        d = self.d
        delta = self.L @ w_hat
        corr = np.vstack([np.zeros(self.p)] + [adj(self.G) @ delta[n*d:(n+1)*d] for n in range(self.N)])
        return self.Opinv @ (u[:, 0, :] - corr).ravel()

    def sin_corregir(self, u):
        return self.Opinv @ u[:, 0, :].ravel()


def rms(errores_cuadrados):
    return float(np.sqrt(np.mean(errores_cuadrados)))


def med(errores_cuadrados):
    """Mediana del error ||f - f_hat|| sobre realizaciones."""
    return float(np.median(np.sqrt(errores_cuadrados)))


def fmt(v, dec=3):
    """Numero con coma decimal para LaTeX."""
    return "$" + f"{v:.{dec}f}".replace(".", "{,}") + "$"


# ============================================================== 4.7.1  la profundidad no reemplaza a la orbita
def experimento_orbita(Ns=(3, 5, 8, 11, 15), M=2, sigma_n=1e-3, reps=2000, seed=11):
    A, G = make_system()
    Sigma, _ = make_sigma("rango_bajo", A, G)
    # instantanea: un solo instante con profundidad M analiza contra M+1 niveles
    rango_inst = np.linalg.matrix_rank(np.vstack([G.T @ np.linalg.matrix_power(A, m) for m in range(M + 1)]))
    filas = []
    for N in Ns:
        esq = Esquema(A, G, Sigma, M, N, sigma_n, kernel_temporal("ar1", N, RHO))
        H = esq.H(esq.Kt, Sigma)                          # media posterior con hiperparametros verdaderos
        esq.preparar_simulacion()
        rng = np.random.default_rng(seed)
        e_sc, e_c = [], []
        for _ in range(reps):
            f, u, z = esq.simular(rng)
            e_sc.append(np.sum((f - esq.sin_corregir(u)) ** 2))
            e_c.append(np.sum((f - esq.reconstruir(u, H @ z)) ** 2))
        fila = dict(N=N, frame=esq.es_frame, alpha=esq.alpha, sc=med(e_sc), c=med(e_c),
                    ident=np.sqrt(esq.mse_exacto(H)) if esq.es_frame else np.nan)
        filas.append(fila)
    return filas, rango_inst


# ============================================================== 4.7.3  eleccion del estimador
def ajustar_hiper(esq, z, Sigma_forma):
    """Maxima verosimilitud marginal de (rho, sigma_w) a partir de los residuos; sigma_n conocido."""
    CSC = esq.C @ Sigma_forma @ esq.C.T
    t = np.arange(1, esq.N + 1)
    Dt = np.abs(t[:, None] - t[None, :])

    def nll(x):
        rho, ls = x
        Kz = np.kron(rho ** Dt, np.exp(2 * ls) * CSC) + esq.Seta
        try:
            Lc = np.linalg.cholesky(Kz)
        except np.linalg.LinAlgError:
            return 1e10
        a = solve_triangular(Lc, z, lower=True)
        return 0.5 * a @ a + np.log(np.diag(Lc)).sum()

    mejor = None
    for x0 in ((0.5, np.log(0.05)), (0.9, np.log(0.2))):
        r = minimize(nll, x0, method="L-BFGS-B", bounds=[(0.0, 0.995), (np.log(1e-3), np.log(1.0))])
        if mejor is None or r.fun < mejor.fun:
            mejor = r
    return mejor.x[0], np.exp(mejor.x[1])


def experimento_estimador(N=12, reps=300, seed=21):
    A, G = make_system()
    Sigma, _ = make_sigma("rango_bajo", A, G)
    Sigma_forma = Sigma / SIGMA_W**2
    filas_def = [("AR(1), $\\rho=0{,}9$", "ar1", RHO, 6, 1e-2),
                 ("AR(1), $\\rho=0{,}9$", "ar1", RHO, 8, 1e-2),
                 ("AR(1), $\\rho=0{,}9$", "ar1", RHO, 8, 1e-3),
                 ("suave (SE)",           "se",  ELL_SE, 6, 1e-2),
                 ("blanco",               "blanco", None, 6, 1e-2),
                 ("blanco",               "blanco", None, 8, 1e-3)]
    filas = []
    for (nombre, tipo, par, M, sn) in filas_def:
        Kt = kernel_temporal(tipo, N, par)
        esq = Esquema(A, G, Sigma, M, N, sn, Kt)
        esq.preparar_simulacion()
        H_oraculo = esq.H(Kt, Sigma)                                  # kernel verdadero
        H_fijo = esq.H(kernel_temporal("ar1", N, RHO), Sigma)          # AR(1) rho = 0,9 fijo
        rng = np.random.default_rng(seed)
        err = {k: [] for k in ("sc", "paso", "movil", "oraculo", "fijo", "ajustado")}
        rhos = []
        for _ in range(reps):
            f, u, z = esq.simular(rng)
            err["sc"].append(np.sum((f - esq.sin_corregir(u)) ** 2))
            Z = z.reshape(N, M * p)
            W_paso = Z @ esq.Cpinv.T                                   # inversion por paso
            W_movil = np.array([W_paso[max(0, n - 1):n + 2].mean(axis=0) for n in range(N)])
            err["paso"].append(np.sum((f - esq.reconstruir(u, W_paso.ravel())) ** 2))
            err["movil"].append(np.sum((f - esq.reconstruir(u, W_movil.ravel())) ** 2))
            err["oraculo"].append(np.sum((f - esq.reconstruir(u, H_oraculo @ z)) ** 2))
            err["fijo"].append(np.sum((f - esq.reconstruir(u, H_fijo @ z)) ** 2))
            rho_h, sw_h = ajustar_hiper(esq, z, Sigma_forma)
            H_aj = esq.H(kernel_temporal("ar1", N, rho_h), sw_h**2 * Sigma_forma)
            err["ajustado"].append(np.sum((f - esq.reconstruir(u, H_aj @ z)) ** 2))
            rhos.append(rho_h)
        fila = {k: med(v) for k, v in err.items()}
        fila.update(nombre=nombre, M=M, sn=sn, rho_hat=float(np.median(rhos)),
                    ident_oraculo=np.sqrt(esq.mse_exacto(H_oraculo)))
        filas.append(fila)
    return filas


# ============================================================== 4.7.2  barrido en M (figura)
def barrido(kind, Ms, N=12, sigma_n=1e-2, sys_seed=SEED_SISTEMA, monte_carlo=True, reps=500, seed=123):
    A, G = make_system(sys_seed)
    Sigma, B = make_sigma(kind, A, G)
    Kt = kernel_temporal("ar1", N, RHO)
    out = []
    for M in Ms:
        esq = Esquema(A, G, Sigma, M, N, sigma_n, Kt)
        H = esq.H(Kt, Sigma)
        r = dict(M=M, kappa=kappa(Sigma, B, M), exacto=np.sqrt(esq.mse_exacto(H)),
                 sin_corregir=np.sqrt(esq.mse_sin_corregir()), solo_ruido=np.sqrt(esq.mse_solo_ruido()),
                 mc=np.nan)
        if monte_carlo:
            esq.preparar_simulacion()
            rng = np.random.default_rng(seed)
            e = []
            for _ in range(reps):
                f, u, z = esq.simular(rng)
                e.append(np.sum((f - esq.reconstruir(u, H @ z)) ** 2))
            r["mc"] = rms(e)
        out.append(r)
    return out


def figura_barrido(Ms=tuple(range(1, 9)), n_sistemas=20):
    kinds = ["rango_bajo", "alineado", "generico"]
    titulos = {"rango_bajo": f"Forzado de rango {RANGO}\n(visible desde $M=2$)",
               "alineado": "Rango completo, varianza decreciente\nalineada con la profundidad",
               "generico": "Rango completo, mismo espectro,\ndirecciones aleatorias"}
    res = {k: barrido(k, Ms) for k in kinds}
    rob = {k: np.array([[r["exacto"] for r in barrido(k, Ms, sys_seed=100 + s, monte_carlo=False)]
                        for s in range(n_sistemas)]) for k in kinds}
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    Mx = np.array(Ms)
    for j, k in enumerate(kinds):
        R, a = res[k], ax[j]
        q1, q3 = np.percentile(rob[k], [25, 75], axis=0)
        a.fill_between(Mx, q1, q3, color="C0", alpha=.18, label=f"identidad, {n_sistemas} sistemas (RIC)")
        a.plot(Mx, [r["exacto"] for r in R], "-", color="C0", lw=2, label="identidad exacta, sistema de referencia")
        a.plot(Mx, [r["mc"] for r in R], "o", color="C3", ms=5, label="Monte Carlo (500)")
        a.plot(Mx, [r["sin_corregir"] for r in R], "--", color="gray", label="sin corregir")
        a.axhline(R[0]["solo_ruido"], color="k", ls=":", lw=1, label="ref.: solo ruido de muestreo")
        mv = [r["M"] for r in R if r["kappa"] < 1e-6]
        if mv:
            a.axvline(mv[0], color="C2", ls="-.", lw=1)
            izq = mv[0] > 3
            a.text(mv[0] - 0.1 if izq else mv[0] + 0.1, 0.03, rf"visible desde $M={mv[0]}$", color="C2",
                   fontsize=8, ha="right" if izq else "left", transform=a.get_xaxis_transform())
        a.set_yscale("log"); a.set_title(titulos[k], fontsize=10); a.set_xlabel("profundidad $M$")
        a.set_yticks([0.04, 0.06, 0.1, 0.2, 0.4, 0.6])
        a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        a.yaxis.set_minor_formatter(NullFormatter())
        if j == 0:
            a.set_ylabel("Error de reconstrucción")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(r"Error de reconstrucción en función de la profundidad $M$  "
                 r"($d=12$, $p=2$, $N=12$, $\rho=0{,}9$, $\sigma_n=10^{-2}$)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, "reconstruccion_vs_M.pdf"), bbox_inches="tight")
    plt.close(fig)
    return res


# ============================================================== main
if __name__ == "__main__":
    # ---------------- 4.7.1
    orb, rango_inst = experimento_orbita()
    print(f"\n=== 4.7.1  (M=2, sigma_n=1e-3)   rango de una instantanea: {rango_inst} < {d}")
    print(" N  frame  alpha_N     sin corregir  corregido  identidad  factor")
    lineas = []
    for r in orb:
        fac = r["sc"] / r["c"]
        print(f"{r['N']:2d}  {str(r['frame']):5s}  {r['alpha']:.2e}   {r['sc']:.4f}       {r['c']:.4f}     "
              f"{r['ident']:.4f}    {fac:.1f}")
        lineas.append(f"{r['N']:2d} & imposible & {fmt(r['sc'])} & {fmt(r['c'])}\\\\")
    with open(os.path.join(OUT, "tabla_orbita.tex"), "w") as fh:
        fh.write("% N & solo instantanea & orbita sin corregir & orbita corregida  (mediana)\n")
        fh.write("\n".join(lineas) + "\n")

    # ---------------- 4.7.3
    est = experimento_estimador()
    print("\n=== 4.7.3  (N=12, forzado de rango 3, medianas de ||f - f_hat|| sobre 300 realizaciones)")
    print(f"{'proceso':22s} M  sigma_n  sin_corr  paso    movil   oraculo  rho=0.9  ajustado  rho_hat  (ident. oraculo)")
    print("    oraculo: media posterior con el kernel temporal verdadero; rho=0.9: con AR(1) rho = 0,9 fijo")
    lineas = []
    for r in est:
        print(f"{r['nombre'][:22]:22s} {r['M']}  {r['sn']:.0e}   {r['sc']:.4f}  {r['paso']:.4f}  {r['movil']:.4f}  "
              f"{r['oraculo']:.4f}   {r['fijo']:.4f}   {r['ajustado']:.4f}    {r['rho_hat']:.2f}   ({r['ident_oraculo']:.4f})")
        sn_txt = "10^{-2}" if r["sn"] == 1e-2 else "10^{-3}"
        # columnas de la Tabla 4.2: la de "rho = 0,9 fijo" usa el modelo AR(1) con rho = 0,9 en todas las filas
        lineas.append(f"{r['nombre']} & $({r['M']};\\,{sn_txt})$ & {fmt(r['sc'])} & {fmt(r['paso'])} & {fmt(r['movil'])} & "
                      f"{fmt(r['fijo'])} & {fmt(r['ajustado'])} & {fmt(r['rho_hat'], 2)}\\\\")
    with open(os.path.join(OUT, "tabla_estimador.tex"), "w") as fh:
        fh.write("% proceso & (M;sigma_n) & sin corregir & por paso & prom. movil & rho = 0,9 fijo & ajustado & rho_hat"
                 "  (medianas)\n")
        fh.write("\n".join(lineas) + "\n")

    # ---------------- 4.7.2
    res = figura_barrido()
    print("\n=== 4.7.2  barrido en M (N=12, sigma_n=1e-2, sistema de referencia)")
    print(" M  " + "  ".join(f"{k:>22s}" for k in res))
    for i in range(len(res["rango_bajo"])):
        print(f"{i+1:2d}  " + "  ".join(f"exacto {res[k][i]['exacto']:.4f} / sc {res[k][i]['sin_corregir']:.4f}"
                                         for k in res))
    print("\nArchivos en:", OUT)
