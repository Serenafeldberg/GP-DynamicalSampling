"""
Operadores del transporte representados en Fourier: advección-difusión (bancos (A^*)^m G no puntuales), pseudoinversa
robusta, y O_N^* O_N por bloques de alias para sensores delta en columnas (OperadorFourier: columnas cada SEP;
OperadorColumnas: columnas x = c mod SEP), con O^+ u = (O^* O)^{-1} O^* u aplicado por FFT sin armar O_N.
"""
import numpy as np
from scipy.linalg import cho_solve

from . import config as cfg
from . import operadores as op
from . import forzado as fz
from . import frames as fr
from . import sensores as sn

DELTA = [("delta",)]


def bancos_ad(G, s, nu, E):
    """Filas (A^*)^e g_i, e = 0..E, aplanadas: (E+1, p, d). Ya no son puntuales."""
    return np.stack([op.A_ad_adj(G, (s, 0), nu, n=e) for e in range(E + 1)]).reshape(E + 1, G.shape[0], -1)


def pinv_gesvd(B, N):
    """O_N^+ como reconstruccion.pinv_O, pero con la SVD de LAPACK gesvd: con algunos O_N estructurados
    (s = 0,75, nu = 0) la rutina por defecto (gesdd) no converge aunque O_N esté bien condicionada."""
    from scipy.linalg import svd
    u, sv, vt = svd(B[:N + 1].reshape(-1, B.shape[-1]), full_matrices=False, lapack_driver="gesvd")
    keep = sv > sv[0] * max(B.shape[1] * (N + 1), B.shape[-1]) * np.finfo(float).eps
    return (vt[keep].T / sv[keep]) @ u[:, keep].T


def ancho_efectivo(B, L_, m):
    """Desvío (x, y) del perfil |(A^*)^m g|^2 de un sensor central, con distancias periódicas a su posición."""
    p = B.shape[1]
    i = p // 2 + (L_ // cfg.SEP) // 2                                   # un sensor cerca del centro
    pos = np.argwhere(B[0, i].reshape(L_, L_) == B[0, i].max())[0]
    w = B[m, i].reshape(L_, L_) ** 2
    w = w / w.sum()
    yy, xx = np.meshgrid((np.arange(L_) - pos[0] + L_ // 2) % L_ - L_ // 2,
                         (np.arange(L_) - pos[1] + L_ // 2) % L_ - L_ // 2, indexing="ij")
    mx, my = np.sum(w * xx), np.sum(w * yy)
    return mx, np.sqrt(np.sum(w * (xx - mx) ** 2)), np.sqrt(np.sum(w * (yy - my) ** 2))


class OperadorFourier:
    """A (símbolo H), sensores delta en columnas y (O_N^* O_N)^{-1} por bloques de alias."""

    def __init__(self, Lc, s, nu, N, sep=cfg.SEP):
        self.L, self.N = Lc, N
        self.H = op.multiplicador_ad(Lc, Lc, (s, 0), nu)
        self.G = sn.banco(Lc, sep, 1, [("delta",)])
        self.pix = self.G.reshape(self.G.shape[0], -1).argmax(axis=1)            # píxel de cada sensor
        self.IY, self.IX = fr.clases(Lc, sep, 1)
        P = Lc * Lc / sep
        phi = (np.fft.fft2(sn.nucleo(Lc, ("delta",))) / Lc)[self.IY, self.IX]
        Hc = np.conj(self.H[self.IY, self.IX])
        B = sum(P * np.einsum("ck,cl->ckl", Hc ** n * phi, np.conj(Hc ** n * phi)) for n in range(N + 1))
        lam = np.linalg.eigvalsh(B)
        self.alfa, self.beta = lam.min(), lam.max()
        self.inv = np.linalg.inv(B)
        self.traza_inv = float(np.sum(1 / lam))                                   # tr (O^* O)^{-1}

    def potencia(self, img, n, adj=False):
        Hn = np.conj(self.H) ** n if adj else self.H ** n
        return np.real(np.fft.ifft2(np.fft.fft2(img, axes=(-2, -1)) * Hn, axes=(-2, -1)))

    def oto_inv_hat(self, bh):
        """(O^* O)^{-1} en Fourier (DFT unitaria): bh (..., L, L) complejo -> imagen real."""
        v = np.einsum("ckl,...cl->...ck", self.inv, bh[..., self.IY, self.IX])
        out = np.zeros_like(bh)
        out[..., self.IY, self.IX] = v
        return np.real(np.fft.ifft2(out, axes=(-2, -1), norm="ortho"))

    def oto_inv(self, b):
        return self.oto_inv_hat(np.fft.fft2(b, axes=(-2, -1), norm="ortho"))

    def O_adj(self, v):
        """O^* v = sum_n A^{*n} G v_n para v (R, N+1, p) -> (R, L, L)."""
        R = v.shape[0]
        acc = np.zeros((R, self.L, self.L), complex)
        for n in range(self.N + 1):
            img = np.zeros((R, self.L * self.L))
            img[:, self.pix] = v[:, n]
            acc += np.fft.fft2(img.reshape(R, self.L, self.L), axes=(-2, -1), norm="ortho") * np.conj(self.H) ** n
        return acc

    def reconstruir(self, u0, delta=None):
        """f_hat = O^+ (u^(0) - G^T delta_n); u0 (R, N+1, p), delta (R, N+1, d)."""
        v = u0 if delta is None else u0 - delta[:, :, self.pix]
        return self.oto_inv_hat(self.O_adj(v)).reshape(u0.shape[0], -1)

    def bancos(self, E):
        """Filas (A^*)^e g_i, e = 0..E (solo para medir con profundidad)."""
        bancos = [self.G] + [self.potencia(self.G, e, adj=True) for e in range(1, E + 1)]    # m = 0: deltas exactas
        return np.stack(bancos).reshape(E + 1, self.G.shape[0], -1)

    def T_soporte(self, idx):
        """T_S (d, N, |S|) = O^+ iota (I ⊗ G^T) L restringido a las columnas de S."""
        Lc, N, d = self.L, self.N, self.L * self.L
        E = np.zeros((len(idx), d))
        E[np.arange(len(idx)), idx] = 1.0
        Eh = np.fft.fft2(E.reshape(-1, Lc, Lc), axes=(-2, -1), norm="ortho")
        mascara = np.zeros(d, bool)
        mascara[self.pix] = True
        Uh = np.zeros_like(Eh)
        TS = np.zeros((d, N, len(idx)))
        for J in range(N):                                                        # U_J, y T_S[:, k-1] con k = N - J
            a = np.real(np.fft.ifft2(Eh * self.H ** J, axes=(-2, -1), norm="ortho")).reshape(-1, d)
            a[:, ~mascara] = 0.0
            Uh += np.fft.fft2(a.reshape(-1, Lc, Lc), axes=(-2, -1), norm="ortho") * np.conj(self.H) ** J
            k = N - J
            TS[:, k - 1] = self.oto_inv_hat(Uh * np.conj(self.H) ** k).reshape(len(idx), d).T
        return TS


def energia(X, Lt, Rs, bloque=512):
    """tr(X K_w X^T) con K_w = (L_t ⊗ R_S)(L_t ⊗ R_S)^T, por bloques de filas."""
    return sum(np.sum((np.einsum("dns,na->das", X[i:i + bloque], Lt) @ Rs) ** 2) for i in range(0, X.shape[0], bloque))


def exacto_fourier(opf, gp, TS, Sigma_S, sigma_n):
    """E||f_hat - f||^2 (sin corregir, GP, oráculo) en el espacio del soporte (como experimento_2D_64.exacto_soporte,
    pero con O^+ aplicado por bloques de Fourier)."""
    N, Mg, p_a = gp.N, gp.M, gp.p_a
    d = TS.shape[0]
    Lt, Rs = np.linalg.cholesky(gp.Kt), fz.raiz_psd(Sigma_S)
    ruido = sigma_n ** 2 * opf.traza_inv
    X = np.einsum("dnq,nk->dkq", TS @ gp.CS.T, gp.Kt).reshape(d, -1)
    Y = cho_solve(gp.cho, X.T).T
    del X
    Y4 = Y.reshape(d, N, Mg, p_a)
    Aw = (Y.reshape(d, N, -1) @ gp.C) - TS
    e_gp = energia(Aw, Lt, Rs)
    del Aw
    YD = np.zeros((d, N + 1, Mg + 1, p_a))
    YD[:, 1:, :Mg] += Y4
    YD[:, :N, 1:] -= Y4
    pix_act = opf.pix[gp.act]
    cruce = 0.0
    for n in range(N + 1):                       # <YD[:, n, 0, i], O^+ col (n, i)> = [A^n (O^*O)^{-1} YD[:, n, 0, i]](x_i)
        img = opf.potencia(opf.oto_inv(YD[:, n, 0, :].T.reshape(p_a, opf.L, opf.L)), n).reshape(p_a, -1)
        cruce += np.sum(img[np.arange(p_a), pix_act])
    ruido_gp = sigma_n ** 2 * (np.sum(YD ** 2) - 2 * cruce + opf.traza_inv)
    return dict(sin_corregir=energia(TS, Lt, Rs) + ruido, gp=e_gp + ruido_gp, oraculo=ruido)


class OperadorColumnas(OperadorFourier):
    """Como barrido_N_64.OperadorFourier, con sensores delta en todas las filas de las columnas x = c mod 4."""

    def __init__(self, Lc, s, nu, N_, columnas, periodo=cfg.SEP):
        self.L, self.N = Lc, N_
        self.H = op.multiplicador_ad(Lc, Lc, (s, 0), nu)
        yy, xx = np.divmod(np.arange(Lc * Lc), Lc)
        self.pix = np.concatenate([np.flatnonzero(xx % periodo == c) for c in columnas])
        self.IY, self.IX = fr.clases(Lc, periodo, 1)
        P = Lc * Lc / periodo
        Hc = np.conj(self.H[self.IY, self.IX])
        phis = [np.exp(-2j * np.pi * self.IX * c / Lc) / Lc for c in columnas]       # DFT unitaria de un delta en (0, c)
        B = sum(P * np.einsum("ck,cl->ckl", Hc ** n * ph, np.conj(Hc ** n * ph)) for n in range(N_ + 1) for ph in phis)
        lam = np.linalg.eigvalsh(B)
        self.alfa, self.beta = lam.min(), lam.max()
        self.inv = np.linalg.inv(B)
        self.traza_inv = float(np.sum(1 / lam))
