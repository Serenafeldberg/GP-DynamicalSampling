"""
Prior GP del forzado y recursión de delta_n.

    w ~ N(0, K_t ⊗ Sigma_w)
    K_t[n,n'] = rho^|n-n'|                         (temporal, AR(1))
    Sigma_w   = sigma_w^2 D (K_y ⊗ K_x) D           (espacial)
    K_x, K_y  : kernel squared exponential de longitud de escala ell
    D         = diag(ventana), ventana gaussiana centrada en la fuente

Se muestrea sin armar ninguna matriz de LL x LL: con raíces K = R R^T,
    W_n = sigma_w * ventana ⊙ sum_k L_t[n,k] R_y Z_k R_x^T ,   Z_k ~ N(0, I),
que es W = L_t Z L_s^T con L_s = sigma_w D (R_y ⊗ R_x) aplicado sin Kronecker.
Luego delta_0 = 0, delta_n = A delta_{n-1} + w_n.
"""
import numpy as np


def kernel_se_1d(L, ell):
    """K[i,j] = exp(-(i-j)^2 / (2 ell^2)) sobre los píxeles 0..L-1."""
    i = np.arange(L)
    return np.exp(-(i[:, None] - i[None, :]) ** 2 / (2 * ell ** 2))


def kernel_temporal(N, rho):
    """K_t[n,n'] = rho^|n-n'| para n, n' = 1..N."""
    n = np.arange(N)
    return rho ** np.abs(n[:, None] - n[None, :])


def ventana_gaussiana(L, centro, ancho):
    """Ventana espacial [y, x] = exp(-|x - c|^2 / (2 ancho^2)), con máximo 1 en la fuente."""
    y, x = np.mgrid[0:L, 0:L]
    return np.exp(-((x - centro[0]) ** 2 + (y - centro[1]) ** 2) / (2 * ancho ** 2))


def raiz_psd(K):
    """R con K = R R^T vía autovalores (el kernel SE está mal condicionado para Cholesky)."""
    lam, U = np.linalg.eigh(K)
    return U * np.sqrt(np.clip(lam, 0, None))


def muestrear_forzado(rng, N, L, ell, rho, sigma_w, ventana):
    """Una realización de w_1..w_N, de forma (N, L, L); w[n-1] = w_n."""
    L_t = np.linalg.cholesky(kernel_temporal(N, rho))
    R = raiz_psd(kernel_se_1d(L, ell))                  # K_x = K_y (imagen cuadrada)
    Z = rng.standard_normal((N, L, L))
    espacial = R @ Z @ R.T                              # (R_y ⊗ R_x) vec(Z_k), por cuadro
    temporal = np.einsum("nk,kij->nij", L_t, espacial)  # correlación AR(1) entre cuadros
    return sigma_w * ventana * temporal


def acumular_delta(w, aplicar_A):
    """delta_0 = 0, delta_n = A delta_{n-1} + w_n; devuelve (N+1, L, L)."""
    N = w.shape[0]
    delta = np.zeros((N + 1,) + w.shape[1:])
    for n in range(1, N + 1):
        delta[n] = aplicar_A(delta[n - 1]) + w[n - 1]
    return delta


def sigma_espacial(K, ventana, sigma_w):
    """Sigma_w = sigma_w^2 D K D explícita (d x d), con D = diag(ventana aplanada); K en el mismo orden."""
    v = np.ravel(ventana)
    return sigma_w ** 2 * v[:, None] * K * v[None, :]


def energia_delta_N(Sigma, Kt, potencia, forma):
    """E||delta_N||^2 = sum_{k,k'} K_t[k,k'] tr(A^{k'-k} Sigma) para A ortogonal (traslación).

    potencia(v, j) aplica A^j (j puede ser negativo) a un arreglo de imágenes con forma (..., *forma).
    """
    N, d = Kt.shape[0], Sigma.shape[0]
    cols = Sigma.reshape(d, *forma)
    t = {j: np.trace(potencia(cols, j).reshape(d, d)) for j in range(-(N - 1), N)}
    return sum(Kt[k, kp] * t[kp - k] for k in range(N) for kp in range(N))


def energia_delta_N_simbolo(Sigma, Kt, H, idx=None):
    """E||delta_N||^2 = sum_{k,k'} K_t[k,k'] tr(A^{N-k} Sigma (A^{N-k'})^*) para A = multiplicador de Fourier H
    (no hace falta que A sea ortogonal).

    tr(B Sigma C^*) = sum_xi b(xi) conj(c(xi)) [F Sigma F^*]_{xi xi}, y esa diagonal es la DFT de las sumas por
    desfasaje s[u] = sum_x Sigma[x, x + u]. Si idx no es None, Sigma está restringida a los píxeles idx.
    """
    Ly, Lx = H.shape
    d, N = Ly * Lx, Kt.shape[0]
    idx = np.arange(d) if idx is None else np.asarray(idx)
    y, x = np.divmod(idx, Lx)
    u = ((y[None, :] - y[:, None]) % Ly) * Lx + (x[None, :] - x[:, None]) % Lx       # u = pos_j - pos_i
    s = np.zeros(d)
    np.add.at(s, u.ravel(), np.asarray(Sigma).ravel())
    diag = np.real(np.fft.ifft2(s.reshape(Ly, Lx)))                                    # [F Sigma F^*]_{xi xi}
    Z = np.stack([H ** (N - k) for k in range(1, N + 1)])                              # a^{N-k}, k = 1..N
    peso = np.real(np.einsum("kyx,kl,lyx->yx", Z, Kt, np.conj(Z)))
    return float(np.sum(diag * peso))
