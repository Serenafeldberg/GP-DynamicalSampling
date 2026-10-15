"""
Identidad exacta del error de reconstrucción (como en el Cap. 4), para el caso 1D.

Si O_N es frame, f_hat - f no depende de f:
    f_hat - f = -T (I - H Psi) w + (T H D - O_N^+ S_u) eps
    T = O_N^+ iota (I_N ⊗ G^T) L ,   L = [A^{n-k}]_{k <= n}  (delta = L w)
    H = K_w Psi^T (Psi K_w Psi^T + sigma_n^2 D D^T)^{-1} ,  Psi = I_N ⊗ C ,  K_w = K_t ⊗ Sigma_w
y entonces E||f_hat - f||^2 = tr(A_w K_w A_w^T) + sigma_n^2 ||A_eps||_F^2.
Arma matrices densas de tamaño N d: solo para d chico.
"""
import numpy as np


def matriz_A(aplicar_A, forma):
    """Matriz densa de A (d x d) aplicando el operador a la base canónica."""
    d = int(np.prod(forma))
    return aplicar_A(np.eye(d).reshape(d, *forma)).reshape(d, d).T


def mse_exacto(A, Gt, Sigma, Kt, M, N, sigma_n):
    """E||f_hat - f||^2 para sin corregir, GP (prior verdadero) y oráculo. Gt: filas g_i^T (p x d).

    Con M = 0 no hay residuos: solo se devuelven sin corregir y oráculo.
    """
    d, p = A.shape[0], Gt.shape[0]
    pw = [np.eye(d)]
    for _ in range(max(N, M)):
        pw.append(A @ pw[-1])
    C = np.vstack([Gt @ pw[m] for m in range(M)]) if M > 0 else np.zeros((0, d))
    Psi = np.kron(np.eye(N), C)
    Kw = np.kron(Kt, Sigma)

    # ruido eps indexado (n = 0..N, m = 0..M, i); eta = D eps ;  S_u selecciona m = 0
    ie = lambda n, m, i: (n * (M + 1) + m) * p + i
    D = np.zeros((N * M * p, (N + 1) * (M + 1) * p))
    Su = np.zeros(((N + 1) * p, (N + 1) * (M + 1) * p))
    for n in range(1, N + 1):
        for m in range(M):
            for i in range(p):
                r = ((n - 1) * M + m) * p + i
                D[r, ie(n, m, i)] += 1.0
                D[r, ie(n - 1, m + 1, i)] -= 1.0
    for n in range(N + 1):
        for i in range(p):
            Su[n * p + i, ie(n, 0, i)] = 1.0

    L = np.zeros((N * d, N * d))
    for n in range(N):
        for k in range(n + 1):
            L[n * d:(n + 1) * d, k * d:(k + 1) * d] = pw[n - k]
    Opinv = np.linalg.pinv(np.vstack([Gt @ pw[n] for n in range(N + 1)]))
    iota = np.vstack([np.zeros((p, N * p)), np.eye(N * p)])
    T = Opinv @ iota @ np.kron(np.eye(N), Gt) @ L

    ruido = sigma_n ** 2 * np.sum((Opinv @ Su) ** 2)
    if M == 0:
        return dict(sin_corregir=np.trace(T @ Kw @ T.T) + ruido, oraculo=ruido, C=C)
    Kz = Psi @ Kw @ Psi.T + sigma_n ** 2 * D @ D.T
    H = np.linalg.solve(Kz, Psi @ Kw).T
    Aw = -T @ (np.eye(N * d) - H @ Psi)
    Ae = T @ H @ D - Opinv @ Su
    return dict(sin_corregir=np.trace(T @ Kw @ T.T) + ruido,
                gp=np.trace(Aw @ Kw @ Aw.T) + sigma_n ** 2 * np.sum(Ae ** 2),
                oraculo=ruido,
                DDt=D @ D.T, C=C)
