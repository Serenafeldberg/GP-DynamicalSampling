"""
Parámetros del ejemplo de transporte, en un solo lugar.

Modelo: y_0 = f, y_n = A y_{n-1} + w_n, w ~ N(0, K_t ⊗ Sigma_w), K_t[n,n'] = rho^|n-n'|, Sigma_w = sigma_w^2 D K D.
Los módulos de comun/ reciben los parámetros como argumentos; solo los experimentos leen este archivo.
"""
# imagen
L = 64                      # la imagen f es de L x L píxeles
IMAGEN = "coffee"           # nombre de skimage.data 

# transporte
S = (1, 0)                  # velocidad (s_x, s_y) en píxeles por paso
SUBPIXEL = False            # True: desplazamiento por FFT (admite s no entero)
N = 16                      # cantidad de pasos: n = 0..N

# forzado
ELL = 3.0                   # longitud de escala del kernel SE espacial (píxeles)
RHO = 0.9                   # correlación temporal AR(1)
SIGMA_W = 0.018             # escala del forzado: max|delta_N| ~ 20-25 % del rango de f
FUENTE_CENTRO = (28, 18)    # posición (c_x, c_y) de la fuente (píxeles) (superficie del café)
FUENTE_ANCHO = 5.0          # desvío de la ventana gaussiana (píxeles)

# reproducibilidad y salidas
SEED = 0
CUADROS_N = (0, 4, 8, 12, 16)   # instantes que se muestran en cuadros.pdf
FPS = 4                         # cuadros por segundo del video

# reconstrucción (residuos + GP, esquema del Cap. 4)
SEP = 4                         # sensores en columnas cada SEP píxeles (m_x = SEP, m_y = 1), núcleo delta
N_REC = 8                       # horizonte de la reconstrucción
SIGMA_N = 1e-3                  # desvío del ruido de medición (como en 4.7.1)
L_1D, M_1D, REPS_1D = 64, (1, 2, 3), 2000
ELL_BARRIDO = {"chica": 1.0, "grande": 8.0}           # ell del forzado en el barrido 1D
FUENTES_1D = {"sobre sensor": 28,                      # 28 = 0 mod SEP
              "copia aguas arriba": 27,                # (A^*) G: x_i - 1
              "lejos": 29}                             # x_i - 3: ninguna copia con M <= 3
FUENTE_ANCHO_POSICION = 1.0    # ancho de la fuente solo para el barrido de posición (error_posicion_1D)
PRESUPUESTOS_1D = (16, 32, 48)  # sensores totales p (M + 1) en la comparación por presupuesto
RAZON_FORZADO = None            # None: usa SIGMA_W; un número: fija sigma_w para que ||delta_N|| / ||f|| (RMS) valga eso
L_2D, M_2D, REPS_2D = 32, (1, 2), 200
FUENTE_2D = (13, 16)            # (c_x, c_y): 13 = 1 mod SEP, fuera de sensores y de copias con M <= 3

# experimentos 2D de 64 x 64 (transporte entero, advección-difusión, barrido en N, comparación justa)
FUENTE_64 = tuple(round(c * 64 / L_2D) for c in FUENTE_2D)    # misma posición relativa que en 32 x 32: (26, 32)
UMBRAL_SOPORTE = 1e-4           # soporte de la fuente: píxeles donde la ventana supera UMBRAL de su máximo
ELLS_64, RAZONES_64, MS_64 = (2.0, 8.0), (0.05, 0.20), (1, 2, 3)
