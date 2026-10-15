# Procesos Gaussianos para la estimación de dinámicas no modeladas y la reconstrucción de señales

Código de la tesis de grado (Informe de Proyecto Final, Ingeniería en Inteligencia Artificial, Universidad de San
Andrés).

**Autora:** Serena Feldberg · **Mentor:** Dr. Ing. Juan Ignacio Giribet

El código se desarrolló con asistencia de herramientas de IA (Claude Code); la autora revisó y validó todos los
resultados.

## Instalación

Requiere Python 3.14 (probado con 3.14.6).

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Cómo correr

Todos los scripts se corren **desde la raíz del repositorio** como módulos (`python -m ...`). Cada uno escribe en
`salida/<capítulo>/<experimento>/`, con rutas relativas a la raíz del repositorio (no al directorio desde donde se
corre). Las semillas son fijas, así que los resultados son reproducibles; la única excepción son los tiempos de la
Figura 2.8, que dependen del equipo.

En `transporte/` el cómputo y las figuras están separados: cada script de `transporte/experimentos/` guarda un
`resultados.npz` y el script homónimo de `transporte/figuras/` lo lee y genera las figuras (PDF) y tablas (LaTeX).

```bash
.venv/bin/python -m cap2.figs_cap2_teoria
.venv/bin/python -m transporte.experimentos.experimento_2D_64
.venv/bin/python -m transporte.figuras.experimento_2D_64
```

Orden necesario en `transporte/`: `comparacion_justa` lee los resultados de `experimento_2D_64` y de
`adveccion_difusion experimento`; `figuras.sensores` y `figuras.resumen_AD` leen los de `comparacion_justa`;
`figuras.adveccion_difusion` usa los de `experimento_2D_64` como referencia entera.

## Estructura

```
cap2/            regresión GP (Capítulo 2): núcleo, figuras y verificación contra scikit-learn y GPy
cap3/            GP temporal sobre un forzado escalar (Capítulo 3)
cap4/            experimentos del Capítulo 4 (Secciones 4.7.1 a 4.7.3)
transporte/      ejemplo de transporte (Sección 4.7.5)
  comun/         módulos compartidos: parámetros (config), operador A, forzado, sensores, frames, reconstrucción,
                 identidad exacta, posterior GP en espacios reducidos, operadores en Fourier, estilo de figuras
  experimentos/  cómputo -> salida/transporte/<experimento>/resultados.npz
  figuras/       figuras y tablas a partir de resultados.npz
salida/          figuras (PDF, GIF, MP4) y tablas (LaTeX) generadas; los .npz no se versionan
```

## Scripts, salidas y secciones

Los tiempos son de un MacBook Pro (8 núcleos, 16 GB) corriendo un script por vez; entre paréntesis, el pico de memoria
cuando supera 1 GB. Cada script imprime sus números y chequeos (conviene guardarlos con `> log.txt`).

| Script | Genera | Sección | Tiempo aprox. |
|---|---|---|---|
| `cap2.figs_cap2_teoria` | Figs. 2.1 `marginal_vs_condicional`, 2.2 `bloques_covarianza`, 2.3 `rbf_lengthscale`, 2.4 `matern_suavidad` | 2.1, 2.4 | 2 s |
| `cap2.figs_cap2_aplicacion` | Figs. 2.5 `prior_posterior`, 2.6 `efecto_lengthscale_lml`, 2.7 `superficie_lml`, 2.8 `costo_cholesky`; números de 2.7.1, Tablas 2.2 y 2.3, 2.7.4 y la observación del jitter (impresos) | 2.6, 2.7 | 10 s |
| `cap3.figs_cap3` | Figs. 3.1 `prior_temporal`, 3.2 `posterior_forzante`; Tabla 3.2 (impresa) | 3.2, 3.6 | 1,5 min |
| `cap4.experimentos_cap4` | Tabla 4.1 `tabla_orbita.tex`, Tabla 4.2 `tabla_estimador.tex`, Fig. 4.1 `reconstruccion_vs_M` | 4.7.1–4.7.3 | 35 s |
| `transporte.experimentos.video_transporte` + `figuras.video_transporte` | `cuadros`, `forzado_fuente`, `forzado_w`, `video.gif`/`.mp4` | 4.7.5 | 5 s |
| `transporte.experimentos.diagnostico_frames` + `figuras.diagnostico_frames` | `alpha_vs_N`, `nucleos`, `invisibles`, `tabla_frames.tex`/`.csv` | 4.7.5 | 15 s |
| `transporte.experimentos.experimento_1D` + `figuras.experimento_1D` | `error_1D`, `error_posicion_1D`, `presupuesto_1D`, `reconstruccion_2D`, `video_sensores.gif` y sus tablas | 4.7.5 | 1 min |
| `transporte.experimentos.cruce_forzado` + `figuras.cruce_forzado` | `cruce_forzado_1D`, `tabla_cruce_1D.tex` | 4.7.5 | 1 min |
| `transporte.experimentos.robustez_prior` + `figuras.robustez_prior` | `robustez_prior_1D`, `tabla_robustez_1D.tex` | 4.7.5 | 10 s |
| `transporte.experimentos.experimento_2D_64` + `figuras.experimento_2D_64` | `error_vs_M_2D`, `reconstruccion_2D_64` (y `_ell8`), `cuadros_2D_64` (y `_sensores`), `cuadro_n8_2D_64`, `sensores_n0_2D_64`, `tabla_2D_64.tex` | 4.7.5 | 5 min (4 GB) |
| `transporte.experimentos.adveccion_difusion diagnostico` + `figuras.adveccion_difusion` | `alpha_vs_nu`, `alpha_vs_s`, `tabla_diagnostico_AD.tex`, `tabla_barrido_s_AD.tex` | 4.7.5 | 5 s |
| `transporte.experimentos.adveccion_difusion experimento` + `figuras.adveccion_difusion` | `error_vs_M_AD`, `reconstruccion_AD`, `cuadros_AD`, `tabla_AD.tex` | 4.7.5 | 1 h (6,5 GB) |
| `transporte.experimentos.barrido_N_64` + `figuras.barrido_N` | `error_vs_N`, `alpha_vs_N_barrido`, `tabla_barrido_N.tex` | 4.7.5 | 20 min (4,6 GB) |
| `transporte.experimentos.comparacion_justa entero` / `ad` + `figuras.comparacion_justa` | `tabla_comparacion_justa.tex`, `figura_presentacion` (entero), `figura_presentacion_reconstruccion` y `_error` (advección-difusión) | 4.7.5 | 1,5 min cada uno |
| `transporte.figuras.sensores` | `sensores_2D_64`, `sensores_AD`, `sensores_comparacion` | 4.7.5 | 5 s |
| `transporte.figuras.resumen_AD` | `resumen_errores_AD` | 4.7.5 | 2 s |
| `transporte.experimentos.presupuesto_AD` + `figuras.presupuesto_AD` | `presupuesto/tabla_presupuesto_AD.tex` (advección-difusión a igual presupuesto total, m en {2, 4, 8}) | 4.7.5 | 8 min (5 GB) |

Los experimentos de `transporte/` verifican con `assert` sus cuentas: el operador (adjunta, unitariedad, `np.roll`),
`alpha_N` por bloques contra el cálculo denso, el posterior y la identidad exacta en espacios reducidos contra
`reconstruccion.py` y `error_exacto.py` en 16 x 16, y la identidad exacta contra Monte Carlo (|z| < 4).

## Cómo citar

```bibtex
@misc{feldberg2026codigo,
  author       = {Feldberg, Serena},
  title        = {Código de la tesis «Procesos Gaussianos para la estimación de dinámicas no modeladas y la
                  reconstrucción de señales»},
  year         = {2026},
  howpublished = {\url{ENLACE_AL_REPOSITORIO}},
  note         = {Versión TAG_DE_LA_VERSION}
}
```
