# Dependencias fijadas

La simulación y el entrenamiento se ejecutan dentro de WSL 2. La carpeta de
Windows contiene solo código fuente; los binarios Linux, el entorno virtual,
MuJoCo Playground y los resultados viven en el runtime WSL.

## Base validada

- Windows 10/11 con WSL 2.
- Ubuntu 24.04 como distribución predeterminada.
- Python `>=3.12,<3.13`, proporcionado por Ubuntu 24.04.
- `uv` 0.11.8 como gestor reproducible de Python.
- MuJoCo Playground fijado al commit
  `9c2dce4a3519cd4bb9d299bf28a6ef3f5086844b`.
- JAX/JAXLIB 0.6.2, MuJoCo/MJX 3.6.0 y las versiones declaradas en
  `tarantulin/hiperparametros.py` para NVIDIA y CPU.

Los paquetes de Ubuntu y `uv` se instalan mediante `install.ps1`. `uv` obtiene
MuJoCo Playground como la distribucion `playground` desde el commit fijado; no
se crea ni se versiona un repositorio Git anidado.

## Red preentrenada reproducible

El repositorio incluye una única red de referencia en
`pretrained/tarantulin_standup_fase2_45932544`. Es un checkpoint Orbax OCDBT
creado con las versiones fijadas en este proyecto: fase 2, semilla 42 y paso
45.932.544 del perfil que entonces se llamaba `lite`, equivalente al perfil
actual `ligero`. Este nombre se conserva únicamente en los metadatos históricos
del checkpoint. Sus archivos y configuraciones se validan mediante `SHA256SUMS`
antes de abrirlos.

```powershell
.\tarantulin.ps1 visualizar-red-preentrenada
```

Este comando usa siempre ese paquete y no consulta
`logs_tarantulin_mjx/ultima_run.txt`. En cambio, `visualizar-resultados` y
`visualizar_ultimo_checkpoint.sh` cargan el último checkpoint generado
localmente; puede proceder de una ejecución parcial y no tiene por qué producir
el mismo movimiento.

Los checkpoints normales de entrenamiento continúan fuera de GitHub. La red
preentrenada es una excepción deliberada y pequeña para que una instalación
nueva pueda reproducir exactamente la misma simulación.

## Perfiles de cómputo en WSL

| Perfil | Estado | Comportamiento |
|---|---|---|
| `auto` | Recomendado | Elige NVIDIA si `nvidia-smi` funciona; en otro caso CPU. |
| `nvidia` | Soportado | Instala el extra CUDA 12 de JAX fijado por este repositorio y exige backend JAX GPU. |
| `cpu` | Soportado | Instala JAX CPU. Sirve para desarrollo y pruebas; no para simulacion masiva. |
| `amd` | No soportado en WSL2 | Se rechaza con un diagnóstico claro; en este sistema se utiliza `cpu`. |
| `intel` | No soportado en WSL | Se rechaza y se recomienda el perfil CPU. |

El proyecto ejecuta únicamente `--impl jax`. `warp-lang==1.11.0` permanece
fijado como dependencia heredada del proyecto, pero la ruta `--impl warp` se
rechaza: está acoplada a NVIDIA y rompería la portabilidad buscada.

## Fuentes oficiales de compatibilidad

- [Instalación y backends de JAX](https://docs.jax.dev/en/latest/installation.html)
- [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html)
- [CUDA sobre WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [Limitaciones actuales de ROCm sobre WSL](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/wsl/howto_wsl.html)
- [Intel Extension for OpenXLA](https://github.com/intel/intel-extension-for-openxla/blob/0.7.0/README.md)
