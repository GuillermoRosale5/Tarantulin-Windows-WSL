# Dependencias fijadas

La simulacion y el entrenamiento se ejecutan dentro de WSL 2. La carpeta de
Windows contiene solo codigo fuente; los binarios Linux, el entorno virtual,
MuJoCo Playground y los resultados viven en el runtime WSL.

## Base validada

- Windows 10/11 con WSL 2.
- Ubuntu 24.04 como distribucion predeterminada.
- Python `>=3.12,<3.13`, proporcionado por Ubuntu 24.04.
- `uv` 0.11.8 como gestor reproducible de Python.
- MuJoCo Playground fijado al commit
  `9c2dce4a3519cd4bb9d299bf28a6ef3f5086844b`.
- JAX/JAXLIB 0.6.2, MuJoCo/MJX 3.6.0 y las versiones declaradas en
  `tarantulin/hiperparametros.py` para NVIDIA y CPU.

Los paquetes de Ubuntu y `uv` se instalan mediante `install.ps1`. `uv` obtiene
MuJoCo Playground como la distribucion `playground` desde el commit fijado; no
se crea ni se versiona un repositorio Git anidado.

## Perfiles de computo en WSL

| Perfil | Estado | Comportamiento |
|---|---|---|
| `auto` | Recomendado | Elige NVIDIA si `nvidia-smi` funciona; en otro caso CPU. |
| `nvidia` | Soportado | Instala el extra CUDA 12 de JAX fijado por este repositorio y exige backend JAX GPU. |
| `cpu` | Soportado | Instala JAX CPU. Sirve para desarrollo y pruebas; no para simulacion masiva. |
| `amd` | No habilitado/validado en WSL | Se rechaza salvo consentimiento `-EnableExperimentalAmdWsl`; requiere ROCm compatible instalado previamente. |
| `intel` | No soportado en WSL | Se rechaza y recomienda CPU o Linux nativo experimental. |

El proyecto ejecuta únicamente `--impl jax`. `warp-lang==1.11.0` permanece
fijado como dependencia heredada del proyecto, pero la ruta `--impl warp` se
rechaza: está acoplada a NVIDIA y rompería la portabilidad buscada.

## Fuentes oficiales de compatibilidad

- [Instalacion y backends de JAX](https://docs.jax.dev/en/latest/installation.html)
- [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html)
- [CUDA sobre WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [Limitaciones actuales de ROCm sobre WSL](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/wsl/howto_wsl.html)
- [Intel Extension for OpenXLA](https://github.com/intel/intel-extension-for-openxla/blob/0.7.0/README.md)
