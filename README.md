# TARANTULIN · Windows + WSL

Entorno reproducible para ejecutar MuJoCo MJX y JAX desde Windows sin calcular
sobre NTFS. Se programa y versiona en una carpeta normal de Windows; los scripts
crean por si solos una copia de ejecucion privada dentro del filesystem Linux de
WSL 2.

Este repositorio es la variante **Windows + WSL**. Para Ubuntu nativo o para
trabajar directamente dentro de WSL sin una carpeta Windows duplicada se usa el
repositorio [Tarantulin-Linux-WSL](https://github.com/GuillermoRosale5/Tarantulin-Linux-WSL).

## Instalacion mas corta

Abre PowerShell en una carpeta Windows vacia y ejecuta:

```powershell
git clone https://github.com/GuillermoRosale5/Tarantulin-Windows-WSL.git .
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

Eso prepara Ubuntu 24.04 en WSL, paquetes de sistema, `uv`, el entorno Python
fijado y el backend de computo. Si Windows solicita reiniciar o Ubuntu solicita
crear el usuario Linux, completa ese paso y vuelve a ejecutar exactamente
`install.ps1`.

Al terminar:

```powershell
.\tarantulin.ps1 doctor
.\tarantulin.ps1 test-mjx
```

No hay que crear ninguna carpeta Linux. El instalador la crea en
`~/.local/share/tarantulin-windows/<id>/workspace`.

## Instalacion automatica desde una carpeta vacia

Si todavia no tienes Git, el bootstrap puede instalar Git con `winget`, clonar
la rama `main` en la carpeta actual y continuar con el instalador:

```powershell
powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/GuillermoRosale5/Tarantulin-Windows-WSL/main/scripts/bootstrap_windows.ps1 | iex"
```

El bootstrap se detiene si la carpeta contiene archivos que no pertenecen a
TARANTULIN; nunca los elimina para poder clonar encima.

## Elige el modo que encaja con tu equipo

### NVIDIA en Windows + WSL

Es la ruta recomendada para simulacion masiva. Instala antes un driver NVIDIA de
Windows con soporte WSL y ejecuta:

```powershell
.\install.ps1 -Accelerator nvidia
.\tarantulin.ps1 doctor
```

No se instala un driver Linux NVIDIA dentro de WSL. La GPU debe aparecer con
`nvidia-smi` desde WSL; el instalador lo comprueba y JAX debe informar backend
`gpu`.

### CPU en cualquier equipo Windows

Sirve para desarrollar, validar XMLs y hacer pruebas cortas cuando no hay una
GPU compatible:

```powershell
.\install.ps1 -Accelerator cpu
```

Es funcional, pero no ofrece el rendimiento necesario para miles de entornos.
El perfil elegido queda guardado fuera de la copia sincronizada, por lo que los
comandos posteriores no cambiaran accidentalmente de CPU a GPU.

### Deteccion automatica

```powershell
.\install.ps1 -Accelerator auto
```

`auto` elige NVIDIA cuando es realmente visible mediante `nvidia-smi`; de lo
contrario elige CPU. No activa AMD ni Intel de forma silenciosa.

### AMD o Intel bajo WSL

JAX/ROCm para AMD no esta actualmente habilitado ni validado oficialmente bajo
WSL. Por eso `-Accelerator amd` se rechaza de forma predeterminada. Solo existe
un modo de investigacion consciente, sin garantia de instalacion ni ejecucion,
para hardware y ROCm previamente compatibles:

```powershell
.\install.ps1 -Accelerator amd -EnableExperimentalAmdWsl
```

Intel GPU no dispone de un backend JAX/XLA utilizable para este flujo en WSL y
se rechaza con una explicacion. En ambos casos usa `-Accelerator cpu` o la
variante Linux nativa cuando proceda.

La matriz y las versiones exactas estan en [DEPENDENCIAS.md](DEPENDENCIAS.md).

## Modos de instalacion y reparacion

Instalacion completa, recomendada en el primer uso:

```powershell
.\install.ps1
```

Crear el runtime y copiar el codigo, pero posponer el entorno Python:

```powershell
.\install.ps1 -NoSetup
```

Sincronizar solamente un runtime ya inicializado:

```powershell
.\install.ps1 -SyncOnly
```

Omitir `apt` porque los paquetes Ubuntu ya estan instalados:

```powershell
.\install.ps1 -SkipSystemPackages
```

Usar otra distribucion Ubuntu ya registrada:

```powershell
.\install.ps1 -Distro Ubuntu-22.04
```

La version fijada y validada es Ubuntu 24.04. El valor predeterminado se puede
cambiar en `.tarantulin/windows.json`.

## Uso diario desde PowerShell

Los comandos de calculo sincronizan primero el codigo Windows al runtime WSL y
luego ejecutan alli el script historico equivalente:

```powershell
.\tarantulin.ps1 doctor
.\tarantulin.ps1 backend
.\tarantulin.ps1 test-mjx
.\tarantulin.ps1 benchmark
.\tarantulin.ps1 train --background --perfil-ppo lite
.\tarantulin.ps1 monitor
.\tarantulin.ps1 stop
.\tarantulin.ps1 view-results
```

Los argumentos existentes siguen funcionando. Cuando PowerShell intente
interpretar uno, coloca `--` antes de los argumentos Linux:

```powershell
.\tarantulin.ps1 train -- --background --num-envs 512 --perfil-ppo debug
```

Atajos y curriculum conservados:

```powershell
.\tarantulin.ps1 curriculum-auto
.\tarantulin.ps1 fases-recompensa
.\tarantulin.ps1 perfiles-ppo
.\tarantulin.ps1 mini-sim
```

Para abrir una shell exactamente en la copia de ejecucion:

```powershell
.\tarantulin.ps1 shell
```

La shell sirve para diagnostico. No edites alli el codigo: el siguiente `sync`
lo reemplazara por la fuente Windows.

`monitor`, `stop` y `pull-results` no sincronizan antes de actuar. Asi pueden
observar, detener o exportar una ejecucion activa sin modificar su workspace.
Antes de enviar `SIGTERM` o `SIGKILL`, `stop` y la proteccion termica contrastan
PID, instante de arranque, comando, runtime y run mediante `/proc`; un PID
obsoleto o reutilizado se rechaza sin senalizarlo.

## Resultados y checkpoints

Durante la ejecucion, logs y checkpoints permanecen en ext4 para no penalizar a
JAX/MuJoCo ni saturar Git. Para copiar la ultima ejecucion a
`artifacts/logs_tarantulin_mjx` en Windows:

```powershell
.\tarantulin.ps1 pull-results
```

Para copiar todas las ejecuciones:

```powershell
.\tarantulin.ps1 pull-results --all
```

`artifacts/` esta ignorado por Git. La exportacion nunca borra resultados de WSL
ni archivos ya presentes en Windows.

## Actualizar el proyecto

La unica copia que se actualiza con Git es la carpeta Windows:

```powershell
git pull --ff-only
.\tarantulin.ps1 sync
.\tarantulin.ps1 doctor
```

El entorno `.venv` se conserva entre sincronizaciones. Si cambia `uv.lock`,
ejecuta de nuevo `install.ps1`; `uv --frozen` garantiza que no se inventen
versiones distintas a las bloqueadas.

## Que hace la doble carpeta

| Lugar | Funcion | Se edita | Se versiona |
|---|---|---:|---:|
| Carpeta elegida en Windows | Codigo fuente canonico, scripts y documentacion | Si | Si |
| Runtime ext4 de WSL | Copia de ejecucion, `.venv`, logs y checkpoints | No | No |
| `artifacts/` en Windows | Exportacion voluntaria de resultados | No | No |

La sincronizacion usa `rsync --delete` solamente dentro de `workspace` y solo
despues de verificar dos marcadores: el de la fuente y el identificador asociado
a esa ruta Windows. Las carpetas `.venv`, `external`, `logs_tarantulin_mjx`,
`checkpoints` y `artifacts` estan protegidas y nunca participan en el borrado.
Cada sincronizacion genera además `sync-manifest.sha256` en el runtime.

Puedes inspeccionar las rutas sin cambiar nada:

```powershell
.\tarantulin.ps1 path
```

Y simular una sincronizacion:

```powershell
.\tarantulin.ps1 sync -DryRunSync
```

## Scripts conservados

- `scripts/tarantulin_wsl.sh`: nucleo de setup, prueba MJX, benchmark,
  entrenamiento, monitor, parada y visualizacion.
- `scripts/lanzar_tarantulin.sh`, `monitor_tarantulin.sh` y
  `parar_tarantulin.sh`: atajos historicos.
- `scripts/curriculum_auto_tarantulin.sh` y
  `cambiar_fase_tarantulin.sh`: supervisor curricular existente.
- `scripts/visualizar_tarantulin.sh`, `graficar_recompensas.sh` y scripts Python:
  visualizacion y analisis existentes.
- `scripts/install_windows.ps1` y
  `scripts/script_instalador_wsl_jax_mujoco_playground.ps1`: nombres historicos
  que ahora redirigen al instalador mantenido.

La implementacion portable es JAX. `--impl warp` se rechaza deliberadamente
porque Warp esta ligado a NVIDIA y haria que el mismo codigo dejara de ser
portable entre perfiles.

## Diagnostico rapido

`doctor` comprueba el sistema operativo, que el runtime no este en `/mnt/c`, los
marcadores, comandos base, perfil solicitado, GPU visible, lock, entorno Python,
imports de JAX/MuJoCo/MJX y backend real:

```powershell
.\tarantulin.ps1 doctor
```

Si WSL no termina de instalarse, abre una vez `Ubuntu-24.04` desde Inicio para
crear usuario y contrasena. Si NVIDIA no aparece, prueba dentro de WSL
`nvidia-smi`; el driver se corrige en Windows. Si solo necesitas continuar con
el codigo, reinstala con `-Accelerator cpu`.
