# TARANTULIN · Windows + WSL

Esta versión está pensada para trabajar como lo hacemos en Windows: editamos y
guardamos el proyecto en una carpeta normal de Windows, pero MuJoCo, MJX y JAX
se ejecutan dentro de Ubuntu mediante WSL2.

La idea es sencilla: tú solo creas una carpeta vacía en Windows. El instalador
prepara la parte de WSL, crea allí su copia de ejecución y los scripts la
mantienen automáticamente. No hace falta copiar archivos a mano ni trabajar
dentro de Ubuntu.

> No hace falta instalar los dos repositorios de TARANTULIN. Elige este si
> quieres guardar y editar el código en Windows. Si quieres que todo viva
> directamente en Ubuntu, usa
> [Tarantulin-Linux-WSL](https://github.com/GuillermoRosale5/Tarantulin-Linux-WSL).

## Qué sistema estoy eligiendo

| Versión | Dónde guardamos y editamos el código | Dónde se calcula |
|---|---|---|
| **Este repositorio: Windows + WSL** | En una carpeta normal de Windows | En una copia automática dentro de WSL2 |
| **Ubuntu nativo / WSL directo** | En una única carpeta dentro de Ubuntu | En esa misma carpeta Linux |

## Si solo queremos empezar

Abrimos PowerShell en una carpeta Windows vacía y ejecutamos el instalador:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm 'https://raw.githubusercontent.com/GuillermoRosale5/Tarantulin-Windows-WSL/main/scripts/bootstrap_windows.ps1' | iex"
```

Cuando termine, comprobamos el sistema:

```powershell
.\tarantulin.ps1 doctor
.\tarantulin.ps1 test-mjx -- --steps 10
```

Ese es el recorrido mínimo. Las secciones siguientes explican qué sucede, qué
hacer si Windows pide reiniciar y cómo usar después el entrenamiento y el visor.

En esta versión hay dos carpetas, pero solo trabajamos sobre una:

```text
Carpeta de Windows que tú eliges
        │
        │  sincronización automática
        ▼
Copia de ejecución dentro de WSL2
        │
        └── MuJoCo + MJX + JAX + GPU/CPU
```

La carpeta Windows es la que se edita y se sube a GitHub. La copia Linux es una
zona de trabajo automática: contiene el entorno Python, los logs y los
checkpoints, pero no tenemos que mantenerla nosotros.

Editamos siempre la carpeta Windows. Si abrimos `tarantulin.ps1 shell`, la usamos
solo para diagnóstico: la siguiente sincronización puede reemplazar cualquier
cambio de código hecho directamente en la copia WSL.

## Qué necesitas antes de empezar

- Windows 10 u 11 de 64 bits con virtualización disponible.
- Conexión a Internet durante la primera instalación.
- Varios GB libres para Ubuntu, Python y las librerías.
- Permisos de administrador si Windows todavía tiene que instalar WSL.
- Si quieres usar NVIDIA, un driver NVIDIA de Windows compatible con WSL.

No necesitas instalar previamente Python, MuJoCo, JAX, `uv` ni crear una
carpeta dentro de Linux. El instalador se encarga.

Si no sabes qué acelerador elegir, usa `auto`. Elegirá NVIDIA si puede usarla de
verdad y, si no, preparará el modo CPU. Si sabes que tu equipo tiene NVIDIA y no
quieres que un problema de driver pase desapercibido, elige `nvidia` de forma
explícita.

## Instalación recomendada desde una carpeta vacía

### Crea la carpeta donde quieras trabajar

Por ejemplo:

```text
C:\Tarantulin
```

La carpeta debe estar vacía la primera vez. Ábrela en el Explorador de archivos,
escribe `powershell` en la barra de direcciones y pulsa Intro. Así PowerShell se
abre directamente en la carpeta correcta.

Para permitir los scripts solo durante esa ventana, sin cambiar la configuración
permanente de Windows:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

### Ejecuta el instalador

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm 'https://raw.githubusercontent.com/GuillermoRosale5/Tarantulin-Windows-WSL/main/scripts/bootstrap_windows.ps1' | iex"
```

Este comando hace todo el recorrido:

- instala Git con `winget` si todavía no está disponible;
- descarga este repositorio dentro de la carpeta actual;
- instala o reutiliza WSL2 y Ubuntu 24.04;
- crea automáticamente la carpeta de ejecución dentro de Linux;
- instala las librerías y crea el entorno Python;
- elige NVIDIA o CPU;
- deja preparados los mismos scripts que ya usamos en TARANTULIN.

El proceso puede tardar porque tiene que descargar Ubuntu y las dependencias.
No cierres la ventana mientras esté instalando.

Si quieres exigir NVIDIA desde el primer momento, usa esta variante:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "& ([scriptblock]::Create((irm 'https://raw.githubusercontent.com/GuillermoRosale5/Tarantulin-Windows-WSL/main/scripts/bootstrap_windows.ps1'))) -Accelerator nvidia"
```

### Si Windows pide reiniciar o Ubuntu pide un usuario

Es normal que una instalación completamente nueva se detenga una vez.

- Reinicia Windows si te lo solicita.
- Abre `Ubuntu-24.04` desde el menú Inicio.
- Crea el usuario y la contraseña de Linux. Mientras escribes la contraseña no
  aparecen caracteres en pantalla; es normal.
- Vuelve a la misma carpeta Windows y ejecuta otra vez el mismo comando.

El instalador reconoce lo que ya está hecho y continúa. No vuelve a empezar ni
borra la carpeta.

Si Windows reconoce el comando `wsl` pero Ubuntu todavía no está instalado, abre
una PowerShell como administrador y ejecuta:

```powershell
wsl --install -d Ubuntu-24.04
```

Después reinicia, abre Ubuntu una vez y repite el comando de instalación de
TARANTULIN desde la misma carpeta Windows. Si Windows no reconoce en absoluto el
comando `wsl`, primero hay que actualizar Windows o habilitar sus componentes de
WSL siguiendo la
[instalación oficial de Microsoft](https://learn.microsoft.com/windows/wsl/install).

### Alternativa si ya tenemos Git

Desde la carpeta Windows vacía podemos ver por separado la descarga y la
instalación:

```powershell
git clone https://github.com/GuillermoRosale5/Tarantulin-Windows-WSL.git .
.\install.ps1 -Accelerator nvidia
```

En un equipo sin NVIDIA sustituimos `nvidia` por `cpu`. El resultado es el mismo;
esta ruta simplemente deja el paso de Git visible.

## Si ya habías clonado el repositorio

Abre PowerShell dentro de la carpeta del proyecto y ejecuta:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\install.ps1 -Accelerator auto
```

Perfiles habituales:

```powershell
.\install.ps1 -Accelerator nvidia
.\install.ps1 -Accelerator cpu
```

`nvidia` obliga a que JAX encuentre una GPU NVIDIA; si no la encuentra, la
instalación falla en vez de continuar por CPU sin avisar. `cpu` sirve para
desarrollar, comprobar XML y hacer pruebas pequeñas, pero no para entrenamiento
masivo.

En WSL, AMD continúa siendo una ruta experimental e Intel GPU no es compatible
con las versiones de JAX usadas por este proyecto. La explicación técnica está
en [DEPENDENCIAS.md](DEPENDENCIAS.md).

## Cómo sabemos que la instalación ha terminado bien

Al final aparecerá un mensaje indicando que la instalación ha terminado.
Después ejecutamos las dos comprobaciones principales:

```powershell
.\tarantulin.ps1 doctor
.\tarantulin.ps1 test-mjx -- --steps 10
```

Si queremos ver además las rutas o consultar solo el motor de cálculo:

```powershell
.\tarantulin.ps1 path
.\tarantulin.ps1 backend
```

`doctor` revisa Windows, WSL2, la carpeta Linux, Python, las librerías y el
dispositivo que está usando JAX. Si elegimos NVIDIA debe indicar backend `gpu`;
si elegimos CPU debe indicar `cpu`.

`test-mjx` es quien carga el XML y hace una simulación corta con acciones cero,
aleatorias y extremas. El resultado que buscamos al final es:

```text
test-mjx OK.
```

La primera prueba puede tardar un poco aunque parezca parada: JAX está compilando
la simulación por primera vez.

## Antes de entrenar: perfiles y fases

Podemos consultar lo que existe sin memorizarlo:

```powershell
.\tarantulin.ps1 perfiles-ppo
.\tarantulin.ps1 fases-recompensa
```

Perfiles principales:

| Perfil | Pasos | Entornos | Para qué lo usamos |
|---|---:|---:|---|
| `debug` | 5 millones | 512 | Primer entrenamiento de prueba |
| `lite` | 100 millones | 512 | Perfil principal actual |
| `lite_fast` | 100 millones | 1024 | Más entornos; necesita más memoria de GPU |
| `full` | 50 millones | 512 | Red y episodios más grandes |

Incluso `debug` recorre 5 millones de pasos: sirve para probar el entrenamiento,
pero no debemos confundirlo con el smoke test corto de `test-mjx`.

Fases de recompensa:

- `0`: base histórica, sin currículo; solo compatibilidad y pruebas.
- `1`: mantener la pose del XML.
- `2`: levantarse desde el suelo hasta la pose del XML.
- `3`: recuperarse de caídas y volver a una pose estable.

Indicamos siempre la fase en el comando para que no se elija la fase `0` por
descuido.

## Primer entrenamiento

Para una primera prueba:

```powershell
.\tarantulin.ps1 train -- --background --run-name primera-prueba --perfil-ppo debug --fase-recompensa 1 --reset-checkpoint
```

Para repetir el caso principal de fase 2:

```powershell
.\tarantulin.ps1 train -- --background --run-name mi-prueba-fase-2 --perfil-ppo lite --fase-recompensa 2 --seed 42 --reset-checkpoint
```

`--background` deja el entrenamiento funcionando en segundo plano. Solo puede
haber uno activo. Antes de empezar, el lanzador ejecuta automáticamente una
prueba MJX de 150 pasos.

El perfil `lite` no es una prueba pequeña: está preparado para 100 millones de
pasos. Conviene usar un nombre distinto para cada ejecución desde cero. Si
reutilizamos un nombre junto con `--reset-checkpoint`, se eliminan sus
checkpoints y se renuevan el log, el estado y los archivos de configuración de
esa run. No usamos esa opción cuando queremos continuarla.

## Ver el progreso, ver el robot y detenerlo

Estas tres cosas son distintas.

### Monitor de texto

```powershell
.\tarantulin.ps1 monitor
```

Lo abrimos en una segunda ventana de PowerShell y la dejamos visible. Enseña
pasos, recompensa, velocidad, GPU, memoria, temperatura y checkpoints. El
comando ocupa esa terminal hasta pulsar `Ctrl+C`; esto cierra únicamente el
monitor y no detiene el entrenamiento.

### Ventana 3D de MuJoCo

Cuando ya existe al menos un checkpoint:

```powershell
.\tarantulin.ps1 view-results -- --episode-length 1500
```

Esto abre una ventana de MuJoCo mediante WSLg con un entorno independiente que
usa la última política guardada. No es una cámara conectada a uno de los 512
entornos internos del entrenamiento. Dentro del visor, `R` reinicia el episodio
y cerrar la ventana termina solo la visualización.

El visor necesita memoria adicional. No abras varios visores y, en un equipo con
poca RAM, detén primero el entrenamiento:

```powershell
.\tarantulin.ps1 stop
.\tarantulin.ps1 view-results -- --episode-length 1500
```

También podemos probar poses concretas:

```powershell
.\tarantulin.ps1 mini-sim -- --reset-preset suelo2 --episode-length 1500
.\tarantulin.ps1 mini-sim -- --reset-preset ideal --episode-length 1500
.\tarantulin.ps1 mini-sim -- --reset-preset caida_lateral --episode-length 1500
.\tarantulin.ps1 mini-sim -- --reset-preset boca_abajo --episode-length 1500
```

### Detener el entrenamiento

```powershell
.\tarantulin.ps1 stop
```

La parada comprueba que el proceso pertenece realmente a la ejecución actual.
Los logs y checkpoints ya creados se conservan.

## Continuar una ejecución

Si queremos continuar `mi-prueba-fase-2`, mantenemos el nombre, el perfil, la
fase y la semilla originales, y no usamos `--reset-checkpoint`:

```powershell
.\tarantulin.ps1 train -- --background --run-name mi-prueba-fase-2 --perfil-ppo lite --fase-recompensa 2 --seed 42 --append-csv
```

También podemos crear una ejecución nueva tomando el último checkpoint detectado:

```powershell
.\tarantulin.ps1 train -- --background --resume-latest --perfil-ppo lite --fase-recompensa 2 --seed 42
```

`--resume-latest` no adivina el perfil ni la fase originales: debemos indicarlos.
No combines una restauración con `--reset-checkpoint`.

## Dónde quedan los resultados

Mientras se entrena, logs y checkpoints permanecen dentro de WSL porque allí el
acceso es más rápido. Para copiar la última ejecución a
`artifacts\logs_tarantulin_mjx` en Windows:

```powershell
.\tarantulin.ps1 pull-results
```

Para copiar todas:

```powershell
.\tarantulin.ps1 pull-results -- --all
```

La exportación no borra los originales de WSL. Si ya existe en Windows un
archivo con el mismo nombre, la copia puede actualizarlo. `artifacts/` no se
sube a GitHub.

## Gráficas y recompensas en directo

Para generar o abrir las gráficas históricas entramos temporalmente en la copia
de ejecución:

```powershell
.\tarantulin.ps1 shell
./scripts/graficar_recompensas.sh --show
```

Para observar las recompensas mientras se entrena:

```powershell
.\tarantulin.ps1 shell
./scripts/ver_recompensas_en_directo.sh
```

Escribimos `exit` para volver a PowerShell. No editamos código dentro de esa
shell.

## Currículo automático

El supervisor de fases que ya teníamos también sigue disponible:

```powershell
.\tarantulin.ps1 curriculum-auto -- --perfil-ppo lite --total-steps 200000000
```

Este supervisor permanece asociado a la terminal; hay que dejarla abierta
mientras controla los cambios de fase.

## Benchmark

El benchmark predeterminado es grande: en NVIDIA recorre 126 combinaciones. No
lo usamos como prueba rápida de instalación. Para eso está `test-mjx`.

Un ejemplo reducido sería:

```powershell
.\tarantulin.ps1 benchmark -- --run-name benchmark-corto --warmup-steps 16 --measure-steps 64 --envs "128 256 512" --precisions "high" --allocators "preallocate" --solver-pairs "12:4"
```

## El trabajo normal del día a día

Después de instalar, el recorrido habitual es:

```powershell
.\tarantulin.ps1 doctor
.\tarantulin.ps1 train -- --background --perfil-ppo lite --fase-recompensa 2
.\tarantulin.ps1 monitor
```

Cuando queramos detenerlo antes de que termine:

```powershell
.\tarantulin.ps1 stop
.\tarantulin.ps1 pull-results
```

Los comandos de cálculo sincronizan automáticamente el código Windows antes de
ejecutarlo. `monitor`, `stop` y `pull-results` no sincronizan, para poder observar
o detener una ejecución sin cambiar nada mientras está funcionando.

## Actualizar o reparar la instalación

Primero detenemos cualquier entrenamiento. Después:

```powershell
.\tarantulin.ps1 stop
git status --short
git pull --ff-only
.\install.ps1 -Accelerator nvidia -SkipSystemPackages
.\tarantulin.ps1 doctor
```

Usamos exactamente el acelerador con el que queramos mantener el runtime:
`nvidia` o `cpu`. Si `git status --short` muestra cambios propios, los guardamos
o revisamos antes de hacer `git pull`. El instalador se puede repetir y no borra
logs ni checkpoints.

Opciones de reparación:

```powershell
.\install.ps1 -NoSetup
.\install.ps1 -SyncOnly
.\tarantulin.ps1 sync -DryRunSync
```

`-NoSetup` prepara y sincroniza la copia Linux sin crear el entorno Python.
`-SyncOnly` sirve solo para un runtime que ya estaba inicializado. `-DryRunSync`
enseña qué se copiaría sin cambiarlo.

Si movemos o renombramos la carpeta Windows, su identificador cambia. Ejecutamos
otra vez `install.ps1`; se creará un runtime nuevo y el anterior se conservará
para no perder datos.

## Si algún día queremos retirar esta instalación

Actualmente no existe un comando de desinstalación automática. Primero usamos
`pull-results`, después `stop` y finalmente `path` para identificar exactamente
el runtime asociado a esta carpeta. Cerramos terminales y visores, ejecutamos
`wsl --shutdown` y solo entonces eliminamos manualmente ese runtime concreto si
estamos seguros de no necesitarlo.

No usamos `wsl --unregister Ubuntu-24.04` para retirar TARANTULIN: esa orden
borra toda la distribución Ubuntu y todos los demás datos que contenga.

## Problemas frecuentes

### PowerShell bloquea los scripts

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

Solo afecta a la ventana actual.

### Ubuntu no termina de arrancar

Abre `Ubuntu-24.04` una vez desde Inicio, termina de crear el usuario y vuelve a
ejecutar `install.ps1`.

### NVIDIA no aparece

Actualiza el driver NVIDIA de Windows y comprueba:

```powershell
wsl -d Ubuntu-24.04 -- nvidia-smi
```

No instales un segundo driver NVIDIA de Linux dentro de WSL. Para continuar sin
GPU, reinstala con `-Accelerator cpu`.

### No se abre el visor

Primero comprueba que ya existe un checkpoint. Después actualiza WSLg desde una
PowerShell y vuelve a abrir Ubuntu:

```powershell
wsl --update
wsl --shutdown
```

### WSL sigue apareciendo después de parar

`stop` detiene el entrenamiento, pero WSL puede seguir abierto por un monitor,
un visor, una terminal o una herramienta que trabaje sobre una ruta
`\\wsl.localhost`. Cierra esas ventanas y ejecuta:

```powershell
wsl --shutdown
```

Esta orden apaga todas las distribuciones y procesos WSL del usuario, no solo
TARANTULIN.

### Dice que ya hay un entrenamiento activo

Usamos `monitor` para observarlo o `stop` para detenerlo. No borramos archivos
PID a mano: el sistema comprueba la identidad del proceso antes de actuar.

### Falta el entorno Python

Repetimos `install.ps1` con el acelerador correcto. `-SyncOnly` no instala
Python ni las librerías.

### Hay demasiada memoria o swap ocupada

Cerramos visores adicionales y comenzamos con menos entornos, por ejemplo:

```powershell
.\tarantulin.ps1 train -- --background --run-name prueba-128 --perfil-ppo debug --fase-recompensa 1 --num-envs 128 --reset-checkpoint
```

## Qué se conserva del proyecto anterior

Se mantienen el entorno TARANTULIN, los XML, las recompensas, el currículo, los
hiperparámetros, PPO y los nombres de los scripts históricos. La parte nueva se
encarga de la instalación, las rutas, la selección del dispositivo y la
seguridad de los procesos.

No guardamos `.venv`, dependencias descargadas, logs ni checkpoints en GitHub
porque pesan mucho. El entorno se reconstruye desde el lock; los logs y
checkpoints se generan durante las ejecuciones y se exportan por separado.
MuJoCo Playground se instala siempre desde el commit fijo
`9c2dce4a3519cd4bb9d299bf28a6ef3f5086844b`.

## Comandos que conviene recordar

```powershell
.\tarantulin.ps1 doctor
.\tarantulin.ps1 test-mjx -- --steps 10
.\tarantulin.ps1 train -- --background --perfil-ppo lite --fase-recompensa 2
.\tarantulin.ps1 monitor
.\tarantulin.ps1 view-results -- --episode-length 1500
.\tarantulin.ps1 stop
.\tarantulin.ps1 pull-results
.\tarantulin.ps1 path
.\tarantulin.ps1 help
```

## Nota final

La idea es que podamos llevar este entorno a otro ordenador, instalarlo sin
reconstruir cada capa a mano y seguir usando los mismos scripts que ya teníamos.
La carpeta Windows es nuestro proyecto; WSL es el lugar donde se hace el trabajo
pesado.
