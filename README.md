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
> [Sim2Real MJX-JAX sobre Linux/WSL](https://github.com/GuillermoRosale5/Sim2Real-MJX-JAX-sobre-Linux-WSL).

## Organización de las dos versiones

| Versión | Dónde guardamos y editamos el código | Dónde se calcula |
|---|---|---|
| **Este repositorio: Windows + WSL** | En una carpeta normal de Windows | En una copia automática dentro de WSL2 |
| **Ubuntu nativo / WSL directo** | En una única carpeta dentro de Ubuntu | En esa misma carpeta Linux |

## Puesta en marcha rápida

Este repositorio es privado. Primero iniciamos sesión en GitHub, descargamos el
ZIP, lo extraemos en una carpeta corta de Windows —por ejemplo
`C:\Sim2Real-Windows`— y hacemos doble clic en:

```text
INSTALAR_WINDOWS.cmd
```

No hay que instalar antes Git, Python, MuJoCo, JAX ni Ubuntu. Windows puede
pedir una confirmación de administrador para WSL y Ubuntu pedirá una vez el
nombre y la contraseña del usuario Linux. Si Windows obliga a reiniciar,
volvemos a ejecutar el mismo archivo después del reinicio.

Cuando termine, comprobamos el sistema:

```powershell
.\TARANTULIN.cmd doctor
.\TARANTULIN.cmd test-mjx -- --steps 10
```

Y podemos abrir directamente la red que dejamos preparada en el repositorio:

```powershell
.\TARANTULIN.cmd visualizar-red-preentrenada
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

La parte desarrollada específicamente para TARANTULIN utiliza nombres en
español: el entorno, el currículo de recompensas, los perfiles PPO, los
comandos de entrenamiento y sus lanzadores. Los nombres antiguos no se
mantienen mediante archivos puente. Si un archivo se ha renombrado, el nombre
anterior deja de existir. Algunos ejemplos son `entorno_tarantulin_mjx.py`,
`curriculo_recompensas.py`, la clase `TarantulinIncorporarse` y los perfiles
`depuracion`, `ligero`, `ligero_rapido` y `completo`.

MuJoCo, MJX, JAX, Brax y Orbax conservan sus nombres y sus interfaces
originales. Por eso dentro del código continúan apareciendo contratos como
`MjxEnv`, `default_config`, `reset`, `step`, las claves de configuración de
Brax PPO y el formato de checkpoint de Orbax. Mantener esta parte sin una
traducción artificial permite comparar nuestra implementación con los
repositorios de los que procede y actualizar las dependencias con menos riesgo.

Editamos siempre la carpeta Windows. Si abrimos `tarantulin.ps1 shell`, la usamos
solo para diagnóstico: la siguiente sincronización puede reemplazar cualquier
cambio de código hecho directamente en la copia WSL.

## Requisitos previos

- Windows 10 2004 o posterior, o Windows 11, sobre arquitectura x86_64.
- Virtualización habilitada en el equipo.
- Conexión a Internet y varios GB libres durante la primera instalación.
- Un driver NVIDIA de Windows compatible con WSL si se quiere entrenar con
  NVIDIA.
- Acceso autorizado a este repositorio privado para descargar el código.

No instalamos previamente Git, Python, MuJoCo, JAX, `uv`, CUDA Toolkit de Linux
ni creamos carpetas dentro de Ubuntu. El instalador prepara lo necesario. La
única parte interactiva que Windows no permite evitar es confirmar el aviso de
administrador y crear una vez el usuario y la contraseña de Ubuntu.

## Instalación desde cero

### Descarga del proyecto privado

Entramos en este repositorio con la cuenta autorizada, pulsamos **Code → Download
ZIP** y extraemos todo, por ejemplo en:

```text
C:\Sim2Real-Windows
```

No ejecutamos el instalador desde dentro del ZIP. Primero hay que extraer la
carpeta completa. Una ruta corta evita los límites de longitud de algunas
versiones de Windows.

El repositorio es privado y por eso no se ofrece un comando `irm ... | iex`:
ese comando fingiría que GitHub puede descargar el código sin iniciar sesión.

### Instalación automática

Hacemos doble clic en:

```text
INSTALAR_WINDOWS.cmd
```

También puede ejecutarse desde PowerShell:

```powershell
.\INSTALAR_WINDOWS.cmd
```

El instalador comprueba Windows, arquitectura y hardware; instala o actualiza
WSL, instala Ubuntu 24.04, crea la copia privada dentro del disco Linux, instala
las dependencias fijadas y comprueba el acelerador. `auto` exige NVIDIA cuando
Windows detecta una tarjeta NVIDIA. Así un driver defectuoso no queda oculto
detrás de una ejecución accidental por CPU.

### Reinicio y primer usuario de Ubuntu

En un Windows completamente limpio pueden hacer falta dos pasadas:

1. Ejecutamos `INSTALAR_WINDOWS.cmd`. Aceptamos el aviso de administrador.
2. Si Windows lo solicita, reiniciamos.
3. Abrimos `Ubuntu-24.04` una vez y creamos el usuario y la contraseña Linux.
   La contraseña no se muestra mientras se escribe; es normal.
4. Volvemos a ejecutar `INSTALAR_WINDOWS.cmd` desde la misma carpeta.

La segunda ejecución continúa sobre lo ya instalado. No vuelve a descargar lo
que está correcto ni borra los resultados. La creación de la contraseña Linux
no se automatiza porque pertenece a la seguridad del usuario.

### Selección explícita del acelerador

Normalmente dejamos `auto`. Para exigir un perfil concreto abrimos PowerShell en
la carpeta y usamos:

```powershell
.\INSTALAR_WINDOWS.cmd -Accelerator nvidia
.\INSTALAR_WINDOWS.cmd -Accelerator cpu
```

`nvidia` exige que la tarjeta sea visible desde WSL y que JAX termine sobre
backend `gpu`. `cpu` sirve para instalación, desarrollo y pruebas pequeñas, no
para entrenamiento masivo. AMD e Intel se dirigen de forma explícita a CPU en
WSL2; no se presenta esa ruta como aceleración GPU validada. La explicación
técnica está en [DEPENDENCIAS.md](DEPENDENCIAS.md).

### Alternativa con Git ya instalado

Git pedirá iniciar sesión porque el repositorio es privado:

```powershell
git -c core.longpaths=true clone https://github.com/GuillermoRosale5/PRIV_Tarantulin-Windows-WSL.git C:\Sim2Real-Windows
cd C:\Sim2Real-Windows
.\INSTALAR_WINDOWS.cmd
```

El script `scripts\bootstrap_windows.ps1` también puede clonar el repositorio e
instalar Git mediante `winget`, pero necesita igualmente la autenticación de
GitHub. No guarda tokens ni contraseñas dentro del proyecto.

## Comprobación de la instalación

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

## Perfiles PPO y fases de recompensa

Podemos consultar lo que existe sin memorizarlo:

```powershell
.\tarantulin.ps1 perfiles-ppo
.\tarantulin.ps1 fases-recompensa
```

Perfiles principales:

| Perfil | Pasos | Entornos | Para qué lo usamos |
|---|---:|---:|---|
| `depuracion` | 5 millones | 512 | Primer entrenamiento de prueba |
| `ligero` | 100 millones | 512 | Perfil principal actual |
| `ligero_rapido` | 100 millones | 1024 | Más entornos; necesita más memoria de GPU |
| `completo` | 50 millones | 512 | Red y episodios más grandes |

Incluso `depuracion` recorre 5 millones de pasos: sirve para probar el
entrenamiento, pero no debemos confundirlo con la prueba breve de `test-mjx`.

Fases de recompensa:

- `0`: base histórica, sin currículo; solo compatibilidad y pruebas.
- `1`: mantener la pose del XML.
- `2`: levantarse desde el suelo hasta la pose del XML.
- `3`: recuperarse de caídas y volver a una pose estable.

Indicamos siempre la fase en el comando para que no se elija la fase `0` por
descuido.

## Inicio del primer entrenamiento

Para una primera prueba:

```powershell
.\tarantulin.ps1 entrenar -- --segundo-plano --nombre-ejecucion primera-prueba --perfil-ppo depuracion --fase-recompensa 1 --desde-cero
```

Para repetir el caso principal de fase 2:

```powershell
.\tarantulin.ps1 entrenar -- --segundo-plano --nombre-ejecucion mi-prueba-fase-2 --perfil-ppo ligero --fase-recompensa 2 --seed 42 --desde-cero
```

`--segundo-plano` deja el entrenamiento funcionando en segundo plano. Solo puede
haber uno activo. Antes de empezar, el lanzador ejecuta automáticamente una
prueba MJX de 150 pasos.

El perfil `ligero` no es una prueba pequeña: está preparado para 100 millones de
pasos. Conviene usar un nombre distinto para cada ejecución desde cero. Si
reutilizamos un nombre junto con `--desde-cero`, se eliminan sus
checkpoints y se renuevan el registro, el estado y los archivos de configuración de
esa ejecución. No usamos esa opción cuando queremos continuarla.

## Seguimiento, visualización y parada

Estas tres cosas son distintas.

### Monitor de texto

```powershell
.\tarantulin.ps1 monitorizar
```

Lo abrimos en una segunda ventana de PowerShell y la dejamos visible. Enseña
pasos, recompensa, velocidad, GPU, memoria, temperatura y checkpoints. El
comando ocupa esa terminal hasta pulsar `Ctrl+C`; esto cierra únicamente el
monitor y no detiene el entrenamiento.

### Red preentrenada incluida: la opción recomendada

Para ver TARANTULIN funcionando sin depender de ningún entrenamiento local:

```powershell
.\tarantulin.ps1 visualizar-red-preentrenada
```

Este comando carga siempre la red de referencia incluida en el repositorio:

- fase de recompensa `2`;
- semilla `42`;
- checkpoint exacto del paso `45.932.544`;
- episodio de `1500` pasos.

La selección está fijada por el propio comando. `visualizar-red-preentrenada` **nunca lee
`logs_tarantulin_mjx/ultima_run.txt`** ni cambia porque hayamos empezado otro
entrenamiento. Antes de abrir el visor también comprueba la integridad del
paquete mediante `SHA256SUMS`. Es la opción adecuada para enseñar el movimiento
de referencia y para comprobar una instalación recién hecha.

Si queremos hacer toda la comprobación, incluida una simulación corta con JAX y
MJX, pero sin abrir todavía la ventana:

```powershell
.\tarantulin.ps1 visualizar-red-preentrenada -- --solo-comprobar
```

La red está avanzada, pero no representa un entrenamiento finalizado: el paso
45.932.544 corresponde al 45,93 % del objetivo de 100 millones de pasos del
perfil histórico `lite`, equivalente al perfil actual `ligero`. Su evaluación
guardada obtuvo una recompensa de `158.104767`.

### Último checkpoint local

Cuando hemos entrenado en este equipo y queremos inspeccionar el resultado más
reciente de esa ejecución:

```powershell
.\tarantulin.ps1 visualizar-resultados -- --longitud-episodio 1500
```

`visualizar-resultados` y `visualizar_ultimo_checkpoint.sh` buscan el
último checkpoint **local**, siguiendo `logs_tarantulin_mjx/ultima_run.txt`. Ese
checkpoint puede pertenecer a una prueba corta, a una ejecución interrumpida o
a un entrenamiento todavía parcial; por eso su movimiento puede ser mucho peor
que el de la red recomendada. Estos comandos nunca sustituyen ni actualizan la
red preentrenada publicada.

Ambos modos abren una ventana de MuJoCo mediante WSLg con un entorno
independiente. No es una cámara conectada a uno de los 512 entornos internos del
entrenamiento. Dentro del visor, `R` reinicia el episodio y cerrar la ventana
termina solo la visualización.

El menú de visualización reúne todos los modos y deja la red recomendada como
primera opción:

```powershell
.\tarantulin.ps1 shell
./scripts/visualizar_tarantulin.sh
```

Escribimos `exit` al volver a la shell. La opción 1 abre la red preentrenada; la
opción 2 abre el último checkpoint local, la 3 permite elegir uno local anterior
y la 4 muestra un XML sin simular.

El visor necesita memoria adicional. No abras varios visores y, en un equipo con
poca RAM, detén primero el entrenamiento:

```powershell
.\tarantulin.ps1 parar
.\tarantulin.ps1 visualizar-resultados -- --longitud-episodio 1500
```

También podemos probar poses concretas:

```powershell
.\tarantulin.ps1 minisimular -- --postura-inicial suelo2 --longitud-episodio 1500
.\tarantulin.ps1 minisimular -- --postura-inicial ideal --longitud-episodio 1500
.\tarantulin.ps1 minisimular -- --postura-inicial caida_lateral --longitud-episodio 1500
.\tarantulin.ps1 minisimular -- --postura-inicial boca_abajo --longitud-episodio 1500
```

### Detener el entrenamiento

```powershell
.\tarantulin.ps1 parar
```

La parada comprueba que el proceso pertenece realmente a la ejecución actual.
Los registros y checkpoints ya creados se conservan.

## Continuar una ejecución

Si queremos continuar `mi-prueba-fase-2`, mantenemos el nombre, el perfil, la
fase y la semilla originales, y no usamos `--desde-cero`:

```powershell
.\tarantulin.ps1 entrenar -- --segundo-plano --nombre-ejecucion mi-prueba-fase-2 --perfil-ppo ligero --fase-recompensa 2 --seed 42 --anexar-csv
```

También podemos crear una ejecución nueva tomando el último checkpoint detectado:

```powershell
.\tarantulin.ps1 entrenar -- --segundo-plano --continuar-ultimo --perfil-ppo ligero --fase-recompensa 2 --seed 42
```

`--continuar-ultimo` no adivina el perfil ni la fase originales: debemos indicarlos.
No combines una restauración con `--desde-cero`.

## Resultados y exportación

Los registros y checkpoints permanecen dentro de WSL porque allí el acceso es
más rápido. Para copiar la última ejecución a
`artifacts\logs_tarantulin_mjx` en Windows:

```powershell
.\tarantulin.ps1 pull-results
```

Para copiar todas:

```powershell
.\tarantulin.ps1 pull-results -- --all
```

La exportación no borra los originales de WSL. Si se ejecuta durante un
entrenamiento, copia los registros y las métricas, pero deja los checkpoints en
WSL para no copiar uno mientras Orbax todavía lo está escribiendo. Al terminar
o detener el entrenamiento repetimos `pull-results` y entonces se copian
también los checkpoints completos. `artifacts/` no se sube a GitHub.

## Gráficas y recompensas en directo

Para generar o abrir las gráficas históricas entramos temporalmente en la copia
de ejecución:

```powershell
.\tarantulin.ps1 shell
./scripts/graficar_recompensas.sh --mostrar
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
.\tarantulin.ps1 curriculo-automatico -- --perfil-ppo ligero --pasos-totales 200000000
```

Este supervisor permanece asociado a la terminal; hay que dejarla abierta
mientras controla los cambios de fase.

## Benchmark

El benchmark predeterminado es grande: en NVIDIA recorre 126 combinaciones. No
lo usamos como prueba rápida de instalación. Para eso está `test-mjx`.

Un ejemplo reducido sería:

```powershell
.\tarantulin.ps1 benchmark -- --nombre-ejecucion benchmark-corto --warmup-steps 16 --measure-steps 64 --envs "128 256 512" --precisions "high" --allocators "preallocate" --solver-pairs "12:4"
```

## Trabajo diario

Después de instalar, el recorrido habitual es:

```powershell
.\tarantulin.ps1 doctor
.\tarantulin.ps1 entrenar -- --segundo-plano --perfil-ppo ligero --fase-recompensa 2
.\tarantulin.ps1 monitorizar
```

Cuando queramos detenerlo antes de que termine:

```powershell
.\tarantulin.ps1 parar
.\tarantulin.ps1 pull-results
```

Los comandos de cálculo sincronizan automáticamente el código Windows antes de
ejecutarlo. `monitorizar`, `parar` y `pull-results` no sincronizan, para poder
observar o detener una ejecución sin cambiar nada mientras está funcionando.

## Actualizar o reparar la instalación

Primero detenemos cualquier entrenamiento. Después:

```powershell
.\tarantulin.ps1 parar
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

## Retirada manual de la instalación

Actualmente no existe un comando de desinstalación automática. Primero usamos
`pull-results`, después `parar` y finalmente `path` para identificar exactamente
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

`parar` detiene el entrenamiento, pero WSL puede seguir abierto por un monitor,
un visor, una terminal o una herramienta que trabaje sobre una ruta
`\\wsl.localhost`. Cierra esas ventanas y ejecuta:

```powershell
wsl --shutdown
```

Esta orden apaga todas las distribuciones y procesos WSL del usuario, no solo
TARANTULIN.

### Dice que ya hay un entrenamiento activo

Usamos `monitorizar` para observarlo o `parar` para detenerlo. No borramos archivos
PID a mano: el sistema comprueba la identidad del proceso antes de actuar.

### Falta el entorno Python

Repetimos `install.ps1` con el acelerador correcto. `-SyncOnly` no instala
Python ni las librerías.

### Hay demasiada memoria o swap ocupada

Cerramos visores adicionales y comenzamos con menos entornos, por ejemplo:

```powershell
.\tarantulin.ps1 entrenar -- --segundo-plano --nombre-ejecucion prueba-128 --perfil-ppo depuracion --fase-recompensa 1 --num-envs 128 --desde-cero
```

## Contenido conservado del proyecto

Se mantienen el entorno TARANTULIN, los XML, las recompensas, el currículo, los
hiperparámetros y PPO. Los nombres propios de esta capa se han cambiado de
verdad al español; no existen copias inglesas que actúen como puentes. La parte
de integración se encarga de la instalación, las rutas, la selección del
dispositivo y la seguridad de los procesos, respetando los contratos originales
de MuJoCo, MJX, JAX, Brax y Orbax.

No guardamos `.venv`, dependencias descargadas, logs ni los checkpoints
generados por cada entrenamiento en GitHub. El entorno se reconstruye desde el
lock; los logs y checkpoints locales se generan durante las ejecuciones y se
exportan por separado. La única excepción es el paquete revisado de
`pretrained/tarantulin_standup_fase2_45932544`: forma parte del sistema para que
todos podamos reproducir la misma demostración, está fijado al paso 45.932.544 y
lleva sus sumas SHA-256. MuJoCo Playground se instala siempre desde el commit fijo
`9c2dce4a3519cd4bb9d299bf28a6ef3f5086844b`.

## Resumen de comandos

```powershell
.\tarantulin.ps1 doctor
.\tarantulin.ps1 test-mjx -- --steps 10
.\tarantulin.ps1 visualizar-red-preentrenada
.\tarantulin.ps1 entrenar -- --segundo-plano --perfil-ppo ligero --fase-recompensa 2
.\tarantulin.ps1 monitorizar
.\tarantulin.ps1 visualizar-resultados -- --longitud-episodio 1500
.\tarantulin.ps1 parar
.\tarantulin.ps1 pull-results
.\tarantulin.ps1 path
.\tarantulin.ps1 help
```

Recordatorio importante: `visualizar-red-preentrenada` abre siempre la red
publicada de fase 2 y paso 45.932.544. `visualizar-resultados` abre el último
checkpoint creado en este equipo y ese resultado puede estar incompleto.

## Nota final

La idea es que podamos llevar este entorno a otro ordenador, instalarlo sin
reconstruir cada capa a mano y seguir usando los mismos scripts que ya teníamos.
La carpeta Windows es nuestro proyecto; WSL es el lugar donde se hace el trabajo
pesado.
