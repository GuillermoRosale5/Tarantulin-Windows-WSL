# Red preentrenada TARANTULIN · fase 2

Esta es la red de referencia incluida con TARANTULIN para poder comprobar el
robot sin entrenar desde cero.

- Perfil PPO: `lite`.
- Fase de recompensa: `2`, levantarse desde el suelo.
- Semilla: `42`.
- Checkpoint: `45.932.544` pasos de un objetivo de 100 millones.
- Recompensa de evaluación guardada: `158,104767`.
- Longitud media de evaluación: `1.432,25` pasos.
- Formato: Orbax OCDBT.

No se presenta como una red terminada al 100 %. Es la red más avanzada y estable
que se conserva actualmente y sustituye como referencia visual al checkpoint
nuevo de 1.548.288 pasos, que todavía estaba demasiado poco entrenado.

El paquete está preparado para inferencia y visualización: conserva los pesos y
las estadísticas de normalización de la política. No incluye un estado completo
del optimizador que garantice continuar el entrenamiento bit a bit.

Desde Windows + WSL:

```powershell
.\tarantulin.ps1 view-pretrained
```

Desde Ubuntu o WSL directo:

```bash
./scripts/visualizar_red_preentrenada.sh
```

El comando comprueba `SHA256SUMS` antes de cargar los pesos. Esta carpeta sí se
versiona de forma deliberada; los checkpoints generados por entrenamientos
normales continúan ignorados por Git.
