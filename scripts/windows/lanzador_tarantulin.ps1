# Este archivo no declara param(): necesitamos recibir literalmente opciones
# Linux como --steps o --fase-recompensa sin que PowerShell intente enlazarlas
# como parametros propios.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$argumentos = @($args)
$comando = if ($argumentos.Count -gt 0) { [string]$argumentos[0] } else { "help" }
$argumentosLinux = @()
if ($argumentos.Count -gt 1) {
  $argumentosLinux = @($argumentos[1..($argumentos.Count - 1)])
}
if ($argumentosLinux.Count -gt 0 -and $argumentosLinux[0] -eq "--") {
  if ($argumentosLinux.Count -gt 1) { $argumentosLinux = @($argumentosLinux[1..($argumentosLinux.Count - 1)]) }
  else { $argumentosLinux = @() }
}

$raiz = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$entrada = Join-Path $raiz "tarantulin.ps1"
$parametros = @{ Command = $comando }
if ($argumentosLinux.Count -gt 0) {
  $parametros.CommandArgs = [string[]]$argumentosLinux
}
& $entrada @parametros
exit $LASTEXITCODE
