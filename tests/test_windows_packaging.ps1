[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))

function Assert-Equal {
  param($Actual, $Expected, [string]$Message)
  if ($Actual -ne $Expected) {
    throw "$Message. Esperado='$Expected'; obtenido='$Actual'."
  }
}

function Assert-Contains {
  param([string]$Text, [string]$Expected, [string]$Message)
  if (-not $Text.Contains($Expected)) { throw "$Message. Falta: $Expected" }
}

$parseErrors = @()
Get-ChildItem -Path $repo -Recurse -Filter *.ps1 | ForEach-Object {
  $tokens = $null
  $errors = $null
  [void][Management.Automation.Language.Parser]::ParseFile($_.FullName, [ref]$tokens, [ref]$errors)
  $parseErrors += $errors
}
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }

$previousImportOnly = [Environment]::GetEnvironmentVariable("TARANTULIN_IMPORT_ONLY", "Process")
try {
  $env:TARANTULIN_IMPORT_ONLY = "1"
  . (Join-Path $repo "tarantulin.ps1")
}
finally {
  if ($null -eq $previousImportOnly) { Remove-Item Env:TARANTULIN_IMPORT_ONLY -ErrorAction SilentlyContinue }
  else { $env:TARANTULIN_IMPORT_ONLY = $previousImportOnly }
}

Assert-Equal (Resolve-WindowsAccelerator -Requested auto -GpuNames @("Intel Iris Xe", "NVIDIA RTX 4050")) nvidia "Auto no prioriza NVIDIA"
Assert-Equal (Resolve-WindowsAccelerator -Requested auto -GpuNames @("AMD Radeon 780M")) cpu "AMD WSL debe caer explicitamente a CPU"
Assert-Equal (Resolve-WindowsAccelerator -Requested auto -GpuNames @("Intel Arc A770")) cpu "Intel WSL debe caer explicitamente a CPU"
Assert-Equal (Resolve-WindowsAccelerator -Requested auto -GpuNames @("Adaptador desconocido")) auto "Hardware desconocido debe delegarse a WSL"
Assert-Equal (Resolve-WindowsAccelerator -Requested cpu -GpuNames @("NVIDIA RTX 4090")) cpu "Una seleccion explicita no se debe cambiar"

Assert-ValidDistroName "Ubuntu-24.04"
$unsafeRejected = $false
try { Assert-ValidDistroName "Ubuntu; Remove-Item C:\" }
catch { $unsafeRejected = $true }
if (-not $unsafeRejected) { throw "Se acepto un nombre de distribucion inseguro." }

$bootstrap = Get-Content -LiteralPath (Join-Path $repo "scripts\bootstrap_windows.ps1") -Raw
Assert-Contains $bootstrap "PRIV_Tarantulin-Windows-WSL.git" "El bootstrap no apunta al repositorio privado actual"
if ((Get-Content -LiteralPath (Join-Path $repo "tarantulin.ps1") -Raw).Contains("EnableExperimentalAmdWsl")) {
  throw "El lanzador conserva la opcion AMD/WSL que esta version no puede soportar."
}
if ($bootstrap.Contains("GuillermoRosale5/Tarantulin-Windows-WSL.git")) {
  throw "El bootstrap conserva la URL antigua del repositorio."
}

$installerCmd = Get-Content -LiteralPath (Join-Path $repo "INSTALAR_WINDOWS.cmd") -Raw
$launcherCmd = Get-Content -LiteralPath (Join-Path $repo "TARANTULIN.cmd") -Raw
Assert-Contains $installerCmd "-ExecutionPolicy Bypass" "El instalador CMD no evita el bloqueo local de scripts"
Assert-Contains $installerCmd "install.ps1" "El instalador CMD no llama al instalador principal"
Assert-Contains $launcherCmd "lanzador_tarantulin.ps1" "El lanzador CMD no llama al adaptador de argumentos"
$rawLauncher = Get-Content -LiteralPath (Join-Path $repo "scripts\windows\lanzador_tarantulin.ps1") -Raw
Assert-Contains $rawLauncher '$parametros.CommandArgs = [string[]]$argumentosLinux' "El adaptador no conserva los argumentos Linux"

$marker = (Get-Content -LiteralPath (Join-Path $repo ".tarantulin\source.marker") -Raw).Trim()
Assert-Equal $marker "tarantulin-windows-source-v1" "Marcador de fuente incorrecto"

$helpProcess = Start-Process -FilePath "powershell.exe" -ArgumentList @(
  "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
  (Join-Path $repo "tarantulin.ps1"), "help"
) -WindowStyle Hidden -Wait -PassThru
Assert-Equal $helpProcess.ExitCode 0 "El punto de entrada no puede mostrar ayuda sin preparar WSL"

Write-Host "WINDOWS_PACKAGING_TESTS_OK"
