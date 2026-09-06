[CmdletBinding()]
param(
  [string]$RepoUrl = "https://github.com/GuillermoRosale5/PRIV_Tarantulin-Windows-WSL.git",
  [string]$Branch = "main",
  [string]$Destination = (Get-Location).Path,
  [ValidateSet("auto", "nvidia", "amd", "intel", "cpu")]
  [string]$Accelerator = "auto",
  [string]$Distro = "Ubuntu-24.04",
  [string]$RuntimePath = "",
  [switch]$NoSetup,
  [switch]$SkipSystemPackages,
  [switch]$SkipWslUpdate,
  [switch]$SkipGpuCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

trap {
  Write-Host ""
  Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
  exit 1
}

function Resolve-Git {
  $git = Get-Command git.exe -ErrorAction SilentlyContinue
  if ($git) { return $git.Source }
  $known = "C:\Program Files\Git\cmd\git.exe"
  if (Test-Path -LiteralPath $known) { return $known }
  $userInstall = Join-Path $env:LOCALAPPDATA "Programs\Git\cmd\git.exe"
  if (Test-Path -LiteralPath $userInstall) { return $userInstall }
  if (-not (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
    throw "Git no esta instalado y winget no esta disponible. Instala Git for Windows y repite."
  }
  Write-Host "==> Instalando Git for Windows" -ForegroundColor Cyan
  & winget.exe install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
  if ($LASTEXITCODE -ne 0) {
    throw "Git no se pudo instalar automaticamente. Instala Git for Windows y repite."
  }
  if (Test-Path -LiteralPath $known) { return $known }
  if (Test-Path -LiteralPath $userInstall) { return $userInstall }
  throw "Git se ha instalado, pero esta terminal aun no encuentra git.exe. Cierra PowerShell, vuelve a abrirla y repite."
}

$Destination = [IO.Path]::GetFullPath($Destination)
if (-not (Test-Path -LiteralPath $Destination)) {
  New-Item -ItemType Directory -Path $Destination | Out-Null
}
$marker = Join-Path $Destination ".tarantulin\source.marker"
if (-not (Test-Path -LiteralPath $marker)) {
  $items = @(Get-ChildItem -LiteralPath $Destination -Force)
  if ($items.Count -ne 0) {
    throw "La carpeta destino no esta vacia y no es una instalacion TARANTULIN: $Destination"
  }
  $git = Resolve-Git
  Write-Host "==> Descargando $RepoUrl ($Branch)" -ForegroundColor Cyan
  & $git -c core.longpaths=true clone --branch $Branch --single-branch $RepoUrl $Destination
  if ($LASTEXITCODE -ne 0) {
    throw @"
No se pudo descargar el repositorio privado (codigo $LASTEXITCODE).
Inicia sesion cuando Git Credential Manager abra el navegador, o descarga el
ZIP desde GitHub con tu cuenta y ejecuta INSTALAR_WINDOWS.cmd dentro de la
carpeta extraida. El instalador no puede saltarse la autenticacion de GitHub.
"@
  }
}

$installer = Join-Path $Destination "install.ps1"
$installParameters = @{
  Distro = $Distro
  Accelerator = $Accelerator
  NoSetup = $NoSetup
  SkipSystemPackages = $SkipSystemPackages
  SkipWslUpdate = $SkipWslUpdate
  SkipGpuCheck = $SkipGpuCheck
}
if ($RuntimePath) { $installParameters.RuntimePath = $RuntimePath }
& $installer @installParameters
exit $LASTEXITCODE
