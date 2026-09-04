[CmdletBinding()]
param(
  [string]$RepoUrl = "https://github.com/GuillermoRosale5/Tarantulin-Windows-WSL.git",
  [string]$Branch = "main",
  [string]$Destination = (Get-Location).Path,
  [ValidateSet("auto", "nvidia", "amd", "intel", "cpu")]
  [string]$Accelerator = "auto",
  [string]$Distro = "Ubuntu-24.04",
  [switch]$NoSetup
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-Git {
  $git = Get-Command git.exe -ErrorAction SilentlyContinue
  if ($git) { return $git.Source }
  $known = "C:\Program Files\Git\cmd\git.exe"
  if (Test-Path -LiteralPath $known) { return $known }
  if (-not (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
    throw "Git no esta instalado y winget no esta disponible. Instala Git for Windows y repite."
  }
  Write-Host "==> Instalando Git for Windows" -ForegroundColor Cyan
  & winget.exe install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $known)) {
    throw "Git no se pudo instalar automaticamente. Instala Git for Windows y repite."
  }
  return $known
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
  & $git clone --branch $Branch --single-branch $RepoUrl $Destination
  if ($LASTEXITCODE -ne 0) { throw "git clone fallo con codigo $LASTEXITCODE." }
}

$installer = Join-Path $Destination "install.ps1"
$installParameters = @{
  Distro = $Distro
  Accelerator = $Accelerator
  NoSetup = $NoSetup
}
& $installer @installParameters
exit $LASTEXITCODE
