[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [string]$Command = "help",

  [string]$Distro = "",

  [ValidateSet("auto", "nvidia", "amd", "intel", "cpu")]
  [string]$Accelerator = "auto",

  [string]$RuntimePath = "",
  [switch]$NoSetup,
  [switch]$SyncOnly,
  [switch]$SkipSystemPackages,
  [switch]$SkipGpuCheck,
  [switch]$EnableExperimentalAmdWsl,
  [switch]$DryRunSync,

  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$CommandArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
  param([Parameter(Mandatory = $true)][string]$Message)
  Write-Host ""
  Write-Host "==> $Message" -ForegroundColor Cyan
}

function Show-Help {
  @"
TARANTULIN para Windows + WSL

La carpeta Windows es la fuente de verdad. Cada orden sincroniza el codigo a
un runtime privado y rapido dentro del filesystem ext4 de WSL.

Primer uso:
  .\install.ps1
  .\install.ps1 -Accelerator nvidia
  .\install.ps1 -Accelerator cpu

Uso diario:
  .\tarantulin.ps1 doctor
  .\tarantulin.ps1 test-mjx -- --steps 10
  .\tarantulin.ps1 entrenar -- --segundo-plano --perfil-ppo ligero --fase-recompensa 1
  .\tarantulin.ps1 monitorizar
  .\tarantulin.ps1 visualizar-red-preentrenada
  .\tarantulin.ps1 visualizar-resultados -- --longitud-episodio 1500
  .\tarantulin.ps1 parar
  .\tarantulin.ps1 pull-results

Mantenimiento:
  .\tarantulin.ps1 sync
  .\tarantulin.ps1 path
  .\tarantulin.ps1 shell

Opciones PowerShell:
  -Distro Ubuntu-24.04
  -Accelerator auto|nvidia|amd|intel|cpu
  -RuntimePath ~/.local/share/tarantulin-windows/mi-runtime
  -NoSetup -SyncOnly -SkipSystemPackages -SkipGpuCheck -DryRunSync
  -EnableExperimentalAmdWsl  (no recomendado; AMD no esta validado en WSL)

Para separar los argumentos Linux de opciones PowerShell se puede usar --:
  .\tarantulin.ps1 entrenar -- --segundo-plano --num-envs 512 --fase-recompensa 1
"@ | Write-Host
}

function Test-NativeCommand {
  param([Parameter(Mandatory = $true)][string]$Name)
  return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Invoke-NativeChecked {
  param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string[]]$Arguments
  )
  & $Executable @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "El comando '$Executable' termino con codigo $LASTEXITCODE."
  }
}

function Get-SourceId {
  param([Parameter(Mandatory = $true)][string]$Path)
  $normalized = ([IO.Path]::GetFullPath($Path)).TrimEnd('\').ToLowerInvariant()
  $bytes = [Text.Encoding]::UTF8.GetBytes($normalized)
  $sha = [Security.Cryptography.SHA256]::Create()
  try {
    $hex = -join ($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString("x2") })
  }
  finally {
    $sha.Dispose()
  }
  return $hex.Substring(0, 16)
}

function ConvertTo-WslPath {
  param(
    [Parameter(Mandatory = $true)][string]$WindowsPath,
    [Parameter(Mandatory = $true)][string]$RequestedDistro
  )
  $full = [IO.Path]::GetFullPath($WindowsPath)
  $previousWslEnv = [Environment]::GetEnvironmentVariable("WSLENV", "Process")
  $previousSource = [Environment]::GetEnvironmentVariable("TARANTULIN_WINDOWS_SOURCE", "Process")
  try {
    $env:TARANTULIN_WINDOWS_SOURCE = $full
    if ($previousWslEnv) {
      $env:WSLENV = "$previousWslEnv`:TARANTULIN_WINDOWS_SOURCE/p"
    }
    else {
      $env:WSLENV = "TARANTULIN_WINDOWS_SOURCE/p"
    }
    $translatedOutput = @(& wsl.exe -d $RequestedDistro -- printenv TARANTULIN_WINDOWS_SOURCE 2>$null)
    $translated = if ($translatedOutput.Count) { [string]$translatedOutput[-1] } else { "" }
    $translated = $translated.Trim()
    if ($LASTEXITCODE -ne 0 -or -not $translated.StartsWith("/")) {
      throw "WSL no pudo traducir la ruta Windows: $full"
    }
    return $translated
  }
  finally {
    if ($null -eq $previousWslEnv) { Remove-Item Env:WSLENV -ErrorAction SilentlyContinue }
    else { $env:WSLENV = $previousWslEnv }
    if ($null -eq $previousSource) { Remove-Item Env:TARANTULIN_WINDOWS_SOURCE -ErrorAction SilentlyContinue }
    else { $env:TARANTULIN_WINDOWS_SOURCE = $previousSource }
  }
}

function Get-InstalledDistros {
  $raw = & wsl.exe --list --quiet 2>$null
  if ($LASTEXITCODE -ne 0) { return @() }
  return @($raw | ForEach-Object { ($_ -replace "`0", "").Trim() } | Where-Object { $_ })
}

function Get-WslDistroVersion {
  param([Parameter(Mandatory = $true)][string]$RequestedDistro)
  $raw = @(& wsl.exe --list --verbose 2>$null)
  if ($LASTEXITCODE -ne 0) { return $null }
  $escaped = [Regex]::Escape($RequestedDistro)
  foreach ($line in $raw) {
    $clean = ([string]$line -replace "`0", "").Trim()
    if ($clean -match "^\*?\s*$escaped\s+\S+\s+([12])$") {
      return [int]$Matches[1]
    }
  }
  return $null
}

function Ensure-Wsl2 {
  param([Parameter(Mandatory = $true)][string]$RequestedDistro)
  $version = Get-WslDistroVersion -RequestedDistro $RequestedDistro
  if ($null -eq $version) {
    throw "No se pudo leer la version WSL de $RequestedDistro mediante 'wsl --list --verbose'."
  }
  if ($version -eq 2) { return }
  Write-Step "Convirtiendo $RequestedDistro de WSL 1 a WSL 2"
  & wsl.exe --set-version $RequestedDistro 2
  if ($LASTEXITCODE -ne 0) {
    throw "No se pudo convertir $RequestedDistro a WSL 2. Actualiza WSL y repite como administrador."
  }
}

function Install-WslIfNeeded {
  param([Parameter(Mandatory = $true)][string]$RequestedDistro)
  if (-not (Test-NativeCommand "wsl.exe")) {
    throw "WSL no esta disponible. Abre PowerShell como administrador, ejecuta 'wsl --install -d $RequestedDistro', reinicia si se solicita y repite la instalacion."
  }

  $distros = Get-InstalledDistros
  if ($distros -contains $RequestedDistro) { return }

  Write-Step "Instalando WSL y $RequestedDistro"
  & wsl.exe --install -d $RequestedDistro
  if ($LASTEXITCODE -ne 0) {
    throw "No se pudo instalar $RequestedDistro automaticamente. Ejecuta PowerShell como administrador y repite."
  }
  Write-Host ""
  Write-Host "WSL ha iniciado la instalacion. Reinicia Windows si lo pide, abre $RequestedDistro una vez para crear el usuario Linux y vuelve a ejecutar .\install.ps1." -ForegroundColor Yellow
  exit 0
}

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
  throw "Este lanzador corresponde al repositorio Windows y debe ejecutarse en PowerShell sobre Windows."
}

$projectRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
$sourceMarker = Join-Path $projectRoot ".tarantulin\source.marker"
if (-not (Test-Path -LiteralPath $sourceMarker -PathType Leaf)) {
  throw "Falta el marcador de fuente '$sourceMarker'. No es seguro sincronizar esta carpeta."
}
if ((Get-Content -LiteralPath $sourceMarker -Raw).Trim() -ne "tarantulin-windows-source-v1") {
  throw "El marcador de fuente no pertenece a TARANTULIN Windows."
}

$configPath = Join-Path $projectRoot ".tarantulin\windows.json"
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if (-not $Distro) { $Distro = [string]$config.distro }
if ($Accelerator -eq "auto" -and $config.accelerator -and $config.accelerator -ne "auto") {
  $Accelerator = [string]$config.accelerator
}
$runtimeBase = [string]$config.runtimeBase
$sourceId = Get-SourceId -Path $projectRoot
if (-not $RuntimePath) { $RuntimePath = "$runtimeBase/$sourceId" }

if ($Command -in @("help", "-h", "--help")) {
  Show-Help
  exit 0
}

Install-WslIfNeeded -RequestedDistro $Distro
Ensure-Wsl2 -RequestedDistro $Distro

$wslHomeOutput = @(& wsl.exe -d $Distro -- printenv HOME 2>$null)
$wslHome = if ($wslHomeOutput.Count) { [string]$wslHomeOutput[-1] } else { "" }
$wslHome = $wslHome.Trim()
if ($LASTEXITCODE -ne 0 -or -not $wslHome.StartsWith("/")) {
  throw "No se pudo determinar HOME dentro de la distro $Distro. Abrela una vez para completar la creacion del usuario Linux."
}
if ($RuntimePath -eq "~") {
  $RuntimePath = $wslHome
}
elseif ($RuntimePath.StartsWith("~/")) {
  $RuntimePath = "$wslHome/$($RuntimePath.Substring(2))"
}
elseif (-not $RuntimePath.StartsWith("/")) {
  throw "RuntimePath debe ser una ruta Linux absoluta o comenzar por ~/ : $RuntimePath"
}

$sourceWsl = ConvertTo-WslPath -WindowsPath $projectRoot -RequestedDistro $Distro
$syncScript = "$sourceWsl/scripts/wsl/sync_runtime.sh"
$bootstrapScript = "$sourceWsl/scripts/wsl/bootstrap_runtime.sh"
$runnerScript = "$sourceWsl/scripts/wsl/run_runtime.sh"
$workspacePath = "$RuntimePath/workspace"

if ($Command -eq "path") {
  Write-Host "Fuente Windows : $projectRoot"
  Write-Host "Fuente en WSL  : $sourceWsl"
  Write-Host "Distro         : $Distro"
  Write-Host "Runtime WSL    : $RuntimePath"
  Write-Host "Workspace WSL  : $workspacePath"
  Write-Host "Acelerador     : $Accelerator"
  exit 0
}

if ($Command -eq "install") {
  $bootstrapArgs = @(
    "-d", $Distro, "--", "bash", $bootstrapScript,
    "--source", $sourceWsl,
    "--runtime", $RuntimePath,
    "--source-id", $sourceId,
    "--accelerator", $Accelerator
  )
  if ($NoSetup) { $bootstrapArgs += "--no-setup" }
  if ($SyncOnly) { $bootstrapArgs += "--sync-only" }
  if ($SkipSystemPackages) { $bootstrapArgs += "--skip-system-packages" }
  if ($SkipGpuCheck) { $bootstrapArgs += "--skip-gpu-check" }
  if ($EnableExperimentalAmdWsl) { $bootstrapArgs += "--enable-experimental-amd-wsl" }
  Write-Step "Preparando runtime WSL aislado"
  Invoke-NativeChecked -Executable "wsl.exe" -Arguments $bootstrapArgs
  Write-Host ""
  Write-Host "Instalacion terminada." -ForegroundColor Green
  Write-Host "Comprueba el sistema con: .\tarantulin.ps1 doctor"
  exit 0
}

if ($Command -eq "sync") {
  $syncArgs = @(
    "-d", $Distro, "--", "bash", $syncScript,
    "--source", $sourceWsl,
    "--runtime", $RuntimePath,
    "--source-id", $sourceId
  )
  if ($DryRunSync) { $syncArgs += "--dry-run" }
  Write-Step "Sincronizando codigo Windows -> WSL"
  Invoke-NativeChecked -Executable "wsl.exe" -Arguments $syncArgs
  exit 0
}

if ($Command -eq "shell") {
  Write-Step "Sincronizando antes de abrir WSL"
  Invoke-NativeChecked -Executable "wsl.exe" -Arguments @(
    "-d", $Distro, "--", "bash", $syncScript,
    "--source", $sourceWsl, "--runtime", $RuntimePath, "--source-id", $sourceId
  )
  & wsl.exe -d $Distro --cd $workspacePath -- bash
  exit $LASTEXITCODE
}

$commandsWithoutAutoSync = @("monitorizar", "parar", "pull-results")
if ($Command -notin $commandsWithoutAutoSync) {
  Write-Step "Sincronizando codigo Windows -> WSL"
  Invoke-NativeChecked -Executable "wsl.exe" -Arguments @(
    "-d", $Distro, "--", "bash", $syncScript,
    "--source", $sourceWsl, "--runtime", $RuntimePath, "--source-id", $sourceId
  )
}

$linuxArgs = @(
  "-d", $Distro, "--", "bash", $runnerScript,
  "--source", $sourceWsl,
  "--runtime", $RuntimePath,
  "--source-id", $sourceId,
  "--accelerator", $Accelerator
)
if ($SkipGpuCheck) { $linuxArgs += "--skip-gpu-check" }
if ($EnableExperimentalAmdWsl) { $linuxArgs += "--enable-experimental-amd-wsl" }
$linuxArgs += "--"
$linuxArgs += $Command
if ($CommandArgs) { $linuxArgs += $CommandArgs }

Invoke-NativeChecked -Executable "wsl.exe" -Arguments $linuxArgs
