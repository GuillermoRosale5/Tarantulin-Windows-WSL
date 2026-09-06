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
  [switch]$SkipWslUpdate,
  [switch]$SkipGpuCheck,
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
  .\tarantulin.ps1 diagnostico-windows
  .\tarantulin.ps1 sync
  .\tarantulin.ps1 path
  .\tarantulin.ps1 shell

Opciones PowerShell:
  -Distro Ubuntu-24.04
  -Accelerator auto|nvidia|cpu
  -RuntimePath ~/.local/share/tarantulin-windows/mi-runtime
  -NoSetup -SyncOnly -SkipSystemPackages -SkipWslUpdate -SkipGpuCheck -DryRunSync

Para separar los argumentos Linux de opciones PowerShell se puede usar --:
  .\tarantulin.ps1 entrenar -- --segundo-plano --num-envs 512 --fase-recompensa 1
"@ | Write-Host
}

function Assert-ValidDistroName {
  param([Parameter(Mandatory = $true)][string]$Name)
  if ($Name -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$') {
    throw "Nombre de distribucion WSL no valido: '$Name'."
  }
}

function Get-WindowsGpuNames {
  try {
    return @(Get-CimInstance Win32_VideoController -ErrorAction Stop |
      ForEach-Object { [string]$_.Name } |
      Where-Object { $_ } |
      Sort-Object -Unique)
  }
  catch {
    Write-Warning "No se pudo consultar el hardware grafico de Windows: $($_.Exception.Message)"
    return @()
  }
}

function Resolve-WindowsAccelerator {
  param(
    [Parameter(Mandatory = $true)][string]$Requested,
    [string[]]$GpuNames = @()
  )
  if ($Requested -ne "auto") { return $Requested }
  $description = ($GpuNames -join "`n").ToLowerInvariant()
  if ($description -match 'nvidia') { return "nvidia" }
  if ($description -match 'amd|radeon|advanced micro devices') {
    return "cpu"
  }
  if ($description -match 'intel') { return "cpu" }
  # Si CIM no identifica la grafica dejamos que WSL resuelva el perfil. De este
  # modo no descartamos una NVIDIA que si sea visible mediante nvidia-smi.
  return "auto"
}

function Show-WindowsCompatibility {
  param(
    [string[]]$GpuNames = @(),
    [string]$SelectedAccelerator = "auto"
  )
  Write-Host "Graficas detectadas en Windows:"
  if ($GpuNames.Count -eq 0) { Write-Host "  - no se pudieron consultar" -ForegroundColor Yellow }
  else { $GpuNames | ForEach-Object { Write-Host "  - $_" } }
  Write-Host "Acelerador elegido para WSL: $SelectedAccelerator"
  if ($SelectedAccelerator -in @("amd", "intel")) {
    Write-Host "La GPU elegida no tiene una ruta JAX/MJX soportada en WSL2. Selecciona cpu de forma explicita." -ForegroundColor Red
    return
  }
  if ($SelectedAccelerator -eq "cpu" -and (($GpuNames -join ' ') -match '(?i)AMD|Radeon|Intel')) {
    Write-Host "La GPU detectada no tiene una ruta JAX/MJX validada en WSL2; se usara CPU de forma explicita." -ForegroundColor Yellow
  }
}

function Get-WslExecutable {
  $command = Get-Command wsl.exe -ErrorAction SilentlyContinue
  if ($command) { return $command.Source }
  $systemWsl = Join-Path $env:SystemRoot "System32\wsl.exe"
  if (Test-Path -LiteralPath $systemWsl -PathType Leaf) { return $systemWsl }
  return $null
}

function Test-IsAdministrator {
  $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
  $principal = [Security.Principal.WindowsPrincipal]::new($identity)
  return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-ElevatedAndWait {
  param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string[]]$Arguments,
    [Parameter(Mandatory = $true)][string]$Description,
    [int[]]$AllowedExitCodes = @(0)
  )
  Write-Host "Windows mostrara una confirmacion de administrador para $Description." -ForegroundColor Yellow
  try {
    $process = Start-Process -FilePath $Executable -ArgumentList $Arguments -Verb RunAs -Wait -PassThru
  }
  catch {
    throw "No se concedieron permisos de administrador para $Description. $($_.Exception.Message)"
  }
  if ($process.ExitCode -notin $AllowedExitCodes) {
    throw "$Description termino con codigo $($process.ExitCode)."
  }
}

function Ensure-WslCommand {
  $wsl = Get-WslExecutable
  if ($wsl) { return $wsl }
  $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
  if (-not $winget) {
    throw "Este Windows no incluye todavia WSL ni winget. Actualiza Windows 10/11 y vuelve a ejecutar el instalador."
  }
  Write-Step "Instalando el componente actual de WSL"
  $arguments = @(
    "install", "--id", "Microsoft.WSL", "--exact", "--source", "winget",
    "--accept-package-agreements", "--accept-source-agreements"
  )
  if (Test-IsAdministrator) {
    & $winget.Source @arguments
    if ($LASTEXITCODE -notin @(0, 1641, 3010)) { throw "winget no pudo instalar Microsoft.WSL (codigo $LASTEXITCODE)." }
  }
  else {
    Invoke-ElevatedAndWait -Executable $winget.Source -Arguments $arguments -Description "instalar WSL" -AllowedExitCodes @(0, 1641, 3010)
  }
  $wsl = Get-WslExecutable
  if (-not $wsl) {
    Write-Host "WSL se ha instalado, pero Windows necesita reiniciarse antes de continuar." -ForegroundColor Yellow
    Write-Host "Reinicia y vuelve a ejecutar INSTALAR_WINDOWS.cmd desde esta misma carpeta." -ForegroundColor Yellow
    return $null
  }
  return $wsl
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
    $translatedOutput = @(& $script:WslExecutable -d $RequestedDistro -- printenv TARANTULIN_WINDOWS_SOURCE 2>$null)
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

function Resolve-WslLinuxPath {
  param(
    [Parameter(Mandatory = $true)][string]$LinuxPath,
    [Parameter(Mandatory = $true)][string]$RequestedDistro
  )
  $output = @(& $script:WslExecutable -d $RequestedDistro -- realpath -m -- $LinuxPath 2>$null)
  $resolved = if ($output.Count) { [string]$output[-1] } else { "" }
  $resolved = $resolved.Trim()
  if ($LASTEXITCODE -ne 0 -or -not $resolved.StartsWith("/")) {
    throw "WSL no pudo normalizar la ruta Linux: $LinuxPath"
  }
  return $resolved
}

function Get-InstalledDistros {
  $raw = & $script:WslExecutable --list --quiet 2>$null
  if ($LASTEXITCODE -ne 0) { return @() }
  return @($raw | ForEach-Object { ($_ -replace "`0", "").Trim() } | Where-Object { $_ })
}

function Get-WslDistroVersion {
  param([Parameter(Mandatory = $true)][string]$RequestedDistro)
  $raw = @(& $script:WslExecutable --list --verbose 2>$null)
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
  & $script:WslExecutable --set-version $RequestedDistro 2
  if ($LASTEXITCODE -ne 0) {
    throw "No se pudo convertir $RequestedDistro a WSL 2. Actualiza WSL y repite como administrador."
  }
}

function Update-WslIfPossible {
  Write-Step "Comprobando actualizaciones de WSL y WSLg"
  & $script:WslExecutable --update
  if ($LASTEXITCODE -eq 0) { return }
  Write-Host "La actualizacion normal de WSL no ha funcionado; se intenta la descarga web." -ForegroundColor Yellow
  & $script:WslExecutable --update --web-download
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Aviso: WSL no se pudo actualizar, pero se continuara si la version instalada funciona." -ForegroundColor Yellow
  }
}

function Install-WslIfNeeded {
  param([Parameter(Mandatory = $true)][string]$RequestedDistro)
  Assert-ValidDistroName -Name $RequestedDistro
  $script:WslExecutable = Ensure-WslCommand
  if (-not $script:WslExecutable) { return $false }

  $distros = Get-InstalledDistros
  if ($distros -contains $RequestedDistro) { return $true }

  Write-Step "Instalando WSL y $RequestedDistro"
  $helpText = ((@(& $script:WslExecutable --help 2>$null) -join "`n") -replace "`0", "")
  $installArguments = @("--install", "--distribution", $RequestedDistro)
  if ($helpText -match '(?m)--no-launch') { $installArguments += "--no-launch" }
  if (Test-IsAdministrator) {
    & $script:WslExecutable @installArguments
    if ($LASTEXITCODE -notin @(0, 1641, 3010)) {
      throw "No se pudo instalar $RequestedDistro automaticamente (codigo $LASTEXITCODE)."
    }
  }
  else {
    Invoke-ElevatedAndWait -Executable $script:WslExecutable -Arguments $installArguments -Description "instalar WSL2 y $RequestedDistro" -AllowedExitCodes @(0, 1641, 3010)
  }

  $distros = Get-InstalledDistros
  if ($distros -contains $RequestedDistro) {
    Write-Host ""
    Write-Host "$RequestedDistro ya esta descargado. Windows puede pedir ahora crear el usuario de Ubuntu." -ForegroundColor Yellow
    & $script:WslExecutable -d $RequestedDistro -- true
    if ($LASTEXITCODE -eq 0) { return $true }
  }

  Write-Host ""
  Write-Host "La primera fase de WSL ha terminado. Reinicia Windows si te lo ha pedido, abre $RequestedDistro una vez para crear el usuario Linux y vuelve a ejecutar INSTALAR_WINDOWS.cmd." -ForegroundColor Yellow
  return $false
}

function Show-WindowsDiagnostic {
  param([Parameter(Mandatory = $true)][string]$RequestedDistro)
  Write-Host "Diagnostico previo de Windows"
  Write-Host "============================="
  $build = [Environment]::OSVersion.Version.Build
  $architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
  Write-Host "Windows build : $build"
  Write-Host "Arquitectura  : $architecture"
  if ($build -lt 19041) {
    Write-Host "[FALLO] WSL2 necesita Windows 10 2004 (build 19041) o una version posterior." -ForegroundColor Red
  }
  else { Write-Host "[OK]    Version de Windows compatible con WSL2" -ForegroundColor Green }
  if ($architecture -eq "X64") { Write-Host "[OK]    Arquitectura x86_64 validada" -ForegroundColor Green }
  else { Write-Host "[FALLO] Esta version fija binarios Linux x86_64 y no esta preparada para $architecture." -ForegroundColor Red }

  try {
    $computer = Get-CimInstance Win32_ComputerSystem -ErrorAction Stop
    $processors = @(Get-CimInstance Win32_Processor -ErrorAction Stop)
    $firmwareEnabled = @($processors | Where-Object { $_.VirtualizationFirmwareEnabled -eq $true }).Count -gt 0
    if ($computer.HypervisorPresent -or $firmwareEnabled) {
      Write-Host "[OK]    Virtualizacion disponible" -ForegroundColor Green
    }
    else {
      Write-Host "[AVISO] Windows no confirma la virtualizacion. Si WSL2 falla, habilita Intel VT-x/AMD-V en UEFI." -ForegroundColor Yellow
    }
  }
  catch {
    Write-Host "[AVISO] No se pudo consultar el estado de virtualizacion." -ForegroundColor Yellow
  }

  $gpuNames = Get-WindowsGpuNames
  Show-WindowsCompatibility -GpuNames $gpuNames -SelectedAccelerator (Resolve-WindowsAccelerator -Requested $Accelerator -GpuNames $gpuNames)

  $wsl = Get-WslExecutable
  if (-not $wsl) {
    Write-Host "[AVISO] WSL todavia no esta instalado; INSTALAR_WINDOWS.cmd intentara instalarlo." -ForegroundColor Yellow
    return
  }
  $script:WslExecutable = $wsl
  Write-Host "[OK]    WSL disponible: $wsl" -ForegroundColor Green
  $distros = Get-InstalledDistros
  if ($distros -contains $RequestedDistro) {
    $version = Get-WslDistroVersion -RequestedDistro $RequestedDistro
    if ($version -eq 2) { Write-Host "[OK]    $RequestedDistro trabaja con WSL2" -ForegroundColor Green }
    else { Write-Host "[AVISO] $RequestedDistro esta en WSL$version y se convertira a WSL2." -ForegroundColor Yellow }
  }
  else {
    Write-Host "[AVISO] $RequestedDistro todavia no esta instalado." -ForegroundColor Yellow
  }
}

function Save-LocalWindowsConfiguration {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$SourceId,
    [Parameter(Mandatory = $true)][string]$SelectedDistro,
    [Parameter(Mandatory = $true)][string]$SelectedRuntimePath
  )
  $directory = Split-Path -Parent $Path
  if (-not (Test-Path -LiteralPath $directory)) {
    New-Item -ItemType Directory -Path $directory | Out-Null
  }
  $temporary = "$Path.tmp.$PID"
  [ordered]@{
    schemaVersion = 1
    sourceId = $SourceId
    distro = $SelectedDistro
    runtimePath = $SelectedRuntimePath
  } | ConvertTo-Json | Set-Content -LiteralPath $temporary -Encoding UTF8
  Move-Item -LiteralPath $temporary -Destination $Path -Force
}

if ($env:TARANTULIN_IMPORT_ONLY -eq "1") { return }

trap {
  Write-Host ""
  Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
  exit 1
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
$distroWasSpecified = -not [string]::IsNullOrWhiteSpace($Distro)
$runtimeWasSpecified = -not [string]::IsNullOrWhiteSpace($RuntimePath)
if (-not $distroWasSpecified) { $Distro = [string]$config.distro }
if ($Accelerator -eq "auto" -and $config.accelerator -and $config.accelerator -ne "auto") {
  $Accelerator = [string]$config.accelerator
}
$runtimeBase = [string]$config.runtimeBase
if ($runtimeBase -ne "~/.local/share/tarantulin-windows") {
  throw "runtimeBase tiene un valor no admitido en .tarantulin/windows.json: $runtimeBase"
}
$sourceId = Get-SourceId -Path $projectRoot
$localConfigPath = Join-Path $projectRoot ".tarantulin-local\windows.json"
if (Test-Path -LiteralPath $localConfigPath -PathType Leaf) {
  try { $localConfig = Get-Content -LiteralPath $localConfigPath -Raw | ConvertFrom-Json }
  catch { throw "La configuracion local esta danada: $localConfigPath. $($_.Exception.Message)" }
  $localProperties = @($localConfig.PSObject.Properties.Name)
  if (($localProperties -contains "sourceId") -and [string]$localConfig.sourceId -eq $sourceId) {
    if (-not $distroWasSpecified -and ($localProperties -contains "distro")) {
      $Distro = [string]$localConfig.distro
    }
    if (-not $runtimeWasSpecified -and ($localProperties -contains "runtimePath")) {
      $RuntimePath = [string]$localConfig.runtimePath
    }
  }
}
if (-not $RuntimePath) { $RuntimePath = "$runtimeBase/$sourceId" }

if ($Command -in @("help", "-h", "--help")) {
  Show-Help
  exit 0
}

Assert-ValidDistroName -Name $Distro
if ($Command -eq "install") {
  if ($RuntimePath -ne "~" -and -not $RuntimePath.StartsWith("~/") -and -not $RuntimePath.StartsWith("/")) {
    throw "RuntimePath debe ser una ruta Linux absoluta o comenzar por ~/ : $RuntimePath"
  }
  Save-LocalWindowsConfiguration -Path $localConfigPath -SourceId $sourceId -SelectedDistro $Distro -SelectedRuntimePath $RuntimePath
}
if ($Command -eq "diagnostico-windows") {
  Show-WindowsDiagnostic -RequestedDistro $Distro
  exit 0
}

if ([Environment]::OSVersion.Version.Build -lt 19041) {
  throw "WSL2 necesita Windows 10 2004 (build 19041) o una version posterior. Actualiza Windows antes de instalar."
}
$architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
if ($architecture -ne "X64") {
  throw "Esta version contiene un entorno reproducible x86_64; la arquitectura $architecture no esta validada."
}
if ($Accelerator -in @("amd", "intel")) {
  throw "La aceleracion GPU $Accelerator no esta soportada por JAX/MJX bajo WSL2 en esta version. Repite con -Accelerator cpu."
}
if ($Command -eq "install") {
  $gpuNames = Get-WindowsGpuNames
  $Accelerator = Resolve-WindowsAccelerator -Requested $Accelerator -GpuNames $gpuNames
  Show-WindowsDiagnostic -RequestedDistro $Distro
}

$wslReady = Install-WslIfNeeded -RequestedDistro $Distro
if (-not $wslReady) { exit 0 }
Ensure-Wsl2 -RequestedDistro $Distro
if ($Command -eq "install" -and -not $SkipWslUpdate) { Update-WslIfPossible }

$wslHomeOutput = @(& $script:WslExecutable -d $Distro -- printenv HOME 2>$null)
$wslHome = if ($wslHomeOutput.Count) { [string]$wslHomeOutput[-1] } else { "" }
$wslHome = $wslHome.Trim()
if ($LASTEXITCODE -ne 0 -or -not $wslHome.StartsWith("/")) {
  throw "No se pudo determinar HOME dentro de $Distro. Abre Ubuntu una vez, crea el usuario y la contrasena de Linux y repite INSTALAR_WINDOWS.cmd."
}
$wslUidOutput = @(& $script:WslExecutable -d $Distro -- id -u 2>$null)
$wslUid = if ($wslUidOutput.Count) { [string]$wslUidOutput[-1] } else { "" }
$wslUid = $wslUid.Trim()
if ($LASTEXITCODE -ne 0 -or $wslUid -notmatch '^\d+$') {
  throw "No se pudo identificar el usuario Linux de $Distro. Abre Ubuntu una vez y termina su configuracion."
}
if ($wslUid -eq "0" -or $wslHome -eq "/root") {
  throw "Ubuntu esta entrando como root y TARANTULIN necesita un usuario normal con sudo. Abre $Distro, crea/configura ese usuario y vuelve a ejecutar INSTALAR_WINDOWS.cmd."
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
$runtimeBase = "$wslHome/.local/share/tarantulin-windows"
$runtimeBase = Resolve-WslLinuxPath -LinuxPath $runtimeBase -RequestedDistro $Distro
$RuntimePath = Resolve-WslLinuxPath -LinuxPath $RuntimePath -RequestedDistro $Distro
if ($RuntimePath -eq $runtimeBase -or -not $RuntimePath.StartsWith("$runtimeBase/")) {
  throw "RuntimePath debe ser una subcarpeta concreta de $runtimeBase y no puede apuntar fuera de ella: $RuntimePath"
}
if ($Command -eq "install") {
  Save-LocalWindowsConfiguration -Path $localConfigPath -SourceId $sourceId -SelectedDistro $Distro -SelectedRuntimePath $RuntimePath
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
  Write-Step "Preparando runtime WSL aislado"
  Invoke-NativeChecked -Executable $script:WslExecutable -Arguments $bootstrapArgs
  Write-Host ""
  Write-Host "Instalacion terminada." -ForegroundColor Green
  Write-Host "Comprueba el sistema con: .\TARANTULIN.cmd doctor"
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
  Invoke-NativeChecked -Executable $script:WslExecutable -Arguments $syncArgs
  exit 0
}

if ($Command -eq "shell") {
  Write-Step "Sincronizando antes de abrir WSL"
  Invoke-NativeChecked -Executable $script:WslExecutable -Arguments @(
    "-d", $Distro, "--", "bash", $syncScript,
    "--source", $sourceWsl, "--runtime", $RuntimePath, "--source-id", $sourceId
  )
  $shellArgs = @(
    "-d", $Distro, "--cd", $workspacePath, "--",
    "flock", "--shared", "$RuntimePath/runtime.lock", "env",
    "TARANTULIN_RUNTIME_ROOT=$RuntimePath",
    "TARANTULIN_RUNTIME_LOCK_HELD=shared",
    "TARANTULIN_BACKEND_PROFILE=$Accelerator",
    "bash"
  )
  & $script:WslExecutable @shellArgs
  exit $LASTEXITCODE
}

$commandsWithoutAutoSync = @("monitorizar", "parar", "pull-results")
if ($Command -notin $commandsWithoutAutoSync) {
  Write-Step "Sincronizando codigo Windows -> WSL"
  Invoke-NativeChecked -Executable $script:WslExecutable -Arguments @(
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
$linuxArgs += "--"
$linuxArgs += $Command
if ($CommandArgs) { $linuxArgs += $CommandArgs }

& $script:WslExecutable @linuxArgs
$codigoWsl = $LASTEXITCODE
if ($Command -eq "curriculo-automatico" -and $codigoWsl -eq 130) {
  Write-Host "El curriculo se ha detenido correctamente."
  exit 0
}
if ($codigoWsl -ne 0) {
  throw "El comando '$($script:WslExecutable)' termino con codigo $codigoWsl."
}
