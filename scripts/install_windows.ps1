# Alias historico. El instalador mantenido esta en la raiz del repositorio.
[CmdletBinding()]
param(
  [string]$RepoUrl = "https://github.com/GuillermoRosale5/Tarantulin-Windows-WSL.git",
  [string]$Distro = "Ubuntu-24.04",
  [string]$InstallPath = "",
  [ValidateSet("auto", "nvidia", "amd", "intel", "cpu")]
  [string]$Accelerator = "auto",
  [switch]$NoSetup,
  [switch]$SyncOnly,
  [switch]$SkipGpuCheck,
  [switch]$EnableExperimentalAmdWsl
)

$ErrorActionPreference = "Stop"
$rootInstaller = Join-Path (Split-Path -Parent $PSScriptRoot) "install.ps1"
$forward = @{
  Distro = $Distro
  Accelerator = $Accelerator
  NoSetup = $NoSetup
  SyncOnly = $SyncOnly
  SkipGpuCheck = $SkipGpuCheck
  EnableExperimentalAmdWsl = $EnableExperimentalAmdWsl
}
if ($InstallPath) { $forward.RuntimePath = $InstallPath }
Write-Verbose "Repositorio esperado: $RepoUrl"
& $rootInstaller @forward
exit $LASTEXITCODE
