# Nombre conservado para no romper accesos directos de fases anteriores.
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

$installer = Join-Path $PSScriptRoot "install_windows.ps1"
$forward = @{
  RepoUrl = $RepoUrl
  Distro = $Distro
  Accelerator = $Accelerator
  NoSetup = $NoSetup
  SyncOnly = $SyncOnly
  SkipGpuCheck = $SkipGpuCheck
  EnableExperimentalAmdWsl = $EnableExperimentalAmdWsl
}
if ($InstallPath) { $forward.InstallPath = $InstallPath }
& $installer @forward
exit $LASTEXITCODE
