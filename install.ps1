[CmdletBinding()]
param(
  [string]$Distro = "",
  [ValidateSet("auto", "nvidia", "amd", "intel", "cpu")]
  [string]$Accelerator = "auto",
  [string]$RuntimePath = "",
  [switch]$NoSetup,
  [switch]$SyncOnly,
  [switch]$SkipSystemPackages,
  [switch]$SkipWslUpdate,
  [switch]$SkipGpuCheck
)

$ErrorActionPreference = "Stop"
$entrypoint = Join-Path $PSScriptRoot "tarantulin.ps1"
$forward = @{
  Command = "install"
  Accelerator = $Accelerator
  NoSetup = $NoSetup
  SyncOnly = $SyncOnly
  SkipSystemPackages = $SkipSystemPackages
  SkipWslUpdate = $SkipWslUpdate
  SkipGpuCheck = $SkipGpuCheck
}
if ($Distro) { $forward.Distro = $Distro }
if ($RuntimePath) { $forward.RuntimePath = $RuntimePath }

& $entrypoint @forward
exit $LASTEXITCODE
