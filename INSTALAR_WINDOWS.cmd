@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Instalacion de TARANTULIN para Windows + WSL2
echo ============================================================
echo.

where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo ERROR: este Windows no encuentra Windows PowerShell.
  echo Se necesita Windows 10 u 11 actualizado.
  set "codigo=1"
  goto :fin
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set "codigo=%ERRORLEVEL%"

:fin
echo.
if "%codigo%"=="0" (
  echo El instalador ha terminado esta fase correctamente.
) else (
  echo El instalador ha terminado con un error. Revisa el mensaje anterior.
)
echo.
if not "%TARANTULIN_SIN_PAUSA%"=="1" pause
exit /b %codigo%
