@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "PS1=%ROOT%set_global_run_mirror_to_eidat_support.ps1"

if not exist "%PS1%" (
  echo [ERROR] Missing script: "%PS1%"
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
exit /b %ERRORLEVEL%

