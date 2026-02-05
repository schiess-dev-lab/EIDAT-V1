@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "PS1=%ROOT%mirror_eidat_support_sqlite.ps1"

if not exist "%PS1%" (
  echo [ERROR] Missing script: "%PS1%"
  exit /b 1
)

rem Optional args:
rem   --RepoRoot "C:\path\to\global\repo"
rem   --IncludeWalShm
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
exit /b %ERRORLEVEL%
