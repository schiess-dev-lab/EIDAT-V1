@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "TOOLS_DIR=%~dp0"
for %%I in ("%TOOLS_DIR%..") do set "RUNTIME_ROOT=%%~fI"

set "APP_ROOT=%RUNTIME_ROOT%\\EIDAT_App_Files"
set "PYTHONPATH=%APP_ROOT%;%PYTHONPATH%"

set "REG=%RUNTIME_ROOT%\\admin_registry.sqlite3"
if not exist "%REG%" (
  echo [WARN] Registry not found at "%REG%". Using default registry path.>&2
  py -3 -m Production.node_mirror --all-from-registry --runtime-root "%RUNTIME_ROOT%"
  exit /b %ERRORLEVEL%
)

py -3 -m Production.node_mirror --all-from-registry --runtime-root "%RUNTIME_ROOT%" --registry "%REG%"
exit /b %ERRORLEVEL%

