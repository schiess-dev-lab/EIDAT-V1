@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "RUNTIME_ROOT=C:\Users\zachs\OneDrive\Documents\DevProjects\PDF Scanner\EIDAT-V1"
set "HERE=%~dp0"
for %%I in ("%HERE%..\..") do set "NODE_ROOT=%%~fI"

set "APP_ROOT=%RUNTIME_ROOT%\EIDAT_App_Files"
set "PYTHONPATH=%APP_ROOT%;%APP_ROOT%\Lib\site-packages;%PYTHONPATH%"
set "EIDAT_NODE_ROOT=%NODE_ROOT%"
set "EIDAT_DATA_ROOT=%NODE_ROOT%\EIDAT\UserData"
set "PYTHONDONTWRITEBYTECODE=1"

set "SYS_PY_EXE="
set "SYS_PY_ARGS="

rem Prefer py launcher (does not require python.exe on PATH)
where py >nul 2>nul && (py -3 -c "import sys" >nul 2>nul && set "SYS_PY_EXE=py" && set "SYS_PY_ARGS=-3")

rem Optional node-local override: EIDAT\Runtime\sys_python.txt contains full path to python.exe
if not defined SYS_PY_EXE (
  set "PYCFG=%NODE_ROOT%\EIDAT\Runtime\sys_python.txt"
  if exist "%PYCFG%" (
    for /f "usebackq tokens=* delims=" %%P in ("%PYCFG%") do (
      set "LINE=%%P"
      if not "!LINE!"=="" if not "!LINE:~0,1!"=="#" if not "!LINE:~0,1!"==";" (
        set "SYS_PY_EXE=!LINE!"
        goto :eidat_have_sys_py
      )
    )
  )
)

rem Fallback: python on PATH (some environments allow this)
if not defined SYS_PY_EXE (
  where python >nul 2>nul && (python -c "import sys" >nul 2>nul && set "SYS_PY_EXE=python")
)

:eidat_have_sys_py
if not defined SYS_PY_EXE (
  echo [ERROR] Python 3 not found.
  echo         Install Python 3 with the 'py' launcher OR set a node-local python path in:
  echo           %NODE_ROOT%\EIDAT\Runtime\sys_python.txt
  pause
  exit /b 1
)

rem Sanity check interpreter
"%SYS_PY_EXE%" %SYS_PY_ARGS% -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python command failed: "%SYS_PY_EXE%" %SYS_PY_ARGS%
  echo         Fix sys_python.txt or reinstall Python.
  pause
  exit /b 1
)

set "VENV_PY=%NODE_ROOT%\EIDAT\Runtime\.venv\Scripts\python.exe"

"%SYS_PY_EXE%" %SYS_PY_ARGS% -m Production.bootstrap_env --node-root "%NODE_ROOT%"
if errorlevel 1 exit /b 1

if not exist "%VENV_PY%" (
  echo [ERROR] Node venv python not found: "%VENV_PY%"
  exit /b 1
)

  "%VENV_PY%" -m Production.launch_ui --node-root "%NODE_ROOT%" --mode files

endlocal & exit /b %ERRORLEVEL%
