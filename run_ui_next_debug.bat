@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
set "APP_ROOT=%ROOT%EIDAT_App_Files\"
set "LOG_FILE=%ROOT%debug_output_new.txt"
set "LOCAL_SITE=%APP_ROOT%Lib\site-packages"
set "VENDOR_MARKER=%LOCAL_SITE%\.eidat_vendor_python.txt"

rem Prefer local venv if present (activate to ensure env vars / PATH are set)
set "PY="
set "VENV_PY=%APP_ROOT%.venv\Scripts\python.exe"
set "VENV_ACTIVATE=%APP_ROOT%.venv\Scripts\activate.bat"
if exist "%VENV_PY%" (
  "%VENV_PY%" -c "import sys" >nul 2>nul
  if errorlevel 1 (
    echo [WARN] Repo-local venv exists but is not runnable: "%VENV_PY%". Falling back to system Python.
  ) else (
    if exist "%VENV_ACTIVATE%" call "%VENV_ACTIVATE%"
    set "PY=%VENV_PY%"
    set "PATH=%APP_ROOT%.venv\Scripts;!PATH!"
  )
)
if not defined PY (
  set "PY=py"
  where %PY% >nul 2>nul || set "PY=python"
)

rem Load optional scanner config (user_inputs\scanner.env) as KEY=VALUE lines
set "CFG=%ROOT%user_inputs\scanner.env"
if exist "%CFG%" (
  if not "%QUIET%"=="1" echo [RUN] Loading config: "%CFG%"
  for /f "usebackq tokens=* delims=" %%L in ("%CFG%") do (
    set "LINE=%%L"
    if not "!LINE!"=="" if not "!LINE:~0,1!"==" " if not "!LINE:~0,1!"=="#" if not "!LINE:~0,1!"==";" if not "!LINE!"=="!LINE:=!" (
      for /f "tokens=1,* delims==" %%A in ("!LINE!") do (
        set "K=%%~A"
        set "V=%%~B"
        if defined K (
          for /f "tokens=1 delims=#;" %%C in ("!V!") do set "V=%%~C"
          set "V=!V:~0!"
          for /f "tokens=* delims= " %%D in ("!V!") do set "V=%%~D"
          if not "!V!"=="" set "!K!=!V!"
        )
      )
    )
  )
)

rem Load optional local overrides (user_inputs\scanner.local.env) after scanner.env
set "CFG_LOCAL=%ROOT%user_inputs\scanner.local.env"
if exist "%CFG_LOCAL%" (
  if not "%QUIET%"=="1" echo [RUN] Loading local overrides: "%CFG_LOCAL%"
  for /f "usebackq tokens=* delims=" %%L in ("%CFG_LOCAL%") do (
    set "LINE=%%L"
    if not "!LINE!"=="" if not "!LINE:~0,1!"==" " if not "!LINE:~0,1!"=="#" if not "!LINE:~0,1!"==";" if not "!LINE!"=="!LINE:=!" (
      for /f "tokens=1,* delims==" %%A in ("!LINE!") do (
        set "K=%%~A"
        set "V=%%~B"
        if defined K (
          for /f "tokens=1 delims=#;" %%C in ("!V!") do set "V=%%~C"
          set "V=!V:~0!"
          for /f "tokens=* delims= " %%D in ("!V!") do set "V=%%~D"
          if not "!V!"=="" set "!K!=!V!"
        )
      )
    )
  )
)

rem Optional venv override via scanner.env (activate to ensure env vars / PATH are set)
if defined VENV_DIR (
  set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
  set "VENV_ACTIVATE=%VENV_DIR%\Scripts\activate.bat"
  if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import sys" >nul 2>nul
    if errorlevel 1 (
      echo [WARN] Configured venv is not runnable: "%VENV_PY%". Rebuilding it now...
      call "%ROOT%install.bat" "%VENV_DIR%"
    )
  ) else (
    if not "%QUIET%"=="1" echo [SETUP] No venv at "%VENV_DIR%". Bootstrapping...
    call "%ROOT%install.bat" "%VENV_DIR%"
  )
  if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import sys" >nul 2>nul
    if errorlevel 1 (
      echo [WARN] Configured venv is still not runnable after install: "%VENV_PY%". Falling back to system Python.
    ) else (
      if exist "%VENV_ACTIVATE%" call "%VENV_ACTIVATE%"
      set "PY=%VENV_PY%"
      set "PATH=%VENV_DIR%\Scripts;!PATH!"
    )
  ) else (
    echo [ERROR] Failed to create venv at "%VENV_DIR%".>&2
    exit /b 1
  )
)

call :configure_pythonpath "%PY%"

rem Force debug mode for UI Next runs
set "DEBUG_MODE=1"

if not "%QUIET%"=="1" (
  echo [RUN] Python: "%PY%"
  echo [RUN] Debug : DEBUG_MODE=1 ^(ui_next^)
  echo [RUN] Log   : "%LOG_FILE%"
)

set "PSHELL=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if exist "%PSHELL%" (
  set "PS_CMD=& { & '%PY%' -m ui_next.qt_main @args 2>&1 ^| Tee-Object -FilePath '%LOG_FILE%'; exit $LASTEXITCODE }"
  "%PSHELL%" -NoLogo -NoProfile -Command "!PS_CMD!" -- %*
) else (
  echo [WARN] PowerShell not found; running without live tee ^(log will overwrite "%LOG_FILE%"^).
  "%PY%" -m ui_next.qt_main %* > "%LOG_FILE%" 2>&1
)

set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%

:configure_pythonpath
set "BASE_PYTHONPATH=%APP_ROOT%"
if defined PYTHONPATH set "BASE_PYTHONPATH=%BASE_PYTHONPATH%;%PYTHONPATH%"
call :resolve_vendored_site "%~1"
if defined VENDORED_SITE_OK (
  set "PYTHONPATH=%LOCAL_SITE%;%BASE_PYTHONPATH%"
) else (
  set "PYTHONPATH=%BASE_PYTHONPATH%"
)
exit /b 0

:resolve_vendored_site
set "VENDORED_SITE_OK="
set "TARGET_MM="
set "VENDOR_MM="
set "VENDOR_WARN="
if not exist "%LOCAL_SITE%" exit /b 0
if not exist "%VENDOR_MARKER%" (
  set "VENDOR_WARN=Vendored site-packages marker is missing. Re-run install.bat to rebuild local dependencies."
  goto debug_vendored_done
)
for /f "usebackq delims=" %%P in (`"%~1" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2^>nul`) do (
  if not defined TARGET_MM set "TARGET_MM=%%P"
)
if not defined TARGET_MM (
  set "VENDOR_WARN=Could not determine the Python version for runtime %~1. Skipping vendored site-packages."
  goto debug_vendored_done
)
for /f "usebackq tokens=1,* delims==" %%A in (`findstr /B /I /C:"major_minor=" "%VENDOR_MARKER%" 2^>nul`) do (
  if /I "%%A"=="major_minor" if not defined VENDOR_MM set "VENDOR_MM=%%B"
)
if not defined VENDOR_MM (
  set "VENDOR_WARN=Vendored site-packages marker is incomplete. Re-run install.bat to rebuild local dependencies."
  goto debug_vendored_done
)
if /I not "!TARGET_MM!"=="!VENDOR_MM!" (
  set "VENDOR_WARN=Vendored site-packages were built for Python !VENDOR_MM!, but runtime %~1 is Python !TARGET_MM!. Re-run install.bat for this interpreter."
  goto debug_vendored_done
)
set "VENDORED_SITE_OK=1"

:debug_vendored_done
if defined VENDOR_WARN echo [WARN] !VENDOR_WARN!
exit /b 0
