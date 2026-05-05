@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
set "APP_ROOT=%ROOT%EIDAT_App_Files\"
set "LOCAL_SITE=%APP_ROOT%Lib\site-packages"
set "VENDOR_MARKER=%LOCAL_SITE%\.eidat_vendor_python.txt"

rem Default master registry inside this deployed runtime root.
rem Override by setting EIDAT_ADMIN_REGISTRY_PATH before launching.
if not defined EIDAT_ADMIN_REGISTRY_PATH set "EIDAT_ADMIN_REGISTRY_PATH=%ROOT%admin_registry.sqlite3"

rem Prefer local venv if present
set "PY="
set "VENV_PY=%APP_ROOT%.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
  "%VENV_PY%" -c "import sys" >nul 2>nul
  if errorlevel 1 (
    echo [WARN] Repo-local venv exists but is not runnable: "%VENV_PY%". Falling back to system Python.
  ) else (
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
  if not "%QUIET%"=="1" echo [RUN] Config parsed.
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
  if not "%QUIET%"=="1" echo [RUN] Local overrides parsed.
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
set "PYTHONDONTWRITEBYTECODE=1"

"%PY%" -m Production.admin_gui
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
  goto vendored_done
)
for /f "usebackq delims=" %%P in (`"%~1" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2^>nul`) do (
  if not defined TARGET_MM set "TARGET_MM=%%P"
)
if not defined TARGET_MM (
  set "VENDOR_WARN=Could not determine the Python version for runtime %~1. Skipping vendored site-packages."
  goto vendored_done
)
for /f "usebackq tokens=1,* delims==" %%A in (`findstr /B /I /C:"major_minor=" "%VENDOR_MARKER%" 2^>nul`) do (
  if /I "%%A"=="major_minor" if not defined VENDOR_MM set "VENDOR_MM=%%B"
)
if not defined VENDOR_MM (
  set "VENDOR_WARN=Vendored site-packages marker is incomplete. Re-run install.bat to rebuild local dependencies."
  goto vendored_done
)
if /I not "!TARGET_MM!"=="!VENDOR_MM!" (
  set "VENDOR_WARN=Vendored site-packages were built for Python !VENDOR_MM!, but runtime %~1 is Python !TARGET_MM!. Re-run install.bat for this interpreter."
  goto vendored_done
)
set "VENDORED_SITE_OK=1"

:vendored_done
if defined VENDOR_WARN echo [WARN] !VENDOR_WARN!
exit /b 0
