@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
set "APP_ROOT=%ROOT%EIDAT_App_Files\"

rem Default master registry inside this deployed runtime root.
rem Override by setting EIDAT_ADMIN_REGISTRY_PATH before launching.
if not defined EIDAT_ADMIN_REGISTRY_PATH set "EIDAT_ADMIN_REGISTRY_PATH=%ROOT%admin_registry.sqlite3"

rem Prefer local venv if present
set "VENV_PY=%APP_ROOT%.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
  set "PY=%VENV_PY%"
  set "PATH=%APP_ROOT%.venv\Scripts;!PATH!"
) else (
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
  if not exist "%VENV_PY%" (
    if not "%QUIET%"=="1" echo [SETUP] No venv at "%VENV_DIR%". Bootstrapping...
    call "%ROOT%install.bat" "%VENV_DIR%"
  )
  if exist "%VENV_PY%" (
    if exist "%VENV_ACTIVATE%" call "%VENV_ACTIVATE%"
    set "PY=%VENV_PY%"
    set "PATH=%VENV_DIR%\Scripts;!PATH!"
  ) else (
    echo [ERROR] Failed to create venv at "%VENV_DIR%".>&2
    exit /b 1
  )
)

rem Make repo + vendored packages available
set "PYTHONPATH=%APP_ROOT%;%APP_ROOT%Lib\site-packages;%PYTHONPATH%"
set "PYTHONDONTWRITEBYTECODE=1"

"%PY%" -m Production.admin_gui
set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%
