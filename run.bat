@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
set "APP_ROOT=%ROOT%EIDAT_App_Files\"

rem Prefer local venv if present
set "VENV_PY=%APP_ROOT%.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
  set "PY=%VENV_PY%"
  set "PATH=%APP_ROOT%.venv\Scripts;!PATH!"
) else (
  set "PY=py"
  where %PY% >nul 2>nul || set "PY=python"
)

rem Make vendored packages available when not installed system-wide
set "PYTHONPATH=%APP_ROOT%Lib\site-packages;%PYTHONPATH%"

rem Detect simple schema file and scaffold if missing
set "TERMS_SIMPLE=%ROOT%user_inputs\terms.schema.simple.xlsx"
set "TERMS_SIMPLE_CSV=%ROOT%user_inputs\terms.schema.simple.csv"

if exist "%TERMS_SIMPLE%" (
  set "TERMS=%TERMS_SIMPLE%"
  goto has_terms
)

if exist "%TERMS_SIMPLE_CSV%" (
  set "TERMS=%TERMS_SIMPLE_CSV%"
  goto has_terms
)

echo [WARN] No simple schema file found.
echo [SETUP] Creating simple schema template (user_inputs\terms.schema.simple.xlsx)
"%PY%" "%APP_ROOT%scripts\generate_terms_schema_simple.py"
if errorlevel 1 exit /b 1
echo Open and edit: "%ROOT%user_inputs\terms.schema.simple.xlsx" and re-run.
exit /b 1

:has_terms
set "IN_DIR=%ROOT%Data Packages"
set "RUN_SCRIPT=%APP_ROOT%scripts\simple_extraction.py"

rem Ensure directories exist
if not exist "%ROOT%run_data_simple" mkdir "%ROOT%run_data_simple"

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

rem Ensure vendored packages path is prepended even if PYTHONPATH was overridden in scanner.env
set "PYTHONPATH=%APP_ROOT%Lib\site-packages;%PYTHONPATH%"

rem Optional venv override via scanner.env
if defined VENV_DIR (
  set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
  if not exist "%VENV_PY%" (
    if not "%QUIET%"=="1" echo [SETUP] No venv at "%VENV_DIR%". Bootstrapping...
    call "%ROOT%install.bat" "%VENV_DIR%"
  )
  if exist "%VENV_PY%" (
    set "PY=%VENV_PY%"
    set "PATH=%VENV_DIR%\Scripts;!PATH!"
  ) else (
    echo [ERROR] Failed to create venv at "%VENV_DIR%".>&2
    exit /b 1
  )
)

if not "%QUIET%"=="1" (
  echo [RUN] Python: "%PY%"
  echo [RUN] Terms : "%TERMS%"  (use .xlsx/.csv)
  echo [RUN] PDFs  : "%IN_DIR%"
  echo [RUN] Out   : "%ROOT%run_data_simple" (per-run outputs saved under run_data_simple)
  echo [RUN] OCR    : OCR_MODE=%OCR_MODE%
  if defined FORCE_OCR echo [RUN] OCR    : FORCE_OCR=%FORCE_OCR%
  if defined OCR_DPI echo [RUN] OCR    : OCR_DPI=%OCR_DPI%
)

rem Default OCR policy
if not defined OCR_MODE set "OCR_MODE=fallback"

set "PDF_ARGS="
for /r "%IN_DIR%" %%F in (*.pdf) do set "PDF_ARGS=!PDF_ARGS! --pdf ^"%%F^""
if "%PDF_ARGS%"=="" (
  echo [ERROR] No PDFs found under "%IN_DIR%".
  exit /b 1
)

"%PY%" "%RUN_SCRIPT%" ^
  --terms "%TERMS%" ^
  %PDF_ARGS% ^
  %*

set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" exit /b %RC%

rem Locate the most recent run_data_simple folder for convenience
set "LAST_RUN="
for /f "delims=" %%D in ('dir /ad /b /o:-d "%ROOT%run_data_simple" 2^>nul') do (
  if not defined LAST_RUN set "LAST_RUN=%%D"
)
if defined LAST_RUN (
  echo [DONE] Run folder: "%ROOT%run_data_simple\\%LAST_RUN%"
) else (
  echo [DONE] Check: "%ROOT%run_data_simple" for this run.
)
endlocal & exit /b 0
