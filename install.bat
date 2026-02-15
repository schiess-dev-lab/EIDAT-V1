@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
set "APP_ROOT=%ROOT%EIDAT_App_Files\"
set "TOOLS_DIR=%ROOT%tools"
set "TESS_ROOT=%TOOLS_DIR%\tesseract"
set "TESS_EXE=%TESS_ROOT%\tesseract.exe"
set "TESSDATA_DIR=%TESS_ROOT%\tessdata"
set "TESS_INSTALLER=%TOOLS_DIR%\tesseract-ocr-w64-setup-5.4.0.20240606.exe"
set "TESS_URL=https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.4.0.20240606.exe"
set "PY=py"
rem Optional first arg: custom venv directory. Defaults to %APP_ROOT%.venv
set "VENV_DIR=%~1"
if "%VENV_DIR%"=="" set "VENV_DIR=%APP_ROOT%.venv"
where %PY% >nul 2>nul || set "PY=python"

echo [SETUP] Creating local virtual environment: "%VENV_DIR%" ...
"%PY%" -m venv "%VENV_DIR%"
if errorlevel 1 (
  echo [ERROR] Failed to create venv. Ensure Python 3 is installed.
  endlocal & exit /b 1
)

set "VPY=%VENV_DIR%\Scripts\python.exe"
if not exist "%VPY%" (
  echo [ERROR] venv python not found: "%VPY%"
  endlocal & exit /b 1
)

echo [SETUP] Upgrading pip...
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 echo [WARN] pip upgrade had warnings.

echo [SETUP] Installing required Python packages (minimal + UI)...
"%VPY%" -m pip install ^
  pymupdf ^
  pandas ^
  openpyxl ^
  XlsxWriter ^
  matplotlib ^
  opencv-python-headless ^
  PySide6
if errorlevel 1 (
  echo [ERROR] Package install failed.
  endlocal & exit /b 1
)

rem --- Optional: repo-local Tesseract install (Windows x64) ---
if not exist "%TOOLS_DIR%" mkdir "%TOOLS_DIR%"
if exist "%TESS_EXE%" (
  echo [INFO] Tesseract already present: "%TESS_EXE%"
) else (
  echo [SETUP] Downloading Tesseract installer...
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ProgressPreference='SilentlyContinue';" ^
    "Invoke-WebRequest -Uri '%TESS_URL%' -OutFile '%TESS_INSTALLER%'"
  if errorlevel 1 (
    echo [WARN] Failed to download Tesseract installer.
  ) else (
    echo [SETUP] Installing Tesseract to "%TESS_ROOT%" ...
    rem Try silent install with common NSIS/Inno flags; fall back to UI if needed.
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$args = @('/S','/D=%TESS_ROOT%','/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/SP-','/DIR=%TESS_ROOT%');" ^
      "Start-Process -FilePath '%TESS_INSTALLER%' -ArgumentList $args -Wait"
    if not exist "%TESS_EXE%" (
      echo [INFO] Silent install did not produce "%TESS_EXE%".
      echo [ACTION] Launching installer UI. Choose "%TESS_ROOT%" as destination.
      start /wait "" "%TESS_INSTALLER%"
    )
  )
)

rem Also vendor runtime deps into repo-local Lib\site-packages (for non-venv runs)
set "LOCAL_SITE=%APP_ROOT%Lib\site-packages"
if not exist "%LOCAL_SITE%" mkdir "%LOCAL_SITE%"
echo [SETUP] Vendoring Python deps to Lib\site-packages (minimal):
echo         pymupdf, pandas, openpyxl, XlsxWriter, matplotlib, opencv-python-headless, PySide6
"%VPY%" -m pip install --upgrade --no-warn-script-location --target "%LOCAL_SITE%" ^
  pymupdf pandas openpyxl XlsxWriter matplotlib opencv-python-headless PySide6
if errorlevel 1 (
  echo [WARN] Vendoring had warnings/failures. Non-venv runs may miss some features.
)

rem --- Scaffold expected folders and sample terms file ---
if not exist "%ROOT%user_inputs" mkdir "%ROOT%user_inputs"
if not exist "%ROOT%Data Packages" mkdir "%ROOT%Data Packages"
if not exist "%ROOT%run_data_simple" mkdir "%ROOT%run_data_simple"
if not exist "%ROOT%Master_Database" mkdir "%ROOT%Master_Database"
if not exist "%ROOT%user_inputs\terms.schema.simple.xlsx" (
  echo [SETUP] Creating simple schema template (user_inputs\terms.schema.simple.xlsx)
  "%VPY%" "%APP_ROOT%scripts\generate_terms_schema_simple.py"
) else (
  echo [INFO] Simple schema template already present.
)

rem Create scanner.env with sensible defaults if missing
if not exist "%ROOT%user_inputs\scanner.env" (
  echo [SETUP] Creating default user_inputs\scanner.env
  >  "%ROOT%user_inputs\scanner.env" echo # Scanner configuration (KEY=VALUE^)
  >> "%ROOT%user_inputs\scanner.env" echo QUIET=1
  >> "%ROOT%user_inputs\scanner.env" echo #VENV_DIR=%APP_ROOT%.venv
  >> "%ROOT%user_inputs\scanner.env" echo #OCR_MODE=fallback   ^# fallback|ocr_only|no_ocr
  >> "%ROOT%user_inputs\scanner.env" echo #OCR_DPI=600
)

rem Create scanner.local.env (machine-specific overrides) if missing
if not exist "%ROOT%user_inputs\scanner.local.env" (
  echo [SETUP] Creating default user_inputs\scanner.local.env
  >  "%ROOT%user_inputs\scanner.local.env" echo # Local overrides (KEY=VALUE^)
  >> "%ROOT%user_inputs\scanner.local.env" echo QUIET=1
)

rem Add local Tesseract config to scanner.local.env if available
if exist "%TESS_EXE%" (
  if not exist "%ROOT%user_inputs\scanner.local.env" (
    > "%ROOT%user_inputs\scanner.local.env" echo # Local overrides (KEY=VALUE^)
  )
  findstr /B /I /C:"TESSERACT_CMD=" "%ROOT%user_inputs\scanner.local.env" >nul || ^
    >> "%ROOT%user_inputs\scanner.local.env" echo TESSERACT_CMD=%TESS_EXE%
  if exist "%TESSDATA_DIR%" (
    findstr /B /I /C:"TESSDATA_PREFIX=" "%ROOT%user_inputs\scanner.local.env" >nul || ^
      >> "%ROOT%user_inputs\scanner.local.env" echo TESSDATA_PREFIX=%TESSDATA_DIR%
  )
)

rem Track the repo root name so the GUI only reads data within this folder
if not exist "%ROOT%user_inputs\repo_root_name.txt" (
  for %%I in ("%ROOT%.") do set "ROOT_NAME=%%~nxI"
  > "%ROOT%user_inputs\repo_root_name.txt" echo !ROOT_NAME!
)

echo.
echo [READY] Local environment set up.
echo   - Python venv: "%VENV_DIR%"
echo   - Vendored Python deps in Lib\site-packages for non-venv runs
echo   - OCR: optional local Tesseract in "%TESS_ROOT%"
echo.
echo Next steps:
echo   1) Point the app at your PDF repository (default: "Data Packages")
echo   2) Edit user_inputs\terms.schema.simple.xlsx (or .csv)
echo   3) Run: run_gui.bat   or  "%VENV_DIR%\Scripts\python.exe" EIDAT_App_Files\ui_next\qt_main.py

endlocal & exit /b 0
