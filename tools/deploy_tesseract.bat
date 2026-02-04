@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem === USER CONFIG ===
rem Set this to the folder that CONTAINS tesseract.exe and tessdata\
set "SOURCE_DIR=C:\Program Files\Tesseract-OCR"
rem Set OVERWRITE=1 to replace an existing tools\tesseract folder
set "OVERWRITE=0"

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT=%%~fI\"
set "TOOLS_DIR=%ROOT%tools"
set "DEST_DIR=%TOOLS_DIR%\tesseract"
set "DEST_EXE=%DEST_DIR%\tesseract.exe"
set "DEST_TESSDATA=%DEST_DIR%\tessdata"
set "SCANNER_ENV=%ROOT%user_inputs\scanner.env"

if not exist "%SOURCE_DIR%" (
  echo [ERROR] SOURCE_DIR not found: "%SOURCE_DIR%"
  exit /b 1
)
if not exist "%SOURCE_DIR%\tesseract.exe" (
  echo [ERROR] tesseract.exe not found in SOURCE_DIR: "%SOURCE_DIR%"
  exit /b 1
)
if not exist "%SOURCE_DIR%\tessdata" (
  echo [ERROR] tessdata folder not found in SOURCE_DIR: "%SOURCE_DIR%"
  exit /b 1
)

if exist "%DEST_DIR%" (
  if "%OVERWRITE%"=="1" (
    echo [SETUP] Removing existing "%DEST_DIR%" ...
    rmdir /s /q "%DEST_DIR%"
  ) else (
    echo [INFO] Destination exists: "%DEST_DIR%"
    echo [INFO] Set OVERWRITE=1 to replace it.
  )
)

if not exist "%TOOLS_DIR%" mkdir "%TOOLS_DIR%"
if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"

echo [SETUP] Copying Tesseract from:
echo         "%SOURCE_DIR%"
echo      to "%DEST_DIR%"
robocopy "%SOURCE_DIR%" "%DEST_DIR%" /E /NFL /NDL /NJH /NJS /NP >nul
set "RC=%ERRORLEVEL%"
if %RC% GEQ 8 (
  echo [ERROR] robocopy failed with code %RC%.
  exit /b 1
)

if not exist "%DEST_EXE%" (
  echo [ERROR] Copy failed: "%DEST_EXE%" not found.
  exit /b 1
)

if not exist "%SCANNER_ENV%" (
  if not exist "%ROOT%user_inputs" mkdir "%ROOT%user_inputs"
  > "%SCANNER_ENV%" echo # Scanner configuration (KEY=VALUE^)
)

findstr /B /I /C:"TESSERACT_CMD=" "%SCANNER_ENV%" >nul || ^
  >> "%SCANNER_ENV%" echo TESSERACT_CMD=%DEST_EXE%
if exist "%DEST_TESSDATA%" (
  findstr /B /I /C:"TESSDATA_PREFIX=" "%SCANNER_ENV%" >nul || ^
    >> "%SCANNER_ENV%" echo TESSDATA_PREFIX=%DEST_TESSDATA%
)

echo [READY] Tesseract deployed to repo.
echo   - %DEST_EXE%
echo   - %DEST_TESSDATA%
echo   - Updated: %SCANNER_ENV%

endlocal & exit /b 0
