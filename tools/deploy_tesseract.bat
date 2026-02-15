@echo off
setlocal EnableExtensions EnableDelayedExpansion

if /I "%EIDAT_DEBUG%"=="1" echo on

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
set "SCANNER_ENV_LOCAL=%ROOT%user_inputs\scanner.local.env"
set "ROBO_LOG=%TOOLS_DIR%\deploy_tesseract.robocopy.log"

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

rem Quick sanity check: ensure we can write to the destination (common OneDrive/ACL issue)
set "WRITE_TEST=%DEST_DIR%\.write_test.tmp"
echo ok> "%WRITE_TEST%" 2>nul
if errorlevel 1 (
  echo [ERROR] Cannot write to destination folder: "%DEST_DIR%"
  echo [HINT] Check OneDrive permissions / folder is not read-only / run in elevated cmd.
  exit /b 1
)
del /q "%WRITE_TEST%" >nul 2>nul

echo [SETUP] Copying Tesseract from:
echo         "%SOURCE_DIR%"
echo      to "%DEST_DIR%"
echo [INFO] robocopy "%SOURCE_DIR%" "%DEST_DIR%" /E /R:1 /W:1> "%ROBO_LOG%"
robocopy "%SOURCE_DIR%" "%DEST_DIR%" /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP >> "%ROBO_LOG%" 2>&1
set "RC=%ERRORLEVEL%"
if %RC% GEQ 8 (
  echo [ERROR] robocopy failed with code %RC% - 0-7 are success; 8+ are failures.
  echo [INFO] See log: "%ROBO_LOG%"
  echo [HINT] Code 16 usually means a serious IO/permission/path issue; the log will say which file/path failed.
  exit /b 1
)

if not exist "%DEST_EXE%" (
  echo [ERROR] Copy failed: "%DEST_EXE%" not found.
  exit /b 1
)

if not exist "%SCANNER_ENV_LOCAL%" (
  if not exist "%ROOT%user_inputs" mkdir "%ROOT%user_inputs"
  > "%SCANNER_ENV_LOCAL%" echo # Local overrides (KEY=VALUE^)
)

findstr /B /I /C:"TESSERACT_CMD=" "%SCANNER_ENV_LOCAL%" >nul || ^
  >> "%SCANNER_ENV_LOCAL%" echo TESSERACT_CMD=%DEST_EXE%
if exist "%DEST_TESSDATA%" (
  findstr /B /I /C:"TESSDATA_PREFIX=" "%SCANNER_ENV_LOCAL%" >nul || ^
    >> "%SCANNER_ENV_LOCAL%" echo TESSDATA_PREFIX=%DEST_TESSDATA%
)

echo [READY] Tesseract deployed to repo.
echo   - %DEST_EXE%
echo   - %DEST_TESSDATA%
echo   - Updated: %SCANNER_ENV_LOCAL%

endlocal & exit /b 0
