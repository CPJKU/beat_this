@echo off
setlocal
cd /d "%~dp0"
echo BPM Finder - source installation and repair
echo For the portable Windows package, extract everything and run BPM Finder.exe.
echo.
set "BPM_UV=uv"
where uv >nul 2>nul
if not errorlevel 1 goto runtime
set "BPM_UV=%CD%\.tools\uv.exe"
if exist "%BPM_UV%" goto runtime
if not exist ".tools" mkdir ".tools"
echo Downloading the Python environment manager from its official source...
curl.exe --fail --location --retry 2 "https://releases.astral.sh/github/uv/releases/download/0.11.7/uv-x86_64-pc-windows-msvc.zip" --output ".tools\uv.zip"
if errorlevel 1 goto failed
tar.exe -xf ".tools\uv.zip" -C ".tools"
if errorlevel 1 goto failed
:runtime
echo Preparing an isolated Python 3.12 environment...
"%BPM_UV%" venv --allow-existing --python 3.12 ".venv-desktop"
if errorlevel 1 goto failed
echo Installing the tested CPU engine...
"%BPM_UV%" pip install --python ".venv-desktop\Scripts\python.exe" "torch==2.14.1+cpu" "torchaudio==2.11.0+cpu" --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
echo Installing UI, audio formats and DBN support...
"%BPM_UV%" pip install --python ".venv-desktop\Scripts\python.exe" -r requirements-desktop.txt
if errorlevel 1 goto failed
"%BPM_UV%" pip install --python ".venv-desktop\Scripts\python.exe" --no-deps -e .
if errorlevel 1 goto failed
echo Checking the runtime and creating the shortcut...
".venv-desktop\Scripts\python.exe" -X utf8 setup_desktop.py
if errorlevel 1 goto failed
echo.
echo Ready. Double-click BPM Finder.lnk. No CMD is needed for normal use.
pause
exit /b 0
:failed
echo.
echo Installation failed. No changes were made to system Python or GPU drivers.
echo Check your internet connection and keep the error above for diagnosis.
echo The portable Windows package does not require this installer.
pause
exit /b 1
