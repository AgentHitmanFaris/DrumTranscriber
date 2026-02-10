@echo off
setlocal enabledelayedexpansion

REM --- Configuration ---
set PYTHON_VER=3.10.11
set PYTHON_ZIP=python-%PYTHON_VER%-embed-amd64.zip
set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VER%/%PYTHON_ZIP%
set PYTHON_DIR=python_embed

REM --- FFmpeg Configuration ---
set FFMPEG_ZIP=ffmpeg-release-essentials.zip
set FFMPEG_URL=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
set BIN_DIR=bin

REM --- Setup Python ---
if not exist "%PYTHON_DIR%\python.exe" (
    echo [INFO] Python environment not found. Setting up portable Python...
    
    if not exist "%PYTHON_DIR%" mkdir "%PYTHON_DIR%"
    
    if not exist "%PYTHON_ZIP%" (
        echo Downloading Python %PYTHON_VER%...
        curl -L -o "%PYTHON_ZIP%" "%PYTHON_URL%"
        if errorlevel 1 (
            echo [ERROR] Failed to download Python. Please check your internet connection.
            pause
            exit /b 1
        )
    )
    
    echo Extracting Python...
    powershell -Command "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"
    
    REM Clean up zip
    del "%PYTHON_ZIP%"
    
    REM Configure python310._pth to uncomment 'import site' to enable pip
    echo Configuring Python for pip...
    set "PTH_FILE=%PYTHON_DIR%\python310._pth"
    powershell -Command "(Get-Content '%PYTHON_DIR%\python310._pth') -replace '#import site', 'import site' | Set-Content '%PYTHON_DIR%\python310._pth'"
    
    echo Downloading get-pip.py...
    curl -L -o get-pip.py https://bootstrap.pypa.io/get-pip.py
    
    echo Installing pip...
    "%PYTHON_DIR%\python.exe" get-pip.py --no-warn-script-location
    del get-pip.py
)

REM --- Setup FFmpeg ---
if exist "ffmpeg\ffmpeg.exe" (
    echo [INFO] Found local FFmpeg in 'ffmpeg' folder.
    set "PATH=%CD%\ffmpeg;%PATH%"
) else if not exist "%BIN_DIR%\ffmpeg.exe" (
    echo [INFO] FFmpeg not found. Downloading...
    
    if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"
    
    if not exist "%FFMPEG_ZIP%" (
        echo Downloading FFmpeg...
        curl -L -o "%FFMPEG_ZIP%" "%FFMPEG_URL%"
    )
    
    echo Extracting FFmpeg...
    powershell -Command "Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath 'ffmpeg_temp' -Force"
    
    echo Installing FFmpeg binaries...
    REM Move ffmpeg.exe and ffprobe.exe from the nested folder to bin
    for /d %%D in (ffmpeg_temp\*) do (
        if exist "%%D\bin\ffmpeg.exe" (
            copy "%%D\bin\ffmpeg.exe" "%BIN_DIR%\" >nul
            copy "%%D\bin\ffprobe.exe" "%BIN_DIR%\" >nul
        )
    )
    
    REM Clean up
    rmdir /s /q ffmpeg_temp
    del "%FFMPEG_ZIP%"
    
    REM Add bin to PATH for this session so yt-dlp can find ffmpeg
    set "PATH=%CD%\%BIN_DIR%;%PATH%"
) else (
    echo [INFO] Found downloaded FFmpeg in '%BIN_DIR%' folder.
    set "PATH=%CD%\%BIN_DIR%;%PATH%"
)
REM --- Install Dependencies ---
echo [INFO] Checking/Installing dependencies...
REM Uninstall standalone keras 3.x if present (conflicts with tf.keras in TF<2.16)
"%PYTHON_DIR%\python.exe" -m pip uninstall -y keras 2>nul
"%PYTHON_DIR%\python.exe" -m pip install --no-warn-script-location "tensorflow==2.15.1"
"%PYTHON_DIR%\python.exe" -m pip install --no-warn-script-location -r requirements.txt
"%PYTHON_DIR%\python.exe" -m pip install --no-warn-script-location gradio yt-dlp plotly gdown

REM --- Check/Download Model ---
if not exist "model\drum_transcriber.h5" (
    echo [INFO] Model file not found. Downloading...
    if not exist "model" mkdir "model"
    
    REM downloading with gdown
    "%PYTHON_DIR%\python.exe" -c "import gdown; url = 'https://drive.google.com/uc?id=1w2fIHeyr-st3sbk1PYrtGOYW6YAD1fsi'; output = 'model/drum_transcriber.h5'; gdown.download(url, output, quiet=False)"
)

REM --- Run App ---
echo [INFO] Starting Drum Transcriber...
echo The app should open in your default browser. Close this window to stop it.
"%PYTHON_DIR%\python.exe" gradio_app.py

pause
