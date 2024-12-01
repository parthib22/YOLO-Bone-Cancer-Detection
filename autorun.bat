@echo off
REM Set the virtual environment folder name
set VENV_DIR=.venv

REM Check if the virtual environment exists
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "%VENV_DIR%\Scripts\activate.bat"
    echo Running the Python program in the virtual environment...
    python test2.py
) else (
    echo Virtual environment not found. Running the Python program globally...
    python test2.py
)

REM Pause to view any output or errors
pause
