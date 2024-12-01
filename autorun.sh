#!/bin/bash

# Set the virtual environment directory
VENV_DIR=".venv"

# Check if the virtual environment exists
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    echo "Running the Python program in the virtual environment..."
    python test2.py
else
    echo "Virtual environment not found. Running the Python program globally..."
    python test2.py
fi

# Pause to view output
read -p "Press Enter to exit..."
