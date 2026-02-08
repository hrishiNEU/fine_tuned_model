@echo off
REM Setup script for News Summarization with Bias Detection Project (Windows)
REM This script sets up the environment and verifies all dependencies

echo ==================================
echo Project Setup (Windows)
echo ==================================
echo.

REM Check Python version
echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)
echo.

REM Create directory structure
echo Creating project directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\cache" mkdir data\cache
if not exist "data\splits" mkdir data\splits
if not exist "models\baseline" mkdir models\baseline
if not exist "models\config1" mkdir models\config1
if not exist "models\config2" mkdir models\config2
if not exist "models\config3" mkdir models\config3
if not exist "models\checkpoints" mkdir models\checkpoints
if not exist "models\final" mkdir models\final
if not exist "results\logs" mkdir results\logs
if not exist "results\visualizations" mkdir results\visualizations
if not exist "notebooks" mkdir notebooks
if not exist "src" mkdir src

echo Directory structure created
echo.

REM Create virtual environment
echo Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
echo.

REM Install requirements
echo Installing Python packages...
if exist "requirements.txt" (
    pip install -r requirements.txt
    echo Python packages installed
) else (
    echo ERROR: requirements.txt not found
    pause
    exit /b 1
)
echo.

REM Check CUDA availability
echo Checking CUDA availability...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
echo.

REM Create .gitignore if it doesn't exist
echo Creating .gitignore...
if not exist ".gitignore" (
    (
        echo # Python
        echo __pycache__/
        echo *.py[cod]
        echo *$py.class
        echo *.so
        echo .Python
        echo venv/
        echo env/
        echo ENV/
        echo.
        echo # Jupyter
        echo .ipynb_checkpoints/
        echo *.ipynb
        echo.
        echo # Data and Models
        echo data/cache/
        echo data/raw/*
        echo !data/raw/.gitkeep
        echo models/*/checkpoint-*
        echo *.bin
        echo *.safetensors
        echo *.ckpt
        echo.
        echo # Results
        echo results/logs/*.log
        echo wandb/
        echo.
        echo # IDE
        echo .vscode/
        echo .idea/
        echo *.swp
        echo *.swo
        echo.
        echo # OS
        echo .DS_Store
        echo Thumbs.db
        echo.
        echo # Misc
        echo *.log
        echo .env
    ) > .gitignore
    echo .gitignore created
) else (
    echo .gitignore already exists
)
echo.

REM Create .gitkeep files
type nul > data\raw\.gitkeep
type nul > models\.gitkeep
type nul > results\.gitkeep

REM Verify installations
echo ==================================
echo Verifying Installation
echo ==================================
echo.

python -c "import sys; packages = {'torch': 'PyTorch', 'transformers': 'Transformers', 'datasets': 'Datasets', 'evaluate': 'Evaluate', 'peft': 'PEFT', 'gradio': 'Gradio', 'wandb': 'Weights ^& Biases'}; all_good = True; [print(f'✓ {name}') if __import__(package) or True else (print(f'✗ {name} - FAILED'), setattr(sys.modules[__name__], 'all_good', False)) for package, name in packages.items()]; sys.exit(0 if all_good else 1)"

if %errorlevel% equ 0 (
    echo.
    echo ==================================
    echo Setup Complete!
    echo ==================================
    echo.
    echo Next steps:
    echo 1. Keep the virtual environment activated
    echo    venv\Scripts\activate.bat
    echo.
    echo 2. Review and customize config.yaml
    echo.
    echo 3. Run a quick test:
    echo    python main.py --all --quick-test
    echo.
    echo 4. For full pipeline:
    echo    python main.py --all
    echo.
    echo 5. For just inference interface:
    echo    python main.py --inference --model-path ^<path-to-model^>
    echo.
) else (
    echo.
    echo ==================================
    echo Setup encountered errors
    echo ==================================
    echo Please check the error messages above and try again.
    pause
    exit /b 1
)

pause