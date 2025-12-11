@echo off
REM Check if virtual environment exists
if not exist "myenv" (
    echo Creating virtual environment...
    python -m venv myenv
)

echo Activating virtual environment...
call myenv\Scripts\activate

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo Starting web server...
start cmd /k "python web_server.py"

echo Starting model training script...
start cmd /k "python run.py --model ViT --dataset UT_HAR_data"