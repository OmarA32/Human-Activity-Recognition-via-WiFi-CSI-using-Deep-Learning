@echo off
call myenv\Scripts\activate
start cmd /k "python web_server.py"
start cmd /k "python run.py --model ViT --dataset UT_HAR_data"