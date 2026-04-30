
@echo off
title Streamlit App Runner

echo Installing requirements...
pip install -r requirements.txt

@REM python train_models.py

echo.
echo Starting Streamlit app...
streamlit run streamlit_app.py

pause