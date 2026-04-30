# Crop Yield Prediction (Dakshina Kannada)

This repo contains:
- Streamlit UI: `streamlit_app.py`
- Model training pipeline: `train_models.py`
- CLI predictor: `predict.py`

## Send this project to someone (no dependency conflicts)

The key is: **don’t install into a global Python**. Always use a **fresh virtual environment**.

### 1) Zip the source (recommended)

Share the whole folder **except** these (they’re environment-specific or large):
- `.venv/` (never share a venv)
- `__pycache__/`
- (optional) `outputs/models/` if you don’t need to ship trained models

### 2) Receiver setup (Windows / PowerShell)

**Prerequisite:** Python 3.11+ installed.

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip

# Install only the Streamlit app deps (recommended for users)
pip install -r requirements.app.txt

# OR install training deps too
pip install -r requirements.txt
```

Run the app:

```powershell
streamlit run streamlit_app.py
```

### 3) If you want maximum reproducibility

On the machine where you verified everything works, generate a lock file:

```powershell
.\.venv\Scripts\Activate.ps1
pip freeze > requirements.lock.txt
```

Then the receiver can do:

```powershell
pip install -r requirements.lock.txt
```

Notes:
- If your app uses TensorFlow, the exact working versions can differ by OS/Python. That’s why TensorFlow is separated into `requirements.train.txt`.
- If you need truly identical environments across machines, use Docker or Conda.
