# Credit card fraud detection app

FastAPI + XGBoost backend, React (Vite) frontend. Dataset: Kaggle `creditcard_2023.csv`.

Open TWO PowerShell terminals in VS Code (Terminal > New Terminal).

## Terminal 1: backend
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
mkdir data
# copy creditcard_2023.csv into backend\data\
python train.py
uvicorn main:app --reload
```
API docs: http://localhost:8000/docs

If activation says "running scripts is disabled", run this once and retry:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

## Terminal 2: frontend (needs Node.js 18+)
```powershell
cd frontend
npm install
npm run dev
```
Open http://localhost:5173 (keep the backend running).

## Order matters
`train.py` must finish first. It creates model.json, meta.json and samples.json, and
`uvicorn main:app` fails without them.
