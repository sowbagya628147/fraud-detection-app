import io
import json
import random
from pathlib import Path

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE = Path(__file__).parent
meta = json.loads((BASE / "meta.json").read_text())
samples = json.loads((BASE / "samples.json").read_text())
FEATURES: list[str] = meta["features"]
THRESHOLD: float = meta["threshold"]

model = xgb.XGBClassifier()
model.load_model(BASE / "model.json")

app = FastAPI(title="Fraud Detection API")
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"]
)


class Transaction(BaseModel):
    features: dict[str, float]


def score(df: pd.DataFrame, explain: bool = False):
    df = df[FEATURES].astype(float)
    proba = model.predict_proba(df)[:, 1]
    contribs = None
    if explain:
        # XGBoost's built-in per-feature contributions (SHAP values); last column is the bias.
        contribs = model.get_booster().predict(xgb.DMatrix(df), pred_contribs=True)
    return proba, contribs


@app.get("/health")
def health():
    return {"status": "ok", "threshold": THRESHOLD, "metrics": meta["metrics"]}


@app.get("/sample")
def sample(kind: str = "legit"):
    if kind not in samples:
        raise HTTPException(400, "kind must be 'fraud' or 'legit'")
    return random.choice(samples[kind])


@app.post("/predict")
def predict(tx: Transaction):
    missing = [f for f in FEATURES if f not in tx.features]
    if missing:
        raise HTTPException(422, f"Missing features: {', '.join(missing)}")
    proba, contribs = score(pd.DataFrame([tx.features]), explain=True)
    p = float(proba[0])
    top = sorted(zip(FEATURES, contribs[0][:-1]), key=lambda t: abs(t[1]), reverse=True)[:5]
    return {
        "probability": p,
        "threshold": THRESHOLD,
        "is_fraud": p >= THRESHOLD,
        "drivers": [{"feature": f, "impact": float(v)} for f, v in top],
    }


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    df = pd.read_csv(io.BytesIO(await file.read()))
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise HTTPException(422, f"CSV is missing columns: {', '.join(missing)}")
    proba, _ = score(df)
    out = df.assign(probability=proba, is_fraud=proba >= THRESHOLD)
    if "id" not in out.columns:
        out["id"] = out.index
    flagged = out[out.is_fraud].sort_values("probability", ascending=False)
    return {
        "total": len(out),
        "flagged": len(flagged),
        "rows": flagged[["id", "Amount", "probability"]].head(200).to_dict("records"),
    }
