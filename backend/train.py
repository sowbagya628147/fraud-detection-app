"""Train the fraud model.

Usage (from the backend folder):
  1. Put creditcard_2023.csv in backend/data/
  2. python train.py
Outputs: model.json, meta.json, samples.json (used by the API).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

BASE = Path(__file__).parent
DATA = BASE / "data" / "creditcard_2023.csv"
TARGET_PRECISION = 0.90  # flag a transaction only when we're right ~9 times in 10

if not DATA.exists():
    raise SystemExit(f"Dataset not found: {DATA}\nPut creditcard_2023.csv inside backend/data/")

df = pd.read_csv(DATA)
# 'id' is just a row number, so it must not be used as a feature.
features = [c for c in df.columns if c not in ("Class", "id")]
X, y = df[features], df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# This dataset is ~50% fraud, so the weight is ~1; the formula still works for imbalanced data.
weight = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=weight,
    eval_metric="aucpr",
    n_jobs=-1,
    random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

proba = model.predict_proba(X_test)[:, 1]
pr_auc = average_precision_score(y_test, proba)
roc_auc = roc_auc_score(y_test, proba)

# Pick the lowest threshold that still meets the precision target (max recall).
prec, rec, thr = precision_recall_curve(y_test, proba)
ok = np.where(prec[:-1] >= TARGET_PRECISION)[0]
threshold = float(thr[ok[0]]) if len(ok) else 0.5
recall_at_threshold = float(rec[ok[0]]) if len(ok) else None

model.save_model(BASE / "model.json")
meta = {
    "features": features,
    "threshold": threshold,
    "metrics": {
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "precision_target": TARGET_PRECISION,
        "recall_at_threshold": recall_at_threshold,
    },
}
(BASE / "meta.json").write_text(json.dumps(meta, indent=2))

# A few held-out examples so the UI can load a real transaction.
fraud = X_test[y_test == 1].sample(5, random_state=1)
legit = X_test[y_test == 0].sample(5, random_state=1)
(BASE / "samples.json").write_text(
    json.dumps({"fraud": fraud.to_dict("records"), "legit": legit.to_dict("records")})
)
print(json.dumps(meta["metrics"], indent=2), "threshold:", round(threshold, 4))
