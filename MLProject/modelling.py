"""
modelling.py — MLProject Entry Point
======================================
Digunakan oleh workflow CI untuk melatih ulang model secara otomatis.
Menerima hyperparameter via argumen CLI.

Contoh:
    python modelling.py 150 4 0.05
"""

import argparse
import json
import os
import sys

import dagshub
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ── CLI Arguments ──────────────────────────────────────────
parser = argparse.ArgumentParser(description="Heart Disease GBM Training")
parser.add_argument("n_estimators",  type=int,   nargs="?", default=150)
parser.add_argument("max_depth",     type=int,   nargs="?", default=4)
parser.add_argument("learning_rate", type=float, nargs="?", default=0.1)
args = parser.parse_args()

# ── DagsHub Tracking ───────────────────────────────────────
os.environ["DAGSHUB_USER_TOKEN"] = os.getenv("DAGSHUB_TOKEN", "")
dagshub.init(
    repo_owner=os.getenv("DAGSHUB_USERNAME"),
    repo_name="HeartDisease-MLProject",
    mlflow=True
)

# ── Load data ───────────────────────────────────────────────
X_train = pd.read_csv("heart_preprocessing/X_train.csv")
X_test  = pd.read_csv("heart_preprocessing/X_test.csv")
y_train = pd.read_csv("heart_preprocessing/y_train.csv").squeeze()
y_test  = pd.read_csv("heart_preprocessing/y_test.csv").squeeze()

print(f"[DATA] Train: {X_train.shape} | Test: {X_test.shape}")

# ── Training ────────────────────────────────────────────────
mlflow.set_experiment("HeartDisease-CI")

with mlflow.start_run(run_name=f"CI-GBM-n{args.n_estimators}-d{args.max_depth}") as run:
    mlflow.log_param("n_estimators",  args.n_estimators)
    mlflow.log_param("max_depth",     args.max_depth)
    mlflow.log_param("learning_rate", args.learning_rate)

    model = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall",    rec)
    mlflow.log_metric("roc_auc",   auc)

    # Confusion Matrix artefak
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="OrRd")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold")
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["No Disease", "Heart Disease"],
           yticklabels=["No Disease", "Heart Disease"],
           title="Confusion Matrix (CI Run)", ylabel="Aktual", xlabel="Prediksi")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100)
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # Simpan run_id
    with open("latest_run_id.txt", "w") as f:
        f.write(run.info.run_id)

    mlflow.sklearn.log_model(model, "model")

    print(f"[RESULT] run_id   : {run.info.run_id}")
    print(f"[RESULT] Accuracy : {acc:.4f}")
    print(f"[RESULT] F1 Score : {f1:.4f}")
    print(f"[RESULT] ROC AUC  : {auc:.4f}")
    print("Training selesai!")
