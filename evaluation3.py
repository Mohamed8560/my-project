# evaluation.py
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import numpy as np

with open("model_results.pkl", "rb") as f:
    results = pickle.load(f)

# Class labels (adjust if necessary)
class_labels = list(range(7))

# Binarize labels for ROC AUC
from sklearn.preprocessing import LabelEncoder
import pandas as pd
y_raw = pd.read_csv("processed_data.csv")["Class"]
le = LabelEncoder()
y_bin = label_binarize(le.fit_transform(y_raw), classes=class_labels)

def plot_roc_curve(models, X, y_true, title):
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f"Fold {i+1} AUC = {roc_auc:.2f}")

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {title}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"roc_{title.lower().replace(' ', '_')}.png")
    plt.show()

# Example: plot ROC for Random Forest on Raw data
from sklearn.model_selection import StratifiedKFold
X_raw = pd.read_csv("processed_data.csv").drop("Class", axis=1).values
y_raw = le.transform(y_raw)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

plot_roc_curve(
    results["Raw"]["Random Forest"]["models"],
    X_raw,
    label_binarize(y_raw, classes=class_labels),
    "Random Forest on Raw Data"
)

