import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load PCA-transformed or LDA-transformed data
df = pd.read_csv("pca_data.csv")  # or 'lda_data.csv'
X = df.drop('Class', axis=1)
y = df['Class']

# Define models and their hyperparameters
models = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10]
        }
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
    }
}

# Outer loop: 5 different random train-test splits
outer_results = {}

for model_name, mp in models.items():
    accuracies = []
    all_preds = []
    all_probas = []
    all_true = []
    
    for i in range(5):
        # New random split each time
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=np.random.randint(10000)
        )

        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        clf = GridSearchCV(mp["model"], mp["params"], cv=inner_cv)
        clf.fit(X_train, y_train)

        best_model = clf.best_estimator_
        preds = best_model.predict(X_test)
        probas = best_model.predict_proba(X_test)

        accuracies.append(accuracy_score(y_test, preds))
        all_preds.extend(preds)
        all_probas.extend(probas)
        all_true.extend(y_test)

    outer_results[model_name] = {
        "accuracy": np.mean(accuracies),
        "std": np.std(accuracies),
        "y_true": all_true,
        "y_pred": all_preds,
        "y_proba": all_probas
    }

    print(f"{model_name} - Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

# Save results
np.save("model_results.npy", outer_results)
