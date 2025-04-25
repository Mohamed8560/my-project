import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, make_scorer

# Load PCA-transformed data
df_pca = pd.read_csv("pca_data.csv")
X = df_pca.drop('Class', axis=1)
y = df_pca['Class']

# Classifiers
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM (RBF)': SVC(kernel='rbf', probability=True),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': GaussianNB()
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    results.append({
        'Model': name,
        'Mean Accuracy': scores.mean(),
        'Std Dev': scores.std()
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("model_performance.csv", index=False)
print(results_df)
