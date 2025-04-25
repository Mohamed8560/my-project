import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Load data
df_pca = pd.read_csv("pca_data.csv")
X = df_pca.drop('Class', axis=1)
y = df_pca['Class']
classes = sorted(y.unique())
y_bin = label_binarize(y, classes=classes)

# Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM (RBF)': SVC(kernel='rbf', probability=True),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': GaussianNB()
}

# Plot ROC curves (One-vs-Rest strategy)
plt.figure(figsize=(10, 8))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train, test in cv.split(X, y):
        model.fit(X.iloc[train], y.iloc[train])
        y_score = model.predict_proba(X.iloc[test])
        fpr, tpr, _ = roc_curve(y_bin[test].ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, alpha=0.3, label=f'{name} Fold AUC = {roc_auc:.2f}')

    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.title("ROC Curves for Classifiers")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curves.png")
plt.show()
