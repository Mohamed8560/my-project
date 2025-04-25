import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

# --- Load your processed data ---
df = pd.read_csv("processed_data.csv")  # this file should include features + 'Class'

# --- Encode class labels ---
le = LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])  # Ensure numeric classes
class_names = le.classes_
n_classes = len(class_names)

# --- Prepare features and labels ---
X = df.drop('Class', axis=1)
y = df['Class']
y_bin = label_binarize(y, classes=np.unique(y))

# --- Train-test split (simulating the best outer fold split) ---
X_train, X_test, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.3, random_state=42, stratify=y)

# --- Train model (replace with your best model if needed) ---
clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
clf.fit(X_train, y_train_bin)
y_score = clf.predict_proba(X_test)

# --- Compute ROC curves and AUC for each class ---
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# --- Plot ROC curves ---
plt.figure(figsize=(10, 8))
colors = plt.cm.get_cmap('tab10', n_classes)

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})", color=colors(i))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (One-vs-All) – Dry Bean Classes')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
