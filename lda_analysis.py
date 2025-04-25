import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
import joblib

# --- LOAD CLEAN DATA ---
df = pd.read_csv("processed_data.csv")  # same file used for PCA
X = df.drop('Class', axis=1)
y = df['Class']

# --- SCALE DATA ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- APPLY LDA ---
lda = LDA(n_components=3)
X_lda = lda.fit_transform(X_scaled, y)

# Save LDA-transformed data
lda_df = pd.DataFrame(X_lda, columns=['LD1', 'LD2', 'LD3'])
lda_df['Class'] = y
lda_df.to_csv("lda_data.csv", index=False)

# --- 2D SCATTER PLOT ---
plt.figure(figsize=(8, 6))
scatter = plt.scatter(lda_df['LD1'], lda_df['LD2'], c=pd.factorize(y)[0], cmap='viridis', alpha=0.7)
plt.title("LDA - First 2 Discriminant Components")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.colorbar(scatter, label='Class')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Save LDA model for reuse ---
joblib.dump(lda, "lda_model.pkl")
