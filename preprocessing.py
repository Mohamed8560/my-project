# --- IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- LOAD DATASET ---
df = pd.read_excel("Dry_Bean_Dataset.xlsx")  # Make sure the .xlsx file is in the same folder

# --- ADD MISSING VALUES ---
df_missing = df.copy()
np.random.seed(42)

# Add 5% missing values to two columns
for col in ['Area', 'Perimeter']:
    df_missing.loc[df_missing.sample(frac=0.05).index, col] = np.nan

# Add 35% missing to one column
df_missing.loc[df_missing.sample(frac=0.35).index, 'Eccentricity'] = np.nan

# --- HANDLE MISSING VALUES ---
# Fill 5% missing with mean or median
df_missing['Area'].fillna(df_missing['Area'].median(), inplace=True)
df_missing['Perimeter'].fillna(df_missing['Perimeter'].mean(), inplace=True)

# Drop column with 35% missing
df_missing.drop(columns=['Eccentricity'], inplace=True)

# --- OUTLIER HANDLING (IQR METHOD) ---
def iqr_outlier_handling(df):
    df_iqr = df.copy()
    for col in df_iqr.columns:
        Q1 = df_iqr[col].quantile(0.25)
        Q3 = df_iqr[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_iqr[col] = np.clip(df_iqr[col], lower, upper)
    return df_iqr

# Apply to numeric features
numerical_features = df_missing.drop(columns=['Class'])
df_iqr_cleaned = iqr_outlier_handling(numerical_features)
df_iqr_cleaned['Class'] = df_missing['Class']  # Add back target column

# --- FEATURE SCALING (STANDARD SCALER) ---
X = df_iqr_cleaned.drop(columns='Class')
y = df_iqr_cleaned['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df['Class'] = y.values

# --- CATEGORICAL ENCODING ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(X_scaled_df['Class'])

# Final dataset
X_final = X_scaled_df.drop(columns='Class')
y_final = y_encoded

# --- OPTIONAL: PRINT INFO ---
print("✅ Data preprocessing complete.")
print(f"Encoded classes: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
print(f"Final shape of X: {X_final.shape}")
print(f"Final shape of y: {y_final.shape}")

# Save cleaned and preprocessed dataset for PCA/LDA and modeling
df.to_csv("processed_data.csv", index=False)
