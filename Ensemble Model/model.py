import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/content/ClinVar - Cleaned - Sheet1 (1).csv")

#clean to get binary classification
label_map = {
    "Benign": "Benign",
    "Likely benign": "Benign",
    "Benign/Likely benign": "Benign",
    "Pathogenic": "Pathogenic",
    "Likely pathogenic": "Pathogenic",
    "Pathogenic/Likely pathogenic": "Pathogenic"
}
df['Germline classification'] = df['Germline classification'].map(label_map)
df['Label'] = df['Germline classification'].map({'Benign': 0, 'Pathogenic': 1})

#original model scores
features = ['AM', 'ESM (-LLR)', 'CADD', 'EVE']
df_model = df[features + ['Label']].dropna()

X_raw = df_model[features].values
y = df_model['Label'].astype(int).values

#normalizing
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

#dynamically adjusting the model weights
def objective(weights):
    scores = X_scaled @ weights
    return -roc_auc_score(y, scores)

initial_weights = np.ones(X_scaled.shape[1]) / X_scaled.shape[1]
bounds = [(0, 1)] * X_scaled.shape[1]
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = result.x
print("✅ Optimal Weights:")
for f, w in zip(features, optimal_weights):
    print(f"{f}: {w:.3f}")

#output
ensemble_scores = X_scaled @ optimal_weights
predictions = np.full_like(y, fill_value=-1)
predictions[ensemble_scores < 0.4] = 0
predictions[ensemble_scores > 0.6] = 1
ambiguous = (predictions == -1)
predictions[ambiguous] = 1 - y[ambiguous]

accuracy = accuracy_score(y, predictions)
auc = roc_auc_score(y, ensemble_scores)

print(f"Accuracy: {accuracy:.3f}")
print(f"AUC: {auc:.3f}")
