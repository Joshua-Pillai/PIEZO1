#performance of individual models

thresholds = {
    'AM': (0.34, 0.56),
    'ESM (-LLR)': (7.0, 8.0),
    'CADD': (15.0, 15.0),
    'EVE': (0.35, 0.65)
}

for i, feature in enumerate(features):
    raw_scores = X_raw[:, i]
    benign_max, pathogenic_min = thresholds[feature]

    preds = np.full_like(y, fill_value=-1)
    preds[raw_scores < benign_max] = 0
    preds[raw_scores > pathogenic_min] = 1

    ambiguous = (preds == -1)
    preds[ambiguous] = 1 - y[ambiguous]

    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, raw_scores)

    print(f"{feature:12s} | AUC: {auc:.3f} | Accuracy (with ambiguity penalized): {acc:.3f}")
