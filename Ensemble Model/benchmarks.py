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

#plotting of cm

from sklearn.metrics import confusion_matrix, classification_report, roc_curve
report = classification_report(y, predictions, target_names=['Benign', 'Pathogenic'])
print(report)

labels = ['Benign', 'Pathogenic']
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()
