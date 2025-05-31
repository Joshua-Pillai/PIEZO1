from matplotlib.ticker import MaxNLocator

benign_mask = (y == 0) & (ensemble_scores < 0.4)
pathogenic_mask = (y == 1) & (ensemble_scores > 0.6)
ambiguous_mask = (ensemble_scores >= 0.4) & (ensemble_scores <= 0.6)

with sns.axes_style("ticks"):
    plt.figure(figsize=(7, 5))
    plt.hist(ensemble_scores[benign_mask], bins=15, alpha=0.3, label='Benign')
    plt.hist(ensemble_scores[pathogenic_mask], bins=20, alpha=0.3, label='Pathogenic', color='red')
    plt.hist(ensemble_scores[ambiguous_mask], bins=15, alpha=0.3, color='gray', label='Ambiguous')
    plt.xlabel("Weighted Pathogenicity Score")
    plt.ylabel("Frequency")
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.savefig("distribution.png", dpi=1200)
    plt.show()

#roc-auc analyses

from sklearn.metrics import roc_curve, roc_auc_score

palette = sns.color_palette("viridis", len(features))

plt.figure(figsize=(10, 7))
plt.style.use("default")

fpr, tpr, _ = roc_curve(y, ensemble_scores)
auc = roc_auc_score(y, ensemble_scores)
plt.plot(fpr, tpr, label=f'Ensemble (AUC = {auc:.4f})', linewidth=2.5, color='Purple')

for i, feature in enumerate(features):
    model_scores = X_scaled[:, i]
    fpr_i, tpr_i, _ = roc_curve(y, model_scores)
    auc_i = roc_auc_score(y, model_scores)
    plt.plot(fpr_i, tpr_i, label=f'{feature} (AUC = {auc_i:.4f})',
             linewidth=1.5, color=palette[i], alpha=0.8)

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.grid(False)
plt.tight_layout()
plt.show()
