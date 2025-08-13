import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import os

all_fold_accuracies = []
all_fold_f1_scores = []
all_fold_pr_aucs = []
all_fold_roc_aucs = []

# Loop through the data for each fold
for fold in range(1, 6):
    print(f"\n--- Processing data for fold {fold} ---")
    preds_file = f'fold_{fold}_preds.npy'
    labels_file = f'fold_{fold}_labels.npy'

    # Load data from files
    try:
        preds = np.load(preds_file)
        true_labels = np.load(labels_file)
    except FileNotFoundError:
        print(f"Warning: Data files for fold {fold} not found. Skipped.")
        continue

    # Extract prediction probabilities for the positive class (class_1)
    positive_class_preds = preds[:, 1]

    # 1. Calculate all metrics
    predicted_classes = (positive_class_preds > 0.5).astype(int)
    accuracy = accuracy_score(true_labels, predicted_classes)
    f1 = f1_score(true_labels, predicted_classes)

    precision, recall, _ = precision_recall_curve(true_labels, positive_class_preds)
    pr_auc = auc(recall, precision)

    fpr, tpr, _ = roc_curve(true_labels, positive_class_preds)
    roc_auc = auc(fpr, tpr)

    # Save metrics for later averaging
    all_fold_accuracies.append(accuracy)
    all_fold_f1_scores.append(f1)
    all_fold_pr_aucs.append(pr_auc)
    all_fold_roc_aucs.append(roc_auc)

    # 2. Save metrics to a text file
    output_file = f'fold_{fold}_evaluation_metrics.txt'
    with open(output_file, 'w') as f:
        f.write(f"Fold {fold} Evaluation Results\n")
        f.write("---------------------\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Precision-Recall AUC: {pr_auc:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")

    print(f"Metrics saved to file: '{output_file}'")

    # 3. Plot and save the Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Fold {fold} Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(f'fold_{fold}_precision_recall_curve.png')
    plt.close()

    # 4. Plot and save the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Fold {fold} Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'fold_{fold}_roc_curve.png')
    plt.close()

# Print the average metrics across all folds as a final summary
print("\n--- Average Evaluation Results Across All Folds ---")
print(f"Average Accuracy: {np.mean(all_fold_accuracies):.4f}")
print(f"Average F1 Score: {np.mean(all_fold_f1_scores):.4f}")
print(f"Average Precision-Recall AUC: {np.mean(all_fold_pr_aucs):.4f}")
print(f"Average ROC AUC: {np.mean(all_fold_roc_aucs):.4f}")