"""
LukOut - zajednicke funkcije za ispis i grafike (koristi ih i lukout.py i lukout_nasaNN.py)
"""

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def print_report(true, pred, p_edible, split_name, class_names, edible_idx, confidence_threshold):
    acc = accuracy_score(true, pred)
    fpr, tpr, _ = roc_curve((true == edible_idx).astype(int), p_edible)
    roc_auc = auc(fpr, tpr)

    print(f"\n{split_name} (prag = {confidence_threshold})")
    print(classification_report(true, pred, target_names=class_names, digits=4))
    print(f"Tacnost (Accuracy): {acc:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    return fpr, tpr, roc_auc


def plot_confusion_matrix(ax, cm, class_names, title):
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    threshold = cm.max() * 0.55
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm[i, j] >= threshold else "black"
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color)
    ax.set_title(title, fontweight="bold", fontsize=12, pad=10)
    ax.set_ylabel("Stvarna klasa", fontsize=10)
    ax.set_xlabel("Predvidjena klasa", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.75)


def full_eval(true, pred, p_edible, split_name, class_names, edible_idx, confidence_threshold):
    cm = confusion_matrix(true, pred)
    fpr, tpr, roc_auc = print_report(true, pred, p_edible, split_name, class_names, edible_idx, confidence_threshold)
    return cm, fpr, tpr, roc_auc


def print_final_report(true_val, pred_val, true_test, pred_test,
                        auc_val, auc_test, class_names, inedible_idx,
                        confidence_threshold, model_name, project_name):
    acc_val_s  = accuracy_score(true_val,  pred_val)
    acc_test_s = accuracy_score(true_test, pred_test)

    f1_val_s   = f1_score(true_val,  pred_val,  average="macro")
    f1_test_s  = f1_score(true_test, pred_test, average="macro")

    prec_val_s  = precision_score(true_val,  pred_val,  average="macro")
    prec_test_s = precision_score(true_test, pred_test, average="macro")

    rec_val_s   = recall_score(true_val,  pred_val,  average="macro")
    rec_test_s  = recall_score(true_test, pred_test, average="macro")

    rec_inedible_val  = recall_score(true_val,  pred_val,  pos_label=inedible_idx)
    rec_inedible_test = recall_score(true_test, pred_test, pos_label=inedible_idx)

    print(f"\nFINALNI IZVESTAJ - {project_name}")
    print(f"Model: {model_name}")
    print(f"Klase: {class_names}")
    print(f"Safety prag: {confidence_threshold} (ispod -> nejestivo)")
    print()
    print(f"{'Metrika':<28} {'Val':>8} {'Test':>8}")
    print(f"{'Tacnost (Accuracy)':<28} {acc_val_s:>8.4f} {acc_test_s:>8.4f}")
    print(f"{'F1-score (macro)':<28} {f1_val_s:>8.4f} {f1_test_s:>8.4f}")
    print(f"{'Preciznost (macro)':<28} {prec_val_s:>8.4f} {prec_test_s:>8.4f}")
    print(f"{'Odziv (macro)':<28} {rec_val_s:>8.4f} {rec_test_s:>8.4f}")
    print(f"{'AUC-ROC':<28} {auc_val:>8.4f} {auc_test:>8.4f}")
    print(f"{'Recall(nejestivo)':<28} {rec_inedible_val:>8.4f} {rec_inedible_test:>8.4f}")
