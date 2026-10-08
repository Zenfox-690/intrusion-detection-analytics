"""Model evaluation and visualization utilities for Intrusion Detection.

Computes accuracy, precision, recall (detection rate), F1-score, false-positive
rate (FPR), per-class metrics, and saves formatted reports and confusion matrices.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from src.data import CLASS_NAMES


def compute_metrics(
    y_true: Union[np.ndarray, List[str]],
    y_pred: Union[np.ndarray, List[str]],
    model_name: str = "Model",
    classes: Optional[List[str]] = None
) -> Dict:
    """Compute comprehensive multiclass and intrusion-detection-specific metrics."""
    if classes is None:
        classes = CLASS_NAMES

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average='macro', zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average='weighted', zero_division=0
    )

    # Detailed per-class classification report
    report_dict = classification_report(
        y_true, y_pred, labels=classes, target_names=classes, output_dict=True, zero_division=0
    )

    # IDS-specific False Positive Rate (FPR) and Detection Rate (DR)
    # Normal is class 'Normal'; Attacks are DoS, Probe, R2L, U2R
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    normal_mask = (y_true_arr == 'Normal')
    total_normal = int(np.sum(normal_mask))
    if total_normal > 0:
        # False Positives: Actual Normal falsely predicted as any attack
        fp_count = int(np.sum((y_pred_arr[normal_mask] != 'Normal')))
        false_positive_rate = float(fp_count / total_normal)
    else:
        fp_count = 0
        false_positive_rate = 0.0

    attack_mask = (y_true_arr != 'Normal')
    total_attack = int(np.sum(attack_mask))
    if total_attack > 0:
        # True Positives: Actual Attack correctly predicted as any attack
        tp_attack_count = int(np.sum((y_pred_arr[attack_mask] != 'Normal')))
        detection_rate = float(tp_attack_count / total_attack)
    else:
        tp_attack_count = 0
        detection_rate = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    return {
        'model_name': model_name,
        'accuracy': acc,
        'precision_macro': float(p_macro),
        'recall_macro': float(r_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(p_weighted),
        'recall_weighted': float(r_weighted),
        'f1_weighted': float(f1_weighted),
        'detection_rate': detection_rate,
        'false_positive_rate': false_positive_rate,
        'total_normal': total_normal,
        'false_positives': fp_count,
        'total_attacks': total_attack,
        'detected_attacks': tp_attack_count,
        'confusion_matrix': cm.tolist(),
        'per_class_report': report_dict,
        'classes': classes
    }


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    classes: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot and optionally save a clean confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_arr = np.array(cm)

    sns.heatmap(
        cm_arr,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        ax=ax
    )

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Class', fontsize=11, labelpad=8)
    ax.set_ylabel('Actual Class', fontsize=11, labelpad=8)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig


def save_evaluation_results(
    results: Dict[str, Dict],
    output_dir: Path
) -> None:
    """Save metrics JSON and comparison summary CSV to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save detailed JSON
    json_path = output_dir / 'evaluation_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    # 2. Build summary comparison table
    summary_rows = []
    for name, m in results.items():
        summary_rows.append({
            'Model': m.get('model_name', name),
            'Accuracy': round(m['accuracy'] * 100, 2),
            'Detection Rate (%)': round(m['detection_rate'] * 100, 2),
            'False Positive Rate (%)': round(m['false_positive_rate'] * 100, 2),
            'Precision (Macro)': round(m['precision_macro'], 4),
            'Recall (Macro)': round(m['recall_macro'], 4),
            'F1-Score (Macro)': round(m['f1_macro'], 4),
            'F1-Score (Weighted)': round(m['f1_weighted'], 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    csv_path = output_dir / 'model_comparison.csv'
    summary_df.to_csv(csv_path, index=False)
