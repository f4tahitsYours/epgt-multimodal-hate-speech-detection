"""
metrics.py — Evaluation metrics untuk EPGT.

Primary metric : Macro F1 (average=macro) per task
Secondary      : Accuracy, Weighted F1
Aggregated     : avg_macro_f1 = mean(f1_intensity, f1_sarcasm, f1_role)

Digunakan untuk early stopping dan checkpoint selection.
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report
)


class EPGTMetrics:
    """
    Evaluasi metrics untuk 3 task EPGT.

    Primary metric untuk model selection: avg_macro_f1
    (rata-rata Macro F1 dari ketiga task)
    """

    TASK_NAMES = {
        "intensity" : "Emotion Intensity",
        "sarcasm"   : "Sarcasm Detection",
        "role"      : "Emoji Role",
    }

    LABEL_NAMES = {
        "intensity" : ["Low", "Medium", "High"],
        "sarcasm"   : ["Non-Sarcastic", "Sarcastic"],
        "role"      : ["Literal", "Exaggeration", "Irony", "Reaction"],
    }

    def compute(
        self,
        preds_intensity : List[int],
        preds_sarcasm   : List[int],
        preds_role      : List[int],
        labels_intensity: List[int],
        labels_sarcasm  : List[int],
        labels_role     : List[int],
    ) -> Dict:
        """
        Hitung semua metrics.

        Returns:
            dict dengan metrics per task + aggregated avg_macro_f1
        """
        results = {}
        macro_f1s = []

        task_data = [
            ("intensity", preds_intensity, labels_intensity),
            ("sarcasm",   preds_sarcasm,   labels_sarcasm),
            ("role",      preds_role,      labels_role),
        ]

        for task, preds, labels in task_data:
            preds_arr  = np.array(preds)
            labels_arr = np.array(labels)

            macro_f1 = f1_score(labels_arr, preds_arr, average="macro",
                                zero_division=0)
            weighted_f1 = f1_score(labels_arr, preds_arr, average="weighted",
                                   zero_division=0)
            acc = accuracy_score(labels_arr, preds_arr)

            results[f"f1_macro_{task}"]    = round(macro_f1, 4)
            results[f"f1_weighted_{task}"] = round(weighted_f1, 4)
            results[f"acc_{task}"]         = round(acc, 4)
            macro_f1s.append(macro_f1)

        # Aggregated metric — digunakan untuk early stopping
        results["avg_macro_f1"] = round(float(np.mean(macro_f1s)), 4)

        return results

    def get_classification_reports(
        self,
        preds_intensity : List[int],
        preds_sarcasm   : List[int],
        preds_role      : List[int],
        labels_intensity: List[int],
        labels_sarcasm  : List[int],
        labels_role     : List[int],
    ) -> Dict[str, str]:
        """Buat classification report per task."""
        reports = {}
        task_data = [
            ("intensity", preds_intensity, labels_intensity),
            ("sarcasm",   preds_sarcasm,   labels_sarcasm),
            ("role",      preds_role,      labels_role),
        ]
        for task, preds, labels in task_data:
            reports[task] = classification_report(
                labels, preds,
                target_names = self.LABEL_NAMES[task],
                zero_division= 0,
            )
        return reports
