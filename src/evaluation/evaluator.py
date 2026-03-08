"""
evaluator.py — Test set evaluator untuk EPGT.

Load checkpoint → forward pass → metrics + classification report.
Mendukung semua model (baseline + ablation) dengan interface seragam.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, accuracy_score,
    classification_report, confusion_matrix
)


class EPGTEvaluator:
    """
    Evaluator untuk test set.
    Load checkpoint dan jalankan inference pada test_loader.
    """

    LABEL_NAMES = {
        "intensity": ["Low", "Medium", "High"],
        "sarcasm"  : ["Non-Sarcastic", "Sarcastic"],
        "role"     : ["Literal", "Exaggeration", "Irony", "Reaction"],
    }

    def __init__(self, device=None):
        self.device = device or torch.device("cpu")

    def load_checkpoint(
        self,
        model,
        checkpoint_path: str,
    ):
        """Load model weights dari checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device)
        model.eval()
        return model, ckpt.get("metrics", {})

    @torch.no_grad()
    def predict(
        self,
        model,
        loader: DataLoader,
    ) -> Dict:
        """
        Jalankan inference pada loader.
        Returns dict of predictions dan labels per task.
        """
        model.eval()
        all_preds  = {"intensity": [], "sarcasm": [], "role": []}
        all_labels = {"intensity": [], "sarcasm": [], "role": []}
        all_probs  = {"intensity": [], "sarcasm": [], "role": []}

        for batch in loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            graph_batch = batch["graph"].to(self.device)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                graph_batch    = graph_batch,
                token_type_ids = token_type_ids,
            )

            import torch.nn.functional as F
            for task, logit_key, label_key in [
                ("intensity", "logits_intensity", "label_intensity"),
                ("sarcasm",   "logits_sarcasm",   "label_sarcasm"),
                ("role",      "logits_role",       "label_emoji_role"),
            ]:
                logits = outputs[logit_key]
                probs  = F.softmax(logits, dim=-1)
                preds  = logits.argmax(dim=-1)
                labels = batch[label_key]

                all_preds[task].extend(preds.cpu().tolist())
                all_labels[task].extend(labels.cpu().tolist())
                all_probs[task].extend(probs.cpu().tolist())

        return {
            "preds" : all_preds,
            "labels": all_labels,
            "probs" : all_probs,
        }

    def compute_metrics(self, pred_dict: Dict) -> Dict:
        """Hitung semua metrics dari pred_dict."""
        preds  = pred_dict["preds"]
        labels = pred_dict["labels"]
        results = {}
        macro_f1s = []

        for task in ["intensity", "sarcasm", "role"]:
            p = np.array(preds[task])
            l = np.array(labels[task])

            macro_f1    = f1_score(l, p, average="macro",    zero_division=0)
            weighted_f1 = f1_score(l, p, average="weighted", zero_division=0)
            acc         = accuracy_score(l, p)

            results["f1_macro_"    + task] = round(macro_f1, 4)
            results["f1_weighted_" + task] = round(weighted_f1, 4)
            results["acc_"         + task] = round(acc, 4)
            macro_f1s.append(macro_f1)

        results["avg_macro_f1"] = round(float(np.mean(macro_f1s)), 4)
        return results

    def get_confusion_matrices(self, pred_dict: Dict) -> Dict:
        """Confusion matrix per task."""
        cms = {}
        for task in ["intensity", "sarcasm", "role"]:
            p = np.array(pred_dict["preds"][task])
            l = np.array(pred_dict["labels"][task])
            n_classes = len(self.LABEL_NAMES[task])
            cms[task] = confusion_matrix(l, p, labels=list(range(n_classes)))
        return cms

    def get_classification_reports(self, pred_dict: Dict) -> Dict:
        """Classification report per task."""
        reports = {}
        n_classes = {"intensity": 3, "sarcasm": 2, "role": 4}
        for task in ["intensity", "sarcasm", "role"]:
            p = np.array(pred_dict["preds"][task])
            l = np.array(pred_dict["labels"][task])
            labels_range = list(range(n_classes[task]))
            reports[task] = classification_report(
                l, p,
                labels       = labels_range,
                target_names = self.LABEL_NAMES[task],
                zero_division= 0,
            )
        return reports

    def evaluate(
        self,
        model,
        loader         : DataLoader,
        checkpoint_path: str = None,
        run_name       : str = "",
    ) -> Dict:
        """
        Full evaluation pipeline:
        load checkpoint → predict → metrics → reports.

        Returns:
            dict dengan metrics, reports, confusion_matrices, pred_dict
        """
        val_metrics = {}
        if checkpoint_path and Path(checkpoint_path).exists():
            model, val_metrics = self.load_checkpoint(model, checkpoint_path)
        else:
            model.to(self.device)
            model.eval()

        pred_dict = self.predict(model, loader)
        metrics   = self.compute_metrics(pred_dict)
        reports   = self.get_classification_reports(pred_dict)
        cms       = self.get_confusion_matrices(pred_dict)

        return {
            "run_name"            : run_name,
            "test_metrics"        : metrics,
            "val_metrics"         : val_metrics,
            "classification_reports": reports,
            "confusion_matrices"  : cms,
            "pred_dict"           : pred_dict,
        }
