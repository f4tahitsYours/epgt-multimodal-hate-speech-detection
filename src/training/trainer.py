import json, time, logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from training.loss    import EPGTLoss
from training.metrics import EPGTMetrics

logger = logging.getLogger(__name__)


class EPGTTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        checkpoint_dir,
        lr               = 2e-5,
        weight_decay     = 0.01,
        max_epochs       = 20,
        patience         = 3,
        warmup_ratio     = 0.10,
        grad_clip        = 1.0,
        accum_steps      = 1,
        device           = None,
        run_name         = "epgt",
        lambda_intensity = 0.40,
        lambda_sarcasm   = 0.35,
        lambda_role      = 0.25,
    ):
        self.model          = model
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.lr             = lr
        self.weight_decay   = weight_decay
        self.max_epochs     = max_epochs
        self.patience       = patience
        self.grad_clip      = grad_clip
        self.accum_steps    = accum_steps
        self.device         = device or torch.device("cpu")
        self.run_name       = run_name

        self.loss_fn = EPGTLoss(
            lambda_intensity = lambda_intensity,
            lambda_sarcasm   = lambda_sarcasm,
            lambda_role      = lambda_role,
        )
        self.metrics = EPGTMetrics()

        bert_params, other_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "text_encoder.bert" in name:
                bert_params.append(param)
            else:
                other_params.append(param)

        self.optimizer = AdamW(
            [
                {"params": bert_params,  "lr": lr},
                {"params": other_params, "lr": lr * 10},
            ],
            weight_decay = weight_decay,
        )

        total_steps  = len(train_loader) * max_epochs // accum_steps
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps   = warmup_steps,
            num_training_steps = total_steps,
        )

        self.best_val_metric  = 0.0
        self.best_epoch       = 0
        self.no_improve_count = 0
        self.training_log     = []

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        n_batches  = 0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            graph_batch      = batch["graph"].to(self.device)
            labels_intensity = batch["label_intensity"].to(self.device)
            labels_sarcasm   = batch["label_sarcasm"].to(self.device)
            labels_role      = batch["label_emoji_role"].to(self.device)

            outputs = self.model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                graph_batch    = graph_batch,
                token_type_ids = token_type_ids,
            )

            loss, loss_dict = self.loss_fn(
                logits_intensity = outputs["logits_intensity"],
                logits_sarcasm   = outputs["logits_sarcasm"],
                logits_role      = outputs["logits_role"],
                labels_intensity = labels_intensity,
                labels_sarcasm   = labels_sarcasm,
                labels_role      = labels_role,
            )

            (loss / self.accum_steps).backward()

            if (step + 1) % self.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss_dict["loss_total"]
            n_batches  += 1

        return {"loss_total": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        all_preds  = {"intensity": [], "sarcasm": [], "role": []}
        all_labels = {"intensity": [], "sarcasm": [], "role": []}
        total_loss = 0.0
        n_batches  = 0

        for batch in loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            graph_batch      = batch["graph"].to(self.device)
            labels_intensity = batch["label_intensity"].to(self.device)
            labels_sarcasm   = batch["label_sarcasm"].to(self.device)
            labels_role      = batch["label_emoji_role"].to(self.device)

            outputs = self.model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                graph_batch    = graph_batch,
                token_type_ids = token_type_ids,
            )

            loss, _ = self.loss_fn(
                logits_intensity = outputs["logits_intensity"],
                logits_sarcasm   = outputs["logits_sarcasm"],
                logits_role      = outputs["logits_role"],
                labels_intensity = labels_intensity,
                labels_sarcasm   = labels_sarcasm,
                labels_role      = labels_role,
            )
            total_loss += loss.item()
            n_batches  += 1

            all_preds["intensity"].extend(outputs["logits_intensity"].argmax(-1).cpu().tolist())
            all_preds["sarcasm"].extend(outputs["logits_sarcasm"].argmax(-1).cpu().tolist())
            all_preds["role"].extend(outputs["logits_role"].argmax(-1).cpu().tolist())
            all_labels["intensity"].extend(labels_intensity.cpu().tolist())
            all_labels["sarcasm"].extend(labels_sarcasm.cpu().tolist())
            all_labels["role"].extend(labels_role.cpu().tolist())

        metrics = self.metrics.compute(
            preds_intensity  = all_preds["intensity"],
            preds_sarcasm    = all_preds["sarcasm"],
            preds_role       = all_preds["role"],
            labels_intensity = all_labels["intensity"],
            labels_sarcasm   = all_labels["sarcasm"],
            labels_role      = all_labels["role"],
        )
        metrics["val_loss"] = total_loss / max(n_batches, 1)
        return metrics

    def _save_checkpoint(self, epoch, metrics):
        ckpt = {
            "epoch"      : epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "metrics"    : metrics,
            "run_name"   : self.run_name,
        }
        torch.save(ckpt, str(self.checkpoint_dir / "best_model.pt"))

    def _save_log(self):
        with open(self.checkpoint_dir / "training_log.json", "w") as f:
            json.dump(self.training_log, f, indent=2)

    def train(self):
        print("Training [" + self.run_name + "]")
        print("  Epochs : " + str(self.max_epochs) + " (patience=" + str(self.patience) + ")")
        print("  LR     : " + str(self.lr))
        print()

        best_metrics = {}

        for epoch in range(1, self.max_epochs + 1):
            t0         = time.time()
            train_loss = self._train_epoch(epoch)
            val_metrics= self._evaluate(self.val_loader)
            val_metric = val_metrics["avg_macro_f1"]
            elapsed    = time.time() - t0
            improved   = " *" if val_metric > self.best_val_metric else ""

            epoch_log = {
                "epoch"        : epoch,
                "train_loss"   : round(train_loss["loss_total"], 4),
                "val_loss"     : round(val_metrics["val_loss"], 4),
                "avg_macro_f1" : round(val_metric, 4),
                "f1_intensity" : val_metrics["f1_macro_intensity"],
                "f1_sarcasm"   : val_metrics["f1_macro_sarcasm"],
                "f1_role"      : val_metrics["f1_macro_role"],
                "elapsed_sec"  : round(elapsed, 1),
            }
            self.training_log.append(epoch_log)
            self._save_log()

            line = (
                "  Epoch " + str(epoch).rjust(2) + "/" + str(self.max_epochs)
                + " | tr_loss=" + str(round(train_loss["loss_total"], 4))
                + " | val_loss=" + str(round(val_metrics["val_loss"], 4))
                + " | avg_f1=" + str(round(val_metric, 4))
                + " | int=" + str(round(val_metrics["f1_macro_intensity"], 3))
                + " | sarc=" + str(round(val_metrics["f1_macro_sarcasm"], 3))
                + " | role=" + str(round(val_metrics["f1_macro_role"], 3))
                + " | " + str(round(elapsed, 0)) + "s" + improved
            )
            print(line)

            if val_metric > self.best_val_metric:
                self.best_val_metric  = val_metric
                self.best_epoch       = epoch
                self.no_improve_count = 0
                best_metrics          = val_metrics.copy()
                self._save_checkpoint(epoch, val_metrics)
            else:
                self.no_improve_count += 1

            if self.no_improve_count >= self.patience:
                print("  Early stopping at epoch " + str(epoch))
                break

        print("Best epoch      : " + str(self.best_epoch))
        print("Best avg_macro_f1: " + str(round(self.best_val_metric, 4)))
        return best_metrics