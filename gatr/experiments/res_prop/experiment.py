# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
# Adapted by Ruibin Liu, ruibinliuphd@gmail.com
from pathlib import Path

import torch
import torcheval.metrics.functional as torch_metrics

from gatr.experiments.base_experiment import BaseExperiment
from gatr.experiments.res_prop.dataset import ResPropDataset


class ResPropExperiment(BaseExperiment):
    """Experiment manager for n-body prediction.

    Parameters
    ----------
    cfg : OmegaConf
        Experiment configuration. See the config folder in the repository for examples.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.exp_type = cfg.exp_type
        assert self.exp_type in ['binary', 'regression', 'multiclass']
        self.label_type = torch.float32
        if self.exp_type == 'regression':
            # Regression tasks
            self._loss_fn = torch.nn.MSELoss()
        elif self.exp_type == 'binary':
            # Binary classification tasks; remember NOT to use Sigmoid in the model last layer
            self._loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            # Multi-class classification tasks
            self._loss_fn = torch.nn.CrossEntropyLoss()
            self.label_type = torch.long
        self._other_metrics = {}
        self.added_metrics = []
        if self.cfg.training.add_metrics is not None:
            self.added_metrics = self.cfg.training.add_metrics.split(',')
        if self.added_metrics and self.exp_type == 'binary':
            if 'auroc' in self.added_metrics:
                self._other_metrics['auroc'] = torch_metrics.binary_auroc
            if 'auprc' in self.added_metrics:
                self._other_metrics['auprc'] = torch_metrics.binary_auprc
            if 'f1' in self.added_metrics:
                self._other_metrics['F1'] = torch_metrics.binary_f1_score
            if 'accuracy' in self.added_metrics:
                self._other_metrics['accuracy'] = torch_metrics.binary_accuracy
            if 'recall' in self.added_metrics:
                self._other_metrics['recall'] = torch_metrics.binary_recall
            if 'precision' in self.added_metrics:
                self._other_metrics['precision'] = torch_metrics.binary_precision
            if 'cm' in self.added_metrics:
                self._other_metrics['confusion_matrix'] = torch_metrics.binary_confusion_matrix
        if self.added_metrics and self.exp_type == 'multiclass':
            if 'auroc' in self.added_metrics:
                self._other_metrics['auroc'] = torch_metrics.multiclass_auroc
            if 'auprc' in self.added_metrics:
                self._other_metrics['auprc'] = torch_metrics.multiclass_auprc
            if 'f1' in self.added_metrics:
                self._other_metrics['F1'] = torch_metrics.multiclass_f1_score
            if 'accuracy' in self.added_metrics:
                self._other_metrics['accuracy'] = torch_metrics.multiclass_accuracy
            if 'recall' in self.added_metrics:
                self._other_metrics['recall'] = torch_metrics.multiclass_recall
            if 'precision' in self.added_metrics:
                self._other_metrics['precision'] = torch_metrics.multiclass_precision
            if 'cm' in self.added_metrics:
                self._other_metrics['confusion_matrix'] = torch_metrics.multiclass_confusion_matrix

    def _load_dataset(self, tag):
        """Loads dataset.

        Parameters
        ----------
        tag : str
            Dataset tag, like "train", "val", or one of self._eval_tags.

        Returns
        -------
        dataset : torch.utils.data.Dataset
            Dataset.
        """

        if tag == "train":
            subsample_fraction = self.cfg.data.subsample
        else:
            subsample_fraction = None

        filename = Path(self.cfg.data.data_dir) / f"{tag}.npz"

        return ResPropDataset(
            filename, subsample=subsample_fraction, label_type=self.label_type,
        )

    def _forward(self, *data):
        """Model forward pass.

        Parameters
        ----------
        data : tuple of torch.Tensor
            Data batch.

        Returns
        -------
        loss : torch.Tensor
            Loss
        metrics : dict with str keys and float values
            Additional metrics for logging
        """

        # Forward pass
        assert self.model is not None
        x, y = data
        y_pred, reg = self.model(x)

         # Compute loss
        loss = self._loss_fn(y_pred, y)
        
        if self.exp_type == 'regression':
            output_reg = torch.mean(reg)
            mse = loss.detach()
            loss += self.cfg.training.output_regularization * output_reg

            # Additional metrics
            mae = torch.nn.L1Loss(reduction="mean")(y_pred, y)
            metrics = dict(
                mse=mse.item(), rmse=loss.item() ** 0.5, output_reg=output_reg.item(), mae=mae.item()
            )
        elif self.exp_type in ['binary', 'multiclass']:
            # Additional metrics
            metrics = {}
            for metric_name, metric_fn in self._other_metrics.items():
                metrics[metric_name] = metric_fn(y_pred, y).item()

        return loss, metrics

    # @property
    # def _eval_dataset_tags(self):
    #     """Eval dataset tags.

    #     Returns
    #     -------
    #     tags : iterable of str
    #         Eval dataset tags
    #     """

    #     # Only evaluate on object_generalization dataset when method supports variable token number
    #     assert self.model is not None
    #     if self.model.supports_variable_items:
    #         return {"eval", "e3_generalization", "object_generalization"}

    #     return {"eval", "e3_generalization"}
