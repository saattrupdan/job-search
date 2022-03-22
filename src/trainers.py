'''HuggingFace Trainer class for multi-label classification'''

from transformers import Trainer
import torch.nn as nn
import torch


class MultiLabelTrainer(Trainer):
    '''HuggingFace Trainer class for multi-label classification'''
    def compute_loss(self, model, inputs, return_outputs: bool = False):
        '''Compute loss for multi-label classification'''
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, logits) if return_outputs else loss


class ClassWeightTrainer(Trainer):
    '''HuggingFace Trainer class for classification with class weights'''
    def __init__(self, pos_weight: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        '''Compute loss for multi-label classification'''
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        weights = torch.tensor([
            self.pos_weight if lbl == 1 else 1 for lbl in labels
        ]).float()
        loss_fct = nn.BCEWithLogitsLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, logits) if return_outputs else loss
