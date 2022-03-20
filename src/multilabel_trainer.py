'''HuggingFace Trainer class for multi-label classification'''

from transformers import Trainer
import torch.nn as nn


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
