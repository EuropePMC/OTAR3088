from loguru import logger
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.optim import AdamW

from datasets import Dataset
from transformers import get_linear_schedule_with_warmup, TrainingArguments


from ..shared.factory import (
                            extract_encoder_layers,
                            extract_model_backbone,
                            compute_training_steps,
                            compute_warmup_steps
                            )



class ReinitLLRDProcessor:
    """
    Implements reinitialisation and layer-wise learning rate decay
    as described in the paper `Revisiting Few-Sample BERT Fine-Tuning`.

    """
    def __init__(self, cfg: DictConfig, model: nn.Module, train_dataset: Dataset = None, training_args: TrainingArguments = None):
        self.cfg = cfg
        self.model = model
        self.train_dataset = train_dataset
        self.training_args = training_args
        self._validate_params()

    def _validate_params(self):
        allowed = {"reinit_only", "llrd_only", "reinit_llrd"}
        strategy = self.cfg.training_strategy.lower()
        if strategy not in allowed:
            raise ValueError(f"Invalid Training Strategy.\n"
                        f"Training strategy should be one of : {allowed}")
        if (not getattr(self.cfg, "llrd", 1.0) < 1.0 
              and not (self.train_dataset and self.training_args)):
            raise ValueError("train_dataset/training_args is required for this training_strategy")

    def _reinit_modules(self, module:nn.Module, initializer_range:float):
        "reinitialises layers to their default pretraining state"
        for layer in module.modules():
          if isinstance(layer, nn.Linear):
              layer.weight.data.normal_(mean=0.0, std=initializer_range)
              if layer.bias is not None:
                  layer.bias.data.zero_()
          elif isinstance(layer, nn.LayerNorm):
              layer.bias.data.zero_()
              layer.weight.data.fill_(1.0)

    def _reinit_classifier(self):
        """
        Reinitialises the classifier head of a Encoder Model(E.g BERT)
        """
        for module in self.model.classifier.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight) #reinit weight to a random
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        logger.info("Classifier head reinitialised to random weights")


    def _reinit_pooler(self):
        _, model_backbone = extract_model_backbone(self.model)
        if hasattr(model_backbone, "pooler"):
          dense = model_backbone.pooler.dense
          dense.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
          dense.bias.data.zero_()
          logger.info("Pooler reinitialised to random weights")

        else:
          logger.info("This model has no pooler layer")

    def _reinit_last_k_layers(self) -> nn.Module:
        """
        Reinitialize the last k layers of a BERT-like models.
        Params: 
            model (nn.Module): The BERT-like models to be reinitialised.
            k (int): The number of layers to be reinitialised.
            reinit_classifier (bool): Whether to reinitialise the classifier head.
            reinit_pooler (bool): Whether to reinitialise the pooler(not applicable to a Token Classification problem).
        """
        if getattr(self.cfg, "reinit_classifier", False):
            self._reinit_classifier()
        if getattr(self.cfg, "reinit_pooler", False):
            self._reinit_pooler()
        encoder_layers = extract_encoder_layers(self.model)
        total_layers = len(encoder_layers)
      
        k = getattr(self.cfg, "reinit_k_layers", 0)
        if k <= 0 or k >= total_layers:
            raise ValueError(f"num_layers_to_reinit must be >0 and < total encoder layers {total_layers}")
      
        logger.info(f"This model has {total_layers} encoder layers")
        logger.info(f"Reinitialising top {k} layer(s)")

        top_k_layers = encoder_layers[-k:]  # Slice the top k layers
        for layer in top_k_layers:
            self._reinit_modules(layer, initializer_range=self.model.config.initializer_range)

        logger.info(f"Last {k} layer(s) reinitialized to random weights")
        return self.model



    def _build_llrd_optim(self):
        """
        Implements Differential Learning Rate to Model layers as descriped in the paper:
        `Revist Bert Finetuning`: [Insert Arxiv Link]
        Parameters(stored in self.cfg):
          llrd: Layerwise Learning rate decay factor(default=0.9) options: [0.9, 0.95, 1.0] as indicated in the paper
          weight_decay: weight decay coefficient(default=0.01) options: [0.0, 0.01, 0.001] as indicated in the paper
          lr: base learning rate (default=5e-5)
        """

        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        model_type = extract_model_backbone(self.model)[0]
        logger.info(f"This model has backbone {model_type.upper()}")
        #no weight decay applied-->same as standard finetuning
        if self.cfg.llrd == 1.0:
          optimizer_grouped_parameters = [
              {
                  "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                  "weight_decay": self.cfg.weight_decay,
                  "lr": self.cfg.lr,
              },
              {
                  "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                  "weight_decay": 0.0,
                  "lr": self.cfg.lr,
              },
          ]
        else:
          #apply layerwise learning rate decay
          #apply to classifier head/pooler head
          optimizer_grouped_parameters = [
              {
                  "params": [p for n, p in self.model.named_parameters() if "classifier" in n or "pooler" in n],
                  "weight_decay": 0.0,
                  "lr": self.cfg.lr,
              },
          ]
          #apply to embeddings and encoder layer
          if model_type in ["bert", "roberta", "electra"]:
            num_layers = self.model.config.num_hidden_layers
            layers = [getattr(self.model, model_type).embeddings] + list(getattr(self.model, model_type).encoder.layer)
            layers.reverse()
            curr_lr = self.cfg.lr
            for layer in layers:
              curr_lr *= self.cfg.llrd
              optimizer_grouped_parameters += [
                  {
                      "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                      "weight_decay": self.cfg.weight_decay,
                      "lr": curr_lr,
                  },
                  {
                      "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                      "weight_decay": 0.0,
                      "lr": curr_lr,
                  },
              ]
          else:
            raise NotImplementedError
        return optimizer_grouped_parameters

    def apply_reinit(self):
        "Reinitialize layers if specified in cfg"
        if getattr(self.cfg, "reinit_k_layers", 0) > 0:
            self.model = self._reinit_last_k_layers()
        return self.model


    def apply_llrd(self):
        "Update optimizer parameters if LLRD is specified in cfg"
        if getattr(self.cfg, "llrd", 1.0) < 1.0:
          optimizer_grouped_params = self._build_llrd_optim()
          #init optimizer
          optimizer = AdamW(optimizer_grouped_params)

          #get training steps and warmup steps
          training_steps = compute_training_steps(
              self.training_args,
              self.train_dataset   
          )
          warmup_steps = compute_warmup_steps(num_training_steps=training_steps, 
                                             warmup_ratio=self.cfg.warmup_ratio)
          custom_scheduler = get_linear_schedule_with_warmup(
              optimizer=optimizer, 
              num_warmup_steps=warmup_steps, 
              num_training_steps=training_steps
          )
          return optimizer, custom_scheduler
        return None, None
  