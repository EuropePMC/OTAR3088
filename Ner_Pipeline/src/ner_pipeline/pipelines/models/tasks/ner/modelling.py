import math
import time
from typing import (List, Dict,
                    Tuple, Union)
from loguru import logger

import evaluate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from transformers import (Trainer, 
                          TrainerCallback, 
                          AutoConfig,
                          AutoModel,
                          AutoModelForTokenClassification, 
                          AutoTokenizer)


from .trainer_config import NerModelConfig
from ...shared.modelling_base import BuildModel
from ...strategies.crf import BERTCRFForTokenClassification                     




class BuildNerModel(BuildModel):
    _MODEL_BUILDER = {
                    "standard": "_build_for_standard",
                    "crf": "_build_for_crf",
                }
    def __init__(self, model_config:NerModelConfig):
        super().__init__(model_config)
        self.ner_head_type = model_config.ner_head_type
        self.num_labels = model_config.num_labels
        self.label2id = model_config.label2id
        self.id2label = model_config.id2label
        


    def build(self):
        if self.ner_head_type not in self._MODEL_BUILDER:
            raise ValueError(f"Unknown model type. Supported types are: {list(self._MODEL_BUILDER.keys())}")

        builder_name = self._MODEL_BUILDER[self.ner_head_type]
        builder = getattr(self, builder_name)
        self._log_model_head_type()

        return builder()

    def _get_common_kwargs(self):
        return {
                "num_labels": self.num_labels,
                "id2label": self.id2label,
                "label2id": self.label2id
                }

    def _build_for_standard(self):
        model = AutoModelForTokenClassification.from_pretrained(
                                    self.checkpoint,
                                    **self._get_common_kwargs()
                                    )
        model = model.to(self.device)
        self._log_trainable_params(model)

        return model


    def _build_for_crf(self):
        config = AutoConfig.from_pretrained(self.checkpoint,
                                            **self._get_common_kwargs()
                                            )

        base_model = AutoModel.from_pretrained(self.checkpoint)

        model = BERTCRFForTokenClassification(config)

        model.bert = base_model
        for name, param in model.named_parameters():
            if "crf" in name:
                print(name, param.shape)

        model = model.to(self.device)
        logger.info(f"Bert info: {model.bert.embeddings.word_embeddings.weight.mean()}")
        
        self._log_trainable_params(model)

        return model

    def _log_model_head_type(self):
        logger.info(f"Ner token classifier for run: {self.ner_head_type}")
    


class BaseTrainer(Trainer):
    """
    Inherits from huggingface Trainer class. 
    Extends some core methods(evaluate, compute_loss),
    and implements other custom methods
    """
    def __init__(self, *args, id2label: Dict[int, str] = None,  **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_predictions = []
        self.epoch_labels = []
        self.epoch_loss = []
        self.id2label = id2label

    def _fetch_logits_and_loss(self, inputs, outputs):
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
        if isinstance(outputs, dict):
            loss = outputs["loss"]
            logits = outputs.get("logits")
        else:
            loss = outputs[0]
            logits = outputs[1:]

        return loss, logits


    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        """
        Adapts hf compute loss method by computing training metrics
        and converting to seqeval format. 
        """
        labels = inputs.get("labels")

        outputs = model(**inputs)
        loss, logits = self._fetch_logits_and_loss(inputs, outputs)
        
        if labels is not None and logits is not None:
            preds = self._decode_predictions(logits)
            self._compute_train_epoch_metrics(preds, labels)

        self.epoch_loss.append(loss.item())
            
        return (loss, outputs) if return_outputs else loss


    def evaluate(self, eval_dataset=None, 
                 ignore_keys=None, 
                 metric_key_prefix: str = "eval",
                 *args, **kwargs):
        """
        Extends HuggingFace trainer.evaluate() method
        by computing eval_dataset predictions and label_ids
        These are further used for downstream purposes in pipeline
        """
        eval_dataset = self.eval_dataset
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        start_time = time.time()

        output = super().evaluation_loop(eval_dataloader, description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix)

        ##next  lines of codes are only useful when implementing thresholding
        # logits = output.predictions
        # labels = output.label_ids
        # preds = logits.argmax(axis=-1)
        #preds = self._apply_threshold(preds, logits)
        # self.eval_predictions = preds
        # self.eval_label_ids = labels
        # if self.compute_metrics is not None:
        #     metrics = self.compute_metrics((preds, labels))
        # else:
        #     metrics = output.metrics
        
        #next lines are for regular evaluation loop 
        self.eval_predictions = output.predictions
        self.eval_label_ids = output.label_ids


        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
            output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        
        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    
    def _decode_predictions(self, logits):
        return logits.argmax(dim=-1)

    
    def _compute_train_epoch_metrics(self, preds, labels):
        # Store predictions and labels in seqeval format
        for pred_seq, label_seq in zip(preds, labels):
            pred_labels, true_labels = [], []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                pred_labels.append(self.id2label[p if isinstance(p, int) else p.item()])
                true_labels.append(self.id2label[l.item()])
     

            self.epoch_predictions.append(pred_labels)
            self.epoch_labels.append(true_labels)


class CRFTrainer(BaseTrainer):
    """
    Inherits from BaseTrainer.
    Adds CRF viberti decoding when using crf head for training
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = outputs.loss
        logits = outputs.logits
        
        mask = inputs["attention_mask"].bool()

        if labels is not None and logits is not None:
            
            preds = self._decode_predictions(model, logits, mask)
            self._compute_train_epoch_metrics(preds, labels)
        
        self.epoch_loss.append(loss.item())

        return (loss, outputs) if return_outputs else loss

    def _decode_predictions(self, model, logits, mask):
        "computes decoding for crf layer using Viberti decoding"
        return model.crf.decode(logits, mask)



class WeightedTrainer(BaseTrainer):
    """
    Inherits from BaseTrainer.
    Implements class weighting technique
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label2id = self.model.config.label2id
        
        class_weights = self._build_boundary_weights(self.label2id)
        self._class_weights = class_weights.to(self.args.device)
        self.loss_fct = nn.CrossEntropyLoss(
                weight=self._class_weights,
                ignore_index=-100
            )


    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = self._fetch_logits_and_loss(inputs, outputs)

        loss = None
        if labels is not None:

            loss = self.loss_fct(logits.view(-1, logits.shape[-1]),
                                 labels.view(-1))

            preds = self._decode_predictions(logits)
            self._compute_train_epoch_metrics(preds, labels)


        if loss is not None:
            self.epoch_loss.append(loss.item())

        return (loss, outputs) if return_outputs else loss


    def _fetch_logits_and_loss(self, inputs, outputs):

        if isinstance(outputs, dict):
            logits = outputs.get("logits")
        else:
            logits = outputs[1:]

        return logits


    def _build_boundary_weights(self, label2id):
        weights = torch.ones(len(label2id))

        for label, idx in label2id.items():
            if label == "O":
                weights[idx] = 1.0

            elif label.startswith("B-"):
                if "Tissue" in label:
                    weights[idx] = 4.0
                else:
                    weights[idx] = 3.0

            elif label.startswith("I-"):
                weights[idx] = 2.0

        return weights



class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        metric = evaluate.load("seqeval")
        preds = self._trainer.epoch_predictions
        labels = self._trainer.epoch_labels
        losses = self._trainer.epoch_loss

        if preds and labels:

            train_results = metric.compute(predictions=preds, references=labels)
            mean_loss = np.mean(losses)

            logger.info("\n======== Training Metrics on Epoch End ========")
            logger.info(f"Train Loss: {mean_loss:.4f}")
            logger.info(f"Training Results:\n {train_results}")
            logger.info("=============================================")

        # Reset storage for next epoch
        self._trainer.epoch_predictions = []
        self._trainer.epoch_labels = []
        self._trainer.epoch_loss = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_predictions = []
        self.epoch_labels = []
        self.epoch_loss = []

