from typing import Optional, Type
from dataclasses import asdict
from loguru import logger
from transformers import Trainer

from ner_pipeline.utils.common import set_seed
from ..ner_metrics import (NervaluateEvaluator, 
                    SeqevalLogger,
                    NervaluateLogger, 
                    decode_all_predictions)

from ..ner_factory import NERPredictions

from ....shared.trainer_base import HFInferenceOrchestrator

class NERInferenceOrchestrator(HFInferenceOrchestrator):
    def __init__(self, runner_conf):
        super().__init__(runner_conf)
        self.cfg = self.builder.cfg

    def execute(self):
        super().execute()
        self._compute_ner_inference_metrics()

    def _build_trainer(self):
        set_seed(self.cfg.seed)
        logger.info("Building Trainer----->")
        trainer_kwargs = self.components.trainer_kwargs
        
        model, args, processing_class = (trainer_kwargs.model,
                                        trainer_kwargs.args,
                                        trainer_kwargs.processing_class,
                                                   )

        data_collator, compute_metrics = (trainer_kwargs.data_collator,
                                          trainer_kwargs.compute_metrics)

        self.inference_trainer = Trainer(model=model,
                                         args=args,
                                         processing_class=processing_class,
                                         data_collator=data_collator,
                                         compute_metrics=compute_metrics)


        logger.success("Trainer built Successfully")
        logger.info("Initialising Trainer------->")

    def _compute_ner_inference_metrics(self):
        self._validate_trainer_built()

        # if not self.results.label_ids:
        #     logger.warning("Metrics is unavailable for this run. \nNo ground truth labels provided for this dataset.")
        #     return 
        
        logits, label_ids = self.results.predictions, self.results.label_ids

        true_labels, pred_labels = decode_all_predictions(
                                            logits=logits,
                                            label_ids=label_ids,
                                            id2label=self.components.id2label
                                            )
        label_names = sorted({
            label[2:]
            for label in self.components.id2label.values()
            if label.startswith(("B-", "I-"))
        })
        
        ner_predictions = NERPredictions(
                                true_labels=true_labels,
                                pred_labels=pred_labels,
                                label_names=label_names
                                )
        
        #nervaluate results
        evaluator = NervaluateEvaluator(ner_predictions)
        nervaluate_results = evaluator.run_evaluation()

        return ner_predictions, nervaluate_results
        
    def _log_to_wandb(self):
        
        #init seqeval_logger and nervaluate_logger
        #fetch prediction results
        ner_predictions, nervaluate_results = self._compute_ner_inference_metrics()
        #seqeval table
        seqeval_logger = SeqevalLogger(ner_predictions, self.wandb_run)

        #nervaluate 
        nervaluate_logger = NervaluateLogger(nervaluate_results, self.wandb_run)
        
        #log metrics to wandb
        seqeval_logger.log()
        nervaluate_logger.log()


