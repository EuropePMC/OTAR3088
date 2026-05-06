from dataclasses import asdict
from loguru import logger
from transformers.trainer_callback import EarlyStoppingCallback

from ...shared.trainer_base import HFTrainingOrchestrator
from .modelling import (BaseTrainer, 
                        CRFTrainer, 
                        CustomCallback)

from .metrics import (NervaluateEvaluator, 
                    SeqevalLogger,
                    NervaluateLogger, 
                    decode_all_predictions)

from .trainer_config import NerPredictions
from .ner_factory import NerTrainerFactory


from ner_pipeline.utils.common import set_seed



class NerTrainingOrchestrator(HFTrainingOrchestrator):
    def __init__(self, runner_conf):
        super().__init__(runner_conf)
        self.cfg = self.builder.cfg
        self._validate_trainer_and_ner_head_type()

    def _validate_trainer_and_ner_head_type(self):
        trainer_type = self.cfg.task.trainer_type
        ner_head_type = self.cfg.task.ner_head_type
        if trainer_type == "crf" and ner_head_type != "crf":
            raise ValueError(f"Trainer type {trainer_type} is not compatible with model type {ner_head_type}")
        
    def execute(self):
        super().execute()
        self._compute_ner_metrics()

    def _build_trainer(self):
        set_seed(self.cfg.seed)
        logger.info("Building Trainer----->")
        trainer_kwargs = self.components.trainer_kwargs
        train_dataset, eval_dataset, id2label, compute_metrics = (trainer_kwargs.train_dataset, 
                                                                  trainer_kwargs.eval_dataset,
                                                                  trainer_kwargs.id2label,
                                                                  trainer_kwargs.compute_metrics
                                                                      )
        model, args, processing_class, data_collator = (trainer_kwargs.model,
                                                        trainer_kwargs.args,
                                                        trainer_kwargs.processing_class,
                                                        trainer_kwargs.data_collator

                                                        )
        TrainerClass = NerTrainerFactory.get_trainer_class(self.cfg.task.trainer_type)

        self.trainer = TrainerClass(**self.components.strategy_kwargs,
                                    train_dataset = train_dataset,
                                    eval_dataset = eval_dataset,
                                    model = model,
                                    args = args,
                                    processing_class = processing_class,
                                    data_collator = data_collator,
                                    id2label = id2label,
                                    compute_metrics = compute_metrics,
                                    )

    
        early_stopping_callback = EarlyStoppingCallback(3)
        self.trainer.add_callback(early_stopping_callback)

        for cb in self.components.callbacks:
            if isinstance(cb, type):
                self.trainer.add_callback(cb(trainer=self.trainer))
            else:
                self.trainer.add_callback(cb)
        
        logger.success("Trainer built Successfully")
        logger.info("Initialising Trainer------->")

    def _compute_ner_metrics(self):
        self._validate_trainer_built()
        
        logits, label_ids = self.trainer.eval_predictions, self.trainer.eval_label_ids
        true_labels, pred_labels = decode_all_predictions(
                                            logits=logits,
                                            label_ids=label_ids,
                                            id2label=self.trainer.model.config.id2label
                                            )
        
        ner_predictions = NerPredictions(
                                true_labels=true_labels,
                                pred_labels=pred_labels,
                                label_names=self.cfg.task.label_names
                                )
                                      


        #nervaluate results
        evaluator = NervaluateEvaluator(ner_predictions)
        nervaluate_results = evaluator.run_evaluation()

        # #compute metrics table for logging to wandb if enabled for run
        # if self.wandb_run:
        #     #seqeval table
        #     self.seqeval_logger = SeqevalLogger(ner_predictions, self.wandb_run)

        #     #nervaluate 
        #     self.nervaluate_logger = NervaluateLogger(nervaluate_results, self.wandb_run)
        if self.wandb_run:
            return ner_predictions, nervaluate_results
        
        return


    def _log_to_wandb(self):
        # if not hasattr(self, "seqeval_logger") or not hasattr(self, "nervaluate_logger"):
        #     self._compute_ner_metrics()

        #init seqeval_logger and nervaluate_logger
        #fetch prediction results
        ner_predictions, nervaluate_results = self._compute_ner_metrics()
        #seqeval table
        seqeval_logger = SeqevalLogger(ner_predictions, self.wandb_run)

        #nervaluate 
        nervaluate_logger = NervaluateLogger(nervaluate_results, self.wandb_run)
        
        #log metrics to wandb
        seqeval_logger.log()
        nervaluate_logger.log()

        #log model artifacts
        if self.wandb_artifact is not None: 
            logger.info("Linking run to wandb registry")
            self.wandb_artifact.add_dir(local_path=self.best_ckpt_path,
                                        name="best_model_checkpoint_path_for_run")
            self.wandb_artifact.save()
        
            
            self.wandb_run.log_artifact(self.wandb_artifact)
            parts = [
                    self.cfg.logging.wandb.run.entity,
                    self.cfg.logging.wandb.registry.registry_name,
                    self.cfg.logging.wandb.registry.collection_name
                                                        ]
            target_save_path = "/".join(parts)
            logger.info(f"Target wandb registry path for this run is set at: {target_save_path}")

            self.wandb_run.link_artifact(artifact=self.wandb_artifact,
                                target_path=target_save_path,
                                aliases=list(self.wandb_run.tags)
                                )
                            
            logger.success("Artifact logged to registry")
