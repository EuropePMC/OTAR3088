

from dataclasses import asdict
from loguru import logger
from transformers.trainer_callback import EarlyStoppingCallback


from .modelling import MLMBaseTrainer
from .trainer_config import MLMTrainerFactory
from ...shared.trainer_base import HFTrainingOrchestrator


from ner_pipeline.utils.common import set_seed



class MLMTrainingOrchestrator(HFTrainingOrchestrator):
    def __init__(self, runner_conf):
        super().__init__(runner_conf)
        self.cfg = self.builder.cfg
    
    def _build_trainer(self):
        set_seed(self.cfg.seed)
        logger.info("Building Trainer----->")
        trainer_kwargs = self.components.trainer_kwargs

        train_dataset, eval_dataset = (trainer_kwargs.train_dataset, 
                                        trainer_kwargs.eval_dataset)
        
        compute_metrics,preprocess_logits_for_metrics = (trainer_kwargs.compute_metrics,
                                                         trainer_kwargs.preprocess_logits_for_metrics)

        model, processing_class, data_collator, args = (trainer_kwargs.model,
                                                        trainer_kwargs.processing_class,
                                                        trainer_kwargs.data_collator,
                                                        trainer_kwargs.args,
                                                        )
        
        trainer_type = getattr(self.cfg.task, "trainer_type", "base")
        TrainerClass = MLMTrainerFactory.get_trainer_class(trainer_type)

        self.trainer = TrainerClass(
                                    #**self.components.strategy_kwargs,
                                    train_dataset=train_dataset,
                                    eval_dataset=eval_dataset,
                                    model=model,
                                    args=args,
                                    processing_class=processing_class,
                                    data_collator=data_collator,
                                    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                                    compute_metrics=compute_metrics,
                                    )

    
        early_stopping_callback = EarlyStoppingCallback(self.cfg.early_stopping_patience)
        self.trainer.add_callback(early_stopping_callback)

        for cb in self.components.callbacks:
            if isinstance(cb, type):
                self.trainer.add_callback(cb(trainer=self.trainer))
            else:
                self.trainer.add_callback(cb)
        
        logger.success("Trainer built Successfully")
        logger.info("Initialising Trainer------->")

    

