
from dataclasses import replace
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import torch.nn as nn
from transformers import (AutoTokenizer,
                          DataCollatorForLanguageModeling,
                          DataCollatorForWholeWordMask,
                            )

from .tokenization_utils import DataCollatorForSpanMasking
from .dataset_loader import PrepareMLMDataset, MLMDatasetArtifact
from .mlm_metrics import (compute_metrics,
                      preprocess_logits_for_metrics,
                      compute_perplexity)


from .modelling import (BuildMLMModel,
                        MLMModelConfig)

from .trainer_config import MLMTrainerKwargs
from ...shared.trainer_config_base import BuildContext, HFTrainingComponents
from ...shared.trainer_builder_base import HFTrainingCompBuilder
from ...shared.modelling_base import TrainingStrategyFactory
from ...shared.factory import build_training_args
from ner_pipeline.utils.common import set_seed



class MLMTrainingCompBuilder(HFTrainingCompBuilder):
    """
    Builder class responsible for constructing and incrementally enriching
    necessary training components used in instantiating model training 
    for MLM using HuggingFace's `Trainer` class.

    Parameters
    ----------
    context : BuildContext
        Execution context providing configuration, device placement,
        output paths, and experiment logging handles.

    Returns
    -------
    HFTrainingComponents
        A Fully constructed and optionally strategy-augmented training
        components ready to be passed to a HuggingFace `Trainer`.

    """
    def __init__(self, context: BuildContext):
        super().__init__(context)
        self.strategy = TrainingStrategyFactory.create(self.context.cfg)
        self.task_cfg = self.cfg.task
        self.use_whole_word_mask = getattr(self.task_cfg, "use_whole_word_mask", False)
        self.use_span_mask = getattr(self.task_cfg, "use_span_mask", False)
        self._components = self._build_components()

    @property
    def cfg(self) -> DictConfig:
        return self.context.cfg

    @property
    def components(self) -> HFTrainingComponents:
        return self._components

    @property
    def trainer_kwargs(self) -> MLMTrainerKwargs:
        return self._components.trainer_kwargs
    
    @property
    def strategy_kwargs(self):
        return self._components.strategy_kwargs

    @property
    def metadata(self):
        return self._components.metadata
    
    @property
    def callbacks(self):
        return self._components.callbacks

    @property
    def dataset_artifact(self) -> MLMDatasetArtifact:
        if not hasattr(self, "_cached_dataset_artifact"):
            dataset_prep = PrepareMLMDataset(self.cfg, self.context.wandb_run)
            self._cached_dataset_artifact = dataset_prep.prepare()
        return self._cached_dataset_artifact
    
    def _build_components(self):
        #init seed for reproducibility
        set_seed(self.cfg.seed)

        wandb_run, wandb_artifact, output_dir, device = (self.context.wandb_run,
                                                self.context.wandb_artifact,
                                                self.context.output_dir,
                                                self.context.device
                                                )

        #load tokenized_datasets
        train_dataset = self.dataset_artifact.train_dataset
        eval_dataset = self.dataset_artifact.eval_dataset

        #build model
        if self.cfg.task.tokenizer_name_or_path:
            tokenizer_ckpt_path = self.cfg.task.tokenizer_name_or_path
        else:
            tokenizer_ckpt_path = self.cfg.task.model_name_or_path 
        model_builder_config = MLMModelConfig(
                                        checkpoint=self.task_cfg.model_name_or_path,
                                        device=device,
                                        tokenizer_checkpoint=tokenizer_ckpt_path)
        
        build_for_hyperparam_tuning = getattr(self.cfg.task, "do_hyperparam_with_trainer", False)
        model_builder = BuildMLMModel(model_builder_config, 
                                      build_for_hyperparam_tuning=build_for_hyperparam_tuning)
        model = model_builder.build()

        #load tokenizer, data collator
        tokenizer, data_collator = self._build_tokenizer_data_collator(tokenizer_ckpt_path)

        #build training args
        training_args = build_training_args(self.cfg, output_dir)

        trainer_kwargs = MLMTrainerKwargs(
                                train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                                model=model,
                                processing_class=tokenizer,
                                args=training_args,
                                data_collator=data_collator,
                                compute_metrics=compute_metrics,
                                preprocess_logits_for_metrics=preprocess_logits_for_metrics
                        )

        components = HFTrainingComponents(
                                       trainer_kwargs=trainer_kwargs,
                                       strategy_kwargs=self._build_trainer_specific_kwargs(),
                                       callbacks=[],
                                       )
        logger.info(f"All training components for this run have been built successfully as below:\n{components}")

        return components


    def _build_tokenizer_data_collator(self, tokenizer_ckpt_path):
        "Builds tokenizer and data collator class for Masked Langague modelling"

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt_path)
        #tokenizer = AutoTokenizer.from_pretrained(self.task_cfg.model_name_or_path)
        mlm_probability = getattr(self.task_cfg, "mlm_probability", 0.15)

        if self.use_whole_word_mask:
            data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, 
                                                            mlm_probability=mlm_probability)

        elif self.use_span_mask:
            data_collator = DataCollatorForSpanMasking(tokenizer=tokenizer,
                                                        mlm_probability=mlm_probability)
        else:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, 
                                                            mlm_probability=mlm_probability)

        return tokenizer, data_collator

    
    def add_metadata(self, **kwargs):
        new_metadata = {**self.metadata, **kwargs}
        self._components = replace(self._components, metadata=new_metadata)
        return self

    def add_strategy_kwargs(self, **kwargs):
        new_strategy_kwargs = {**self.strategy_kwargs, **kwargs}
        self._components = replace(self._components, strategy_kwargs=new_strategy_kwargs)
        return self

    def add_callback(self, *args):
        new_callbacks = [*self.callbacks, *args]
        self._components = replace(self._components, callbacks=new_callbacks)
        return self
    
   
    def apply_strategy(self) -> HFTrainingComponents:
        "Applies training specific strategy as defined in cfg"
        self.strategy.apply(self)

        return self._components
