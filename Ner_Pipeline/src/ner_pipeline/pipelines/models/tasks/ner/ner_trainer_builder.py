from dataclasses import replace
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import torch.nn as nn
from transformers.trainer_callback import EarlyStoppingCallback

from .ner_dataset_loader import PrepareNERDataset
from .ner_metrics import seqeval_metrics
from .modelling import (
                BuildNERModel,
                NERTrainerCallback
                        )
from .ner_factory import build_tokenizer_data_collator
from .ner_trainer_config import NERTrainerKwargs, NERTrainingComponents
from .ner_factory import NERModelConfig
from ...shared.trainer_config_base import BuildContext
from ...shared.trainer_builder_base import HFTrainingCompBuilder
from ...shared.modelling_base import TrainingStrategyFactory
from ...shared.factory import build_training_args
from ner_pipeline.utils.common import set_seed


class NERTrainingCompBuilder(HFTrainingCompBuilder):
    """
    Builder class responsible for constructing and incrementally enriching
    necessary training components used in instantiating model training 
    for NER using HuggingFace's `Trainer` class.

    Parameters
    ----------
    context : BuildContext
        Execution context providing configuration, device placement,
        output paths, and experiment logging handles.

    Returns
    -------
    NERTrainingComponents
        A Fully constructed and optionally strategy-augmented NER training
        components ready to be passed to a HuggingFace `Trainer`.

    """
    def __init__(self, context: BuildContext):
        super().__init__(context)
        self.strategy = TrainingStrategyFactory.create(self.context.cfg)
        self._components = self._build_components()


    @property
    def cfg(self) -> DictConfig:
        return self.context.cfg

    @property
    def components(self) -> NERTrainingComponents:
        return self._components

    @property
    def trainer_kwargs(self):
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
    def dataset_artifact(self):
        if not hasattr(self, "_cached_dataset_artifact"):
            dataset_prep = PrepareNERDataset(self.cfg, self.context.wandb_run)
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
        #Load datasets + labels
        train_dataset = self.dataset_artifact.train_dataset
        eval_dataset = self.dataset_artifact.eval_dataset
        unique_tags = self.dataset_artifact.unique_tags
        label2id =  self.dataset_artifact.label2id
        id2label = self.dataset_artifact.id2label


        #build model
        model_builder_config = NERModelConfig(
                                        checkpoint=self.cfg.task.model_name_or_path,
                                        device=device,
                                        ner_head_type=getattr(self.cfg.task, "ner_head_type", "standard"),
                                        num_labels=len(unique_tags),
                                        label2id=label2id,
                                        id2label=id2label
                                                        )

        build_for_hyperparam_tuning = getattr(self.cfg.task, "do_hyperparam_with_trainer", False)
        model_builder = BuildNERModel(model_builder_config, 
                                      build_for_hyperparam_tuning=build_for_hyperparam_tuning)
        model = model_builder.build()
        
        # #max_pos_emb = model.config.max_position_embeddings
        # max_pos_emb = 256

        #load tokenizer, data_collator
        tokenizer, data_collator = build_tokenizer_data_collator(self.cfg.task.model_name_or_path)

    
        #build training args
        training_args = build_training_args(self.cfg, output_dir)

        #prepare metrics
        compute_metrics = seqeval_metrics(unique_tags)

        # Optional log to wandb
        if getattr(self.cfg, "use_wandb", False) and wandb_run is not None:
            wandb_run.log({"Model checkpoint used for this run": self.cfg.task.model_name_or_path})
            # Convert ListConfig to native Python list for JSON serialization
            unique_tags_list = OmegaConf.to_container(unique_tags) \
                                            if hasattr(unique_tags, '__class__') \
                                            and 'ListConfig' in str(type(unique_tags)) \
                                            else list(unique_tags)
            wandb_run.log({
                "Unique labels": unique_tags_list,
                "Num classes": len(unique_tags)
            })
        trainer_kwargs = NERTrainerKwargs(
                                train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                                model=model,
                                processing_class=tokenizer,
                                args=training_args,
                                data_collator=data_collator,
                                compute_metrics=compute_metrics,
                                id2label=id2label
                        )


        components = NERTrainingComponents(
                                       trainer_kwargs=trainer_kwargs,
                                       strategy_kwargs=self._build_trainer_specific_kwargs(),
                                       callbacks=[NERTrainerCallback],
                                       )
        logger.info(f"All training components for this run have been built successfully as below:\n{components}")

        return components

    def _build_trainer_specific_kwargs(self):
        trainer_type = getattr(self.cfg.task, "trainer_type", "base").lower()
        if trainer_type != "focal":
            return {}

        alpha = getattr(self.cfg.task, "focal_loss_alpha", None)
        if alpha is not None and OmegaConf.is_config(alpha):
            alpha = OmegaConf.to_container(alpha, resolve=True)

        return {
            "focal_loss_gamma": getattr(self.cfg.task, "focal_loss_gamma", 2.0),
            "focal_loss_alpha": alpha,
        }

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

    def update_model(self, model:nn.Module):
        "Updates a model if reinit training strategy is used"
        new_trainer_kwargs = replace(self.trainer_kwargs, model=model)
        self._components = replace(
                            self._components,
                            trainer_kwargs=new_trainer_kwargs
                                    )


    def apply_strategy(self) -> NERTrainingComponents:
        "Applies training specific strategy as defined in cfg"
        self.strategy.apply(self)

        return self._components
