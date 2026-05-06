

import os
import math
from typing import Dict, Tuple, List
from abc import ABC, abstractmethod

from omegaconf import DictConfig
from loguru import logger

import numpy as np


from transformers import (Trainer, 
                          TrainerCallback, 
                          TrainingArguments,
                          AutoModelForTokenClassification, 
                          DataCollatorForTokenClassification, 
                          AutoTokenizer)

from .trainer_config_base import TrainingStrategyName, HFModelConfig
from .factory import count_trainable_params
from ..strategies.reinit_llrd import ReinitLLRDProcessor


class BuildModel(ABC):
    "Abstract class for initialising HF Based model"
    def __init__(self, model_config:HFModelConfig):
        self.checkpoint = model_config.checkpoint
        self.device = model_config.device

    @abstractmethod
    def build(self):
        pass

    def _log_trainable_params(self, model):
        trainable_params = count_trainable_params(model)
        logger.info(f"Model initialised with {trainable_params:,} trainable parameters.")




class TrainingStrategy(ABC):
    """
    Abstract base class for training strategy implementations.
    """
    @abstractmethod
    def apply(self, builder: "HFTrainingCompBuilder") -> None :
        """
        Apply the training strategy to the builder.

        Parameters
        ----------
        builder : HFTrainingCompBuilder
            Builder instance whose components will be modified in-place.
        """
        pass


class BaseStrategy(TrainingStrategy):
    "Base class utility logging Base training strategy attributes"
    def apply(self, builder):
        builder.add_metadata(strategy="base")


class ReinitStrategy(TrainingStrategy):
    """
    Utility class to apply reinit training strategy
    and log strategy attributes
    
    """
    def apply(self, builder):
        cfg = builder.cfg
        model = builder.trainer_kwargs.model

        strategy_impl = ReinitLLRDProcessor(cfg=cfg, model=model)
        #apply reinitalisation of model layers
        builder.update_model()
        builder.add_metadata(
            strategy = "reinit_only",
            reinit_classifier = cfg.get("reinit_classifier", False),
            reinit_k_layers = cfg.get("reinit_k_layers", None)
                )
        

class ReinitLLRDStrategy(TrainingStrategy):
    """
    Utility class to apply reinit_llrd training strategy
    and log strategy attributes
    """
    def apply(self, builder):
        cfg = builder.cfg
        model = builder.trainer_kwargs.model
        args = builder.trainer_kwargs.args
        train_dataset = builder.trainer_kwargs.train_dataset

        strategy_impl = ReinitLLRDProcessor(cfg=cfg, model=model, 
                                    train_dataset=train_dataset, training_args=args)
        builder.update_model(model)
        optimizer, lr_scheduler = strategy_impl.apply_llrd()

        builder.add_strategy_kwargs(
            optimizers = (optimizer, lr_scheduler)
        )

        builder.add_metadata(
            strategy = "reinit_llrd",
            reinit_k_layers = cfg.get("reinit_k_layers", 0),
            reinit_classifier = cfg.get("reinit_classifier", False),
            llrd_factor = cfg.get("llrd", 1.0),

        )
        


class LLRDStrategy(TrainingStrategy):
    """
    Utility class to apply LLRD training strategy
    and log strategy attributes
    """
    def apply(self, builder):
        cfg = builder.cfg
        model = builder.trainer_kwargs.model
        args = builder.trainer_kwargs.args
        train_dataset = builder.trainer_kwargs.train_dataset

        strategy_impl = ReinitLLRDProcessor(cfg=cfg, model=model, 
                                    train_dataset=train_dataset, training_args=args)

        optimizer, lr_scheduler = strategy_impl.apply_llrd()
        builder.add_strategy_kwargs(
            optimizers = (optimizer, lr_scheduler)
        )

        builder.add_metadata(
            strategy = "llrd_only",
            llrd_factor = cfg.get("llrd", 1.0)
        )

class GroupedLLRDStrategy(TrainingStrategy):
    """
    Utility class to apply grouped_llrd/ulmfit training strategy
    and log strategy attributes
    """
    def apply(self, builder):
        pass



class TrainingStrategyFactory:
    """
    Factory responsible for instantiating the correct training strategy.
    """
    _registry = {
        TrainingStrategyName.BASE: BaseStrategy,
        TrainingStrategyName.REINIT: ReinitStrategy,
        TrainingStrategyName.REINIT_LLRD: ReinitLLRDStrategy,
        TrainingStrategyName.LLRD: LLRDStrategy,
        TrainingStrategyName.GROUPED_LLRD: GroupedLLRDStrategy

    }

    @classmethod
    def create(cls, cfg:DictConfig):
        strategy = TrainingStrategyName(cfg.training_strategy.lower())
        return cls._registry[strategy]()





