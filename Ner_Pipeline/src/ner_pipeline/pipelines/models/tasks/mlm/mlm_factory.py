

from pathlib import Path
from dataclasses import dataclass
from omegaconf import DictConfig
from loguru import logger

from .experiment_manager import MLMBaseExperimentSubfolderBuilder
from .logging_manager import MLMBaseLoguruHelper, MLMBaseWandbRunManager

from ...shared.trainer_base import HFTrainingOrchestratorConfig
from ...shared.factory import (TrainingStrategyName, 
                               BaseStrategyFactory,
                               )



def clean_text(example, text_col):
    sentences = [x.strip() for x in example[text_col]]
    return {text_col: sentences}

    
@dataclass(frozen=True)
class MLMTrainingOrchestratorConfig(HFTrainingOrchestratorConfig):
    pass


class MLMExperimentSubfolderFactory(BaseStrategyFactory):
    _registry = {
        TrainingStrategyName.BASE: MLMBaseExperimentSubfolderBuilder,
    }


class MLMLoguruHelperFactory(BaseStrategyFactory):
    _registry = {
        TrainingStrategyName.BASE: MLMBaseLoguruHelper,
    }

    @classmethod
    def create(cls, cfg: DictConfig, *args, **kwargs):
        helper = super().create(cfg, *args, **kwargs)
        print(f"Loguru Helper for {cfg.task_type.upper()}-task set to: {helper.__class__.__name__}()")
        return helper


class MLMWandbRunManagerFactory(BaseStrategyFactory):
    _registry = {
        TrainingStrategyName.BASE: MLMBaseWandbRunManager,
    }

    @classmethod
    def create(cls, cfg: DictConfig, log_dir: Path, *args, **kwargs):
        manager = super().create(cfg, log_dir, *args, **kwargs)
        logger.info(f"Wandb Manager set to: {manager}")
        return manager


class MLMDataCollatorFactory(BaseStrategyFactory):
    _registry = {

    }
    @classmethod
    def create(cls, cfg:DictConfig):
        pass