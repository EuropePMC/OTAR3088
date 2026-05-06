from abc import ABC, abstractmethod
from dataclasses import replace
from typing import List, Any, Dict, Optional
from omegaconf import DictConfig

from .modelling_base import TrainingStrategyFactory
# from .metrics_logger import MetricsLogger

# from .trainer_config_base import BuildComponents, BuildContext, BaseTrainerKwargs



class HFTrainingCompBuilder(ABC):
    def __init__(self, context):
        self.context = context

    @abstractmethod
    def _build_components(self):
        pass

    @abstractmethod
    def apply_strategy(self):
        pass


        

