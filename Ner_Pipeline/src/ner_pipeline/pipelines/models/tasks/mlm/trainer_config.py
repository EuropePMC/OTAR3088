
from enum import Enum
from dataclasses import dataclass
from typing import Union, Callable

from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask

from .modelling import MLMBaseTrainer
from ...shared.trainer_config_base import (BaseTrainerKwargs,
                                            HFModelConfig,
                                            BaseTrainerFactory)

from ...shared.trainer_config_base import HFTrainingComponents


class MLMTrainerType(str, Enum):
    STANDARD = "base"


@dataclass(frozen=True)
class MLMTrainerKwargs(BaseTrainerKwargs):
    data_collator: Union[DataCollatorForLanguageModeling, DataCollatorForWholeWordMask]
    preprocess_logits_for_metrics: Callable



class MLMTrainerFactory(BaseTrainerFactory):
    """
    Factory responsible for instantiating the correct
    trainer class for MLM model training.
    """
    _enum_class = MLMTrainerType
    
    _registry = {
        MLMTrainerType.STANDARD: MLMBaseTrainer
    }


@dataclass(kw_only=True)
class MLMTrainingComponents(HFTrainingComponents):
    pass