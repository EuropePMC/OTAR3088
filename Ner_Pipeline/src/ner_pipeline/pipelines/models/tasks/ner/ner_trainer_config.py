from typing import Dict
from enum import Enum
from dataclasses import dataclass

from datasets import Dataset
from transformers import DataCollatorForTokenClassification

from .modelling import (BaseNERTrainer,
                       CRFNERTrainer,
                       FocalLossNERTrainer,
                       WeightedNERTrainer
                       )


from ...shared.trainer_config_base import (BaseTrainerKwargs,
                                            HFModelConfig,
                                            BaseTrainerFactory)
                                            
from ...shared.trainer_base import HFTrainingOrchestratorConfig
from ...shared.trainer_config_base import HFTrainingComponents


@dataclass(frozen=True)
class NERTrainerKwargs(BaseTrainerKwargs):
    """
    NER container object for keyword arguments passed directly to the
    HuggingFace `Trainer` constructor specific for NER Training.
    Inherits from `BaseTrainerKwargs`
    """
    train_dataset: Dataset
    eval_dataset: Dataset
    data_collator: DataCollatorForTokenClassification
    id2label: Dict[int, str]


class NERTrainerType(str, Enum):
    STANDARD = "base"
    CRF = "crf"
    FOCAL = "focal"
    WEIGHTED = "weighted"


class NERTrainerFactory(BaseTrainerFactory):
    """
    Factory responsible for returning the correct
    trainer class for NER model training.
    """
    _enum_class = NERTrainerType
    _registry = {
        NERTrainerType.STANDARD: BaseNERTrainer,
        NERTrainerType.CRF: CRFNERTrainer,
        NERTrainerType.FOCAL: FocalLossNERTrainer,
        NERTrainerType.WEIGHTED: WeightedNERTrainer
    }

@dataclass(kw_only=True)
class NERTrainingComponents(HFTrainingComponents):
    pass