from enum import Enum
from dataclasses import dataclass

from typing import (
                    List, Dict, 
                    Union, Optional, 
                    Callable, Any,
                    Type, Literal
                )


from transformers import DataCollatorForTokenClassification

from  ...shared.trainer_config_base import BaseTrainerKwargs, HFModelConfig
from ...shared.trainer_base import HFTrainingOrchestratorConfig



@dataclass(frozen=True)
class NerTrainerKwargs(BaseTrainerKwargs):
    """
    Ner container object for keyword arguments passed directly to the
    HuggingFace `Trainer` constructor specific for NER Training.
    Inherits from `BaseTrainerKwargs`
    """
    data_collator: DataCollatorForTokenClassification
    id2label: Dict[int, str]



@dataclass(frozen=True)
class NerTrainingOrchestratorConfig(HFTrainingOrchestratorConfig):
    pass




@dataclass
class NerPredictions:
    true_labels: List[List[str]]
    pred_labels: List[List[str]]
    label_names: List[str]


@dataclass(kw_only=True)
class NerModelConfig(HFModelConfig):
    num_labels: int
    id2label: Dict[int, str]
    label2id: Dict[str, int]
    ner_head_type: Literal["standard", "crf"] 


class NerTrainerType(str, Enum):
    STANDARD = "base"
    CRF = "crf"
    WEIGHTED = "weighted"

