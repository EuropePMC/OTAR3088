
from typing import (List, Dict, 
                    Union, Callable)
from dataclasses import dataclass

import torch.nn as nn
from datasets import Dataset
from transformers import DataCollatorForTokenClassification

from ....shared.trainer_config_base import BaseTrainerKwargs, HFInferenceComponents


@dataclass(frozen=True)
class NERInferenceTrainerKwargs(BaseTrainerKwargs):
    data_collator: DataCollatorForTokenClassification


@dataclass
class NERHFInferenceComponents(HFInferenceComponents):
    label2id: Dict[str, int]
    id2label: Dict[int, str]
    label_list: List[str]
    trainer_kwargs: NERInferenceTrainerKwargs
    
