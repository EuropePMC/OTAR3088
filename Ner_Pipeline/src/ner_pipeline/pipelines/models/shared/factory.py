
import os
import math

from typing import Union

from enum import Enum
from dataclasses import dataclass
from abc import abstractmethod, ABC

from omegaconf import DictConfig

import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from transformers import (PreTrainedTokenizer, 
                          PreTrainedTokenizerFast,
                          TrainingArguments)





def split_dataset(example:Dataset, test_size:float = 0.2) -> DatasetDict:
  """
  Splits a dataset into train and validation sets
  Args:
    example: A huggingface dataset class
    test_size: Ratio to split dataset by
  returns:
  A huggingface dataset with train and validation splits

  """
  example = example.train_test_split(test_size=test_size, seed=42)
  example["validation"] = example["test"]
  example.pop("test")
  return example


def format_model_checkpoint_name(ckpt:str):

    base_name = os.path.basename(ckpt)
    if not "-" in base_name:
      ckpt_name = base_name

    else:  
      base_name = base_name.split("-")
      if len(base_name) >= 10:
        ckpt_name = "_".join(base_name[:8])
      else:
        ckpt_name = "_".join(base_name)
    return ckpt_name



def build_training_args(cfg:DictConfig, output_dir:str):
    report_to = "wandb" if cfg.use_wandb else "none"
    remove_unused_columns = True
    if cfg.task_type.lower() == "mlm" and getattr(cfg.task, "use_whole_word_mask", False):
        remove_unused_columns = False

    return TrainingArguments(
        output_dir=output_dir,
        logging_dir=f"{output_dir}/logs",
        report_to = report_to,
        remove_unused_columns=remove_unused_columns,
        **cfg.task.args
    )


def compute_training_steps(args:TrainingArguments, train_dataset:Dataset):

  """
  Calculate total training steps based on dataset size, batch size, gradient accumulation steps and number of epochs
  Args:
    args: TrainingArguments object
    train_dataset: Huggingface dataset object
  Returns:
    Total number of training steps as integer
  """
  # Effective batch size accounts for accumulation and number of devices
  effective_batch_size = (
      args.per_device_train_batch_size *
      args.gradient_accumulation_steps *
      max(1, torch.cuda.device_count())  # default 1 if no GPU
  )
  #get total num steps per epoch
  num_update_steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)

  #get total training steps
  num_training_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
  
  return num_training_steps



def compute_warmup_steps(num_training_steps, warmup_ratio):
    "Return warmup steps given a ratio (e.g., 0.1 = 10%)."
    return int(num_training_steps * warmup_ratio)


def extract_model_backbone(model: nn.Module):
    """
    Return (backbone_name, backbone) for common architectures.
    """
    supported = {"bert", "roberta", "distilbert"}
    if hasattr(model, "bert"):
        return "bert", model.bert
    if hasattr(model, "roberta"):
        return "roberta", model.roberta
    if hasattr(model, "distilbert"):
        return "distilbert", model.distilbert
    raise ValueError("Unsupported Bert backbone. Inspect model to find name of backbone.",
                    f"Supported backbones are: {supported}")



def extract_encoder_layers(model: nn.Module):
  "Finds encoder layer for BERT model variants"
  model_name, backbone = extract_model_backbone(model)
  if model_name == "bert":
    encoder_layer = backbone.encoder.layer
  elif model_name == "roberta":
    encoder_layer = backbone.encoder.layer
  elif model_name == "distilbert":
    encoder_layer = backbone.transformer.layer
  return encoder_layer


def count_trainable_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class TrainingStrategyName(str, Enum):
    "Training Strategy types"
    BASE = "base"
    REINIT = "reinit_only"
    LLRD = "llrd_only"
    REINIT_LLRD = "reinit_llrd"
    GROUPED_LLRD = "grouped_llrd" 



@dataclass
class BaseTokenizationTransform(ABC):
    """
    Base tokenization transform.

    Shared by task-specific tokenization transforms such as:
    - NERTokenizationTransform
    - MLMTokenizationTransform
    """
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    text_col: str
    do_truncate: bool
    max_length: int = 512

    @abstractmethod
    def __call__(self):
        raise NotImplementedError("Subclasses must implement this method")



@dataclass(frozen=True)
class ReinitStrategyParams:
    reinit_classifier: bool=False
    reinit_k_layers: int=0

    @classmethod
    def from_cfg(cls, cfg: DictConfig):
        reinit_classifier = getattr(cfg, "reinit_classifier", False)
        reinit_k_layers = getattr(cfg, "reinit_k_layers", 0)
        return cls(reinit_classifier=reinit_classifier, reinit_k_layers=reinit_k_layers)


@dataclass(frozen=True)
class LLRDStrategyParams:
    llrd_factor: float=1.0

    @classmethod
    def from_cfg(cls, cfg: DictConfig):
        llrd_factor = getattr(cfg, "llrd_factor", 1.0)
        return cls(llrd_factor=llrd_factor)


@dataclass(frozen=True)
class ReinitLLRDStrategyParams(ReinitStrategyParams, LLRDStrategyParams):
    reinit_classifier: bool=False
    reinit_k_layers: int=0
    llrd_factor: float=1.0
    @classmethod
    def from_cfg(cls, cfg: DictConfig):
        return cls(
            reinit_classifier=getattr(cfg, "reinit_classifier", False),
            reinit_k_layers=getattr(cfg, "reinit_k_layers", 1),
            llrd_factor=getattr(cfg, "llrd_factor", 1.0),
        )


class StrategyParamsMixin:
    """
    Shared Mixin class for resolving training-strategy parameters from cfg.

    This mixin assumes the consuming class has:
        self.cfg
        self.training_strategy
    """

    def _get_reinit_params(self) -> ReinitStrategyParams:
        return ReinitStrategyParams.from_cfg(self.cfg)
    
    def _get_llrd_params(self) -> LLRDStrategyParams:
        return LLRDStrategyParams.from_cfg(self.cfg)
    
    def _get_reinit_llrd_params(self) -> ReinitLLRDStrategyParams:
        return ReinitLLRDStrategyParams.from_cfg(self.cfg)



@dataclass
class BaseResolvedConfig(ABC):
    """
    Base resolved config dataclass.

    Shared by task-specific resolved config dataclasses such as:
    - NERResolvedConfig
    - MLMResolvedConfig
    """
    cfg: DictConfig
    dataset_label: str
    model_architecture: str
    task_type: str

    @classmethod
    def _common_kwargs(cls, cfg: DictConfig) -> dict:
        task_cfg = cfg.task
        task_type = task_cfg.task_type.lower()

        dataset_name = task_cfg.data.name
        dataset_version = getattr(task_cfg.data, "version", "")
        dataset_label = f"{dataset_name}_{dataset_version}" if dataset_version else dataset_name

        model_name_or_path = (
            getattr(task_cfg, "model_name_or_path", None)
            or getattr(cfg, "model_name_or_path", None)
        )
        model_architecture = format_model_checkpoint_name(model_name_or_path)

        return {
            "cfg": cfg,
            "dataset_label": dataset_label,
            "model_architecture": model_architecture,
            "task_type": task_type,
        }

    @classmethod
    def from_cfg(cls, cfg: DictConfig):
        return cls(**cls._common_kwargs(cfg))      



class BaseStrategyFactory(ABC):
    _enum_class = TrainingStrategyName
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls is BaseStrategyFactory:
            return

        if cls._registry is None:
            raise TypeError(f"{cls.__name__} must define a `_registry` class attribute.")

    @classmethod
    def _get_strategy(cls, resolved_cfg: BaseResolvedConfig):
        return cls._enum_class(resolved_cfg.training_strategy.lower())
    
    @classmethod
    def create(cls, resolved_cfg: BaseResolvedConfig, *args, **kwargs):
        strategy = cls._get_strategy(resolved_cfg)
        target_cls = cls._registry[strategy]
        print(f"Initialised Helper: {target_cls.__name__}()")
        return target_cls(resolved_cfg, *args, **kwargs)


