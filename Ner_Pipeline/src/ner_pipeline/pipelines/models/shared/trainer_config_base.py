from typing import (
                    List, Dict, 
                    Union, Optional, 
                    Callable, Any,
                    Type
                )
            
from abc import ABC
from dataclasses import field, dataclass
from omegaconf import DictConfig

from wandb.sdk.wandb_run import Run as WandbRun
from wandb import Artifact as WandbArtifact

import torch.nn as nn
from datasets import Dataset
from transformers import (
    PreTrainedTokenizerBase, 
    PreTrainedTokenizerFast, 
    TrainingArguments,
    TrainerCallback

)

@dataclass(frozen=True)
class BaseTrainerKwargs:
    """
    Base Container object for keyword arguments passed directly to the
    HuggingFace `Trainer` constructor.


    Attributes
    ----------
    train_dataset : Dataset
        Tokenized training dataset.
    eval_dataset : Dataset
        Tokenized validation dataset.
    model : torch.nn.Module
        Model instance used for training.
    processing_class : PreTrainedTokenizerBase | PreTrainedTokenizerFast
        Tokenizer used for preprocessing and postprocessing.
    """

    model: nn.Module
    processing_class: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast]
    args: TrainingArguments
    compute_metrics: Callable




@dataclass
class BuildContext:
    """
    Container for all external inputs required to build HuggingFace
    training components.

    This object represents the *execution context* for a training run,
    separating environment- and run-specific concerns from the
    component construction logic.

    Attributes
    ----------
    cfg : DictConfig
        Hydra configuration object containing all experiment, model,
        data, and training parameters.
    output_dir : str
        Directory where model checkpoints, logs, and artifacts
        will be written.
    device : str
        Device identifier (e.g. "cpu", "cuda", "cuda:0") on which
        the model will be instantiated.
    wandb_run : Optional[WandbRun], default=None
        Active Weights & Biases run for logging, if enabled via cfg.
    run_artifact : Optional[RunArtifact], default=None
        W&B artifact used to log model checkpoints or datasets metadata.
    """
    
    cfg: DictConfig
    output_dir: str
    device: str
    wandb_run: Optional[WandbRun] = None
    wandb_artifact: Optional[WandbArtifact] = None


@dataclass(kw_only=True)
class HFTrainingComponents:
    trainer_kwargs: BaseTrainerKwargs
    strategy_kwargs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    callbacks: List[TrainerCallback] = field(default_factory=list)

@dataclass(kw_only=True)
class HFInferenceComponents:
    test_dataset: Dataset
    trainer_kwargs: BaseTrainerKwargs


@dataclass
class PushToHubParams:
    "Default HF params"
    repo_id:str
    push_to_org_repo: bool = False
    is_private: bool = False
    token: str = None
    commit_message: str = "Add model to hub"


@dataclass(kw_only=True)
class HFModelConfig:
    checkpoint: str
    device: str


class BaseTrainerFactory(ABC):
    _enum_class = None
    _registry = {}
    

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls is BaseTrainerFactory:
            return

        missing = []
        if cls._registry is None:
            missing.append("_registry")
        if cls._enum_class is None:
            missing.append("_enum_class")

        if missing:
            raise TypeError(
                f"{cls.__name__} must define: {', '.join(missing)}"
            )

    @classmethod
    def get_trainer_class(cls, trainer_type:str):
        """
        Fetches and returns the trainer class associated with the requested trainer type.
        """
        trainer_name = cls._enum_class(trainer_type.lower())
        trainer_cls = cls._registry[trainer_name]

        return trainer_cls