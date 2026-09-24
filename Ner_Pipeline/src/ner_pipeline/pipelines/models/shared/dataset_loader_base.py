
from dataclasses import dataclass
from abc import ABC, abstractmethod
from omegaconf import DictConfig

from datasets import Dataset


class DatasetLoader(ABC):
    def __init__(self, cfg, hf_cache_dir=None):
        self.cfg = cfg
        self.hf_cache_dir = hf_cache_dir
        
    @abstractmethod
    def load(self):
        pass


@dataclass(frozen=True)
class DatasetArtifact(ABC):
    """
    Base class stores dataset artifacts shared across NLP tasks
    e.g train_dataset, eval_dataset
    Used for easy retrieval and compatible with huggingface trainer class
    Child classes extends this class and adds task-specific artifacts.
    e.g label2id,id2label 
    """
    train_dataset: Dataset
    eval_dataset: Dataset


class PrepareDataset(ABC):
    """
    Base class for task-specific dataset preparation.
    Handles dataset preparation for modelling shared across NLP tasks, e.g:
    - Tokenization
    - Data augmentation
    - Data splitting
    Child classes implement task-specific dataset preparation.
    
    """
    def __init__(self, cfg, wandb_run=None):
        self.cfg = cfg
        self.wandb_run = wandb_run

    @abstractmethod
    def prepare(self):
        raise NotImplementedError("Task-specific dataset preparation class must be implemented in child classes.")

    @abstractmethod
    def _apply_tokenization(self, dataset):
        pass



class DatasetConfigValidator(ABC):
    """
    Base dataset config validator.

    Handles validation shared across NLP tasks:
    - source_type
        - HuggingFace datasetsource validation
        - local dataset source validation

    Child classes implements task-specific validation.
    """

    _SUPPORTED_SOURCE_TYPES = {"hf", "local"}

    @classmethod
    def validate(cls, cfg:DictConfig):
        """
        Validates the dataset configuration.

        Args:
            cfg: The configuration object containing dataset parameters.

        Raises:
            ValueError: If the source_type is unsupported or if required parameters are missing.
        """
        data_cfg = cfg.task.data
        source_type = getattr(data_cfg, "source_type", None)
        source_type = source_type.lower()
        
        cls.validate_source_type(source_type)
        cls.validate_task_config(data_cfg)

        if source_type == "hf":
            cls.validate_hf_source(getattr(data_cfg, "hf_path", None))

        elif source_type == "local":
            cls.validate_local_source(data_cfg)


    @classmethod
    @abstractmethod
    def validate_task_config(cls, data_cfg: DictConfig):
        """
        Validates task-specific dataset configuration.
        Examples:
        - NER: text_col, label_col
        - MLM: text_col

        Args:
            data_cfg: The configuration object containing dataset parameters.
        """

        raise NotImplementedError("Task-specific validation must be implemented in child classes.")


    @classmethod
    def validate_source_type(cls, source_type: str) -> None:
        """
        Validates the source_type parameter.

        Args:
            source_type: The source type to validate.

        Raises:
            ValueError: If the source_type is unsupported.
        """
        
        if not source_type:
            raise ValueError(
                "Source type missing from data config.\n"
                "Source type is required for loading the dataset using the appropriate method.\n"
                "Use one of: `local` or `hf`."
            )
        
        
        if source_type not in cls._SUPPORTED_SOURCE_TYPES:
            raise ValueError(
                "Invalid source_type. Supported `source_type` values are:\n"
                f"{cls._SUPPORTED_SOURCE_TYPES}"
            )


    @classmethod
    def validate_hf_source(cls, hf_path: str) -> None:
        """
        Validates the HuggingFace dataset source configuration.

        Args:
            hf_path: The path to the HuggingFace dataset.

        Raises:
            ValueError: If hf_path is not provided.
        """
        if not hf_path:
            raise ValueError(
                "Hugging Face path is required when `source_type == 'hf'`.\n"
                "Example hf_path format: `OTAR3088/CeLLate1.0`."
            )


    @classmethod
    def validate_local_source(cls, data_cfg: DictConfig) -> None:
        """
        Validates the local dataset source configuration.

        Args:
            data_cfg: The configuration object containing dataset parameters.

        Raises:
            ValueError: If required parameters for local source are missing.
        """
        data_dir = getattr(data_cfg, "data_dir", None)


        if not data_dir:
            raise ValueError(
                "Local data folder is required when `source_type == 'local'`.\n"
                "Expected format: `/absolute/path/to/folder/`."
            )

    


