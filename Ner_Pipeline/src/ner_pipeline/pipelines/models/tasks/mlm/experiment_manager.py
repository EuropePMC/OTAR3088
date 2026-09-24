
from pathlib import Path
from omegaconf import DictConfig

from .tokenization_utils import MLMMaskingMixin
from ...shared.experiment_manager_base import ExperimentSubfolderBuilder


class MLMExperimentSubfolderBuilder(MLMMaskingMixin, ExperimentSubfolderBuilder):
    """
    Experiment subfolder utility class for Masked Language Modeling tasks. 
    Inherits from ExperimentSubfolderBuilder.
    Implements task-specific functionality.

    """
    task_type = "mlm"

    def __init__(self, resolved_cfg: MLMResolvedConfig):
        super().__init__(resolved_cfg)
        self.resolved_cfg = resolved_cfg
        self.task_cfg = self.cfg.task
        self.mlm_task_type = resolved_cfg.mlm_task_type
        self.mlm_trainer_type = resolved_cfg.mlm_trainer_type
        self.tokenizer_path = resolved_cfg.tokenizer_path
       
    
    def _build_base_subfolder(self) -> Path:
        """
        Constructs the base subfolder name for MLM experiments based on the configuration.
        Overrides super class method.

        Returns:
            Path: The base subfolder path.
        """
        normalised_training_strategy_name = self._normalise_training_strategy_name(self.training_strategy)
        masking_method = self._add_masking_method(self.task_cfg)

        parts = [
            self.task_type.upper(),
            self.mlm_task_type.upper(),
            masking_method,
            self.dataset_label,
            self.model_architecture,
            ("Adapted_Tokenizer" if self.tokenizer_path else ""),
            (f"{normalised_training_strategy_name}Strategy")
        ]
        return Path(*parts)



class MLMBaseExperimentSubfolderBuilder(MLMExperimentSubfolderBuilder):
    """
    Concrete implementation of MLMExperimentSubfolderBuilder for base training strategy.
    Constructs the experiment subfolder name based on base training strategy configuration.
    """
    def build(self) -> Path:
        if self._is_built:
            return self.subfolder
        subfolder = self._build_base_subfolder()
        subfolder = self._add_training_kwargs(subfolder)
        self._subfolder = subfolder
        self._is_built = True
        return self.subfolder
