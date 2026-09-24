
from pathlib import Path
from omegaconf import DictConfig

from .logging_manager import NERResolvedConfig
from ...shared.experiment_manager_base import (ExperimentSubfolderBuilder,
                                              ReinitStrategySubfolderMixin,
                                              LLRDStrategySubfolderMixin,
                                              ReinitLLRDStrategySubfolderMixin)


class NERExperimentSubfolderBuilder(ExperimentSubfolderBuilder):
    """
    Experiment subfolder utility class for NER tasks. 
    Inherits from ExperimentSubfolderBuilder.
    Implements task-specific functionality.

    """
    task_type = "ner"
    def __init__(self, resolved_cfg: NERResolvedConfig):
        super().__init__(resolved_cfg)
        self.ner_head_type = resolved_cfg.ner_head_type
        self.ner_trainer_type = resolved_cfg.ner_trainer_type
        self.use_data_aug = resolved_cfg.use_data_aug
        self.ner_data_aug_method = resolved_cfg.ner_data_aug_method
        # self.ner_head_type = getattr(cfg.task, "ner_head_type", "standard")
        # self.trainer_type = getattr(cfg.task, "trainer_type", "base")
        # self.use_data_aug = getattr(cfg.task, "use_data_aug", False)
        # self.data_aug_method = getattr(cfg.task, "data_aug_method", None)


    def _build_base_subfolder(self) -> Path:
        """
        Constructs the base subfolder name for NER experiments based on the configuration.
        Returns:
            Path: The base subfolder path.
        """

        base_parts = super()._build_base_subfolder()
        ner_parts = [
            (f"{self.ner_head_type.capitalize()}NerHead"),
            (f"{self.ner_trainer_type.capitalize()}Trainer")
        ]
        return Path(*base_parts, *ner_parts)


    def _add_data_aug_method(self, subfolder: Path) -> Path:
        """
        Appends NER data augmentation method to the subfolder name if it exists in the configuration.
        """
        if self.use_data_aug and self.ner_data_aug_method:
            return subfolder / f"with_{self.ner_data_aug_method}"
        return subfolder / "no_data_aug"


class NERBaseExperimentSubfolderBuilder(NERExperimentSubfolderBuilder):
    """
    Concrete implementation of NERExperimentSubfolderBuilder for base training strategy.
    Constructs the experiment subfolder name based on base training strategy configuration.
    """
    def build(self) -> Path:
        if self._is_built:
            return self.subfolder
        subfolder = self._build_base_subfolder()
        subfolder = self._add_data_aug_method(subfolder)
        subfolder = self._add_training_kwargs(subfolder)
        self._subfolder = subfolder
        self._is_built = True
        return self.subfolder


class NERReinitExperimentSubfolderBuilder(ReinitStrategySubfolderMixin, NERExperimentSubfolderBuilder):
    """
    Concrete implementation of NERExperimentSubfolderBuilder for Reinit training strategy.
    Constructs the experiment subfolder name based on Reinit training strategy configuration.
    """
    
    def build(self) -> Path:
        if self._is_built:
            return self.subfolder
        subfolder = self._build_base_subfolder()
        subfolder = self._add_data_aug_method(subfolder)
        subfolder = self._add_training_kwargs(subfolder)
        subfolder = self._add_reinit_params(subfolder)
        self._subfolder = subfolder
        self._is_built = True
        return self.subfolder


class NERLLRDExperimentSubfolderBuilder(LLRDStrategySubfolderMixin, NERExperimentSubfolderBuilder):
    """
    Concrete implementation of NERExperimentSubfolderBuilder for LLRD training strategy.
    Constructs the experiment subfolder name based on LLRD training strategy configuration.
    """
    
    def build(self) -> Path:
        if self._is_built:
            return self.subfolder
        subfolder = self._build_base_subfolder()
        subfolder = self._add_data_aug_method(subfolder)
        subfolder = self._add_training_kwargs(subfolder)
        subfolder = self._add_llrd_params(subfolder)
        self._subfolder = subfolder
        self._is_built = True
        return self.subfolder


class NERReinitLLRDExperimentSubfolderBuilder(ReinitLLRDStrategySubfolderMixin, NERExperimentSubfolderBuilder):
    """
    Concrete implementation of NERExperimentSubfolderBuilder for ReinitLLRD training strategy.
    Constructs the experiment subfolder name based on combined Reinit and LLRD training strategy configuration.
    """
    
    def build(self) -> Path:
        if self._is_built:
            return self.subfolder
        subfolder = self._build_base_subfolder()
        subfolder = self._add_data_aug_method(subfolder)
        subfolder = self._add_training_kwargs(subfolder)
        subfolder = self._add_reinit_llrd_params(subfolder)
        self._subfolder = subfolder
        self._is_built = True
        return self.subfolder



