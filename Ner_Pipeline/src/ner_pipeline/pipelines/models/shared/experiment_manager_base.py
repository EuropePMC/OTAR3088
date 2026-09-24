
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from pathlib import Path
from omegaconf import DictConfig

from .factory import (BaseResolvedConfig,
                      ReinitStrategyParams, 
                      LLRDStrategyParams, 
                      ReinitLLRDStrategyParams,
                      StrategyParamsMixin)



class ExperimentSubfolderBuilder(ABC):
    """
    Base class for building experiment subfolder names based on configuration.
    Used to organise and store model experimental artifacts
    """
    def __init__(self, resolved_cfg:BaseResolvedConfig):
        self.cfg = resolved_cfg.cfg
        self._is_built = False
        self._subfolder = None

        self.dataset_label = resolved_cfg.dataset_label

        #modelling attributes
        self.model_architecture = resolved_cfg.model_architecture
        self.training_strategy = resolved_cfg.training_strategy.lower()
        
        self.task_type = resolved_cfg.task_type
        self.training_kwargs = getattr(self.cfg, "training_kwargs", "")

    @property
    def subfolder(self) -> Path:
        if not self._is_built or self._subfolder is None:
            raise ValueError("Experiment subfolder not built. Call `.build()` first.")
        return self._subfolder

    @abstractmethod
    def build(self) -> None:
        """
        Main method responsible for building the experiment subfolder name based on the configuration.
        Subclasses must implement this method to define their specific logic for constructing the subfolder name.
        """
        raise NotImplementedError("Subclasses must implement the `build` method.")


    def _build_base_subfolder(self) -> List[str]:
        """
        Uses pipeline common metadata to construct a base subfolder
        Child classes extends this method to add additional information to the subfolder name
        Returns:
            List[str]: A list of strings representing the base subfolder components.
        """
        normalised_training_strategy_name = self._normalise_training_strategy_name(self.training_strategy)

        parts = [
            self.task_type.upper(),
            self.dataset_label,
            self.model_architecture,
            (f"{normalised_training_strategy_name}Strategy")
        ]
        return parts

    def _add_training_kwargs(self, subfolder: Path) -> Path:
        """
        Appends training kwargs to the subfolder name if they exist in the configuration.
        Args:
            subfolder (Path): The current subfolder path.
        Returns:
            Path: The updated subfolder path with training kwargs appended.
        """
        if self.training_kwargs:
            return subfolder / self.training_kwargs
        return subfolder

    def _normalise_training_strategy_name(self, strategy_name) -> str:
        if "_" in strategy_name:
            return strategy_name.title().replace("_", "")
    
        return strategy_name.title()


class ReinitStrategySubfolderMixin(StrategyParamsMixin):
    "Mixin class for handling Reinit experiment subfolder functionality."

    def _add_reinit_params(self, subfolder: Path) -> Path:
        """
        Appends reinitialisation strategy parameters to the subfolder name.
        Args:
            subfolder (Path): The current subfolder path.
        Returns:
            Path: The updated subfolder path with reinitialisation strategy parameters appended.
        """
        reinit_params = self._get_reinit_params()
        subfolder = subfolder / f"Reinit_{reinit_params.reinit_k_layers}_layers"

        if reinit_params.reinit_classifier:
            return subfolder / "with_classifier"
        return subfolder / "no_classifier"


class LLRDStrategySubfolderMixin(StrategyParamsMixin):
    "Mixin class for handling LLRD experiment subfolder functionality."
    def _add_llrd_params(self, subfolder: Path) -> Path:
        """
        Appends LLRD strategy parameters to the subfolder name.
        Args:
            subfolder (Path): The current subfolder path.
        Returns:
            Path: The updated subfolder path with LLRD strategy parameters appended.
        """
        llrd_params = self._get_llrd_params()
        return subfolder / f"LLRD_{llrd_params.llrd_factor}"



class ReinitLLRDStrategySubfolderMixin(ReinitStrategySubfolderMixin, LLRDStrategySubfolderMixin):
    "Mixin class for handling Reinit + LLRD experiment subfolder functionality."

    def _add_reinit_llrd_params(self, subfolder: Path) -> Path:
        """
        Appends combined reinitialisation and LLRD strategy parameters to the subfolder name.
        Args:
            subfolder (Path): The current subfolder path.
        Returns:
            Path: The updated subfolder path with combined strategy parameters appended.
        """
        reinit_llrd_params = self._get_reinit_llrd_params()
        subfolder = self._add_reinit_params(subfolder)
        subfolder = self._add_llrd_params(subfolder)
        return subfolder
