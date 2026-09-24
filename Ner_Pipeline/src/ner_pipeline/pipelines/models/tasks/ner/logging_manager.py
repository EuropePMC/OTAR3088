
from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path
from omegaconf import DictConfig

from ...shared.factory import BaseResolvedConfig
from ...shared.logging_manager_base import (LoguruHelper,
                                            WandbRunManager,
                                            ReinitStrategyLoguruMixin,
                                            LLRDStrategyLoguruMixin,
                                            ReinitLLRDStrategyLoguruMixin,
                                            ReinitStrategyWandbRunMixin,
                                            LLRDStrategyWandbRunMixin,
                                            ReinitLLRDStrategyWandbRunMixin)



@dataclass
class NERResolvedConfig(BaseResolvedConfig):
    ner_head_type: str = "standard"
    ner_trainer_type: str = "base"
    use_data_aug: bool = False
    ner_data_aug_method: str = None
    lr: float = None
    training_strategy: str = "base"
    training_kwargs: str = ""
    run_wandb_sweep: bool = False

    @classmethod
    def from_cfg(cls, cfg:DictConfig):
        """
        Extracts NER specific config parameters from main cfg 
        for easy retrieval in downstream pipeline use-cases
        """
        kwargs = cls._common_kwargs(cfg)
        return cls(**kwargs,
                    ner_head_type=getattr(cfg.task, "ner_head_type", "standard"),
                    ner_trainer_type=getattr(cfg.task, "trainer_type", "base"),
                    use_data_aug=getattr(cfg.task, "use_data_aug", False),
                    ner_data_aug_method=getattr(cfg.task, "data_aug_method", None),
                    lr=getattr(cfg, "lr", None),
                    training_strategy=getattr(cfg, "training_strategy", "base"),
                    training_kwargs=getattr(cfg, "training_kwargs", ""),
                    run_wandb_sweep=getattr(cfg.task, "run_wandb_sweep", False)
                        )

class NERLoguruHelper(LoguruHelper):
    """
    Loguru helper class for NER tasks. 
    Inherits from LoguruHelper.
    Implements task-specific functionality.

    """
    task_type = "ner"

    def __init__(self, resolved_cfg: NERResolvedConfig, base_dir: str = None, run_id = None):
        super().__init__(resolved_cfg, base_dir, run_id)
        # self.ner_head_type = getattr(cfg.task, "ner_head_type", "standard")
        # self.trainer_type = getattr(cfg.task, "trainer_type", "base")
        # self.use_data_aug = getattr(cfg.task, "use_data_aug", False)
        # self.data_aug_method = getattr(cfg.task, "data_aug_method", None)
        self.ner_head_type = resolved_cfg.ner_head_type
        self.ner_trainer_type = resolved_cfg.ner_trainer_type
        self.use_data_aug = resolved_cfg.use_data_aug
        self.ner_data_aug_method = resolved_cfg.ner_data_aug_method

    def _log_data_aug_method(self, base_log_dir: Path) -> Path:
        if self.use_data_aug and self.ner_data_aug_method:
            return base_log_dir / f"with_{self.ner_data_aug_method}"
        return base_log_dir / "no_data_aug"

    def _generate_filename_parts_and_log_dir(self) -> Tuple[List[str], Path]:
        filename_parts = self._build_base_filename_parts()
        base_log_dir_parts = self._build_base_log_dir_parts()
        ner_log_dir_parts = base_log_dir_parts + [
            (f"{self.ner_head_type.capitalize()}NerHead"),
            (f"{self.ner_trainer_type.capitalize()}Trainer")
        ]
        log_dir = Path(*ner_log_dir_parts)
        log_dir = self._log_data_aug_method(log_dir)
        log_dir = self._log_training_kwargs(log_dir)
        return filename_parts, log_dir


class NERBaseLoguruHelper(NERLoguruHelper):
    """
    Concrete implementation of NERLoguruHelper for base training strategy.
    Implements task-specific logging functionality for base training strategy.
    """
    def configure(self):
        if self._is_configured:
            return
        filename_parts, log_dir = self._generate_filename_parts_and_log_dir()
        self._log_filename, self._log_dir = self._setup_sink(filename_parts, log_dir)
        self._is_configured = True


class NERReinitLoguruHelper(ReinitStrategyLoguruMixin, NERLoguruHelper):
    """
    Concrete implementation of NERLoguruHelper for reinit training strategy.
    Implements task-specific logging functionality for reinit training strategy.
    """
    def configure(self):
        if self._is_configured:
            return
        filename_parts, log_dir = self._generate_filename_parts_and_log_dir()
        filename_parts, log_dir = self._add_reinit_params(filename_parts, log_dir)
        self._log_filename, self._log_dir = self._setup_sink(filename_parts, log_dir)
        self._is_configured = True


class NERLLRDLoguruHelper(LLRDStrategyLoguruMixin, NERLoguruHelper):
    """
    Concrete implementation of NERLoguruHelper for LLRD training strategy.
    Implements task-specific logging functionality for LLRD training strategy.
    """
    def configure(self):
        if self._is_configured:
            return
        filename_parts, log_dir = self._generate_filename_parts_and_log_dir()
        filename_parts, log_dir = self._add_llrd_params(filename_parts, log_dir)
        self._log_filename, self._log_dir = self._setup_sink(filename_parts, log_dir)
        self._is_configured = True


class NERReinitLLRDLoguruHelper(ReinitLLRDStrategyLoguruMixin, NERLoguruHelper):
    """
    Concrete implementation of NERLoguruHelper for Reinit + LLRD training strategy.
    Implements task-specific logging functionality for Reinit + LLRD training strategy.
    """
    def configure(self):
        if self._is_configured:
            return
        filename_parts, log_dir = self._generate_filename_parts_and_log_dir()
        filename_parts, log_dir = self._add_reinit_llrd_params(filename_parts, log_dir)
        self._log_filename, self._log_dir = self._setup_sink(filename_parts, log_dir)
        self._is_configured = True


class NERWandbRunManager(WandbRunManager):
    """
    Wandb run manager class for NER tasks. 
    Inherits from WandbRunManager.
    Implements task-specific functionality.

    """
    task_type = "ner"
    task_type = task_type.upper()

    def __init__(self, resolved_cfg: NERResolvedConfig, log_dir: Path):
        super().__init__(resolved_cfg, log_dir)
        # self.ner_head_type = getattr(cfg.task, "ner_head_type", "standard")
        # self.trainer_type = getattr(cfg.task, "trainer_type", "base")
        # self.use_data_aug = getattr(cfg.task, "use_data_aug", False)
        # self.data_aug_method = getattr(cfg.task, "data_aug_method", None)
        self.ner_head_type = resolved_cfg.ner_head_type
        self.ner_trainer_type = resolved_cfg.ner_trainer_type
        self.use_data_aug = resolved_cfg.use_data_aug
        self.ner_data_aug_method = resolved_cfg.ner_data_aug_method


    def _add_data_aug_tags(self, tags: List[str]) -> List[str]:
        if self.use_data_aug and self.ner_data_aug_method:
            tags.append(f"with_{self.ner_data_aug_method}")
        else:
            tags.append("no_data_aug")
        return tags

    def _build_run_tags(self) -> List[str]:
        base_tags = super()._build_run_tags()
        data_aug_tags = self._add_data_aug_tags(base_tags)
        ner_tags = [
            f"{self.ner_head_type.upper()}NerHead",
            f"{self.ner_trainer_type.capitalize()}Trainer"
        ]
        new_tags = data_aug_tags + ner_tags
        return new_tags



class NERBaseWandbRunManager(NERWandbRunManager):
    """
    Concrete implementation of NERWandbRunManager for base training strategy.
    """
    
    pass


class NERReinitWandbRunManager(ReinitStrategyWandbRunMixin, NERWandbRunManager):
    """
    Concrete implementation of NERWandbRunManager for reinit training strategy.
    Implements task-specific wandb run management functionality for reinit training strategy.
    """
    

    def _build_run_tags(self) -> List[str]:
        base_tags = super()._build_run_tags()
        reinit_tags = self._add_reinit_tags(base_tags)
        return reinit_tags

    def _generate_artifact_components(self):
        artifact_name, artifact_desc, artifact_metadata = super()._generate_artifact_components()
        reinit_params = self._get_reinit_params()

        artifact_name += f"_{reinit_params.reinit_k_layers}K"
        artifact_metadata["Num Reinit Layers"] = reinit_params.reinit_k_layers
        artifact_metadata["Reinit Classifier"] = reinit_params.reinit_classifier
        return artifact_name, artifact_desc, artifact_metadata


class NERLLRDWandbRunManager(LLRDStrategyWandbRunMixin, NERWandbRunManager):
    """
    Concrete implementation of NERWandbRunManager for LLRD training strategy.
    Implements task-specific wandb run management functionality for LLRD training strategy.
    """

    def _build_run_tags(self) -> List[str]:
        base_tags = super()._build_run_tags()
        llrd_tags = self._add_llrd_tags(base_tags)
        return llrd_tags
    
    def _generate_artifact_components(self):
        artifact_name, artifact_desc, artifact_metadata = super()._generate_artifact_components()
        
        llrd_params = self._get_llrd_params()
        llrd_factor = llrd_params.llrd_factor

        artifact_name += f"_llrd-{str(llrd_factor)}"
        artifact_metadata["LLRD Factor"] = str(llrd_factor)

        return artifact_name, artifact_desc, artifact_metadata


class NERReinitLLRDWandbRunManager(ReinitLLRDStrategyWandbRunMixin, NERWandbRunManager):
    """
    Concrete implementation of NERWandbRunManager for Reinit + LLRD training strategy.
    Implements task-specific wandb run management functionality for Reinit + LLRD training strategy.
    """

    def _build_run_tags(self) -> List[str]:
        base_tags = super()._build_run_tags()
        reinit_llrd_tags = self._add_reinit_llrd_tags(base_tags)
        return reinit_llrd_tags
    
    def _generate_artifact_components(self):
        artifact_name, artifact_desc, artifact_metadata = super()._generate_artifact_components()
        
        reinit_params = self._get_reinit_params()
        llrd_params = self._get_llrd_params()
        llrd_factor = llrd_params.llrd_factor

        artifact_name += f"_{reinit_params.reinit_k_layers}K"
        artifact_name += f"_llrd-{str(llrd_factor)}"

        artifact_metadata["Num Reinit Layers"] = reinit_params.reinit_k_layers
        artifact_metadata["Reinit Classifier"] = reinit_params.reinit_classifier
        artifact_metadata["LLRD Factor"] = str(llrd_factor)

        return artifact_name, artifact_desc, artifact_metadata