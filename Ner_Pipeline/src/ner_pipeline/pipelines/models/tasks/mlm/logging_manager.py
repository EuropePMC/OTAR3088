
from typing import List, Tuple
from pathlib import Path
from omegaconf import DictConfig

from .tokenization_utils import MLMMaskingMixin
from ...shared.logging_manager_base import LoguruHelper, WandbRunManager
from ...shared.factory import format_model_checkpoint_name, BaseResolvedConfig



@dataclass
class MLMResolvedConfig(BaseResolvedConfig):
    mlm_task_type: str = "tapt"
    tokenizer_path: str = None
    mlm_trainer_type: str = "base"
    run_wandb_sweep: bool = False
    lr: float = None

    @classmethod
    def from_cfg(cls, cfg: DictConfig):
        """
        Extracts MLM specific config parameters from main cfg
        for easy retrieval in downstream pipeline use-cases
        """
        kwargs = cls._common_kwargs(cfg)
        return cls(**kwargs,
                    mlm_task_type=getattr(cfg.task, "mlm_task_type", "tapt"),
                    mlm_trainer_type=getattr(cfg.task, "trainer_type", "base"),
                    tokenizer_path=getattr(cfg.task, "tokenizer_name_or_path", None),
                    run_wandb_sweep=getattr(cfg.task, "run_wandb_sweep", False),
                    lr=getattr(cfg, "lr", None),
                        )


class MLMLoguruHelper(MLMMaskingMixin, LoguruHelper):
    """
    Loguru helper class for Masked Language Modeling tasks. 
    Inherits from LoguruHelper.
    Implements task-specific functionality.

    """
    task_type = "mlm"
    task_type = task_type.upper()

    def __init__(self, resolved_cfg: MLMResolvedConfig, base_dir:str=None, run_id=None):
        super().__init__(resolved_cfg, base_dir, run_id)
        self.resolved_cfg = resolved_cfg
        self.mlm_task_type = resolved_cfg.mlm_task_type
        self.mlm_trainer_type = resolved_cfg.mlm_trainer_type
        self.tokenizer_path = resolved_cfg.tokenizer_path
        

    def _generate_filename_parts_and_log_dir(self) -> Tuple[List[str], Path]:
        filename_parts = self._build_base_filename_parts()
        mlm_log_dir_parts = self._build_log_dir_parts()

        log_dir = Path(*mlm_log_dir_parts)
        log_dir = self._log_training_kwargs(log_dir)
        return filename_parts, log_dir

    def _build_log_dir_parts(self) -> List[str]:
        normalised_training_strategy = self._normalise_training_strategy_name(self.training_strategy)
        mlm_masking_strategy = self._add_masking_method(self.task_cfg)

        parts = [
            (self.base_dir if self.base_dir
             else ""),
            "logs",
            "loguru_logs",
            self.task_type,
            self.mlm_task_type.upper(),
            mlm_masking_strategy,
            ("wandb_sweep_run" if self.is_wandb_sweep else ""),
            self.resolved_cfg.dataset_label,
            self.resolved_cfg.model_architecture,
            ("Adapted_Tokenizer" if self.tokenizer_path else ""),
            (f"{normalised_training_strategy}Strategy"),
            (f"{self.mlm_trainer_type.capitalize()}Trainer")
        ]
    
        return parts


class MLMBaseLoguruHelper(MLMLoguruHelper):
    """
    Concrete implementation of MLMLoguruHelper for base training strategy.
    Implements task-specific logging functionality for base training strategy.
    """
    def configure(self):
        if self._is_configured:
            return
        filename_parts, log_dir = self._generate_filename_parts_and_log_dir()
        self._log_filename, self._log_dir = self._setup_sink(filename_parts, log_dir)
        self._is_configured = True


class MLMWandbRunManager(MLMMaskingMixin, WandbRunManager):
    """
    Wandb run manager class for MLM tasks. 
    Inherits from WandbRunManager.
    Implements task-specific functionality.

    """

    task_type = "mlm"
    task_type = task_type.upper()

    def __init__(self, cfg: DictConfig, log_dir: Path):
        super().__init__(cfg, log_dir)
        self.resolved_cfg = resolved_cfg
        self.mlm_task_type = resolved_cfg.mlm_task_type
        self.mlm_trainer_type = resolved_cfg.mlm_trainer_type
        self.tokenizer_path = resolved_cfg.tokenizer_path
        

    def _build_run_tags(self) -> List[str]:
        base_tags = super()._build_run_tags()
        mlm_masking_strategy = self._add_masking_method(self.task_cfg)
        mlm_tags = [
            self.mlm_task_type.upper(),
            f"{self.mlm_trainer_type.capitalize()}Trainer",
            mlm_masking_strategy
        ]

        if self.tokenizer_path:
            mlm_tags.extend(["Adapted_Tokenizer", 
                            format_model_checkpoint_name(self.tokenizer_path)]
                            )
        new_tags = base_tags + mlm_tags
        return new_tags


class MLMBaseWandbRunManager(MLMWandbRunManager):
    """
    Concrete implementation of MLMWandbRunManager for base training strategy.
    """
    
    pass