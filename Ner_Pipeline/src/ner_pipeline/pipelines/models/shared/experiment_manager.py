
from pathlib import Path
from omegaconf import DictConfig

from .trainer_config_base import TrainingStrategyName
from .factory import format_model_checkpoint_name





class BaseExperimentSubfolderBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        #data attributes
        self.data_name = cfg.task.data.name
        self.data_version = getattr(cfg.task.data, "version", "")

        #modelling attributes
        self.task_type = cfg.task_type
        self.model_architecture = format_model_checkpoint_name(cfg.task.model_name_or_path)
        self.training_strategy = cfg.training_strategy.lower()
        
        self.ner_head_type = getattr(cfg.task, "ner_head_type", "standard")
        self.trainer_type = getattr(cfg.task, "trainer_type", "base")

        self.use_data_aug = getattr(cfg.task, "use_data_aug", False)
        self.data_aug_method = getattr(cfg.task, "data_aug_method", None)
        self.training_kwargs = getattr(cfg, "training_kwargs", "")
        
        self._is_built = False

    @property
    def subfolder(self) -> Path:
        if not self._is_built:
            raise ValueError("Experiment subfolder not built. Call `.build()` first.")
        return self._subfolder

    def build(self) -> None:
        if self._is_built:
            return
        self._subfolder = self._build_base_subfolder()
        self._subfolder = self._add_data_aug_method(self._subfolder)
        self._subfolder = self._add_training_kwargs(self._subfolder)
        self._is_built = True

  
    def _build_base_subfolder(self) -> Path:
        parts = [
            self.task_type,
            (f"{self.data_name}_{self.data_version}"
            if self.data_version
            else self.data_name),
            self.model_architecture,
            (f"{self.training_strategy}Strategy"),
            (f"{self.ner_head_type.upper()}NerHead"),
            (f"{self.trainer_type.capitalize()}Trainer")
        ]
        return Path(*parts)


    def _add_training_kwargs(self, subfolder: Path):
        if self.training_kwargs:
            return subfolder / self.training_kwargs
        return subfolder


    def _add_data_aug_method(self, subfolder: Path) -> Path:
        if self.use_data_aug and self.data_aug_method:
            return subfolder / f"with_{self.data_aug_method}"
        return subfolder / "no_data_aug"



class ReinitExperimentSubfolderBuilder(BaseExperimentSubfolderBuilder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.reinit_classifier = getattr(cfg, "reinit_classifier", False)
        self.reinit_k_layers = cfg.reinit_k_layers

    def build(self):
        if self._is_built:
            return
        subfolder = self._build_base_subfolder()
        subfolder = self._add_data_aug_method(subfolder)
        subfolder = self._add_training_kwargs(subfolder)
        
        subfolder = self._add_reinit_classifier(subfolder)
        self._subfolder = self._add_k_layers(subfolder)
        self._is_built = True


    def _add_k_layers(self, subfolder:Path) -> Path:
        return subfolder / f"reinit_{self.reinit_k_layers}K"


    def _add_reinit_classifier(self, subfolder:Path) -> Path:
        if self.reinit_classifier:
            return subfolder / "with_classifier"
        return subfolder / "no_classifier"



class LLRDExperimentSubfolderBuilder(BaseExperimentSubfolderBuilder):
    def __init__(self, cfg:DictConfig):
        super().__init__(cfg)
        self._llrd_factor = getattr(cfg, "llrd_factor", 1.0)

    def build(self):
        if self._is_built:
            return
        subfolder = self._build_base_subfolder()
        subfolder = self._add_data_aug_method(subfolder)
        subfolder = self._add_training_kwargs(subfolder)
        
        self._subfolder = self._add_llrd_factor(subfolder)
        self._is_built = True

    def _add_llrd_factor(self, subfolder):
        return subfolder / f"llrd_{self._llrd_factor}"


class ReinitLLRDExperimentSubfolderBuilder(ReinitExperimentSubfolderBuilder, LLRDExperimentSubfolderBuilder):
    def __init__(self, cfg:DictConfig):
        super().__init__(cfg)

    def build(self):
        if self._is_built:
            return
        self._subfolder = self._build_base_subfolder()
        self._subfolder = self._add_data_aug_method(self._subfolder)
        self._subfolder = self._add_training_kwargs(self._subfolder)
        self._subfolder = self._add_reinit_classifier(self._subfolder)
        self._subfolder = self._add_k_layers(self._subfolder)
        self._subfolder = self._add_llrd_factor(self._subfolder)
        self._is_built = True

class GroupedLLRDExperimentSubfolderBuilder(BaseExperimentSubfolderBuilder):
    def __init__(self, cfg):
        pass
    
    def build(self):
        pass


class ExperimentSubfolderFactory:
    """
    Factory class responsible for instantiating the correct 
    Experiment Subfolder builder
    """
    _registry = {
        TrainingStrategyName.BASE: BaseExperimentSubfolderBuilder,
        TrainingStrategyName.REINIT: ReinitExperimentSubfolderBuilder,
        TrainingStrategyName.LLRD: LLRDExperimentSubfolderBuilder,
        TrainingStrategyName.REINIT_LLRD: ReinitLLRDExperimentSubfolderBuilder,
        TrainingStrategyName.GROUPED_LLRD: GroupedLLRDExperimentSubfolderBuilder,
    }
    @classmethod
    def create(cls, cfg:DictConfig):
        training_strategy = TrainingStrategyName(cfg.training_strategy.lower())
        return cls._registry[training_strategy](cfg)