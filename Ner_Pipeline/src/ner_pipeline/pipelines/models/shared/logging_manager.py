
from typing import List
from pathlib import Path
from abc import ABC, abstractmethod
from omegaconf import DictConfig, OmegaConf

from loguru import logger
import wandb
from wandb.sdk.wandb_run import Run as WandbRun
from wandb import Artifact as WandbArtifact

from .trainer_config_base import TrainingStrategyName
from .factory import format_model_checkpoint_name



class BaseLoguruHelper:
    """
    """

    def __init__(self, cfg:DictConfig, base_dir:str=None, run_id=None):
        self.cfg = cfg
        self.lr = cfg.lr
        self.data_name = cfg.task.data.name
        self.data_version = getattr(cfg.task.data, "version", "")

        #modelling
        self.model_architecture = format_model_checkpoint_name(cfg.task.model_name_or_path)
        self.task_type = cfg.task_type
        self.training_strategy = cfg.training_strategy
        #self.weighted_trainer = getattr(cfg.task, "use_weighted_trainer", False)

        self.ner_head_type = getattr(cfg.task, "ner_head_type", "standard")
        self.trainer_type = getattr(cfg.task, "trainer_type", "base")

        self.use_data_aug = getattr(cfg.task, "use_data_aug", False)
        self.data_aug_method = getattr(cfg.task, "data_aug_method", None)

        self.training_kwargs = getattr(cfg, "training_kwargs", "")

        #only used if running wandb sweep
        self.run_id = run_id

        #where to save logs to. Defaults to path.cwd if not defined
        self.base_dir = base_dir

        self._is_configured = False

    
    @property
    def log_dir(self):
        if not self._is_configured:
            raise ValueError("Logging params are not yet configured."
                            "Call `.configure()` before accessing class attributes")
        return self._log_dir

    @property
    def log_filename(self):
        if not self._is_configured:
            raise ValueError("Logging params are not yet configured."
                            "Call `.configure()` before accessing class attributes")
        return self._log_filename

    def configure(self) -> None:
        """
        Configures Loguru logging and updates the cfg in-place with:
        - logging directory and filename

        This method mutates self.cfg and does not return anything.
        """
        if self._is_configured:
            return
        
        self._log_filename, self._log_dir = self._generate_log_filename_and_log_dir()
        
        self._setup_sink(self._log_filename, self._log_dir)
        self._is_configured = True
        
        

    def _setup_sink(self, log_filename:List, log_dir:Path) -> None:
        print(f"Logging dir set to: {log_dir}")

        if self.run_id:
            log_filename.append(f"run={self.run_id}")

        log_filename = "_".join(str(x) for x in log_filename)
        log_path = log_dir / f"{log_filename}.log"
        log_dir.mkdir(parents=True, exist_ok=True)
        

        logger.remove()
        logger.add(log_path,
            format="[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] | <level>{level}</level> | <cyan>{message}</cyan>",
            mode="w", 
            level="DEBUG",
            retention="4 months"
            )
        logger.success(f"Loguru initialised at: {log_path}")

    def _generate_log_filename_and_log_dir(self):
        log_dir = self._build_log_dir()
        log_dir = self._log_data_aug_method(log_dir)
        log_dir = self._log_training_kwargs(log_dir)
        log_filename = self._build_log_filename()
        return log_filename, log_dir


    def _build_log_filename(self) -> List:
        parts = [
            (f"{self.data_name}_{self.data_version}"
            if self.data_version
            else self.data_name),
            f"lr={self.lr}",

        ]
        
        return parts


    def _log_data_aug_method(self, base_log_dir: Path) -> Path:
        if self.use_data_aug and self.data_aug_method:
            return base_log_dir / f"with_{self.data_aug_method}"
        return base_log_dir / "no_data_aug"


    def _log_training_kwargs(self, base_log_dir: Path) -> Path:
        if self.training_kwargs:
            return base_log_dir / self.training_kwargs
        return base_log_dir


    def _build_log_dir(self) -> Path:
        parts = [
            (self.base_dir if self.base_dir
             else ""),
            "logs",
            "loguru_logs",
            self.task_type,
            (f"{self.data_name}_{self.data_version}"
            if self.data_version
            else self.data_name),
            self.model_architecture,
            (f"{self.training_strategy.title()}_Strategy"),
            (f"{self.ner_head_type.capitalize()}NerHead"),
            (f"{self.trainer_type.capitalize()}Trainer")
        ]
    
        return Path(*parts)
    
    @classmethod
    def __repr__(cls) -> str:
        return f"{cls.__name__}()"


class ReinitLoguruHelper(BaseLoguruHelper):
    def __init__(self, cfg:DictConfig, base_dir:str=None, run_id=None):
        super().__init__(cfg, base_dir, run_id)
        self._reinit_k_layers = cfg.reinit_k_layers
        self._reinit_classifier = getattr(cfg, "reinit_classifier", False)

    def configure(self):
        """
        Builds on BaseHelper and adds reinitialisation of
        classifier flags to log_filename
        """
        if self._is_configured:
            return
        self._log_filename, self._log_dir = self._generate_log_filename_and_log_dir()
        self._setup_sink(self._log_filename, self._log_dir)
        self._is_configured = True

    def _generate_log_filename_and_log_dir(self):
        log_filename, log_dir = super()._generate_log_filename_and_log_dir()
        
        log_filename = self._log_k_layers(log_filename)
        log_dir = self._log_reinit_classifier(log_dir) 
        return log_filename, log_dir    

    def _log_k_layers(self, filename_parts:List) -> List:
        return filename_parts + [f"{self._reinit_k_layers}K"]

    def _log_reinit_classifier(self, log_dir:Path) -> Path:
        if self._reinit_classifier:
          return log_dir / "with_classifier"
        return log_dir / "no_classifier"



class LLRDLoguruHelper(BaseLoguruHelper):
    def __init__(self, cfg:DictConfig, base_dir:str=None, run_id=None):
        super().__init__(cfg, base_dir, run_id)
        self._llrd_factor = cfg.llrd_factor

    def configure(self):
        """
        Builds on BaseHelper on adds LLRD flags to log_filename
        """
        if self._is_configured:
            return
        
        self._log_filename, self._log_dir = self._generate_log_filename_and_log_dir()
        self._setup_sink(self._log_filename, self._log_dir)
        self._is_configured = True
        
    def _generate_log_filename_and_log_dir(self):
        log_filename, log_dir = super()._generate_log_filename_and_log_dir()
        log_filename = self._log_llrd_factor(log_filename)
        return log_filename, log_dir

    def _log_llrd_factor(self, filename_parts:List) -> List:
        "Appends LLRD Factor from configure to log_filename"
        return filename_parts + [f"llrd={self._llrd_factor}"]


class ReinitLLRDLoguruHelper(ReinitLoguruHelper, LLRDLoguruHelper):
    def __init__(self, cfg:DictConfig, base_dir:str=None, run_id=None):
        super().__init__(cfg, base_dir, run_id)

    def configure(self) -> None:
        """
        Inherits from both `ReinitLoguruHelper & LLRDLoguruHelper`,
        adds `reinit_classifier` and `llrd_factor flags to
        `log_dir` and/or `log_filename`

        """
        if self._is_configured:
            return
        self._log_filename = self._log_k_layers(self._build_log_filename())
        self._log_filename = self._log_llrd_factor(self._log_filename)
        self._log_dir = self._log_reinit_classifier(self._build_log_dir())
        self._setup_sink(self._log_filename, self._log_dir)
        self._is_configured = True


class GroupedLLRDLoguruHelper(BaseLoguruHelper):
    def __init__(self, cfg:DictConfig, base_dir:str=None, run_id=None):
        super().__init__(cfg, base_dir, run_id)
        pass


class LoguruHelperFactory:
    "Factory class responsible for instantiating the correct logging helper"
    _registry = {
        TrainingStrategyName.BASE: BaseLoguruHelper,
        TrainingStrategyName.REINIT: ReinitLoguruHelper,
        TrainingStrategyName.LLRD: LLRDLoguruHelper,
        TrainingStrategyName.REINIT_LLRD: ReinitLLRDLoguruHelper,
        TrainingStrategyName.GROUPED_LLRD: GroupedLLRDLoguruHelper,
    }
    @classmethod
    def create(cls, cfg:DictConfig, base_dir=None, run_id=None):
        strategy = TrainingStrategyName(cfg.training_strategy.lower())
        helper = cls._registry[strategy](cfg, base_dir, run_id)
        print(f"Logging helper set to : {helper}")
        return helper
  


class BaseWandbRunManager:
    """
    Utility manager for orchestrating Weights & Biases  
    run configuration and setup for experiments.
    """
    def __init__(self, cfg, log_dir):
        """
        Initialise the run manager with experiment configuration.

        Args:
            cfg: Hydra configuration object containing experiment parameters.
            log_dir: Local directory path for wandb logs.
        """
        self.cfg = cfg
        self.log_dir = log_dir
        self.job_type = cfg.logging.wandb.run.job_type

        # Resolve OmegaConf to a plain Python dict for wandb config logging
        self.plain_cfg = OmegaConf.to_container(cfg, resolve=True)
         
        # Data identifiers
        self.data_name = cfg.task.data.name
        self.data_version = getattr(cfg.task.data, "version", "")
        self.dataset_label = f"{self.data_name}_{self.data_version}" if self.data_version else self.data_name

        # Extract training and task-specific attributes
        self.task_type = cfg.task_type
        self.training_strategy = cfg.training_strategy.lower()
        self.training_kwargs = getattr(cfg, "training_kwargs", "")
        
        # Modelling specific attributes
        self.model_architecture = format_model_checkpoint_name(cfg.task.model_name_or_path)
        self.ner_head_type = getattr(cfg.task, "ner_head_type", "standard")
        self.trainer_type = getattr(cfg.task, "trainer_type", "base")

        self.use_data_aug = getattr(cfg.task, "use_data_aug", False)
        self.data_aug_method = getattr(cfg.task, "data_aug_method", None)

        self._is_initialised = False


    @property
    def run_tags(self):
        if not self._is_initialised:
            raise ValueError("Wandb Run not initialised yet."
                            "Call `.setup_run() first")
        return self._build_run_tags()
    
    def build_run_payload(self):
        return {
            "name": self._generate_run_name(),
            "tags": self._build_run_tags(),
            "config": self.plain_cfg,
        }

    def attach_to_existing_run(self, run):
        payload = self.build_run_payload()
        run.name = payload["name"]
        print("Payload tags", payload["tags"])

        existing_tags = tuple(run.tags or ())
        new_tags = tuple(t for t in payload["tags"] if t not in existing_tags)
        if new_tags:
            run.tags += new_tags

        run.config.update(payload["config"], allow_val_change=True)
        self._is_initialised = True
        return run

    def setup_run(self):
        "Initialise a new wandb run with a generated name, tags, and configuration."
        
        run_identifier = self._generate_run_name()

        run_tags = self._build_run_tags()

        run = wandb.init(
            name=run_identifier,
            reinit=True,
            config=self.plain_cfg,
            tags=run_tags,
            dir=self.log_dir,
            **self.cfg.logging.wandb.run
        )
        self._is_initialised = True

        return run

    def create_artifact(self):
        "creates a run aritifact instance"
        
        artifact_name, artifact_desc, artifact_metadata = self._generate_artifact_components()
        
        artifact = wandb.Artifact(
                    name=artifact_name,
                    description=artifact_desc,
                    type="model",
                    metadata=artifact_metadata
                )
        return artifact


    def _generate_run_name(self):
        "Generates a unique run identifier name for the UI"
        parts = [
                self.dataset_label,
                self.task_type,
                self.training_strategy,
                f"{wandb.util.generate_id()}"
                    ]
        if self.training_kwargs:
            parts.append(self.training_kwargs)
        run_name = "_".join(parts)
        
        return run_name


    def _generate_artifact_components(self):
        """
        Generates the necessary components for creating a wandb artifact.

        Returns:
            A tuple (artifact_name, artifact_description, artifact_metadata).
        """

        parts = [
                self.dataset_label,
                self.training_strategy,
                self.training_kwargs
                    ]
        artifact_name = "_".join(parts)
        artifact_desc = f"{self.task_type.upper()} {self.job_type} using \
                        {self.dataset_label} and \
                        strategy: {self.training_strategy}"

        artifact_metadata = {
                            "Model architecture": self.model_architecture,
                            "Dataset": self.dataset_label,
                            "Task": self.task_type,
                            "Mode": self.job_type,
                            "Strategy" : self.training_strategy
                        }

        return artifact_name, artifact_desc, artifact_metadata


    def _build_run_tags(self):
        """
        Builds a list of tags for the wandb run.

        Tags include job type, dataset, task, strategy, model name, and
        optional data augmentation details.

        Returns:
            A list of tag strings.
        """

        tags = [
            self.job_type,
            self.dataset_label,
            self.task_type,
            (f"{self.training_strategy.title()}_Strategy"),
            self.model_architecture,
            str(self.cfg.lr),
            (f"{self.ner_head_type.capitalize()}NerHead"),
            (f"{self.trainer_type.capitalize()}Trainer")
        ]

        
        if self.training_kwargs:
            tags.append(self.training_kwargs)

        if self.use_data_aug:
            tags.append(self.data_aug_method)
        else:
            tags.append("no_data_aug")

        return tags


class ReinitWandbRunManager(BaseWandbRunManager):
    def __init__(self, cfg, log_dir):
        super().__init__(cfg, log_dir)
        self.reinit_classifier = getattr(cfg, "reinit_classifier", False)
        self.k_layers = getattr(cfg, "reinit_k_layers", 0)
    
    def _build_run_tags(self):
        base_tags = super()._build_run_tags()
        additional_tags = [
            f"reinit_classifier-{self.reinit_classifier}",
            f"reinit_k_layers-{str(self.k_layers)}",

        ]
        new_tags = base_tags + additional_tags

        return new_tags

    def _generate_artifact_components(self):
        artifact_name, artifact_desc, artifact_metadata = super()._generate_artifact_components()
        artifact_name = f"{artifact_name}_K-Layers_{str(self.k_layers)}"
        artifact_metadata["K_layers"] = str(self.k_layers)
        artifact_metadata["Reinit_Classifier"] = self.reinit_classifier

        return artifact_name, artifact_desc, artifact_metadata
    


class LLRDWandbRunManager(BaseWandbRunManager):
    def __init__(self, cfg, log_dir):
        super().__init__(cfg, log_dir)
        self._llrd_factor = getattr(cfg, "llrd_factor", 1.0)

    def _build_run_tags(self):
        tags = super()._build_run_tags()
        tags.append(str(self._llrd_factor))
        return tags

    def _generate_artifact_components(self):
        artifact_name, artifact_desc, artifact_metadata = super()._generate_artifact_components()
        artifact_name = f"{artifact_name}_LLRD_{str(self._llrd_factor)}"
        artifact_metadata["LLRD_Factor"] = str(self._llrd_factor)

        return artifact_name, artifact_desc, artifact_metadata



class ReinitLLRDWandbRunManager(ReinitWandbRunManager, LLRDWandbRunManager):
    def __init__(self, cfg, log_dir):
        super().__init__(cfg, log_dir)
        pass


class WandbRunManagerFactory:
    "Factory class responsible for instantiating the correct Wandb Manager"
    _registry = {
        TrainingStrategyName.BASE: BaseWandbRunManager,
        TrainingStrategyName.REINIT: ReinitWandbRunManager,
        TrainingStrategyName.LLRD: LLRDWandbRunManager,
        TrainingStrategyName.REINIT_LLRD: ReinitLLRDWandbRunManager,
    }


    @classmethod
    def create(cls, cfg:DictConfig, log_dir:Path):
        strategy = TrainingStrategyName(cfg.training_strategy.lower())
        manager = cls._registry[strategy](cfg, log_dir)
        logger.info(f"Wandb Manager set to : {manager}")
        return manager