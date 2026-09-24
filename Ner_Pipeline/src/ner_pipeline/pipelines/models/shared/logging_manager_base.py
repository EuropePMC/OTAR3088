
from typing import List, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from omegaconf import DictConfig, OmegaConf

from loguru import logger
import wandb
from wandb.sdk.wandb_run import Run as WandbRun
from wandb import Artifact as WandbArtifact

from .factory import (BaseResolvedConfig, 
                      TrainingStrategyName,
                      ReinitStrategyParams,
                      LLRDStrategyParams,
                      ReinitLLRDStrategyParams,
                      StrategyParamsMixin,)




class ReinitStrategyLoguruMixin(StrategyParamsMixin):
    """
    Mixin class for logging Reinit training strategy parameters to log filename and directory.
    """
    def _add_reinit_params(self, filename_parts:List, log_dir:Path) -> (List, Path):
        """
        Appends Reinit strategy parameters to log filename and directory.

        Args:
            filename_parts (List): List of parts for the log filename.
            log_dir (Path): Base log directory path.
        
        Returns:
            Tuple[List, Path]: Updated filename parts and log directory path.
        """
        reinit_params = self._get_reinit_params()
        filename_parts.append(f"{reinit_params.reinit_k_layers}K")
        log_dir = log_dir / ("with_classifier" if reinit_params.reinit_classifier else "no_classifier")
        return filename_parts, log_dir


class LLRDStrategyLoguruMixin(StrategyParamsMixin):
    """
    Mixin class for logging LLRD training strategy parameters to log filename and directory.
    """
    def _add_llrd_params(self, filename_parts:List) -> List:
        """
        Appends LLRD strategy parameters to log filename and directory.

        Args:
            filename_parts (List): List of parts for the log filename.
            
        Returns:
            List: Updated filename parts.
        """
        llrd_params = self._get_llrd_params()
        filename_parts.append(f"llrd={llrd_params.llrd_factor}")
        return filename_parts


class ReinitLLRDStrategyLoguruMixin(ReinitStrategyLoguruMixin, LLRDStrategyLoguruMixin):
    """
    Mixin class for logging Reinit + LLRD training strategy parameters to log filename and directory.
    """
    def _add_reinit_llrd_params(self, filename_parts:List, log_dir:Path) -> (List, Path):
        """
        Appends Reinit + LLRD strategy parameters to log filename and directory.

        Args:
            filename_parts (List): List of parts for the log filename.
            log_dir (Path): Base log directory path.
        Returns:
            Tuple[List, Path]: Updated filename parts and log directory path.
        """
        filename_parts, log_dir = self._add_reinit_params(
            filename_parts,
            log_dir,
        )
        filename_parts = self._add_llrd_params(filename_parts)
        return filename_parts, log_dir


class LoguruHelper(ABC):
    """
    Abstract base class for Loguru logging helpers.
    Provides a common interface for configuring logging and generating log filenames and directories.
    """
    def __init__(self, resolved_cfg:BaseResolvedConfig, base_dir: str = None, run_id = None):
        self.cfg = resolved_cfg.cfg
        self.resolved_cfg = resolved_cfg
        self.lr = resolved_cfg.lr
        self.dataset_label = resolved_cfg.dataset_label
       
        self.model_architecture = resolved_cfg.model_architecture
        self.training_strategy = resolved_cfg.training_strategy
        self.training_kwargs = resolved_cfg.training_kwargs
        
        #only used if running wandb sweep
        self.run_id = run_id
        self.is_wandb_sweep = getattr(self.cfg.task, "run_wandb_sweep", False)

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
    

    @abstractmethod
    def configure(self) -> None:
        """
        Configures Loguru logging and updates the cfg in-place with:
        - logging directory and filename

        """
        raise NotImplementedError("Subclasses must implement the `configure` method to set up logging.")

    
    def _setup_sink(self, log_filename_parts:List, log_dir:Path) -> Tuple[str, Path]:
        print(f"Logging dir set to: {log_dir}")

        if self.run_id:
            log_filename_parts.append(f"run={self.run_id}")

        log_filename = "_".join(str(x) for x in log_filename_parts)
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

        return log_filename, log_dir
    
    def _build_base_filename_parts(self) -> List:
        parts = [
            self.dataset_label,
            f"lr={self.lr}",

        ]
        
        return parts
    

    def _log_training_kwargs(self, base_log_dir: Path) -> Path:
        if self.training_kwargs:
            return base_log_dir / self.training_kwargs
        return base_log_dir

    def _normalise_training_strategy_name(self, strategy_name) -> str:
        if "_" in strategy_name:
            return strategy_name.title().replace("_", "")
    
        return strategy_name.title()

    def _build_base_log_dir_parts(self) -> List[str]:
        normalised_training_strategy = self._normalise_training_strategy_name(self.training_strategy)
        parts = [
            (self.base_dir if self.base_dir
             else ""),
            "logs",
            "loguru_logs",
            self.task_type,
            (self.mlm_task_type.upper() if self.task_type.lower()=="mlm" else ""),
            ("wandb_sweep_run" if self.is_wandb_sweep else ""),
            self.dataset_label,
            self.model_architecture,
            (f"{normalised_training_strategy}Strategy"),
        ]
    
        return parts
    
    @classmethod
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"



class ReinitStrategyWandbRunMixin(StrategyParamsMixin):
    """
    Mixin class for managing Weights & Biases (wandb) runs with Reinit training strategy parameters.
    """
    def _add_reinit_tags(self, tags: List[str]) -> List[str]:
        """
        Appends Reinit strategy parameters to the 
        list of wandb run tags.
        """
        reinit_params = self._get_reinit_params()
        tags.append(f"{reinit_params.reinit_k_layers}K")
        if reinit_params.reinit_classifier:
            tags.append("with_classifier")
        else:
            tags.append("no_classifier")
        return tags


class LLRDStrategyWandbRunMixin(StrategyParamsMixin):
    """
    Mixin class for managing Weights & Biases (wandb) runs with LLRD training strategy parameters.
    """
    def _add_llrd_tags(self, tags: List[str]) -> List[str]:
        """
        Appends LLRD strategy parameters to the 
        list of wandb run tags.
        """
        llrd_params = self._get_llrd_params()
        tags.append(f"llrd-{str(llrd_params.llrd_factor)}")
        return tags


class ReinitLLRDStrategyWandbRunMixin(ReinitStrategyWandbRunMixin, LLRDStrategyWandbRunMixin):
    """
    Mixin class for managing Weights & Biases (wandb) runs with Reinit + LLRD training strategy parameters.
    """
    def _add_reinit_llrd_tags(self, tags: List[str]) -> List[str]:
        """
        Appends Reinit + LLRD strategy parameters to the 
        list of wandb run tags.
        """
        tags = self._add_reinit_tags(tags)
        tags = self._add_llrd_tags(tags)

        return tags




class WandbRunManager(ABC):
    """
    Abstract base class for managing Weights & Biases (wandb) runs.
    Provides a common interface for configuring wandb runs and logging experiment parameters.
    """
    def __init__(self, resolved_cfg: BaseResolvedConfig, log_dir:str):
        self.resolved_cfg = resolved_cfg
        self.cfg = resolved_cfg.cfg
        self.log_dir = log_dir
        self.job_type = self.cfg.logging.wandb.run.job_type

        # Resolve OmegaConf to a plain Python dict for wandb config logging
        self.plain_cfg = OmegaConf.to_container(self.cfg, resolve=True)
         
        # Data identifiers
        self.dataset_label = resolved_cfg.dataset_label

        # Extract training and task-specific attributes
        self.task_type = resolved_cfg.task_type
        self.training_strategy = resolved_cfg.training_strategy.lower()
        self.training_kwargs = resolved_cfg.training_kwargs
        
        # Modelling specific attributes
        self.model_architecture = resolved_cfg.model_architecture

        self._is_initialised = False


    @property
    def run_tags(self):
        if not self._is_initialised:
            raise ValueError("Wandb Run not initialised yet."
                            "Call `.setup_run() first")
        return self._build_run_tags()
    
    
    def build_run_payload(self):
        """
        Constructs a payload dictionary containing 
        the run name, tags, and configuration for wandb logging.
        Returns:
            dict: A dictionary containing the run name, tags, 
                and configuration.
        """
        return {
            "name": self._generate_run_name(),
            "tags": self._build_run_tags(),
            "config": self.plain_cfg,
        }

    
    
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

    
    def _build_run_tags(self) -> List[str]:
        tags = [
            self.task_type,
            self.training_strategy,
            self.dataset_label,
            (self.model_architecture[:64] if len(self.model_architecture) > 64 
            else self.model_architecture),
            str(self.resolved_cfg.lr),
        ]
        if self.training_kwargs:
            tags.append(self.training_kwargs)
        return tags

    
    def attach_to_existing_run(self, run):
        """
        Attaches to an existing wandb run and updates its name,
        tags, and configuration using a payload.
        """
        if not self._is_initialised:
            self._is_initialised = True

        payload = self.build_run_payload()
        if hasattr(run, "name"):
            run.name = payload["name"]

        existing_tags = tuple(getattr(run, "tags", ()) or ())
        new_tags = tuple(t for t in payload["tags"] if t not in existing_tags)
        if new_tags:
            if getattr(run, "tags", None) is None:
                run.tags = new_tags
            else:
                run.tags += new_tags

        if hasattr(run, "config"):
            run.config.update(payload["config"], allow_val_change=True)
            
        return run

    

    def _generate_artifact_components(self):
        """
        Generates the necessary components for creating a wandb artifact.

        Returns:
            A tuple (artifact_name, artifact_description, artifact_metadata).
        """

        artifact_name_parts = [
                self.dataset_label,
                self.training_strategy,
                self.training_kwargs
                    ]
        artifact_name = "_".join(artifact_name_parts)
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
            **self.resolved_cfg.cfg.logging.wandb.run
        )
        self._is_initialised = True

        return run
