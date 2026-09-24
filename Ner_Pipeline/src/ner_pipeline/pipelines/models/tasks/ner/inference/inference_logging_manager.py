
from typing import List, Tuple
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import wandb
from wandb.sdk.wandb_run import Run as WandbRun
from wandb import Artifact as WandbArtifact

from .inference_factory import NERInferenceResolvedConfig
from ....shared.factory import format_model_checkpoint_name, BaseResolvedConfig





class NERInferenceLoguruManager:
    def __init__(self, resolved_cfg: NERInferenceResolvedConfig, base_dir: str=None):
        self.resolved_cfg = resolved_cfg
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
    
    def configure(self):
        if self._is_configured:
            return
        filename_parts = self._build_filename_parts()
        log_dir = self._build_log_dir_parts()
        self._log_filename, self._log_dir = self._setup_sink(filename_parts, log_dir)
        self._is_configured = True


    def _setup_sink(self, filename_parts: List, log_dir:Path):

        log_filename = "_".join(str(x) for x in filename_parts)
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


    def _build_filename_parts(self) -> List[str]:
        
        parts = [
            self.resolved_cfg.dataset_label,
            self.resolved_cfg.task_type,
        ]
        return parts

    def _build_log_dir_parts(self) -> List[str]:
        parts = [
            (self.base_dir if self.base_dir
             else ""),
            "logs",
            "loguru_logs",
            "ner"
            "inference",
            self.resolved_cfg.dataset_label,
            self.resolved_cfg.model_architecture,
        ]
        return Path(*parts)


    @classmethod
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"




class NERInferenceWandbManager:
    def __init__(self, resolved_cfg: NERInferenceResolvedConfig, log_dir: Path = None):
        self.resolved_cfg = resolved_cfg
        self.cfg = resolved_cfg.cfg
        self.log_dir = log_dir
        self.job_type = "Inference"
        self.plan_cfg = OmegaConf.to_container(self.cfg, resolve=True)

        self._is_initialised = False


    def _generate_run_name(self):
        parts = [
            self.resolved_cfg.dataset_label, 
            self.job_type,
        ]

        run_name = "_".join(parts)
        return run_name

    def _build_run_tags(self):
        tags = [
            self.resolved_cfg.dataset_label,
            self.job_type,
            self.resolved_cfg.model_architecture,
        ]

        return tags



    def setup_run(self):
        if self._is_initialised:
            logger.warning("Wandb run is already initialised. Skipping.")
            return self.run

        run_identifier = self._generate_run_name()
        run_tags = self._build_run_tags()

        run = wandb.init(
                    name=run_identifier,
                    reinit=True,
                    config=self.plan_cfg,
                    tags=run_tags,
                    dir=self.log_dir,
                    project=self.cfg.logging.wandb.run.project,
                    entity=self.cfg.logging.wandb.run.entity,
                    job_type=self.job_type
                )

        self._is_initialised = True
        self.run = run

        return run



class NERInferenceSubfolderBuilder:
    def __init__(self, resolved_cfg: NERInferenceResolvedConfig):
        self.cfg = resolved_cfg.cfg
        self.resolved_cfg = resolved_cfg
        self._is_built = False
        self._subfolder = None
    
    @property
    def subfolder(self) -> Path:
        if not self._is_built or self._subfolder is None:
            raise ValueError("Experiment subfolder not built. Call `.build()` first.")
        return self._subfolder
    
    def build(self) -> Path:
        if self._is_built:
            return self.subfolder

        self._subfolder = self._build_subfolder()
        self._is_built = True
        return self.subfolder
    
    def _build_subfolder(self) -> Path:

        parts = [
            "NER",
            "Inference",
            self.resolved_cfg.dataset_label,
            self.resolved_cfg.model_architecture
        ]

        return Path(*parts)

    