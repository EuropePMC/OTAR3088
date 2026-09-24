
import os
from pathlib import Path

from typing import Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

import hydra
import wandb
from loguru import logger

from omegaconf import DictConfig, OmegaConf

import torch

from ner_pipeline.utils.common import create_output_dir, set_seed
from ner_pipeline.pipelines.models.shared.trainer_config_base import (BuildContext, 
                                                                      PushToHubParams)

from ner_pipeline.pipelines.models.shared.metrics_base import MetricsLogger
from ner_pipeline.pipelines.models.shared.factory import BaseResolvedConfig


@dataclass(frozen=True)
class TaskComponents:
    """
    Stores common training pipeline components 
    required to initialse a run
    """
    resolve_cfg: Any
    experiment_subfolder_factory: Any
    loguru_helper_factory: Any
    wandb_run_manager_factory: Any
    training_comp_builder_cls: Any
    orchestrator_config_cls: Any
    orchestrator_cls: Any

class BaseTaskSpec(ABC):
    """
    Task-level wiring for the training pipeline.

    Each task (e.g NER, MLM) define their own component mapping, but the runner
    stays generic.
    """

    components: TaskComponents

    def resolve_task_name(self, cfg: DictConfig) -> str:
        task_type = getattr(cfg.task, "task_type", None) or getattr(cfg.task, "task_name", None)
        if not task_type:
            raise ValueError("Missing task type name in config.")
        return task_type.lower()

    def apply_sweep_overrides(self, cfg: DictConfig, wb_cfg: dict) -> DictConfig:
        """
        Default sweep override logic. Task-specific specs can override this
        """

        cfg.lr = wb_cfg["learning_rate"]
        cfg.weight_decay = wb_cfg["weight_decay"]
        cfg.warmup_ratio = wb_cfg["warmup_ratio"]
        cfg.batch_size = wb_cfg["train_batch_size"]
        cfg.eval_batch_size = wb_cfg["eval_batch_size"]
        cfg.seed = wb_cfg["seed"]

        cfg.task.args.lr_scheduler_type = wb_cfg["lr_scheduler_type"]

        return cfg


class TrainingPipelineRunner:
    def __init__(self, task_spec: BaseTaskSpec):
        if isinstance(task_spec, type):
            task_spec = task_spec()
        self.task_spec = task_spec
        self.components = task_spec.components

    def _resolve_cfg(self, cfg: DictConfig) -> BaseResolvedConfig:
        return self.components.resolve_cfg.from_cfg(cfg) 

    def _create_loguru_helper(self, resolved_cfg: BaseResolvedConfig, base_dir: str, run_id: str = None):
        kwargs = {"resolved_cfg": resolved_cfg, "base_dir": base_dir}
        if run_id is not None:
            kwargs["run_id"] = run_id
        return self.components.loguru_helper_factory.create(**kwargs)

    def _create_subfolder_builder(self, resolved_cfg: BaseResolvedConfig):
        return self.components.experiment_subfolder_factory.create(resolved_cfg)

    def _create_wandb_manager(self, resolved_cfg: BaseResolvedConfig, log_dir: Path):
        return self.components.wandb_run_manager_factory.create(resolved_cfg, log_dir)

    def execute_once(
        self,
        cfg: DictConfig,
        base_dir: str,
        loguru_helper=None,
        wandb_run=None,
        wandb_artifact=None,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #components = self.task_spec.components
        resolved_cfg = self._resolve_cfg(cfg)

        # Logging
        if loguru_helper is None:
            loguru_helper = self._create_loguru_helper(resolved_cfg, base_dir)
            loguru_helper.configure()
            logger.info(f"Current device set as: {device}")

        log_dir = loguru_helper.log_dir

        # Experiment subfolder
        subfolder_builder = self._create_subfolder_builder(resolved_cfg)
        subfolder_builder.build()
        experiment_subfolder = subfolder_builder.subfolder

        # W&B (if not already provided by a sweep)
        if wandb_run is None:
            if cfg.use_wandb:
                run_manager = self._create_wandb_manager(resolved_cfg, log_dir)
                wandb_run = run_manager.setup_run()
                
                wandb_artifact = run_manager.create_artifact()

                logger.info(
                    "Logging to W&B is enabled for this run.\n"
                    f"Run logs and metadata will be logged to: {cfg.logging.wandb.run.project}"
                )
                wandb_run.log({"Current device for run": device})
            else:
                logger.info("Logging to W&B is disabled for this run.")

        # Output dir
        output_dir = create_output_dir(base_path=base_dir,
                                        experiment_subfolder=experiment_subfolder)

        # Build training context
        context = BuildContext(cfg=cfg,
                            output_dir=output_dir,
                            device=device,
                            wandb_run=wandb_run,
                            wandb_artifact=wandb_artifact,
                        )

        # Build task-specific training components
        training_comp = self.components.training_comp_builder_cls(context)
        
        #init training metrics logger
        metrics_logger = MetricsLogger()

        hub_params = None
        if cfg.publish_model:
            #set push to hub params
            hub_params = PushToHubParams(repo_id=cfg.repo_id,
                                        push_to_org_repo=cfg.push_to_org_repo,
                                        commit_message=cfg.commit_message,
                                    )

        orchestrator_conf = self.components.orchestrator_config_cls(
            context=context,
            builder=training_comp,
            metrics_logger=metrics_logger,
            hub_params=hub_params,
            publish_model=cfg.publish_model,
            wandb_run=wandb_run,
            wandb_artifact=wandb_artifact,
        )

        #pass orchestrator config to main training orchestrator
        training_orchestrator = self.components.orchestrator_cls(orchestrator_conf)
        
        #execute training
        return training_orchestrator.execute()

    
    def execute_sweep(self, cfg: DictConfig, base_dir: str):
        sweep_cfg = OmegaConf.to_container(cfg.task.sweeps_config, resolve=True)
        
        local_tmp = os.environ.get("TMPDIR", "/tmp")

        sweep_id = wandb.sweep(
            sweep=sweep_cfg,
            project=cfg.logging.wandb.run.project,
            entity=cfg.logging.wandb.run.entity,
        )

        def make_sweep_trial():
            trial_sink_id = None

            with wandb.init(
                project=cfg.logging.wandb.run.project,
                entity=cfg.logging.wandb.run.entity,
                job_type=cfg.logging.wandb.run.job_type,
                dir=local_tmp,
            ) as run:
                try:
                    os.environ["HF_DATASETS_CACHE"] = os.path.join(
                        local_tmp, "hf_datasets", run.id
                    )
                    os.environ["TRANSFORMERS_CACHE"] = os.path.join(
                        local_tmp, "hf_hub"
                    )
                    os.environ["HF_HOME"] = os.path.join(local_tmp, "hf_hub")

                    #clone and patch cfg from sampled hyperparameters
                    trial_cfg = OmegaConf.create(
                        OmegaConf.to_container(cfg, resolve=False)
                    )
                    trial_cfg = self.task_spec.apply_sweep_overrides(
                        trial_cfg, run.config
                    )
                    resolved_trial_cfg = self._resolve_cfg(trial_cfg)

                    #build trial-specific logging structure from patched cfg
                    trial_loguru_helper = self._create_loguru_helper(
                        resolved_trial_cfg, base_dir, run_id=run.id
                    )
                    trial_sink_id = trial_loguru_helper.configure()

                    #reuse existing manager logic for final identity/metadata
                    run_manager = self._create_wandb_manager(
                        resolved_trial_cfg, trial_loguru_helper.log_dir
                    )

                    #attach updated run parameters to existing run
                    run_manager.attach_to_existing_run(run)
                    
                    #create run artifact
                    artifact = run_manager.create_artifact()

                    #execute one full training run with the trial run
                    return self.execute_once(
                        trial_cfg,
                        base_dir=base_dir,
                        loguru_helper=trial_loguru_helper,
                        wandb_run=run,
                        wandb_artifact=artifact,
                    )
                finally:
                    logger.complete()
                    if trial_sink_id is not None:
                        logger.remove(trial_sink_id)

        wandb.agent(
            sweep_id,
            function=make_sweep_trial,
            count=getattr(cfg.task, "sweep_count", 20),
        )



def setup_hf_cache_for_trial(run_id: str) -> str:
    local_tmp = os.environ.get("TMPDIR", "/tmp")
    hf_root = Path(local_tmp) / "hf_cache" / run_id
    datasets_cache = hf_root / "datasets"
    hub_cache = hf_root / "hub"

    datasets_cache.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_root)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)

    return str(datasets_cache)

    