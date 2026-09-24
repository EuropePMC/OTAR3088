
import os
from pathlib import Path
from dotenv import load_dotenv

from loguru import logger
import wandb 

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch

from ner_pipeline.cli.pipeline_common import (TaskComponents,
                              TrainingPipelineRunner, 
                              BaseTaskSpec,
                              setup_hf_cache_for_trial)

from ner_pipeline.utils.common import create_output_dir, set_seed
from ner_pipeline.pipelines.models.tasks.ner.ner_trainer import NERTrainingOrchestrator
from ner_pipeline.pipelines.models.tasks.ner.ner_trainer_builder import NERTrainingCompBuilder

from ner_pipeline.pipelines.models.tasks.ner.ner_factory import (NERResolvedConfig,
                                                                NERTrainingOrchestratorConfig,
                                                                NERExperimentSubfolderFactory,
                                                                NERLoguruHelperFactory,
                                                                NERWandbRunManagerFactory)


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
os.environ["WANDB_CACHE_DIR"] = Path.cwd() / ".cache"




class NERSweepConfigMixin:
    def validate_training_strategy(self, cfg):
        allowed_training_strategy = {"reinit_only", "reinit_llrd"}
        if not cfg.training_strategy.lower() in allowed_training_strategy:
            return False
        return True
    

    def update_sweep_config(self, sweep_cfg):

        sweep_cfg["parameters"]["reinit_k_layers"] = {
                                                        "values": [3, 4, 5, 6]
                                                        }
        # sweep_cfg["parameters"]["reinit_classifier"] = {
        #                                                 "values": [True, False]
        #                                                 }
        return sweep_cfg        



class NERTaskSpec(BaseTaskSpec):
    components = TaskComponents(
        resolve_cfg=NERResolvedConfig,
        experiment_subfolder_factory=NERExperimentSubfolderFactory,
        loguru_helper_factory=NERLoguruHelperFactory,
        wandb_run_manager_factory=NERWandbRunManagerFactory,
        training_comp_builder_cls=NERTrainingCompBuilder,
        orchestrator_config_cls=NERTrainingOrchestratorConfig,
        orchestrator_cls=NERTrainingOrchestrator,
    )

    def apply_sweep_overrides(self, cfg: DictConfig, wb_cfg: dict, override_reinit_llrd_config: bool = False) -> DictConfig:
        """
        Default sweep override logic. Task-specific specs can override this
        """
        cfg = super().apply_sweep_overrides(cfg, wb_cfg)
        cfg.task.use_data_aug = wb_cfg["use_data_aug"]

        if override_reinit_llrd_config:
            #cfg.reinit_classifier = wb_cfg["reinit_classifier"]
            cfg.reinit_k_layers = wb_cfg["reinit_k_layers"]

        return cfg


class NERTrainingPipelineRunner(TrainingPipelineRunner, NERSweepConfigMixin):
    def __init__(self, task_spec: NERTaskSpec):
        if isinstance(task_spec, type):
            task_spec = task_spec()
        super().__init__(task_spec)
        self.task_spec = task_spec

                 
    def execute_sweep(self, cfg: DictConfig, base_dir: str):
        sweep_cfg = OmegaConf.to_container(cfg.task.sweeps_config, resolve=True)

        override_reinit_llrd_config = self.validate_training_strategy(cfg)
        if override_reinit_llrd_config:
            sweep_cfg = self.update_sweep_config(sweep_cfg)
        
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
                        trial_cfg, run.config, override_reinit_llrd_config=override_reinit_llrd_config
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


   

@hydra.main(config_path="../../config", config_name="training_common", version_base=None)
@logger.catch
def run(cfg: DictConfig):
    #seed reproducibility seed
    set_seed(cfg.seed)
    #load environment variables
    load_dotenv()
    
    BASE_DIR = os.environ.get("BASE_DIR")

    if cfg.use_wandb:
        wandb.login(key=os.environ["WANDB_TOKEN"])
    else:
        os.environ["WANDB_MODE"] = "disabled"

    runner = NERTrainingPipelineRunner(NERTaskSpec)

    do_sweep = getattr(cfg.task, "run_wandb_sweep", False)
    if do_sweep:
        if not cfg.use_wandb:
            raise ValueError(
                "A run sweep cannot be initiated when W&B is disabled.\n"
                "Set `use_wandb=True` in config to execute a sweep."
            )
        runner.execute_sweep(cfg, base_dir=BASE_DIR)
        return

    runner.execute_once(cfg, base_dir=BASE_DIR)

if __name__ == "__main__":
    run()

