
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
                              BaseTaskSpec)

from ner_pipeline.utils.common import create_output_dir, set_seed

from ner_pipeline.pipelines.models.tasks.mlm.experiment_manager import MLMBaseExperimentSubfolderBuilder
from ner_pipeline.pipelines.models.tasks.mlm.logging_manager import MLMBaseLoguruHelper

from ner_pipeline.pipelines.models.tasks.mlm.trainer_builder import MLMTrainingCompBuilder
from ner_pipeline.pipelines.models.tasks.mlm.mlm_trainer import MLMTrainingOrchestrator

from ner_pipeline.pipelines.models.tasks.mlm.mlm_factory import (
    MLMTrainingOrchestratorConfig,
                                                                MLMExperimentSubfolderFactory,
                                                                MLMLoguruHelperFactory,
                                                                MLMWandbRunManagerFactory)


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
os.environ["WANDB_CACHE_DIR"] = "/nfs/production/literature/amina-mardiyyah/.cache"





class MLMTaskSpec(BaseTaskSpec):
    components = TaskComponents(
        experiment_subfolder_factory=MLMExperimentSubfolderFactory,
        loguru_helper_factory=MLMLoguruHelperFactory,
        wandb_run_manager_factory=MLMWandbRunManagerFactory,
        training_comp_builder_cls=MLMTrainingCompBuilder,
        orchestrator_config_cls=MLMTrainingOrchestratorConfig,
        orchestrator_cls=MLMTrainingOrchestrator,
    )



@hydra.main(config_path="../config", config_name="training_common", version_base=None)
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

    runner = TrainingPipelineRunner(MLMTaskSpec)

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