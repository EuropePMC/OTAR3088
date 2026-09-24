
import os
from pathlib import Path
from dotenv import load_dotenv

from loguru import logger
import wandb 

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch


from ner_pipeline.utils.common import create_output_dir, set_seed
from ner_pipeline.pipelines.models.shared.trainer_config_base import BuildContext

from ner_pipeline.pipelines.models.tasks.ner.inference.inference_trainer import NERInferenceOrchestrator
from ner_pipeline.pipelines.models.tasks.ner.inference.inference_trainer_builder import NERInferenceCompBuilder

from ner_pipeline.pipelines.models.tasks.ner.inference.inference_factory import (NERInferenceOrchestratorConfig, 
                                                                                NERInferenceResolvedConfig)

from ner_pipeline.pipelines.models.tasks.ner.inference.inference_logging_manager import (NERInferenceLoguruManager,
                                                                                        NERInferenceWandbManager,
                                                                                        NERInferenceSubfolderBuilder)




os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
os.environ["WANDB_CACHE_DIR"] = Path.cwd() / ".cache"


@hydra.main(config_path="../../config", config_name="inference_common", version_base=None)
@logger.catch(reraise=True)
def run(cfg:DictConfig):
    #seed reproducibility seed
    set_seed(cfg.seed) 

    #load environment variables
    load_dotenv()

    BASE_DIR = os.environ.get("BASE_DIR")
    print(BASE_DIR)

    #resolve configure for helpers
    resolved_cfg = NERInferenceResolvedConfig.from_cfg(cfg)

    #initialise logger helper
    loguru_helper = NERInferenceLoguruManager(resolved_cfg, BASE_DIR)
    loguru_helper.configure()

    #experiment subfolder helper
    subfolder_builder = NERInferenceSubfolderBuilder(resolved_cfg)
    subfolder_builder.build()

    #init device
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    logger.info(f"Current device set as: {device}")


    wandb_run = None
    #disable wandb if not set to true in config
    if not cfg.use_wandb:
        os.environ["WANDB_MODE"] = "disabled"
        logger.info(f"Logging to Wandb is disabled for this run.")
      

    else:
        #init wandb if set to true in config
        wandb_token = os.environ.get("WANDB_TOKEN")
        wandb.login(key=wandb_token)

        #fetch log_dir from loguru helper
        log_dir = loguru_helper.log_dir

        #init wandb manager
        run_manager = NERInferenceWandbManager(resolved_cfg, log_dir)
        wandb_run = run_manager.setup_run()
    
        logger.info(f"Logging to Wandb is enabled for this run. \
                    Run logs and metadata will be logged to: {cfg.logging.wandb.run.project}")


    output_dir = create_output_dir(base_path=BASE_DIR, 
                                  experiment_subfolder=subfolder_builder.subfolder)


    #build context
    context = BuildContext(
                cfg = cfg,
                output_dir = output_dir,
                device = device, 
                wandb_run =  wandb_run,
            )
    #build inference components
    inference_comp = NERInferenceCompBuilder(context)


    #pass everything to inference orchestrator config
    orchestrator_conf = NERInferenceOrchestratorConfig(
                                context = context,
                                builder = inference_comp,
                                wandb_run = wandb_run,
                                wandb_artifact = None
                                )

    #pass orchestrator config to main inference orchestrator
    inference_orchestrator = NERInferenceOrchestrator(orchestrator_conf)

    #execute inference run
    inference_orchestrator.execute()

if __name__ == "__main__": run()