
import os
from pathlib import Path
from dotenv import load_dotenv

from loguru import logger
import wandb 

import hydra
from omegaconf import DictConfig, OmegaConf

import torch

from ner_pipeline.utils.common import create_output_dir, set_seed
from ner_pipeline.pipelines.models.tasks.ner.ner_trainer import NerTrainingOrchestrator
from ner_pipeline.pipelines.models.tasks.ner.trainer_builder import NerTrainingCompBuilder
from ner_pipeline.pipelines.models.shared.trainer_config_base import (BuildContext, 
                                                                      PushToHubParams)
from ner_pipeline.pipelines.models.shared.metrics_base import MetricsLogger
from ner_pipeline.pipelines.models.shared.trainer_base import HFTrainingOrchestratorConfig
from ner_pipeline.pipelines.models.shared.experiment_manager import ExperimentSubfolderFactory
from ner_pipeline.pipelines.models.shared.logging_manager import (LoguruHelperFactory, 
                                                                    WandbRunManagerFactory
                                                                                            )

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
os.environ["WANDB_CACHE_DIR"] = Path.cwd() / ".cache"





def apply_sweep_overrides(cfg, wb_cfg):
    """
    Dynamically Overrides training cfg hyperparameters with 
    wandb_trials config when generated
    """
    #override training arguments hyperparameters
    cfg.lr = wb_cfg["learning_rate"]
    cfg.weight_decay = wb_cfg["weight_decay"]
    cfg.warmup_ratio = wb_cfg["warmup_ratio"]
    cfg.task.args.lr_scheduler_type = wb_cfg["lr_scheduler_type"]
    cfg.task.args.per_device_train_batch_size = wb_cfg["train_batch_size"]
    cfg.task.args.per_device_eval_batch_size = wb_cfg["eval_batch_size"]
    cfg.task.args.num_train_epochs = wb_cfg["epochs"]

    #override global hyperparameters used elsewhere in script
    cfg.batch_size = wb_cfg["train_batch_size"]
    cfg.eval_batch_size = wb_cfg["eval_batch_size"]
    cfg.num_epochs = wb_cfg["epochs"]

    return cfg





def execute_once(cfg:DictConfig, base_dir,
                 loguru_helper=None, 
                 wandb_run=None, 
                 wandb_artifact=None):
    """
    Executes a single training lifecycle

    """
   
    #init device
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    #initialise logger
    if loguru_helper is None:
        loguru_helper = LoguruHelperFactory.create(cfg=cfg, base_dir=base_dir)
        loguru_helper.configure()

        logger.info(f"Current device set as: {device}")
        #fetch log_dir from loguru helper
        log_dir = loguru_helper.log_dir

    #build experiment subfolder
    subfolder_builder = ExperimentSubfolderFactory.create(cfg)
    subfolder_builder.build()
    experiment_subfolder = subfolder_builder.subfolder

    if wandb_run is None:
        
        #if wandb_run is none and use_wandb is True, 
        #then we assume it's not a sweep run and create a run
        if cfg.use_wandb:
            #init wandb manager
            run_manager = WandbRunManagerFactory.create(cfg, log_dir)
            wandb_run = run_manager.setup_run()
            wandb_artifact = run_manager.create_artifact()

            logger.info(f"""Logging to Wandb is enabled for this run. \n
                        Run logs and metadata will be logged to: {cfg.logging.wandb.run.project}""")
            wandb_run.log({"Current device for run" : device})

        #wandb_run is none and use_wandb is false
        else:
            logger.info(f"Logging to Wandb is disabled for this run.")

    #create experiment output_dir using subfolder and basepath
    output_dir = create_output_dir(base_path=base_dir, 
                                  experiment_subfolder=experiment_subfolder)

    
    #build training context
    context = BuildContext(
                cfg = cfg,
                output_dir = output_dir,
                device = device, 
                wandb_run =  wandb_run,
                wandb_artifact = wandb_artifact
            )
    #build training components
    training_comp = NerTrainingCompBuilder(context)

    #init training metrics logger
    metrics_logger = MetricsLogger()

    hub_params = None

    if cfg.publish_model:
        #set push to hub params
        hub_params = PushToHubParams(
                                repo_id = cfg.repo_id,
                                push_to_org_repo = cfg.push_to_org_repo,
                                commit_message = cfg.commit_message
                                )

    #pass everything to training orchestrator config
    orchestrator_conf = HFTrainingOrchestratorConfig(
                                context = context,
                                builder = training_comp,
                                metrics_logger = metrics_logger,
                                hub_params = hub_params,
                                publish_model = cfg.publish_model,
                                wandb_run = wandb_run,
                                wandb_artifact = wandb_artifact
                                )

    #pass orchestrator config to main training orchestrator
    training_orchestrator = NerTrainingOrchestrator(orchestrator_conf)

    #execute training
    return training_orchestrator.execute()




def execute_sweep(cfg, base_dir):
    sweep_cfg = OmegaConf.to_container(cfg.task.sweeps_config, resolve=True)
    local_tmp = os.environ.get("TMPDIR", "/tmp")
    
    
    sweep_id = wandb.sweep(
        sweep=sweep_cfg,
        project=cfg.logging.wandb.run.project,
        entity=cfg.logging.wandb.run.entity,
    )

    def make_sweep_trial():
        #init sweep run
        with wandb.init(
            project=cfg.logging.wandb.run.project,
            entity=cfg.logging.wandb.run.entity,
            job_type=cfg.logging.wandb.run.job_type,
            dir=local_tmp,
        ) as run:
            try:
                os.environ["HF_DATASETS_CACHE"] = os.path.join(local_tmp, "hf_datasets", run.id)
                os.environ["TRANSFORMERS_CACHE"] = os.path.join(local_tmp, "hf_hub")
                os.environ["HF_HOME"] = os.path.join(local_tmp, "hf_hub")

            
                #clone and patch cfg from sampled hyperparameters
                trial_cfg = OmegaConf.create(
                    OmegaConf.to_container(cfg, resolve=False)
                )
                trial_cfg = apply_sweep_overrides(trial_cfg, run.config)

                #build trial-specific logging structure from patched cfg
                trial_loguru_helper = LoguruHelperFactory.create(cfg=trial_cfg, 
                                                                base_dir=base_dir,
                                                                run_id=run.id)
                trial_sink_id = trial_loguru_helper.configure()

                #reuse existing manager logic for final identity/metadata
                run_manager = WandbRunManagerFactory.create(trial_cfg, trial_loguru_helper.log_dir)

                run_manager.attach_to_existing_run(run)

                artifact = run_manager.create_artifact()

                #execute one full training run with the open run
                return execute_once(
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




@hydra.main(config_path="../config", config_name="common", version_base=None)
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
    
    do_sweep = getattr(cfg.task, "run_wandb_sweep", False)

    if do_sweep:
        if not cfg.use_wandb:
            raise ValueError("A run sweep cannot be initiated when Wandb is disabled"
                        "Set `use_wandb` to True in config to execute a sweep trial")
        execute_sweep(cfg, base_dir=BASE_DIR)
        return

    execute_once(cfg, base_dir=BASE_DIR)

if __name__ == "__main__":
    run()