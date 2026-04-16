from typing import Type, Optional
from abc import abstractmethod, ABC
from dataclasses import dataclass
from loguru import logger
from .metrics_base import MetricsLogger
from .trainer_builder_base import HFTrainingCompBuilder
from .trainer_config_base import BuildContext, PushToHubParams
from ner_pipeline.utils.common import set_seed

@dataclass(frozen=True)
class HFTrainingOrchestratorConfig:
    """
    Configuration container for orchestrating a Hugging Face training run.

    This dataclass encapsulates all dependencies required by the training runner,
    including the builder responsible for constructing training components,
    runtime context, metrics logging utilities, and optional publishing settings.

    The runner uses this configuration to:

    - Construct training components via the provided builder
    - Execute training and evaluation lifecycle
    - Log metrics to configured backends (e.g., W&B)
    - Optionally publish trained models to the Hugging Face Hub

    Attributes:
        context (BuildContext):
            Execution context containing configuration, device, logging,
            artifact tracking, and experiment metadata.

        builder (Type[HFTrainingCompBuilder]):
            Builder class responsible for constructing training components
            such as model, datasets, trainer arguments, callbacks, and
            strategy-specific kwargs.

            Examples:
                - HFTrainingCompBuilder
                - TAPTTrainingCompBuilder
                - NerTrainingCompBuilder

        metrics_logger (MetricsLogger):
            Metrics logging utility responsible for persisting training and
            evaluation metrics to configured backends (Trainer, W&B, etc.).

        hub_params (Optional[PushToHubParams]):
            Parameters controlling publication of the trained model to the
            Hugging Face Hub. Required only when publish_model=True.

        publish_model (bool, default=False):
            Whether the trained model should be pushed to the Hugging Face Hub
            after training completes.
    """
    context: Type[BuildContext]
    builder: Type[HFTrainingCompBuilder] 
    metrics_logger: Type[MetricsLogger]
    hub_params: Optional[PushToHubParams]
    publish_model: bool
    wandb_run: Optional[None]
    wandb_artifact: Optional[None]



class HFTrainingOrchestrator(ABC):
    """
    Abstract base class for orchestrating HuggingFace training workflows.

    This class implements the template method pattern to standardise training,
    evaluation, metrics logging, and model publishing while allowing subclasses
    to define trainer construction logic.

    Workflow:
        - Construct training components via configured builder
        - Initialise Hugging Face Trainer
        - Execute training and evaluation
        - Log training and evaluation metrics
        - Persist checkpoints and trainer state
        - Log experiment metadata (e.g., W&B)
        - Optionally publish trained model to Hugging Face Hub
    
    Subclasses must implement:
        _build_trainer() and/or optional logging to experiment trackers(e.g, W&B)

    Attributes:
        runner_conf (HFTrainingRunnerConfig):
           Runner Configuration container controlling execution behavior.

        context (BuildContext):
            Execution context containing runtime configuration and artifacts.

        builder (Type[BaseHFTrainingCompBuilder]):
            Builder used to construct training components.

        components (HFTrainerComponents):
            Fully constructed training components including model, datasets,
            trainer kwargs, and strategy metadata.

        trainer (Trainer):
            Hugging Face Trainer instance used for training and evaluation.

        train_results (TrainOutput):
            Results returned from trainer.train().

        eval_results (Dict[str, float]):
            Results returned from trainer.evaluate().
        """

    def __init__(self, runner_conf: HFTrainingOrchestratorConfig):
        """
        Initialises the training runner and constructs trainer components.

        Args:
            runner_conf (HFTrainerOrchestratorConfig):
                Configuration container defining builder, context,
                metrics logger, and publishing behaviour.
        """

        self.runner_conf = runner_conf
        self.context = self.runner_conf.context
        self.wandb_run = self.context.wandb_run
        self.wandb_artifact = self.context.wandb_artifact
        self.builder = self.runner_conf.builder
        self.components = self.builder.apply_strategy()
        self.metrics_logger = self.runner_conf.metrics_logger
        self.publish_model = self.runner_conf.publish_model
        self.hub_params = self.runner_conf.hub_params
        self.trainer = None
        self._is_trained = False

    
    def _validate_training_completed(self):
        """
        Checks that trainer has been intialised and 
        training cycle completed before returning 
        related artifacts
        """
        if not self._is_trained:
            raise RuntimeError(
                "Training not initialised.\n"
                "Call `execute()` to run training step first, "
                "before accessing trainer artifacts."
            )

    def _validate_trainer_built(self):
        if self.trainer is None:
            raise RuntimeError(
                "Trainer not built. `_build_trainer()` must assign self.trainer."
            )


    @abstractmethod
    def _build_trainer(self):
        """
        Constructs HuggingFace Trainer instance.

        Must be implemented by subclasses to define trainer initialisation logic.

        Expected to assign:
            self.trainer
        """

        raise NotImplementedError("Subclass must implement `build_trainer` method")
    



    @property
    def trainer_log_history(self):
        """
        Returns Trainer log history.

        Useful for experiment analysis, debugging, and downstream reporting.

        Returns:
            List[Dict]: Trainer log history entries containing train and eval metrics.
        """
        self._validate_training_completed()
        
        return self.trainer.state.log_history

    def _run_training(self):
        """
        Executes model training.

        Returns:
            TrainOutput: HuggingFace training output.
        """
        logger.success("Trainer initialised")
        logger.info("Training commencing...")
        self.train_results = self.trainer.train()
        logger.info(f"Train Results: \n{self.train_results}")
        return self.train_results

    
    def _run_evaluation(self):
        """
        Executes model evaluation on validation dataset.

        Returns:
            dict: Evaluation metrics dictionary.
        """
        self.eval_results = self.trainer.evaluate()
        logger.info(f"Final evaluation results: {self.eval_results}")
        return self.eval_results

    def _push_to_hf(self):
        """
        Publishes trained model to HuggingFace Hub.

        Uses PushToHubParams configuration to control repository name,
        authentication, privacy settings, and commit metadata.

        Raises:
            ValueError: If hub_params is not configured.
        """
        self._validate_training_completed()
        #pass hub params to trainer
        self.trainer.args.hub_model_id = self.hub_params.repo_id
        self.trainer.args.hf_token = self.hub_params.token
        if self.hub_params.push_to_org_repo:
            self.trainer.args.push_to_hub_organization = self.hub_params.repo_id

        self.trainer.args.push_to_hub_model_id = self.hub_params.repo_id
        self.trainer.args.push_to_hub_token = self.hub_params.token
        logger.info(f"Repo ID: {self.trainer.args.hub_model_id}")
        self.trainer.push_to_hub(self.hub_params.commit_message)

    def _log_to_wandb(self):
        """
        Optional log to wandb method.
        Can be implemented by subclasses
        """
        pass


    def execute(self):
        """
        Executes full training lifecycle.

        This is the main entry point for training execution.

        Steps:
            1. Set reproducibility seed
            2. Build Trainer instance
            3. Train model
            4. Save best checkpoint
            5. Evaluate model
            6. Log metrics
            7. Publish model to huggingface hub (optional)
        """
        if self._is_trained:
            raise RuntimeError("Training has already been executed.")

        #set reproducibility seed
        set_seed(self.builder.cfg.seed)

        #build trainer
        self._build_trainer()

        self._validate_trainer_built()


        #init training
        self.train_results = self._run_training()

        #get best checkpoint
        self.best_ckpt_path = self.trainer.state.best_model_checkpoint
        logger.info(f"Best checkpoint for this run: {self.best_ckpt_path}")
        self.trainer.save_model(self.best_ckpt_path)
        self.trainer.save_state()
        logger.info("Training completed...")
        logger.info("Running final evaluation on validation set...")
        
        #init evaluation
        self.eval_results = self._run_evaluation()

        #log metrics
        self.metrics_logger.log_training_metrics(self.trainer, self.train_results)
        self.metrics_logger.log_eval_metrics(self.trainer, self.eval_results)

        #validate training lifecycle completed
        self._is_trained = True

        #optional experiment logging to WANDB
        if self.builder.cfg.use_wandb:
            self._log_to_wandb() 

        #optional publish model to Huggingface hub
        if self.publish_model:
            if not self.hub_params:
                raise ValueError("`hub_params` is required when `publish_model` is set to True.")

            logger.info(f"Pushing Model to hub repository --> {self.hub_params.repo_id}")
            self._push_to_hf()

       
        return {
            "train_results": self.train_results,
            "eval_results": self.eval_results,
            "best_checkpoint_path": self.best_ckpt_path,
            "trainer_log_history": self.trainer_log_history
        }

    