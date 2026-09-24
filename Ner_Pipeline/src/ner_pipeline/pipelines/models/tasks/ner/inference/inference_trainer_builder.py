
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import torch.nn as nn
from transformers import AutoModelForTokenClassification

from .inference_trainer_config import NERInferenceTrainerKwargs, NERHFInferenceComponents
from .inference_dataloader import PrepareNERInferenceDataset, NERInferenceDatasetArtifact

from ..ner_factory import build_tokenizer_data_collator
from ..ner_metrics import seqeval_metrics


from ....shared.trainer_config_base import BuildContext
from ....shared.trainer_builder_base import HFInferenceCompBuilder
from ....shared.factory import build_training_args
from ner_pipeline.utils.common import set_seed


class NERInferenceCompBuilder(HFInferenceCompBuilder):
    """
    Builder class responsible for constructing all necesary
    components required for inference of an NER Model using
    HuggingFace's `Trainer` class.

    Parameters:
    """
    def __init__(self, context: BuildContext):
        self.context = context
        self._components = None
        self._is_built = False


    def _require_built(self):
        if self._components is None:
            raise RuntimeError(
                "Components not built yet."
                "\nPlease call the `.build_components()` method before accessing attributes."
            )

    @property
    def cfg(self) -> DictConfig:
        return self.context.cfg

    @property
    def components(self) -> NERHFInferenceComponents:
        self._require_built()
        return self._components

    @property
    def trainer_kwargs(self):
        self._require_built()
        return self._components.trainer_kwargs
      
    @property
    def inference_dataset_artifact(self) -> NERInferenceDatasetArtifact:
        if not hasattr(self, "_cached_dataset_artifact"):
            dataset_prep = PrepareNERInferenceDataset(cfg=self.cfg, 
                                                    label2id=self.label2id, 
                                                    wandb_run=self.context.wandb_run)

            self._cached_dataset_artifact = dataset_prep.prepare()  
        return self._cached_dataset_artifact


    def build_components(self):
        if self._components is not None:
            return self._components
        
        #init seed for reproducibility
        set_seed(self.cfg.seed)

        wandb_run, wandb_artifact, output_dir, device = (self.context.wandb_run,
                                                  self.context.wandb_artifact,
                                                  self.context.output_dir,
                                                  self.context.device)
         
         
        #load tokenizer and data collator
        tokenizer, data_collator = build_tokenizer_data_collator(self.cfg.model_name_or_path)
        self.tokenizer = tokenizer

        # The checkpoint mapping is authoritative for optional dataset labels.
        model = AutoModelForTokenClassification.from_pretrained(self.cfg.model_name_or_path)
        model.to(device)
        model.eval()

        self.label2id = dict(model.config.label2id)
        id2label = dict(model.config.id2label)

        label_list = list(model.config.id2label.values())
        logger.success(f"Model Loaded successfully. This model has been trained with the following entities: {label_list}")  

        compute_metrics = seqeval_metrics(label_list)


        #build training args
        inference_args = build_training_args(self.cfg, output_dir)

        #load inference dataset
        test_dataset =  self.inference_dataset_artifact.test_dataset


        trainer_kwargs = NERInferenceTrainerKwargs(
            model=model,
            args=inference_args,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        components = NERHFInferenceComponents(trainer_kwargs=trainer_kwargs, 
                                            test_dataset=test_dataset,
                                            label2id=self.label2id,
                                            id2label=id2label,
                                            label_list=label_list
                                            )


        logger.info(f"Inference components build successfully as: \n{components}")

        self._components = components
        self._is_built = True
        
        return self._components