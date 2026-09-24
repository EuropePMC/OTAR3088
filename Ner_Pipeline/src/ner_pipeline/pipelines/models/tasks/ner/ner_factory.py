from typing import (Tuple, Dict, 
                    Union, List,
                    Literal)

from dataclasses import dataclass
from pathlib import Path
from omegaconf import DictConfig
from loguru import logger

from transformers import (PreTrainedTokenizerBase,
                         PreTrainedTokenizerFast, 
                         AutoTokenizer,
                         DataCollatorForTokenClassification)


from .experiment_manager import (NERBaseExperimentSubfolderBuilder,
                                 NERReinitExperimentSubfolderBuilder,
                                 NERLLRDExperimentSubfolderBuilder,
                                 NERReinitLLRDExperimentSubfolderBuilder)


from .logging_manager import (NERResolvedConfig,
                              NERBaseLoguruHelper, 
                              NERReinitLoguruHelper,
                              NERLLRDLoguruHelper,
                              NERReinitLLRDLoguruHelper,
                              NERBaseWandbRunManager,
                              NERReinitWandbRunManager,
                              NERLLRDWandbRunManager,
                              NERReinitLLRDWandbRunManager
                              )

from  ...shared.trainer_config_base import BaseTrainerKwargs, HFModelConfig
from ...shared.trainer_base import HFTrainingOrchestratorConfig
from ...shared.factory import (TrainingStrategyName, 
                                BaseStrategyFactory, 
                                )



def build_label2id_id2label(label_list:Dict) -> Tuple[Dict, Dict]:

  label2id = {label:i for i,label in enumerate(label_list)}
  #id2label = {i:label for label,i in label2id.items()}
  id2label = dict(enumerate(label_list))

  return label2id, id2label


def build_tokenizer_data_collator(checkpoint_name: str) -> Tuple[Union[PreTrainedTokenizerBase, 
                                                                        PreTrainedTokenizerFast], 
                                                                DataCollatorForTokenClassification]:
  "Initialises tokenizer, data collator and applies tokenization function to dataset"
  tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
  data_collator  = DataCollatorForTokenClassification(tokenizer=tokenizer)

  return tokenizer, data_collator



@dataclass
class NERPredictions:
    true_labels: List[List[str]]
    pred_labels: List[List[str]]
    label_names: List[str]


@dataclass(kw_only=True)
class NERModelConfig(HFModelConfig):
    num_labels: int
    id2label: Dict[int, str]
    label2id: Dict[str, int]
    ner_head_type: Literal["standard", "crf"] 


@dataclass(frozen=True)
class NERTrainingOrchestratorConfig(HFTrainingOrchestratorConfig):
    pass



# class NERTrainerFactory:
#     """
#     Factory responsible for instantiating the correct
#     trainer class for NER model training.
#     """
#     _registry = {
#         NERTrainerType.STANDARD: BaseTrainer,
#         NERTrainerType.CRF: CRFTrainer,
#         NERTrainerType.FOCAL: FocalLossTrainer,
#         NERTrainerType.WEIGHTED: WeightedTrainer
#     }

#     @classmethod
#     def get_trainer_class(cls, trainer_type:str):
#         """
#         Creates and initialise a specific huggingface Trainer
#         Args:
#             trainer_type: type of hf trainer to use. 
#                             Options[Standard Trainer, CRFTrainer, WeightedTrainer, FocalLossTrainer]
#             trainer_kwargs: Other default keyword parameters traditionally accepted 
#                             by huggingface trainer. E.g model, training_arguments, train_dataset, etc
#         """
#         trainer_name = NERTrainerType(trainer_type.lower())
#         trainer_cls = cls._registry[trainer_name]

#         return trainer_cls


class NERExperimentSubfolderFactory(BaseStrategyFactory):
    _registry = {
        TrainingStrategyName.BASE: NERBaseExperimentSubfolderBuilder,
        TrainingStrategyName.REINIT: NERReinitExperimentSubfolderBuilder,
        TrainingStrategyName.LLRD: NERLLRDExperimentSubfolderBuilder,
        TrainingStrategyName.REINIT_LLRD: NERReinitLLRDExperimentSubfolderBuilder,
    }


class NERLoguruHelperFactory(BaseStrategyFactory):
    _registry = {
        TrainingStrategyName.BASE: NERBaseLoguruHelper,
        TrainingStrategyName.REINIT: NERReinitLoguruHelper,
        TrainingStrategyName.LLRD: NERLLRDLoguruHelper,
        TrainingStrategyName.REINIT_LLRD: NERReinitLLRDLoguruHelper,
    }

    @classmethod
    def create(cls, resolved_cfg: NERResolvedConfig, base_dir=None, run_id=None, *args, **kwargs):
        helper = super().create(resolved_cfg, base_dir, run_id, *args, **kwargs)
        print(f"Loguru Helper for NER task created: {helper.__class__.__name__}()")
        return helper


class NERWandbRunManagerFactory(BaseStrategyFactory):
    _registry = {
        TrainingStrategyName.BASE: NERBaseWandbRunManager,
        TrainingStrategyName.REINIT: NERReinitWandbRunManager,
        TrainingStrategyName.LLRD: NERLLRDWandbRunManager,
        TrainingStrategyName.REINIT_LLRD: NERReinitLLRDWandbRunManager,
    }

    @classmethod
    def create(cls, resolved_cfg: NERResolvedConfig, log_dir: Path, *args, **kwargs):
        manager = super().create(resolved_cfg, log_dir, *args, **kwargs)
        logger.info(f"Wandb Manager set to: {manager.__class__.__name__}()")
        return manager