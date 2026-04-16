from typing import Tuple, Dict, Union

from transformers import (PreTrainedTokenizerBase,
                         PreTrainedTokenizerFast, 
                         AutoTokenizer,
                         DataCollatorForTokenClassification)

from .modelling import (BaseTrainer,
                       CRFTrainer,
                       WeightedTrainer
                       )
from .trainer_config import NerTrainerType



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



class NerTrainerFactory:
    """
    Factory responsible for instantiating the correct
    trainer class for NER model training.
    """
    _registry = {
        NerTrainerType.STANDARD: BaseTrainer,
        NerTrainerType.CRF: CRFTrainer,
        NerTrainerType.WEIGHTED: WeightedTrainer
    }

    @classmethod
    def get_trainer_class(cls, trainer_type:str):
        """
        Creates and initialise a specific huggingface Trainer
        Args:
            trainer_type: type of hf trainer to use. 
                            Options[Standard Trainer, CRFTrainer, WeightedTrainer, SWATrainer]
            trainer_kwargs: Other default keyword parameters traditionally accepted 
                            by huggingface trainer. E.g model, training_arguments, train_dataset, etc
        """
        trainer_name = NerTrainerType(trainer_type.lower())
        trainer_cls = cls._registry[trainer_name]

        return trainer_cls
