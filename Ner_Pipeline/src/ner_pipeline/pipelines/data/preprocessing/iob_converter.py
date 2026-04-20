from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union, Any
import pandas as pd

import spacy
import scispacy
from tqdm import tqdm

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast
from ner_pipeline.schemas.ner_dataset import IOBConfig, nlp, EntityDict, NerDataset




class IOBConverter(ABC):
  def __init__(
    self, 
    data:Union[pd.DataFrame, NerDataset, Dataset, DatasetDict], 
    config: IOBConfig):
    self.data = data
    self.config = config

  
  def __repr__(self):
    return f"{self.__class__.__name__}(num_samples={(len(self.data))}, config_params={self.params})"


  def _validate_entity_schema(self, entities: List[EntityDict]) -> None:
    if not isinstance(entities, list) or not all(isinstance(item, dict) for item in entities):
      raise ValueError(f"Expected list of dicts, got {type(entities)}")

    required_keys = {"start", "end", self.config.schema.ent_label_key}
    for ent in entities:
      missing_key = required_keys - ent.keys()
      if missing_key:
        raise KeyError(f"Missing required entity dict key(s): {missing_key}")
    
  @abstractmethod
  def _tokenize_with_offsets(self, sentence: str):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_tokens_offsets()"
        )


  def _process_single_sentence(self, sentence: str, entities: List[EntityDict]):
      self._validate_entity_schema(entities)
      tokens, offsets = self._tokenize_with_offsets(sentence)
      iob_tags = ["O"] * len(tokens)

      for ent in entities:
          ent_start = ent["start"]
          ent_end = ent["end"]
          label = ent[self.config.schema.ent_label_key]

          is_first_token = True
          for idx, (tok_start, tok_end) in enumerate(offsets):
        
              if tok_start >= ent_start and tok_end <= ent_end:

                if is_first_token:
                    iob_tags[idx] = f"B-{label}"
                    is_first_token = False
                else:
                    iob_tags[idx] = f"I-{label}"

      return tokens, iob_tags


  def batch_process(self, batch):
    all_tokens = []
    all_iob_tags = []

    for sentences, entities in zip(batch[self.config.schema.text_col], batch[self.config.schema.entity_col]):
      tokens, iob_tags = self._process_single_sentence(sentences, entities)
      all_tokens.append(tokens)
      all_iob_tags.append(iob_tags)

    return {"tokens": all_tokens, "tags": all_iob_tags}


  def _remove_cols(self, example: Union[Dataset, DatasetDict]):
    if isinstance(example, DatasetDict):
      remove_cols = [v.column_names for k, v in example.items()][0]
    else:
      remove_cols = example.column_names

    return remove_cols

  def convert(self) -> Union[Dataset, DatasetDict, NerDataset]:
    
    if isinstance(self.data, pd.DataFrame):
      ds = Dataset.from_pandas(self.data)
    elif isinstance(self.data, list) and all(isinstance(item, dict) for item in self.data):
      ds = Dataset.from_list(self.data)
    else:
        
      ds = self.data


  
    processed_ds = ds.map(
      self.batch_process,
      batched=True,
      remove_columns=self._remove_cols(ds),
      load_from_cache_file=False
    )

    if not self.config.as_hf_dataset:
        if isinstance(processed_ds, DatasetDict):
            return {k: v.to_list() for k, v in processed_ds.items()}
        return processed_ds.to_list() 

    return processed_ds 





class SpacyIOBConverter(IOBConverter):
  def __init__(
    self,
    data:Union[pd.DataFrame, NerDataset, Dataset, DatasetDict], 
    config: IOBConfig
    )-> Union[Dataset, DatasetDict, NerDataset]:
    super().__init__(data, config)


  def _tokenize_with_offsets(self, sentences: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    doc = self.config.tokenizer_backend(sentences) 
    tokens = [tok.text for tok in doc]
    offsets = [(tok.idx, tok.idx + len(tok.text)) for tok in doc]
    return tokens, offsets
    
    
class HFIOBConverter(IOBConverter):
  def __init__(
    self,
    data:Union[pd.DataFrame, NerDataset, Dataset, DatasetDict], 
    config: IOBConfig
    )-> Union[Dataset, DatasetDict, NerDataset]:
    super().__init__(data, config)

  def _tokenize_with_offsets(self, sentences: str):
    encoded = self.config.tokenizer_backend(
      sentences,
      add_special_tokens=True,
      return_offsets_mapping=True,
      return_special_tokens_mask=True,
      return_tensors=None,
    )

    tokens = encoded.tokens()
    offsets = encoded.offset_mapping
    special_tokens_mask = encoded.special_tokens_mask
    
    cleaned_tokens, cleaned_offsets = [], []
    for token, (token_start, token_end), is_special in zip(tokens, offsets, special_tokens_mask):
        # Skip special tokens (like [CLS], [SEP]), but keep everything else
        if is_special:
            continue
        cleaned_tokens.append(token)
        cleaned_offsets.append((token_start, token_end))

    return cleaned_tokens, cleaned_offsets





