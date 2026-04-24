from dataclasses import dataclass, field
from typing import Literal, Union, Dict, Any, TypedDict, List

import spacy

from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast
from ..utils.common import inherit_docstring



nlp = spacy.load("en_core_sci_md", disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"]) 
nlp.add_pipe("sentencizer")
nlp.max_length = 10_000_000


EntityDict = Dict[str, Any]

class SentenceNerRecord(TypedDict):
  """
  Defines a typed attribute of a sentence-level NER dataset
  sentence(str): Name of field/column containing sentence 
  entitities: Name of field/column containing entities. Structure: Dict[str, Any]

  """
  sentence: str
  entities: EntityDict

NerDataset = List[SentenceNerRecord]


@dataclass
class RawNerSchema:
    """
    Defines the schema of NER dataset prior to preprocessing.

    Attributes:
        text_col: Name of Column or field containing sentence or document text
        entity_col: Name of Column or field containing raw entity annotations dict
        ent_label_key: Key inside entities column dict that stores the entity class
    """
    text_col: str
    entity_col: str
    ent_label_key: str


@inherit_docstring(RawNerSchema)
@dataclass
class BratConfig(RawNerSchema):
  "Default configuration for processing BRAT files to IOB"
  do_filter: bool = True
  do_rename: bool = True
  converter_type: Literal["spacy", "hf"] = "hf"
  save_dest: Literal["hf", "local"] = "hf"
  rename_map: Union[Dict[str, str], None] = field(default_factory=lambda: {"Anatomy": "Tissue"})

@dataclass
class IOBConfig:
  """
  Configuration for converting sentence-level NER into IOB format.
  """
  schema: RawNerSchema
  tokenizer_backend: Union[
      spacy.language.Language,
      PreTrainedTokenizerBase,
      PreTrainedTokenizerFast,
  ]
  as_hf_dataset: bool = False

  def __post_init__(self):
    if isinstance(self.schema, dict):
      try:
        self.schema = RawNerSchema(**self.schema)
      except TypeError as e:
        raise ValueError(f"Invalid RawNerSchema: {e}")


