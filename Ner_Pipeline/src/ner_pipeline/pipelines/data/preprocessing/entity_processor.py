
from typing import Union, Optional, List, Dict, Any, Type
import pyarrow as pa
import spacy
import scispacy

from ner_pipeline.schemas.ner_dataset import nlp, EntityDict

DatasetColumn = pa.ChunkedArray

def flatten_singleton_labels(entities_lst, ent_label_key="labels"):

    if not isinstance(entities_lst, list):
        raise ValueError(f"Expected List of Dictionaries but got : {type(entities_lst)}")
    cleaned_entities_lst = []
    for ent in entities_lst:
        if isinstance(ent, dict) and ent_label_key in ent and isinstance(ent[ent_label_key] , list) and len(ent[ent_label_key]) == 1:
            cleaned_ent = ent.copy()
            cleaned_ent[ent_label_key] = cleaned_ent[ent_label_key][0]
            cleaned_entities_lst.append(cleaned_ent)
        else:
            cleaned_entities_lst.append(ent)

    return cleaned_entities_lst



def rename_ent(
    ent_col: Union[DatasetColumn, List[List[str]], List[EntityDict]],
    rename_map: Dict[str, str],
    is_iob: Optional[bool]=None,
    ent_label_key: Optional[str]=None,
    *,
    dataset_format: str
) -> Union[DatasetColumn, List[List[str]],
    List[EntityDict]
    ]:
    
    
    """
    Renames entities in a huggingface dataset, pandas dataframe column or a spacy-like object
    
    Args:
        ent_col: The column containing the list of entities. A hf dataset column(e.g dataset['labels']), 
                a pandas dataframe column(e.g df['labels']) represented as a spacy json object or  brat doc. 
                N.b spacy_like_df assumes a list of list of dict object, while spacy_brat assumes a list of dict. 
                Inspect your data column to select the right format

        param rename_map: A dictionary mapping old entity(ies) name to new --> {"old_name": "new_name"} e.g {"CELL": "CellType", "Anatomy": "Tissues"}
        is_iob: Optional(bool) Whether dataset is in IOB format or not(only required for HF datasets)
        ent_label_key: the name of the key carrying the entity label. Only required for spacy_like objects
        dataset_format: Whether dataset is HF dataset or spacy_like(use spacy_like for pandas): Options["hf", "spacy_like"]
    Return: 
        Any | List[List[str]] | List[List[Dict[str, str]]]

    """
    dataset_format = dataset_format.lower().strip()
    rename_map = {k.lower().strip():v for k,v in rename_map.items()} #normalises dict keys to catch possible typing errors


    if dataset_format == "spacy_like" and ent_label_key is None:
        raise ValueError("Name of entity label key in dict is required for spacy dataset. Typically 'label' or 'labels'.")

    if dataset_format=="hf" and is_iob is None:
        raise ValueError("is_iob required if dataset is hf dataset format")

    if dataset_format not in ["hf", "spacy_like"]:
        raise ValueError("Unsupported dataset format")

    if dataset_format == "spacy_like":
        rename_func_name = __get_spacy_subtype(ent_col)
    else:
        rename_func_name = __get_hf_subtype(dataset_format, is_iob)

    rename_strategy = __RENAME_REGISTRY[rename_func_name]

    if rename_func_name.startswith("hf"):
        return rename_strategy(ent_col, rename_map)
    else:
        return rename_strategy(ent_col, rename_map, ent_label_key)




#func needs refactoring
def filter_ent(
    ent_list: List[EntityDict], 
    relevant_ent:Dict,
    ent_label_key) -> List[EntityDict]:
    """
    Keep only entities with relevant labels: CellLine, CellType, Tissue.
    Handles casing and whitespace.
    Args:
        ent_list
        relevant_ent: ....... e.g{"cellline", "celltype", "tissue"}
    """
    
    return [
        ent for ent in ent_list
        if ent.get(ent_label_key, "").strip().lower() in relevant_ent
        ]




def __rename_spacy_like_df(
    ent_col:Union[List[EntityDict], List[List[EntityDict]]], 
    rename_map:Dict[str, str], 
    ent_label_key:str
    )-> Union[List[EntityDict], List[List[EntityDict]]]:
  
  """
  Renames entity labels in a Pandas dataframe or Spacy type format with list of dicts. e.g [{"start": 22, "end": 46, "text": "non-small cell lung cancer", "labels": "Disease"},
  {"start": 52, "end": 62, "text": "BRAF V600E", "labels": "Mutation"}]
  """
  __validate_col_struc(ent_col, dict, "entities")
  
  new_ent_lst = []
  for ent_lst in ent_col:
    replaced_ent_lst = []
    for ent_dict in ent_lst:
      if ent_label_key is None:
        raise ValueError("Entity key name missing. Typically 'label' or 'labels'. Check datastructure to confirm")

      ent_label_key = ent_label_key.lower()
      old_label = ent_dict[ent_label_key]
      new_label = rename_map.get(old_label.lower().strip(), old_label)
      replaced_ent_lst.append(
          {**ent_dict, ent_label_key: new_label}
          )
      
      
    new_ent_lst.append(replaced_ent_lst)

  return new_ent_lst


def __rename_spacy_like_brat(ent_col, rename_map, ent_label_key):
    if not all(isinstance(ent, dict) for ent in ent_col):
        raise TypeError("Expected List[Dict] for BRAT-style spaCy entities")

    new_ent_lst = []

    for i, ent_dict in enumerate(ent_col):
        if ent_label_key not in ent_dict:
            raise KeyError(
                f"Missing key '{ent_label_key}' in entity at index {i}"
            )

        old_label = ent_dict[ent_label_key].strip()
        new_label = rename_map.get(old_label.lower(), old_label)

        new_ent_lst.append(
            {**ent_dict, ent_label_key: new_label}
        )

    return new_ent_lst


def __rename_hf_plain(
    ent_col:Union[DatasetColumn, List[List[str]]],
    rename_map:Dict[str, str]
    )-> Union[DatasetColumn, List[List[str]]]:

    """
    Renames a HF Dataset NER Label colum containing lists of Tags without IOB labels.
    e.g. [["Disease", "Protein"], ["Gene", "Disease", "Chemical"], ......]
    Args:
        ent_col: The dataset col containing labels which can be a list of entity labels (str) as in hf dataset format 
        rename_map: Dictionary mapping old entity names to new ones.
    """
  
    __validate_col_struc(ent_col, str, "label_list")
    new_ent_lst = [
                [
                rename_map.get(ent.lower().strip(), ent) for ent in ent_lst
                ] for ent_lst in ent_col
        ]
    return new_ent_lst


def __rename_hf_iob(
    ent_col:Union[DatasetColumn, List[List[str]]], 
    rename_map:Dict[str, str]
    ) -> Union[DatasetColumn, List[List[str]]]:

    """
    Renames a HF Dataset NER Label colum containing lists of IOB Tags.
    e.g. [["B-Disease", "O"], ["I-Gene", "I-Disease", "O"], ......]
    Args:
        ent_col: The dataset col containing labels which can be a list of entity labels (str) as in hf dataset format 
        rename_map: Dictionary mapping old entity names to new ones.
    """

    __validate_col_struc(ent_col, str, "label_list")

    new_ent_list = []
    for label_list in ent_col:
        renamed_label_list = []
        for ent in label_list:
            if ent == "O":
                renamed_label_list.append(ent)
            else:
                prefix, label = ent.split("-", 1)
                new_label = rename_map.get(label.lower().strip(), label)
                renamed_label_list.append(f"{prefix}-{new_label}")
        new_ent_list.append(renamed_label_list)

    return new_ent_list



def __validate_col_struc(
    obj:Union[DatasetColumn, List[List[str]], List[EntityDict]], 
    inner_type:Type, 
    name:str
    )-> None:
    """
    Validate if outer and inner structure of an entity column conforms to the expected format
    for each subtypes(hf datasets, spacy_like)
    Args: 
        obj: The dataset col to validate
        inner_type: Either of str(for hf dataset) or dict(for spacy_like) expected
        name: Only used for debugging purposes. Typically "entities", "hf_labels", or "label_list"

    """

    if not isinstance(obj, list):
        raise TypeError(f"Expected {name} as a list but got {type(obj).__name__}")
    
    for i, item in enumerate(obj):
        if not isinstance(item, list):
            raise TypeError(f"Expected item: {name}{[item]} as a list, but got wrong type: {type(item).__name__} at index {i}")

        for j, value in enumerate(item):
            if not isinstance(value, inner_type):
                raise TypeError(f"Expected {name}{[i]}{[j]} to be {inner_type} but got {type(value).__name__} ") 


def __get_hf_subtype(dataset_format, is_iob):
  
  if dataset_format == "hf":
    return "hf_iob" if is_iob else "hf_plain"

  else:
    raise ValueError("Unsupported dataset format")


def __get_spacy_subtype(ent_col):
    
    if not isinstance(ent_col, list):
        raise TypeError("spaCy-like data must be a list")

    if len(ent_col) == 0:
        raise ValueError("Empty spaCy entity list")

    # List[Dict] -> BRAT
    if all(isinstance(item, dict) for item in ent_col):
        return "spacy_like_brat"

    # List[List[Dict]] -> DataFrame-style
    if all(isinstance(item, list) for item in ent_col):
        return "spacy_like_df"

    raise TypeError(
        "Unsupported spaCy-like structure. Expected List[Dict] or List[List[Dict]]"
    )


__RENAME_REGISTRY = {"hf_iob": __rename_hf_iob,
               "hf_plain": __rename_hf_plain,
               "spacy_like_df": __rename_spacy_like_df,
               "spacy_like_brat": __rename_spacy_like_brat 
               }




def sentencize_and_align_entity_spans(document: str, doc_annotations, ent_label_key="label", nlp:scispacy=nlp):
    """
    Sentencizes a document and aligns entity annotations to the new sentence-level offsets.

    Args:
        document (str): The full text document.
        doc_annotations (List[Dict[str, Any]]): A list of entity dictionaries,
            each with 'start', 'end', 'label', and 'text' keys, relative to the document.
        label_field (str, optional): The key in the entity dictionary that holds the label.
            Defaults to 'label'.
        nlp_model (spacy.language.Language): The spaCy language model for sentencization.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents
        a sentence with its text and aligned entities (offsets relative to the sentence).
    """
    doc = nlp(document)
    sentence_annotations = []
    for sent in doc.sents:
        sent_start = sent.start_char
        sent_end = sent.end_char
        sent_entities = []

        for entity in doc_annotations:
            if entity["start"] >= sent_start and entity["end"] <= sent_end:
              sent_entities.append({
                 "start": entity["start"] - sent_start,
                  "end": entity["end"] - sent_start,
                  "label": entity[ent_label_key]
            })
        sentence_annotations.append({
            "sentence": sent.text,
            "entities": sent_entities
        })
    return sentence_annotations







def convert_str_2_lst(col):
    """
    Converts a string representation of a list back to an actual list.
    If the input is not a string representation of a list, it returns the input unchanged. 
    Args:
        col (a str or pandas Dataframe col): Input that may be a string representation of a list.
    Returns:
        list or any: The converted list if input was a string representation of a list, otherwise the original input.
    Args:
        col: a pandas Dataframe col or a list-like string
    Returns:
        A evaluated python list
    """

    if isinstance(col, str) and col.startswith("[") and col.endswith("]"):
        return literal_eval(col)
    else:
        return col