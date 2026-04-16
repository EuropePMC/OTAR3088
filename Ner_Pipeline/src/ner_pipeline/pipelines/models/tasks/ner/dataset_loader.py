
import random
from ast import literal_eval
from collections import Counter
from typing import (
                    List, Tuple,
                    Dict, Union
                        )
from pathlib import Path

from dataclasses import dataclass


from omegaconf import DictConfig
from hydra.utils import to_absolute_path

import pandas as pd
import numpy as np

from datasets import (
                      Dataset, DatasetDict, 
                      Sequence, Value, ClassLabel,
                      load_dataset, concatenate_datasets
                      )

from loguru import logger
from wandb.sdk.wandb_run import Run as WandbRun
from ner_pipeline.utils.common import set_seed
from ner_pipeline.utils.io.readers import read_conll
from ner_pipeline.pipelines.data.preprocessing.entity_processor import convert_str_2_lst
from .ner_factory import build_label2id_id2label
from .data_augmentation import GazetteerConfig, GazetteerAugmentationStrategy
from ...shared.factory import split_dataset
from ...shared.dataset_loader_base import DatasetLoader, PrepareDataset



def cast_to_class_labels(dataset:Dataset, label_col:str, text_col:str):
    """
    Casts dataset columns to int, primarily for classification tasks
    (e.g token classification).

    This function updates the dataset features by setting the text column to a 
    sequence of strings and the label column to a sequence of ClassLabels based 
    on the provided unique tags. It then returns the dataset with these new 
    feature schemas applied.

    Args:
        dataset (Dataset): The Hugging Face dataset to be transformed.
        label_col (str): The name of the column containing the labels/tags.
        text_col (str): The name of the column containing the input text.
        unique_tags (List[str]): A list of unique strings representing the 
            class names for the ClassLabel feature.

    Returns:
        Dataset: A new dataset object with updated feature types.
    """    
    features = dataset.features.copy()
    features[text_col] = Sequence(Value("string"))
    features[label_col] = Sequence(Value("int64"))
    return dataset.cast(features, load_from_cache_file=False)


def encode_labels(example, label_col, label2id):
    """Encode labels to IDs. Handles both single examples and batches."""
    labels = example[label_col]
    
    # Check if batched (list of lists) or single example (list)
    if labels and isinstance(labels[0], list):
        # Batched: [[tag1, tag2, ...], [tag3, tag4, ...], ...]
        example[label_col] = [[label2id[tag] for tag in label_sequence] for label_sequence in labels]
    else:
        # Single example: [tag1, tag2, ...]
        example[label_col] = [label2id[tag] for tag in labels]
    
    return example


def downsample_O_ent(batch, label_col="labels", keep_ratio):
    labels_array = batch[label_col]

    is_o_only = np.array([set(l) == {"O"} for l in labels_array])

    random_vals = np.random.rand(len(labels_array))

    keep_mask = np.where(is_o_only, random_vals < keep_ratio, True)

    return {"keep": keep_mask}


def update_counters(labels: List,
                   label_counter_iob: Counter,
                   label_counter_wo_iob: Counter) -> Tuple[Counter, Counter]:
  """
  Update counters for labels with and without IOB tags
  """
  label_counter_iob.update(labels)
  entity_labels_wo_iob = [label.split("-")[-1] if "-" in label else label for label in labels]
  label_counter_wo_iob.update(entity_labels_wo_iob)
  return label_counter_iob, label_counter_wo_iob


def count_entity_labels(dataset:Dataset, label_col:str) -> Counter:
  """
  Count instances of labels per row of Dataset
  Expects list of labels per row
  Returns: Counters of labels with and without IOB tags
  """
  label_counter_iob = Counter()
  label_counter_wo_iob = Counter()

  for labels in dataset[label_col]:
    if isinstance(labels, list):
      label_counter_iob, label_counter_wo_iob = update_counters(
         labels,
         label_counter_iob,
         label_counter_wo_iob
         )
    else:
      try:
        labels = literal_eval(labels)
        label_counter_iob, label_counter_wo_iob = update_counters(
           labels,
           label_counter_iob,
           label_counter_wo_iob
           )
      except:
        raise ValueError(f"Expected list of labels per example, got {type(labels)}")

  return label_counter_iob, label_counter_wo_iob



@dataclass
class DatasetArtifact:
  """
  Container object holding prepared datasets and label metadata
    required for downstream model training.

    Attributes
    ----------
    train_dataset : Dataset
        Token-level annotated training dataset.
    eval_dataset : Dataset
        Token-level annotated validation dataset.
    unique_tags : List[str]
        List of unique entity labels present across train and validation splits.
    label2id : Dict[str, int]
        Mapping from string labels to integer IDs.
    id2label : Dict[int, str]
        Reverse mapping from integer IDs to string labels.
  """
  train_dataset: Dataset
  eval_dataset: Dataset
  unique_tags: List[str]
  label2id: Dict[str, int]
  id2label: Dict[int, str]



class NerDatasetConfigValidator:
       
    _SUPPORTED_FILE_TYPES = {"conll", "txt", "csv", "tsv"}
    _SUPPORTED_SOURCE_TYPES= {"hf", "local"}

    @staticmethod
    def validate(cfg:DictConfig) -> None:
        data_cfg = cfg.task.data
        source_type = data_cfg.source_type

        NerDatasetConfigValidator.validate_source_type(source_type)
        NerDatasetConfigValidator.validate_columns(data_cfg.text_col,
                                                    data_cfg.label_col)

        if source_type.lower() == "hf":
            NerDatasetConfigValidator.validate_hf_source(data_cfg.hf_path)
        
        elif source_type.lower() == "local":
            NerDatasetConfigValidator.validate_local_source(data_cfg.file_type,
                                                            data_cfg.data_dir)


    @staticmethod
    def validate_source_type(source_type:str):
        if not source_type:
            raise ValueError("Source type missing from data config.\n"
                        "Source type is required for loading dataset using the appropraite method.\n"
                        "Use one of `local` or `hf` to specify.")

        if source_type not in NerDatasetConfigValidator._SUPPORTED_SOURCE_TYPES:
            raise ValueError("Invalid source_type. Supported `source_type` are:\n"
                      f"`{NerDatasetConfigValidator._SUPPORTED_SOURCE_TYPES}`")

    @staticmethod
    def validate_columns(text_col:str, label_col:str):
        if not text_col or not label_col:
            raise ValueError("Config missing dataset `text_col` or `label_col`")

    @staticmethod
    def validate_hf_source(hf_path:str):
        if not hf_path:
            raise ValueError("HuggingFace path is required when `source_type`==`hf`.\n "
                      "Example hf_path format: `OTAR3088/CeLLate1.0`")

    @staticmethod
    def validate_local_source(file_type:str, data_dir:Path):
        if not file_type:
            raise ValueError("file_type required for local datasets")
            
        if file_type not in NerDatasetConfigValidator._SUPPORTED_FILE_TYPES:
            raise ValueError(f"Unsupported file_type: {file_type}")

        if not data_dir:
            raise ValueError("Local data folder is required when `source_type`==`local`.\n"
                    "Format==/absolute/path/to/folder/in/local/")


    

class NerDatasetLoader(DatasetLoader):
    """

    """
    _SOURCE_LOADERS = {
      "hf": lambda self: self._load_hf_dataset(),
      "local": lambda self: self._load_local_dataset()
      }
      
    _FILE_PARSERS = {
        "conll": lambda self, path: self._parse_conll(path),
        "txt": lambda self, path: self._parse_conll(path),
        "csv": lambda self, path: self._parse_csv_tsv(path),
        "tsv": lambda self, path: self._parse_csv_tsv(path)
    }

    def __init__(self, cfg):
        super().__init__(cfg)

        self._validator = NerDatasetConfigValidator()
        self._validator.validate(cfg)
  
        self.text_col = getattr(cfg.task.data, "text_col")
        self.label_col = getattr(cfg.task.data, "label_col")
        
        self.data_dir = getattr(cfg.task.data, "data_dir", None)
        self.hf_path = getattr(cfg.task.data, "hf_path", None)

        source_type = getattr(cfg.task.data, "source_type")
        file_type = getattr(cfg.task.data, "file_type", None)

        self.file_type = file_type.lower() if file_type else None
        self.source_type = source_type.lower()

    
    def load(self):
        return self._SOURCE_LOADERS[self.source_type](self)

    def _load_hf_dataset(self):
        ds = load_dataset(self.hf_path, 
                          trust_remote_code=True, 
                          download_mode="force_redownload")

        return self._normalise_hf_dataset_dict(ds)

    def _load_local_dataset(self) -> DatasetDict:
        """
        Loads dataset from a directory in local path
        Returns them as a hf DatasetDict
        """

        files = self._discover_files()

        dataset_dict = {}
        for file in files:
            split = self._normalise_split_name(file.stem)
            dataset_dict[split] = self._load_single_local_file(file_path)
        return DatasetDict(dataset_dict)

    
    def _load_single_local_file(self, file_path:Path):
      
        return self._FILE_PARSERS[self.file_type](self, file_path)

    
    def _discover_files(self) -> List[Path]:
        files = list(Path(self.data_dir).glob(f"*.{self.file_type}"))
      
        if len(files) == 0:
            raise ValueError(f"No files found in {self.data_dir} with extension {self.file_type}")
      
        return files

    def _normalise_hf_dataset_dict(self, dataset_dict:DatasetDict) -> DatasetDict:
        normalised = {}
        for split, dataset in dataset_dict.items():
            normalised[self._normalise_split_name(split)] = dataset
        return DatasetDict(normalised)

    def _normalise_split_name(self, name: str) -> str:
        name = name.lower()
        known_val_names = {"validation", "val", "eval", "dev"}
        known_train_names = {"train", "training"}
    
        if any(x in name for x in known_val_names) :
            return "validation"
        if any(x in name for x in known_train_names):
            return "train"
        if "test" in name:
            return "test"
        raise ValueError(f"Unrecognized split name: {name}")

    def _parse_conll(self, file_path:Path) -> Dataset:
        
        tokens, labels = read_conll(file_path)
        
        return Dataset.from_dict({self.text_col: tokens, self.label_col: labels})

    def _parse_csv_tsv(self, file_path:Path) -> Dataset:

        sep = "\t" if self.file_type == "tsv" else ","

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
      
        has_header = self._has_header(lines[0], sep)
        data_lines = lines[1:] if has_header else lines

        if self._is_conll_format(data_lines, sep):
            return self._parse_conll(file_path)

        df = pd.read_csv(file_path, sep=sep, header=0 if has_header else None)

        if not has_header:
            if df.shape[1] < 2:
                raise ValueError(f"Tabular file {file_path} must have at least 2 columns.")
            elif df.shape[1] > 2:
                df.columns = [self.text_col, self.label_col] + list(df.columns[2:])
            else:
                df.columns = [self.text_col, self.label_col]

        for col in [self.text_col, self.label_col]:
            sample_val = df[col].iloc[0]
            if isinstance(sample_val, str) and sample_val.strip().startswith("["):
                df[col] = df[col].apply(convert_str_2_lst)

        return Dataset.from_pandas(df)


    def _has_header(self, header_line: str, sep: str) -> bool:
        known_headers = {"word", "token", "words", "tokens",
                       "ner", "ner_tag", "ner_tags", "label", "labels"}
        return any(col.lower() in known_headers for col in header_line.strip().split(sep))

    def _is_conll_format(self, data_lines: list[str], sep: str, threshold: float = 0.9) -> bool:
        non_empty = [line for line in data_lines if line.strip()]
        two_col_lines = [line for line in non_empty if len(line.strip().split(sep)) == 2]
        return (len(two_col_lines) / max(len(non_empty), 1)) > threshold



class PrepareNerDataset(PrepareDataset):
    """
        Utility class responsible for loading, validating, filtering,
        and normalising NER datasets for model training.

        This class supports loading datasets from either HuggingFace Hub
        or local filesystem sources and prepares them into a standardised
        format suitable for HuggingFace Trainer-based pipelines.
    """

    def __init__(self, cfg: DictConfig, wandb_run: WandbRun = None):
        """
        Initialise the dataset preparation utility.

        Parameters
        ----------
        cfg : DictConfig
            Configuration object containing dataset source information,
            preprocessing options, and runtime settings.
        wandb_run : wandb.sdk.wandb_run.Run, optional
            Active Weights & Biases run for logging dataset statistics.
        """
        super().__init__(cfg, wandb_run)

        dataset_loader = NerDatasetLoader(cfg)
        self.dataset = dataset_loader.load()

        self.apply_downsample = getattr(cfg.task, "apply_downsample", False)
        self.downsample_ratio = getattr(cfg.task, "downsample_ratio", 0.5)
        self.apply_augmentation = getattr(self.cfg.task, "use_data_aug", False)
        
        self.test_size = getattr(cfg.task.data, "test_size", 0.2)
        self.text_col = "tokens"
        self.label_col = "tags"



    def _require_prepared(self):
        if not hasattr(self, "_dataset_artifact"):
            raise RuntimeError(
                "Datasets have not been prepared yet. "
                "Call `prepare()` before accessing dataset properties."
            )

    @property
    def train_ent_iob(self):
        self._require_prepared()
        return self._train_ent_iob

    @property
    def eval_ent_iob(self):
        self._require_prepared()
        return self._eval_ent_iob

    @property
    def train_ent_wo_iob(self):
        self._require_prepared()
        return self._train_ent_wo_iob

    @property
    def eval_ent_wo_iob(self):
        self._require_prepared()
        return self._eval_ent_wo_iob

    @property
    def dataset_artifact(self):
        self._require_prepared()
        return self._dataset_artifact

    @property
    def train_dataset(self):
        self._require_prepared()
        return self._dataset_artifact.train_dataset

    @property
    def eval_dataset(self):
        self._require_prepared()
        return self._dataset_artifact.eval_dataset

    @property
    def unique_tags(self):
        self._require_prepared()
        return self._dataset_artifact.unique_tags

    @property
    def label2id(self):
        self._require_prepared()
        return self._dataset_artifact.label2id

    @property
    def id2label(self):
        self._require_prepared()
        return self._dataset_artifact.id2label


    def prepare(self) -> DatasetArtifact:
        """
        Executes the full dataset preparation pipeline for modelling.

        This includes loading datasets, downsampling,
        data augmentation, one-hot encoding, 
        computing label statistics, generating label mappings, normalising
        column names, and optionally logging dataset metadata to Weights & Biases.

        Returns
        -------
        DatasetKwargs
            Prepared datasets and associated label metadata ready for
            downstream model training.
        """
        if hasattr(self, "_dataset_artifact"):
            return self._dataset_artifact

        set_seed(self.cfg.seed)
        train_dataset, eval_dataset = self._get_or_create_splits()

        unique_tags = list(self.cfg.task.ner_tag_list)
        logger.info(f"Unique Tag list: {unique_tags}")

        label2id, id2label = build_label2id_id2label(self.cfg.task.ner_tag_list) 
        logger.info(f"Train dataset: {train_dataset}")
        logger.info(f"Eval dataset: {eval_dataset}")

        if self.apply_downsample:
            train_dataset = self._downsample_train_dataset(train_dataset, label2id)
            logger.info(f"Train dataset with {self.downsample_ratio*100}% `O` entity downsampling: {train_dataset}")
        

        self._train_ent_iob, self._train_ent_wo_iob = self._compute_label_stats(train_dataset)
        self._eval_ent_iob, self._eval_ent_wo_iob = self._compute_label_stats(eval_dataset)

        train_dataset = self._encode_labels_and_cast_types(train_dataset, label2id)
        eval_dataset = self._encode_labels_and_cast_types(eval_dataset, label2id)

        if self.apply_augmentation:
            train_dataset = self._augment_train_dataset(train_dataset, 
                                                        id2label=id2label,
                                                        label2id=label2id)
            logger.info(f"Train dataset after data augmentation is applied: {train_dataset}")

        
        
        self._dataset_artifact = DatasetArtifact(
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    unique_tags=unique_tags,
                    label2id=label2id,
                    id2label=id2label
        )

        self._log_to_wandb()
        
        return self._dataset_artifact


    def _normalise_columns(self, train_dataset: Dataset, eval_dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """
        Normalises dataset column names to a standard pipeline-accepted schema.

        Renames dataset columns to `tokens` and `tagss` to ensure
        compatibility with downstream NER processing utilities.

        Parameters
        ----------
        train_dataset : Dataset
            Training dataset with original column names.
        eval_dataset : Dataset
            Validation dataset with original column names.

        Returns
        -------
        Tuple[Dataset, Dataset]
            Datasets with normalised column names.
        """
        def rename(ds):
            cols = ds.column_names

            if self.text_col in cols and self.label_col in cols:
                return ds 

            if self.cfg.task.data.text_col not in cols:
                raise ValueError(f"Expected {self.cfg.task.data.text_col}"
                                f"But got {ds.column_names}")

            if self.cfg.task.data.label_col not in cols:
                raise ValueError(f"Expected {self.cfg.task.data.label_col}"
                                f"But got {ds.column_names}")

            return ds.rename_columns({
                self.cfg.task.data.text_col: self.text_col,
                self.cfg.task.data.label_col: self.label_col
            })
        
        return rename(train_dataset), rename(eval_dataset)


    def _get_or_create_splits(self):
        """
        Fetches dataset train and validation set if present
        Otherwise creates one

        """

        splits = list(self.dataset.keys())
        if "train" not in splits:
            raise ValueError("No training split found in dataset")

        
        train_dataset = self.dataset["train"]
        
        if "validation" in splits:
            eval_dataset = self.dataset["validation"]
        
        elif "test" in splits:
            logger.warning("No validation dataset found in dataset. Using test set instead")
            eval_dataset = self.dataset["test"]

        else:
            
            test_size = self.test_size if self.test_size else 0.2
            
            logger.warning(f"No validation set found in dataset. \
                       Auto-generating validation split using {test_size*100}% \
                       of training set")
           
            dataset = split_dataset(train_dataset, test_size=test_size)

            train_dataset, eval_dataset = dataset["train"], dataset["validation"]
        
        train_dataset, eval_dataset = self._normalise_columns(train_dataset, eval_dataset)
       
          
        return train_dataset, eval_dataset


    def _augment_train_dataset(self, train_dataset, id2label, label2id):

        #use gazetteer as default as other methods are yet to be implemented
        data_aug_method =  getattr(self.cfg.task, "data_aug_method", "gazetteer")
        max_entities_per_type = getattr(self.cfg.task, "max_entities_per_type", 5000)
        augment_prob = getattr(self.cfg.task, "augment_prob", 0.3)

        logger.info("Applying Gazetteer to train_dataset ----->")
        #apply gazeeteer
        gaz_config = GazetteerConfig(train_dataset,
                                text_col = self.text_col,
                                label_col = self.label_col,
                                id2label = id2label,
                                label2id = label2id,
                                max_entities_per_type = max_entities_per_type,
                                augment_prob = augment_prob
                                          )

        gazetteer_builder = GazetteerAugmentationStrategy(gaz_config)

        logger.info("generating new samples")

        augmented_train_samples = []
        num_aug = getattr(self.cfg.task, "num_gaz_aug", 3)

        for _ in range(num_aug):
            augmented_train_samples.append(gazetteer_builder.augment())

        logger.success(f"New samples generated successfully \
                        Generated dataset: {augmented_train_samples}")
        logger.info("Appending samples to train dataset")

        train_dataset = concatenate_datasets([train_dataset] + augmented_train_samples)
        logger.info(f"New train_dataset: {train_dataset}")

        return train_dataset


    def _downsample_train_dataset(self, train_dataset, label2id):
        """
        Downsample `O` entity in train dataset by a ratio of

        """
        train_dataset = train_dataset.filter(
                lambda x: downsample_O_ent(x, label_col=self.label_col, 
                keep_ratio=self.downsample_ratio)
                  )
                  
        return train_dataset


    def _encode_labels_and_cast_types(self, dataset, label2id):
        dataset = dataset.map(
            lambda x: encode_labels(x, label_col=self.label_col, 
                                    label2id=label2id), batched=True)
        dataset = cast_to_class_labels(dataset, self.label_col, self.text_col)
        return dataset


    def _compute_label_stats(self, dataset):
        return count_entity_labels(dataset, self.label_col)

    
    def _log_to_wandb(self):
        if getattr(self.cfg, "use_wandb", False) and self.wandb_run is not None:
            self.wandb_run.log({
                "Text column in dataset": self.text_col,
                "Labels column in dataset": self.label_col,
                "Unique labels in dataset": list(self.unique_tags),
            })

