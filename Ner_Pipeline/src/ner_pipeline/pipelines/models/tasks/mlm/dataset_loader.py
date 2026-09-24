from itertools import chain
from typing import List, Tuple
from dataclasses import dataclass

from pathlib import Path
from omegaconf import DictConfig

from loguru import logger
from wandb.sdk.wandb_run import Run as WandbRun

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from .tokenization_utils import MLMTokenizationTransform

from ...shared.factory import split_dataset
from ...shared.dataset_loader_base import (DatasetLoader,
                                           DatasetArtifact,
                                           PrepareDataset,
                                           DatasetConfigValidator)


from ner_pipeline.utils.common import set_seed


@dataclass(frozen=True)
class MLMDatasetArtifact(DatasetArtifact):
    pass



class MLMDatasetConfigValidator(DatasetConfigValidator):
    """
    Dataset config Validator for Masked Language Modelling task(MLM)

    """
    
    @classmethod
    def validate_task_config(cls, data_cfg:DictConfig) -> None:
        text_col = getattr(data_cfg, "text_col", None)
        if not text_col:
            raise ValueError("MLM dataset config must include a `text_col`.")




class MLMDatasetLoader(DatasetLoader):
    """
    Dataset loader class for Masked Language Modeling tasks. 
    Inherits from DatasetLoader.
    Implements task-specific functionality.
    """
    
    _SOURCE_LOADERS = {
      "hf": lambda self: self._load_from_hf(),
      "local": lambda self: self._load_from_local()
      }

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._validator = MLMDatasetConfigValidator()
        self._validator.validate(cfg)
        self.text_col = getattr(cfg.task.data, "text_col")
        self.data_dir = getattr(cfg.task.data, "data_dir", None)
        self.hf_path = getattr(cfg.task.data, "hf_path", None)

        source_type = getattr(cfg.task.data, "source_type")
        file_type = getattr(cfg.task.data, "file_type", None)

        self.file_type = file_type.lower() if file_type else None
        self.source_type = source_type.lower()

    
    def load(self):
        """
        Loads the dataset based on the source type specified in the configuration.
        Returns:
            DatasetDict: A dictionary containing the loaded datasets for each split.
        """
        
        return self._SOURCE_LOADERS[self.source_type](self)

    
    def _load_from_hf(self) -> DatasetDict:
        """
        Loads the dataset from Hugging Face datasets based on the hf_path specified in the configuration.
        Returns:
            DatasetDict: A dictionary containing the loaded datasets for each split.
        """
        logger.info(f"loading dataset from HF from path: {self.hf_path}")
        dataset = load_dataset(self.hf_path)
        dataset = self._normalise_hf_dataset_dict(dataset)
        return dataset

    
    def _load_from_local(self):
        """
        Loads dataset from a directory in local path
        Returns them as a hf DatasetDict
        """
        logger.info(f"loading dataset from local from directory: {self.data_dir}")
        files = self._discover_files()

        data_files = {}
        for file in files:
            split = self._normalise_split_name(file.stem)
            data_files[split] = str(file)
        dataset = load_dataset(self.file_type, data_files=data_files)
        
        return dataset

    
    def _discover_files(self) -> List[Path]:
        files = list(Path(self.data_dir).glob(f"*.{self.file_type}"))
      
        if len(files) == 0:
            raise ValueError(f"No files found in {self.data_dir} with extension {self.file_type}")
      
        return files

    
    def _normalise_split_name(self, name: str) -> str:
        name = name.lower()
        known_val_names = {"validation", "val", "eval", "dev", "test"}
        known_train_names = {"train", "training"}
    
        if any(x in name for x in known_val_names) :
            return "validation"
        if any(x in name for x in known_train_names):
            return "train"

        if "test" in name:
            return "test"

        raise ValueError(f"Unrecognized split name: {name}")

    
    def _normalise_hf_dataset_dict(self, dataset_dict:DatasetDict) -> DatasetDict:
        normalised_ds = {}
        for split, dataset in dataset_dict.items():
            normalised_ds[self._normalise_split_name(split)] = dataset
   
        return DatasetDict(normalised_ds)


class PrepareMLMDataset(PrepareDataset):
    """
    """
    def __init__(self, cfg: DictConfig, wandb_run: WandbRun = None):
        super().__init__(cfg, wandb_run)

        dataset_loader = MLMDatasetLoader(cfg)
        self.dataset = dataset_loader.load()
        data_cfg = getattr(cfg.task, "data")
        self.test_size = getattr(data_cfg, "test_size", 0.2)
        self.text_col = getattr(data_cfg, "text_col")
        self.do_group_text = getattr(cfg.task, "do_group_text", False)

    
    def _require_prepared(self):
        if not hasattr(self, "_mlm_dataset_artifact"):
            raise RuntimeError(
                "Datasets have not been prepared yet. "
                "Call `prepare()` before accessing dataset properties."
            )
    
    
    @property
    def mlm_dataset_artifact(self):
        self._require_prepared()
        return self._mlm_dataset_artifact

    @property
    def train_dataset(self):
        self._require_prepared()
        return self._mlm_dataset_artifact.train_dataset

    @property
    def eval_dataset(self):
        self._require_prepared()
        return self._mlm_dataset_artifact.eval_dataset
    
    def prepare(self):
        if hasattr(self, "_mlm_dataset_artifact"):
            return self._mlm_dataset_artifact
        
        set_seed(self.cfg.seed)

        logger.info("Preparing MLM dataset")
        train_dataset, eval_dataset = self._get_or_create_splits()

        #tokenize datasets
        tokenized_train = self._apply_tokenization(train_dataset)
        tokenized_eval = self._apply_tokenization(eval_dataset)

        #save dataset artifacts
        self._mlm_dataset_artifact = MLMDatasetArtifact(
                                            train_dataset=tokenized_train,
                                            eval_dataset=tokenized_eval
                                            )

        return self._mlm_dataset_artifact
    
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
       
          
        return train_dataset, eval_dataset

    
    def _apply_tokenization(self, dataset):
        if self.cfg.task.tokenizer_name_or_path:
            ckpt_path = self.cfg.task.tokenizer_name_or_path
        else:
            ckpt_path = self.cfg.task.model_name_or_path 
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        tokenizer_transform = MLMTokenizationTransform(
                                    tokenizer=tokenizer,
                                    text_col=self.text_col,
                                    do_truncate=False if self.do_group_text else True,
                                    )
        logger.info(f"Dataset before tokenization: {dataset}")
        tokenized_dataset = dataset.map(tokenizer_transform, batched=True,
                                        remove_columns=dataset.features,
                                        load_from_cache_file=False
                                        )
        logger.info(f"Dataset after tokenization: {tokenized_dataset}")                              

        
        
        if self.do_group_text:
            tokenized_dataset = tokenized_dataset.map(self._group_texts,
                                                batched=True,
                                                batch_size=len(tokenized_dataset),
                                                desc="Applying group text to dataset"
                                                )
            logger.info(f"Tokenized dataset of group texts: {tokenized_dataset}")
        tokenized_dataset = tokenized_dataset.filter(
                                    lambda example: len(example["input_ids"]) > 2
                                                    )
        
        return tokenized_dataset

    def _group_texts(self, examples, max_seq_len: int = 512):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // max_seq_len) * max_seq_len
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_len] for i in range(0, total_length, max_seq_len)]
            for k, t in concatenated_examples.items()
        }
        return result