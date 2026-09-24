
from dataclasses import dataclass
from loguru import logger

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from ..ner_tokenization_utils import NERTokenizationTransform
from ..ner_dataset_loader import (cast_to_class_labels, 
                                  encode_labels,
                                  count_entity_labels,
                                  NERDatasetConfigValidator,
                                  NERDatasetLoader,
                                  #NERDatasetMixin
                                  )

from ....shared.dataset_loader_base import (DatasetLoader,
                                            DatasetArtifact,
                                            PrepareDataset,
                                            DatasetConfigValidator
                                            )

from ner_pipeline.utils.common import set_seed


@dataclass
class NERInferenceDatasetArtifact:
    test_dataset: Dataset


class NERInferenceDatasetLoader(NERDatasetLoader):
    """
    NER Inference Dataset Loader class. 
    Inherits from NERDatasetLoader, while modifying specific methods
    to handle inference-only datasets. 
    """
    def __init__(self, cfg):
        super().__init__(cfg)

    def _normalise_hf_dataset_dict(self, dataset_dict: DatasetDict) -> DatasetDict:
        inference_splits = [
            split for split in dataset_dict
            if any(name in split.lower() for name in ("test", "inference"))
        ]
        if not inference_splits:
            raise ValueError(
                "Inference requires a Hugging Face dataset split named "
                "`test` or `inference`. Available splits: "
                f"{list(dataset_dict)}"
            )
        dataset = DatasetDict({"test": dataset_dict[inference_splits[0]]})
        return dataset["test"]

    

class PrepareNERInferenceDataset(PrepareDataset):
    def __init__(self, cfg, label2id, wandb_run=None):
        super().__init__(cfg, wandb_run)
        self.data_cfg = getattr(cfg.task, "data", None)
        dataset_loader = NERInferenceDatasetLoader(cfg)
        self.dataset = dataset_loader.load()
        self.label2id = label2id

        self.label_all_tokens = getattr(self.cfg.task, "label_all_tokens", False)
        self.text_col = "tokens"
        self.label_col = "ner_tags"

    def _require_prepared(self):
        if not hasattr(self, "_inference_dataset_artifact"):
            raise RuntimeError(
                "Dataset has not been prepared yet."
                "\nPlease call the `.prepare()` method before accessing dataset attributes."
            )
    
    @property
    def inference_dataset_artifact(self) -> NERInferenceDatasetArtifact:
        self._require_prepared()
        return self._inference_dataset_artifact


    def prepare(self) -> NERInferenceDatasetArtifact:
        if hasattr(self, "_inference_dataset_artifact"):
            return self._inference_dataset_artifact

        set_seed(self.cfg.seed)
        dataset = self._normalise_columns(self.dataset)

        #dataset label stats
        label_stats = self._compute_label_stats(dataset)

        #encode and cast label 
        dataset = self._encode_labels_and_cast_types(dataset, self.label2id)

        #apply tokenization
        tokenized_dataset = self._apply_tokenization(dataset)

        #save prepared dataset artifact
        self._inference_dataset_artifact = NERInferenceDatasetArtifact(test_dataset=tokenized_dataset)

        return self._inference_dataset_artifact

    def _normalise_columns(self, dataset: Dataset) -> Dataset:
        """
        Normalises a dataset column names to a standard pipeline-accepted schema.

        Renames dataset columns to `tokens` and `tagss` to ensure
        compatibility with downstream NER processing utilities.

        Parameters
        ----------
        dataset : Dataset
                A HuggingFace NER dataset object
        Returns
        -------
        Dataset
            Dataset with normalised column names.
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
        
        return rename(dataset)



    def _encode_labels_and_cast_types(self, dataset, label2id):
        """
        Encodes labels to IDs 
        and casts dataset columns to appropriate huggingface dataset types.
        """
        logger.info(f"Dataset sample before casting is applied: {dataset[0]}")
        dataset = dataset.map(
            lambda x: encode_labels(x, label_col=self.label_col, 
                                    label2id=label2id), batched=True)
                                      #Transform datasets to ensure dataset labels are aligned to model labels
        logger.info("Casting Dataset from strings to integers")
    
        dataset = cast_to_class_labels(dataset, self.label_col, self.text_col)
        logger.info(f"Dataset sample after casting is applied: {dataset[0]}")

        return dataset


    def _compute_label_stats(self, dataset):
        """
        Computes label statistics for a given dataset. 
        Returns a counter dict with label and stats counts in dataset
        """
        return count_entity_labels(dataset, self.label_col)

    
    def _apply_tokenization(self, dataset):
        
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name_or_path)

        tokenizer_transform = NERTokenizationTransform(
                                    tokenizer=tokenizer,
                                    text_col=self.text_col,
                                    label_col=self.label_col,
                                    label_all_tokens=self.label_all_tokens,
                                    do_truncate=True,
                                    )
        tokenized_dataset = dataset.map(tokenizer_transform, batched=True,
                                        #remove_columns=dataset.features,
                                        remove_columns=[self.text_col, self.label_col],
                                        load_from_cache_file=False
                                        )

        return tokenized_dataset
