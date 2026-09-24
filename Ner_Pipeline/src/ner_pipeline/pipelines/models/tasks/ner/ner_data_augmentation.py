import random
from typing import Dict, List, Optional
from collections import defaultdict

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from datasets import Dataset


@dataclass
class GazetteerConfig:
  dataset: Dataset
  text_col: str
  label_col: str
  id2label: Dict[int, str]
  label2id: Dict[str, int]
  seed: int = 42
  external_vocab: Optional[Dict] = field(default_factory=dict)
  max_seq_len: int = 512
  augment_prob: float = 0.3
  max_entities_per_type: int = 5000


class BaseAugmentationStrategy:
    "Abstract Base class for data augmentation"

    @abstractmethod
    def augment(self) -> Dataset:
        pass

class GazetteerAugmentationStrategy(BaseAugmentationStrategy):
    """
    Gazetteer-driven data augmentation strategy for token-level NER datasets.

    This class builds an internal gazetteer from labeled BIO-formatted data
    (optionally extended with an external vocabulary) and applies stochastic
    entity replacement augmentation while preserving BIO tag consistency.
    It:
    - Ensures reproducibility via a controlled RNG
    - Supports batched and non-batched HuggingFace Dataset.map modes

    Assumes:
    - dataset[text_col] is a list of tokens
    - dataset[label_col] is a list of BIO tags (e.g. B-Tissue, I-Tissue)

    """
    def __init__(self, config):
        self.config = config
        self.dataset = config.dataset
        self.text_col = config.text_col
        self.label_col = config.label_col

        self._gazetteer: Dict[str, List[List[str]]] = defaultdict(list)
        self._built: bool = False

        # Deterministic RNG
        self._rng = random.Random(self.config.seed)
    
    def _decode_labels(self, labels: List[int]) -> List[str]:
        return [self.config.id2label[i] for i in labels]

    def _encode_labels(self, labels: List[str]) -> List[int]:
        return [self.config.label2id[i] for i in labels]

    def _build_gazetteer(self) -> None:
        """
        Builds gazetteer from dataset and optional external vocab.
        """
        if self._built:
            return
        for example in self.dataset:
            tokens = example[self.text_col]
            label_ids = example[self.label_col]
            tags = self._decode_labels(label_ids)

            current_tokens = []
            current_type = None

            for token, tag in zip(tokens, tags):
                if tag.startswith("B-"):
                    if current_tokens:
                        self._gazetteer[current_type].append(current_tokens)
                    current_tokens = [token]
                    current_type = tag[2:]

                elif tag.startswith("I-") and current_type == tag[2:]:
                    current_tokens.append(token)

                else:
                    if current_tokens:
                        self._gazetteer[current_type].append(current_tokens)
                    current_tokens = []
                    current_type = None
            if current_tokens:
                self._gazetteer[current_type].append(current_tokens)
        if self.config.external_vocab:
            self._add_external_vocab()
        
        self._deduplicate_and_cap()

        self._built = True
        
    def _add_external_vocab(self):
        "Add entities from an external vocab or dictionary if defined"
        for ent_type, values in self.config.external_vocab.items():
            for v in values:
                tokens = v.split() if isinstance(v, str) else list(v)
                self._gazetteer[ent_type].append(tokens)

    def _deduplicate_and_cap(self):
        for ent_type, entries in self._gazetteer.items():
            unique = list({tuple(e) for e in entries})
            self._rng.shuffle(unique)
            capped = unique[: self.config.max_entities_per_type]
            self._gazetteer[ent_type] = [list(e) for e in capped]

    def _augment_single(self, tokens, label_ids):
        "Augments a single example in dataset"
        if self._rng.random() > self.config.augment_prob:
            return tokens, label_ids

        tags = self._decode_labels(label_ids)

        entity_starts = [i for i, t in enumerate(tags) if t.startswith("B-")]
        if not entity_starts:
            return tokens, label_ids

        start = self._rng.choice(entity_starts)
        ent_type = tags[start][2:]

        end = start
        while (
            end + 1 < len(tags)
            and tags[end + 1].startswith("I-")
            and tags[end + 1][2:] == ent_type
        ):
            end += 1

        if ent_type not in self._gazetteer:
            return tokens, label_ids

        replacement = self._rng.choice(self._gazetteer[ent_type])

        if replacement == tokens[start : end + 1]:
            return tokens, label_ids

        new_tokens = tokens[:start] + replacement + tokens[end + 1 :]

        if len(new_tokens) > self.config.max_seq_len:
            return tokens, label_ids

        new_tags = (
            tags[:start]
            + [f"B-{ent_type}"]
            + [f"I-{ent_type}"] * (len(replacement) - 1)
            + tags[end + 1 :]
        )

        new_label_ids = self._encode_labels(new_tags)

        return new_tokens, new_label_ids

    def augment(self) -> Dataset:
        self._build_gazetteer()

        def _augment_batch(batch):
            new_tokens_batch = []
            new_labels_batch = []

            for tokens, labels in zip(batch[self.text_col], batch[self.label_col]):
                new_tokens, new_label_ids = self._augment_single(tokens, labels)
                new_tokens_batch.append(new_tokens)
                new_labels_batch.append(new_label_ids)

            return {
                self.text_col: new_tokens_batch,
                self.label_col: new_labels_batch,
            }

        return self.dataset.map(_augment_batch, batched=True)
