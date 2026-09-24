

from typing import Union
from dataclasses import dataclass

from loguru import logger

import torch
import torch.nn as nn

from transformers import (PreTrainedTokenizerBase, 
                          PreTrainedTokenizerFast, 
                          Trainer,
                          AutoTokenizer,
                          AutoModelForMaskedLM,)

from .mlm_metrics import compute_perplexity
from ...shared.modelling_base import BuildModel
from ...shared.trainer_config_base import HFModelConfig


@dataclass(kw_only=True)
class MLMModelConfig(HFModelConfig):
    tokenizer_checkpoint: str


def resize_model_embeddings(model, tokenizer):
    ""
    return model.resize_token_embeddings(len(tokenizer))
 


def initialise_new_embeddings(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast],
    init_strategy: str = "match_old",  # ["match_old", "normal"]
):
    """
    Reinitialise embeddings for newly added tokens in-place.

    Assumes:
    - tokenizer has been extended BEFORE calling this
    - model.resize_token_embeddings(len(tokenizer)) has been called

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizer
    init_strategy : str
        - "match_old": match mean/std of original embeddings (recommended)
        - "normal": N(0, initializer_range)
    """

    embedding_layer = model.get_input_embeddings()
    embedding_weight = embedding_layer.weight

    old_vocab_size = embedding_weight.shape[0] - (
        len(tokenizer) - tokenizer.vocab_size
    )
    new_vocab_size = embedding_weight.shape[0]

    if new_vocab_size <= old_vocab_size:
        logger.info("No new tokens detected. Skipping embedding reinitialisation.")
        return model

    new_token_ids = list(range(old_vocab_size, new_vocab_size))
    logger.info(f"Reinitialising {len(new_token_ids)} new token embeddings")

    with torch.no_grad():
        if init_strategy == "match_old":
            old_embs = embedding_weight[:old_vocab_size]
            mean = old_embs.mean(dim=0)
            std = old_embs.std(dim=0)

            embedding_weight[new_token_ids] = (
                torch.randn_like(embedding_weight[new_token_ids]) * std + mean
            )

        elif init_strategy == "normal":
            initializer_range = getattr(
                model.config, "initializer_range", 0.02
            )
            embedding_weight[new_token_ids].normal_(
                mean=0.0, std=initializer_range
            )

        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")

    logger.success(
        f"New embeddings initialised | std={embedding_weight[new_token_ids].std().item():.4f}"
    )

    return model


class BuildMLMModel(BuildModel):
    def __init__(self, config: MLMModelConfig, build_for_hyperparam_tuning:bool=False):
        super().__init__(config, build_for_hyperparam_tuning)
        self.config = config
        self.build_for_hyperparam_tuning = build_for_hyperparam_tuning
        self.tokenizer_checkpoint = config.tokenizer_checkpoint
    
    def build(self):
        builder = self._build_for_standard
        if self.build_for_hyperparam_tuning:
            builder = self._build_for_model_init(builder)
            
        return builder()

    def _build_for_standard(self):
        model = AutoModelForMaskedLM.from_pretrained(self.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_checkpoint)
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
            model = initialise_new_embeddings(model, tokenizer)
        
        return model

    def _build_for_model_init(self, builder):
        """
        Returns a model initialisation function compatible with Hugging Face Trainer.
        
        Usage:
            trainer = Trainer(
                model_init=model_init_func,
                ...
            )
        """
        def init(trial=None):
            return builder()
        return init



class MLMBaseTrainer(Trainer):
    """
    Trainer subclass that logs MLM perplexity during training/evaluation.

    Adds:
    - train_perplexity from logged training loss
    - eval_perplexity from eval_loss
    """

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        logs = dict(logs)  # avoid mutating caller-owned dict unexpectedly

        if "loss" in logs:
            logs["train_perplexity"] = compute_perplexity(logs["loss"])

        if "eval_loss" in logs:
            logs["eval_perplexity"] = compute_perplexity(logs["eval_loss"])

        return super().log(logs, *args, **kwargs)




