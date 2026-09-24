
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase, DataCollatorWithPadding

from ...shared.factory import BaseTokenizationTransform



@dataclass
class MLMTokenizationTransform(BaseTokenizationTransform):
    stride: int = 128
    return_attention_mask: bool = True
    return_special_tokens_mask: bool = True
    return_overflowing_tokens: bool = True

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        tokenizer_kwargs = {
            "truncation": self.do_truncate,
            "max_length": self.max_length,
            "stride": self.stride,
            "return_attention_mask": self.return_attention_mask,
            "return_special_tokens_mask": self.return_special_tokens_mask,
        }

        return self.tokenizer(batch[self.text_col],
                                **tokenizer_kwargs)




@dataclass
class DataCollatorForSpanMasking:
    """
    Span masking collator for standard BERT/RoBERTa MLM.

    Characteristics
    ---------------
    • Approximately 15% of valid tokens are masked.
    • Contiguous spans.
    • Geometric span-length distribution.
    • No overlapping spans.
    • Standard MLM labels.
    • Span-level 80/10/10 replacement.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    geometric_p: float = 0.26
    max_span_length: int = 10

    def _sample_span_length(self) -> int:
        """Sample a truncated geometric span length."""

        length = 1

        while (
            length < self.max_span_length
            and torch.rand(()) > self.geometric_p
        ):
            length += 1

        return length

    def _get_special_tokens_mask(
        self,
        input_ids: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        if "special_tokens_mask" in batch:
            return batch["special_tokens_mask"].bool()

        masks = [
            self.tokenizer.get_special_tokens_mask(
                ids.tolist(),
                already_has_special_tokens=True,
            )
            for ids in input_ids
        ]

        return torch.tensor(
            masks,
            dtype=torch.bool,
            device=input_ids.device,
        )

    def _build_span_mask(
        self,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor,
    ) -> torch.Tensor:

        valid = attention_mask & (~special_tokens_mask)

        valid_positions = valid.nonzero(as_tuple=False).flatten()

        if valid_positions.numel() == 0:
            return torch.zeros_like(valid), []

        budget = max(
            1,
            round(valid_positions.numel() * self.mlm_probability),
        )

        mask = torch.zeros_like(valid)

        available = valid.clone()

        spans: List[List[int]] = []

        masked = 0

        while masked < budget:

            candidates = available.nonzero(as_tuple=False).flatten()

            if candidates.numel() == 0:
                break

            start = candidates[
                torch.randint(
                    candidates.numel(),
                    (1,),
                )
            ].item()

            target_span_length = self._sample_span_length()

            span = []

            for pos in range(start, valid.size(0)):

                if len(span) >= target_span_length:
                    break

                if not available[pos]:
                    break

                span.append(pos)

            remaining = budget - masked

            span = span[:remaining]

            if not span:
                available[start] = False
                continue

            mask[span] = True
            available[span] = False

            spans.append(span)

            masked += len(span)

        return mask

    def _apply_mlm_replacement(self,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
        vocab_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the standard BERT MLM replacement strategy.

        The span mask determines *which* tokens are predicted.
        Replacement itself remains token-level:

            80% -> [MASK]
            10% -> random token
            10% -> unchanged
        """

        masked_positions = mask.nonzero(as_tuple=False).flatten()

        if masked_positions.numel() == 0:
            return input_ids

        probs = torch.rand(
            masked_positions.numel(),
            device=input_ids.device,
        )

        # 80% -> [MASK]
        mask_positions = masked_positions[probs < 0.8]
        input_ids[mask_positions] = self.tokenizer.mask_token_id

        # 10% -> random token
        random_positions = masked_positions[
            (probs >= 0.8) & (probs < 0.9)
        ]

        if random_positions.numel():

            random_ids = vocab_candidates[
                torch.randint(
                    vocab_candidates.numel(),
                    (random_positions.numel(),),
                    device=input_ids.device,
                )
            ]

            input_ids[random_positions] = random_ids

        # Remaining 10% are left unchanged

        return input_ids

    def __call__(
        self,
        features: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:

        batch = self.tokenizer.pad(
            features,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]

        labels = input_ids.clone()

        if self.tokenizer.mask_token_id is None:
            raise ValueError(
                "Tokenizer must define mask_token_id."
            )

        attention_mask = batch.get(
            "attention_mask",
            torch.ones_like(input_ids),
        ).bool()

        special_tokens_mask = self._get_special_tokens_mask(
            input_ids,
            batch,
        )

        special_ids = set(self.tokenizer.all_special_ids)

        vocab_candidates = torch.tensor(
            [
                idx
                for idx in range(len(self.tokenizer))
                if idx not in special_ids
            ],
            device=input_ids.device,
            dtype=torch.long,
        )

        for i in range(input_ids.size(0)):

            mask = self._build_span_mask(
                attention_mask[i],
                special_tokens_mask[i],
            )

            labels[i] = -100
            labels[i, mask] = input_ids[i, mask]

            input_ids[i] = self._apply_mlm_replacement(input_ids[i],
                                                        mask,
                                                        vocab_candidates,
                                                    )

        batch["input_ids"] = input_ids
        batch["labels"] = labels

        return batch



class MLMMaskingMixin:
    def _add_masking_method(self, task_cfg):
        use_whole_word_mask = getattr(task_cfg, "use_whole_word_mask", False)
        use_span_mask = getattr(task_cfg, "use_span_mask", False)
        
        if use_whole_word_mask:
            return "WholeWordMasking"
        elif use_span_mask:
            return "SpanMasking"
        else:
            return "StandardMasking"