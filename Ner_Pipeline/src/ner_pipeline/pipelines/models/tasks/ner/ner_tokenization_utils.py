from typing import Union, List
from dataclasses import dataclass

from datasets import Dataset
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from ...shared.factory import BaseTokenizationTransform



@dataclass
class NERTokenizationTransform(BaseTokenizationTransform):
    label_col: str = "tags"
    label_all_tokens: bool = False
    is_split_into_words: bool = True
    

    def __call__(self, batch):
        tokenizer_kwargs = {
          "truncation": self.do_truncate,
          "is_split_into_words": self.is_split_into_words,
          "max_length": self.max_length,
          }
        tokenized_inputs = self.tokenizer(batch[self.text_col],
                                            **tokenizer_kwargs)

        all_labels = batch[self.label_col]
        aligned_labels = []

        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)

            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None: #special token
                    label_ids.append(-100)

                elif word_idx != previous_word_idx:
                    #start of a new word
                    label_ids.append(labels[word_idx])
                
                else:
                    if self.label_all_tokens:

                        label = labels[word_idx]
                        if label % 2 == 1:
                            label += 1
                        label_ids.append(label)

                    else:
                        label_ids.append(-100)

                previous_word_idx = word_idx

            aligned_labels.append(label_ids)


        tokenized_inputs['labels'] = aligned_labels
        return tokenized_inputs





# def tokenize_and_align(example: Dataset,
#                        tokenizer: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast],
#                        block_size: int = 512,
#                        text_col: str = 'tokens',
#                        label_col: str ='tags'):
    

    
#     tokenized_inputs = tokenizer(
#         example[text_col],
#         max_length=block_size,
#         truncation=True,
#         is_split_into_words=True
#     )
#     new_labels = []

#     for i, labels in enumerate(example[label_col]):
#       word_ids = tokenized_inputs.word_ids(i)
#       new_labels.append(align_labels_with_tokens(labels, word_ids))

#     tokenized_inputs['labels'] = new_labels
#     return tokenized_inputs