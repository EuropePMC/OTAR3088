from typing import Union, List
from datasets import Dataset
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast


def _shift_label(label):
    if label % 2 == 1:
      label += 1
    return label


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
      if word_id is None:
        new_labels.append(-100)
      elif word_id != current_word:
        current_word = word_id # Start of a new word!
        new_labels.append(labels[word_id])
      else:
        new_labels.append(_shift_label(labels[word_id]))
    return new_labels


def tokenize_and_align(example: Dataset,
                       tokenizer: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast],
                       block_size: int = 512,
                       text_col: str = 'tokens',
                       label_col: str ='tags'):
    

    
    tokenized_inputs = tokenizer(
        example[text_col],
        max_length=block_size,
        truncation=True,
        is_split_into_words=True
    )
    new_labels = []

    for i, labels in enumerate(example[label_col]):
      word_ids = tokenized_inputs.word_ids(i)
      new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs['labels'] = new_labels
    return tokenized_inputs