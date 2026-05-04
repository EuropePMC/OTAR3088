import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional
import logging

#load_brat------> if loading a single file, pass both file path and annotation file path. 
# If loading multiple files, then pass only directory path. 
# The function will take care of reading the files and their respective annotations using the file id name 



#refactor load_brat func
def load_brat(input_path: Union[str, Path], ann_path: Optional[str] = None):
    input_path = Path(input_path)
    dataset = []

    if input_path.is_file() and input_path.suffix == ".txt":
        ann_path = input_path.with_suffix(".ann")
        if ann_path.exists():
            file_id = input_path.stem
            full_text, entities = _read_brat(input_path, ann_path)
            dataset.append({"id":file_id, "doc": full_text, "entities": entities})
        else:
            raise FileNotFoundError(f"Annotation file not found for {input_path}")

    elif input_path.is_dir():
        for file in os.listdir(input_path):
            if file.endswith(".txt"):
                file_id = file[:-4]
                text_file = input_path / f"{file_id}.txt"
                ann_file = input_path / f"{file_id}.ann"
                if text_file.exists() and ann_file.exists():
                    full_text, entities = _read_brat(text_file, ann_file)
                    # Check if _read_brat returned the expected values
                    if full_text is not None and entities is not None:
                        dataset.append({"text": full_text, "entities": entities})
                    else:
                        raise FileNotFoundError(f"Skipping {file_id} due to error in _read_brat")
                else:
                    raise ValueError(f"Missing .txt or .ann for file ID: {file_id}")
    else:
        raise FileNotFoundError(f"Invalid input path: {input_path}")

    return dataset


def _read_brat(txt_path: Path, ann_path: Optional[str] = None):
    try:
      file = Path(txt_path)
      full_text = file.read_text()

      entities = []
      with open(ann_path, "r", encoding="utf-8") as f:
          for line in f:
              line = line.strip()
              if not line.startswith("T"):
                  continue
              parts = line.strip().split("\t")
              if len(parts) < 3:
                  continue
              entity_info, entity_text = parts[1], parts[2]

              # Some annotations have spans like: "Entity 0 10;12 15"
              try:
                  span = entity_info.split()
                  entity_label = span[0]
                  start = span[1]
                  end = span[-1]
              except Exception as e:
                  raise ValueError(f"Skipping malformed span: {entity_info}")
                  continue

              entities.append({
                  "text": entity_text,
                  "label": entity_label,
                  "start": int(start),
                  "end": int(end)
                  })

      return full_text, entities

    except Exception as e:
        raise ValueError(f"Error reading {txt_path.name}: {e}")
        return None, [] # Return None for both to signal an error




def read_conll(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Reads a CoNLL-formatted file and extracts tokens, labels grouped by sentences.

    Args:
        file_path (str): Path to the CoNLL .txt file.

    Returns:
        Tuple[List[List[str]], List[List[str]]]: 
            - A list of token sequences per sentence.
            - A corresponding list of label sequences per sentence.

    Raises:
        ValueError: If a line format is invalid (missing token or label).
    """
    all_tokens, all_labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        sentence_tokens, sentence_labels = [], []

        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                # End of sentence
                if sentence_tokens and sentence_labels:
                    all_tokens.append(sentence_tokens)
                    all_labels.append(sentence_labels)
                    sentence_tokens, sentence_labels = [], []
            else:
                #parts = re.split(r'\s+', line)
                parts = line.split()
                if len(parts) >= 2:
                    sentence_tokens.append(parts[0])   # First part = token
                    sentence_labels.append(parts[-1])  # Last part = label
                else:
                    raise ValueError(f"Invalid line format on line {i + 1}: '{line}'")
                

        # Handle last sentence if file doesn't end with newline
        if sentence_tokens:
            all_tokens.append(sentence_tokens)
            all_labels.append(sentence_labels)

    return all_tokens, all_labels



