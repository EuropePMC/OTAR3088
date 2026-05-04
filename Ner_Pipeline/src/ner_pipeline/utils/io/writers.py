from typing import List, Dict, Optional, Union
from pathlib import Path




def write_to_conll(
    sentences: Union[List[List[Dict[str, str]]], List[Dict[str, Union[List[str], List[str]]]]],
    text_col: str,
    label_col:str,
    output_path: Optional[str] = None,
    file_name: Optional[str] = None
) -> None:
    """
    Writes token-label pairs in CoNLL format to a file.
    Supports both:
        - List[List[Dict[str, str]]] with keys 'text_col(e.g "tokens")' and 'label_col(e.g "ner_tags")'
        - List[Dict] format 

    Args:
        sentences: Input data (either token dicts or HuggingFace-style format).
        output_path: Directory where file will be saved. Defaults to 'outputs/'.
        file_name: Name of output file (without extension). Defaults to 'conll_file'.

    Returns:
        None
    """
    output_path = Path(output_path) if output_path else Path("outputs")
    output_path.mkdir(parents=True, exist_ok=True)

    if not file_name:
        file_name = "conll_file"
    output_file = output_path / f"{file_name}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in sentences:
            # Case 1: {'words': [...], 'labels': [...]} --> HF Dataset format
            if isinstance(sentence, dict) and text_col in sentence and label_col in sentence:
                for word, label in zip(sentence[text_col], sentence[label_col]):
                    f.write(f"{word}\t{label}\n")
            # Case 2: [{'text': ..., 'label': ...}, ...] --> Other dataset annotation syle
            elif isinstance(sentence, list) and all(isinstance(token, dict) for token in sentence):
                for token in sentence:
                    f.write(f"{token[text_col]}\t{token[label_col]}\n")
            else:
                raise ValueError("Unsupported dataset format.")
            f.write("\n")  # Sentence boundary

