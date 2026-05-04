
from __future__ import annotations

import requests
from requests.exceptions import HTTPError, RequestException
import functools
import tarfile
from pathlib import Path
from typing import Optional, Union, List, Dict
import re
import os
import sys
import random
from ast import literal_eval


import torch
from datasets import Dataset, DatasetDict


import numpy as np
import pandas as pd

from loguru import logger
from omegaconf import DictConfig
import wandb


##missing typ int and docstrings for some func
##Todo

def catch_request_errors(func):
    """
    Wrapper function to catch request errors
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HTTPError as http_err:
            print(f"HTTP error in {func.__name__}: {http_err}")
        except RequestException as req_err:
            print(f"Request error in {func.__name__}: {req_err}")
        return None
    return wrapper



           
def extract_tar_gz_local(file_path: Path, extract_to: Optional[Path] = None):
    """
    Extracts a .tar.gz archive to the specified directory.

    Args:
        file_path (Path): Path to the .tar.gz file.
        extract_to (Optional[Path]): Directory to extract into. Defaults to a folder with the same name as the archive.
    """
    if extract_to is None:
        extract_to = file_path.with_suffix('').with_suffix('')  # Remove both .gz and .tar
    extract_to.mkdir(parents=True, exist_ok=True)

    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print(f"Extracted {file_path.name} → {extract_to}")


def clean_text(text:str) -> str:
    """
    This function cleans a text by filtering reference patterns in text, 
    extra whitespaces, escaped latex-style formatting appearing in text body instead of predefined latex tags

    Args: 
    text(str): The text to be cleaned
    
    Returns: 
    tex(str): The cleaned text 
    
    """
   
    # Remove LaTeX-style math and formatting tags #already filtered from soup content but some still appear
    text = re.sub(r"\{.*?\}", "", text)  # Matches and removes anything inside curly braces {}
    text = re.sub(r"\\[a-zA-Z]+", "", text)  # Matches and removes characters that appears with numbers
    
    # Remove reference tags like [34] or [1,2,3]
    text = re.sub(r"\[\s*(\d+\s*(,\s*\d+\s*)*)\]", "", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text



def set_seed(seed: int):
    """Ensure reproducibility across Python, NumPy, and PyTorch"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def create_output_dir(base_path:str,  
                      *, 
                      name:str=None,
                      is_model:bool=True, 
                      is_datasets:bool=False,
                      include_type_dir:bool=True,
                      experiment_subfolder: str = None):
  
  
    """
    Creates output directory for saving model outputs or datasets.
    Parameters:
        base_path (str): Base directory to create outputs in.
        name (str): Name of the output directory (e.g model or dataset name/version).
        is_model (bool): If True, creates under 'model_outputs' if a model type folder.
        is_datasets (bool): If True, creates under 'Datasets' if dataset type folder.
        include_type_dir (bool): Whether to create under a subfolder for a model/dataset type.
        experiment_subfolder (str, optional): 
            Optional sub-directory under 'model_outputs' or 'Datasets' 
            for organizing runs (e.g., different training strategies or experiments).

    Returns:
        Path: Path object of the created directory.

    """
    if not is_model and not is_datasets:
        raise ValueError("Either `is_model` or `is_datasets` must be True.")
    if is_model and is_datasets:
        raise ValueError("Only one of `is_model` or `is_datasets` can be True at a time.")
    if name is None and experiment_subfolder is None:
        raise ValueError("Either Name or experiment subfolder must be provided")
  

    try:
        base_path = Path(base_path)
        base_type = Path("model_outputs") if is_model else Path("Datasets")
    except TypeError:
        raise ValueError("base_path must be a valid str or path")

    subfolder = base_type if include_type_dir else Path()
    if experiment_subfolder:
        subfolder /= experiment_subfolder
  
  
    try:
        output_dir = base_path / subfolder if experiment_subfolder else base_path / subfolder / name
        output_dir.mkdir(parents=True, exist_ok=True)
    
    except Exception as e:
        raise RuntimeError(f"Failed to create output directory at {output_dir}") from e
    print(f"Output directory created at {output_dir}")
    return output_dir



def setup_loguru(config: Optional[DictConfig]):
    "Setup loguru to a central directory if specified in hydra config or default to current directory(useful for inference only)"
    log_dir = Path(config.loguru.log_dir) if config and "loguru" in config else Path.cwd()
    print(log_dir)
    log_filename = config.loguru.log_filename if config and "loguru" in config else "run.log"

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_filename

    logger.remove(0)
    logger.add(
        log_path,
        format="[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] | <level>{level}</level> | <cyan>{message}</cyan>",
        mode="w",

    level=config.loguru.level
    )
    logger.success(f"Loguru initialised at: {log_path}")

    

def convert_str_2_lst(col):
    """
    Converts a string representation of a list back to an actual list.
    If the input is not a string representation of a list, it returns the input unchanged. 
    Args:
        col (a str or pandas Dataframe col): Input that may be a string representation of a list.
    Returns:
        list or any: The converted list if input was a string representation of a list, otherwise the original input.
    Args:
        col: a pandas Dataframe col or a list-like string
    Returns:
        A evaluated python list
    """

    if isinstance(col, str) and col.startswith("[") and col.endswith("]"):
        return literal_eval(col)
    else:
        return col


def inherit_docstring(parent_class):
  def fetch_docstring(child_class):
    if parent_class.__doc__:
        child_doc = child_class.__doc__ if child_class.__doc__ else "" 
        child_class.__doc__ = f"{parent_class.__name__}({parent_class.__doc__})\n\n{child_doc}"
    return child_class
  return fetch_docstring