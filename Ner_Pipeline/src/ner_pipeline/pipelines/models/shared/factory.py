
import os
import math
from omegaconf import DictConfig
import torch.nn as nn
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments





def split_dataset(example:Dataset, test_size:float = 0.2) -> DatasetDict:
  """
  Splits a dataset into train and validation sets
  Args:
    example: A huggingface dataset class
    test_size: Ratio to split dataset by
  returns:
  A huggingface dataset with train and validation splits

  """
  example = example.train_test_split(test_size=test_size, seed=42)
  example["validation"] = example["test"]
  example.pop("test")
  return example


def format_model_checkpoint_name(ckpt:str):

    base_name = os.path.basename(ckpt)
    if not "-" in base_name:
      ckpt_name = base_name

    else:  
      base_name = base_name.split("-")
      if len(base_name) >= 3:
        ckpt_name = "_".join(base_name[:3])
      else:
        ckpt_name = "_".join(base_name)
    return ckpt_name



def build_training_args(cfg:DictConfig, output_dir:str):
  report_to = "wandb" if cfg.use_wandb else "none"
  return TrainingArguments(
      output_dir=output_dir,
      logging_dir=f"{output_dir}/logs",
      report_to = report_to,
      **cfg.task.args
  )


def compute_training_steps(args:TrainingArguments, train_dataset:Dataset):

  """
  Calculate total training steps based on dataset size, batch size, gradient accumulation steps and number of epochs
  Args:
    args: TrainingArguments object
    train_dataset: Huggingface dataset object
  Returns:
    Total number of training steps as integer
  """
  # Effective batch size accounts for accumulation and number of devices
  effective_batch_size = (
      args.per_device_train_batch_size *
      args.gradient_accumulation_steps *
      max(1, torch.cuda.device_count())  # default 1 if no GPU
  )
  #get total num steps per epoch
  num_update_steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)

  #get total training steps
  num_training_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
  
  return num_training_steps



def compute_warmup_steps(num_training_steps, warmup_ratio):
    "Return warmup steps given a ratio (e.g., 0.1 = 10%)."
    return int(num_training_steps * warmup_ratio)


def extract_model_backbone(model: nn.Module):
    """
    Return (backbone_name, backbone) for common architectures.
    """
    supported = {"bert", "roberta", "distilbert"}
    if hasattr(model, "bert"):
        return "bert", model.bert
    if hasattr(model, "roberta"):
        return "roberta", model.roberta
    if hasattr(model, "distilbert"):
        return "distilbert", model.distilbert
    raise ValueError("Unsupported Bert backbone. Inspect model to find name of backbone.",
                    f"Supported backbones are: {supported}")



def extract_encoder_layers(model: nn.Module):
  "Finds encoder layer for BERT model variants"
  model_name, backbone = extract_model_backbone(model)
  if model_name == "bert":
    encoder_layer = backbone.encoder.layer
  elif model_name == "roberta":
    encoder_layer = backbone.encoder.layer
  elif model_name == "distilbert":
    encoder_layer = backbone.transformer.layer
  return encoder_layer


def count_trainable_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)