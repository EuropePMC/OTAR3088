import pandas as pd
import ast
import spacy
import json

import sys, os
sys.path.insert(0, os.path.normpath(os.path.join(os.getcwd(), '../../Ner_Pipeline/src/')))
from ner_pipeline.pipelines.data.preprocessing.iob_converter import SpacyIOBConverter
from ner_pipeline.schemas.ner_dataset import IOBConfig, RawNerSchema, nlp

df = pd.read_csv("variant_ner_dataset.csv")

def parse_labels(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

df["entities"] = df["entities"].apply(parse_labels)

schema = RawNerSchema(
    text_col="sentence",
    entity_col="entities",
    ent_label_key="label"
)

config = IOBConfig(
    schema=schema,
    tokenizer_backend=nlp,
    as_hf_dataset=True
)

converter = SpacyIOBConverter(data=df, config=config)
hf_dataset = converter.convert()

# Export for Review
output_file = "variant_iob_dataset.jsonl"
hf_dataset.to_json(output_file, orient="records", lines=True)
print(f"Successfully saved Hugging Face dataset to {output_file} for review!")
