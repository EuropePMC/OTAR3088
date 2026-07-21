import pandas as pd
import json
from typing import List

# Screening to drop GeneProtein annotations (for now)

def parse_labels(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

def screen_annotations(labels: List, drop: str):
    return_labels = []
    for l in labels:
        if drop in l['labels']:
            continue
        else:
            return_labels.append(l)
    return return_labels

# Load LabelStudio TSV data
df_raw = pd.read_csv("/Users/withers/Downloads/ray-variants-annotations.tsv", sep="\t")
df_raw['label'] = df_raw['label'].apply(parse_labels)
df_raw['label'] = df_raw['label'].apply(screen_annotations, drop="GeneProtein")
with_res = df_raw[df_raw['label'].apply(len) > 0]
print("Rows remaining:", len(with_res))
