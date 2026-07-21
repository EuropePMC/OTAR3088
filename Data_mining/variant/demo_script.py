import pandas as pd
import json
import sys, os

# Fix import path for Ner_Pipeline
sys.path.insert(0, os.path.normpath(os.path.join(os.getcwd(), '../../Ner_Pipeline/src/')))
from ner_pipeline.pipelines.data.preprocessing.article_normaliser import ArticleNormaliser, detect_section_headers, NERDatasetAnalyser

class DemoParams:
    text_col = "text"
    label_col = "label"
    ent_label_key = "labels"

# 1. Load the raw TSV data
df_raw = pd.read_csv("/Users/withers/Downloads/ray-variants-annotations.tsv", sep="\t")

# 2. Convert string JSON in 'label' to actual Python lists
def parse_labels(x):
    if pd.isna(x) or x == "":
        return []
    try:
        return json.loads(x)
    except:
        return []
df_raw["label"] = df_raw["label"].apply(parse_labels)

# 3. Initialise the normaliser
normaliser = ArticleNormaliser(
    params=DemoParams(),
    section_header_func=detect_section_headers,
    max_len=500
)

# 4. Run the normalisation! Process the first 50 rows of TSV so it goes fast
sample_df = df_raw.head(50)
df_normalised = normaliser.normalise(sample_df)

# 5. Let's see the result
print(f"Original massive text rows: {len(sample_df)}")
print(f"Bite-sized Natural Sentences: {len(df_normalised)}\n")

print("--- EXAMPLES OF NEW CHUNKS ---")
for i, row in df_normalised.head(3).iterrows():
    print(f"\nSentence {i+1} (Length {len(row['sentence'])} chars):")
    print(row['sentence'])
    print(f"Found Entities: {[ent['text'] for ent in row['entities']]}")

# Print basic stats from NERDatasetAnalyser
analyser = NERDatasetAnalyser(df_normalised, sent_col="sentence", ent_col="entities")
stats = analyser.compute_entity_stats()
print("\n--- DATASET HEALTH SUMMARY ---")
print(f"Total Labels Available: {stats.get('total_number_labels')}")
pd.set_option('display.max_columns', None)
print("\nLabel Distribution:")
print(stats.get('labels_count'))

