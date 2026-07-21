import json
import os
import pandas as pd
import re
import spacy

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import sys
    print("spaCy model not found. Please ensure it is installed.")
    sys.exit(1)

input_path = '/Users/withers/Downloads/combined_iaa.tsv'
output_path = '/Users/withers/Downloads/combined_iaa.tsv'

if not os.path.exists(input_path):
    print(f"Error: Input file {input_path} does not exist.")
    sys.exit(1)

df = pd.read_csv(input_path, sep='\t')

def parse_entities(entities_str):
    if pd.isna(entities_str) or not isinstance(entities_str, str) or entities_str.strip() in ('', '[]', 'nan'):
        return []
    try:
        return json.loads(entities_str)
    except Exception:
        import ast
        try:
            return ast.literal_eval(entities_str)
        except Exception as e:
            print(f"Error parsing entities: {entities_str}, error: {e}")
            return []

new_rows = []
unmatched_entities_count = 0
total_entities_count = 0

for index, row in df.iterrows():
    pmcid = row['PMCID']
    text = str(row['sentence']) if not pd.isna(row['sentence']) else ""
    entities_str = row['entities']
    data_source = row['data_source']
    
    # 1. Clean the paragraph text by removing the trailing "\nSource paper: PMC..."
    cleaned_text = re.sub(r'\n?Source paper:\s*PMC\d+', '', text)
    
    # 2. Parse the entities
    entities = parse_entities(entities_str)
    total_entities_count += len(entities)
    
    # Track which entities have been mapped to a sentence
    mapped_entity_indices = set()
    
    # 3. Split the paragraph into sentences using spaCy
    doc = nlp(cleaned_text)
    
    for sent in doc.sents:
        sent_start = sent.start_char
        sent_end = sent.end_char
        sent_text_raw = sent.text
        
        # Strip leading/trailing whitespaces of the sentence and compute how much was stripped from the left
        lstrip_len = len(sent_text_raw) - len(sent_text_raw.lstrip())
        sent_text = sent_text_raw.strip()
        
        # Ignore completely empty sentences (e.g. if the original text was just whitespace)
        if not sent_text:
            continue
            
        sent_entities = []
        for i, ent in enumerate(entities):
            # Check if the entity falls within this sentence boundaries
            if sent_start <= ent['start'] and ent['end'] <= sent_end:
                mapped_entity_indices.add(i)
                
                # Adjust offsets relative to the stripped sentence text
                new_start = ent['start'] - sent_start - lstrip_len
                new_end = ent['end'] - sent_start - lstrip_len
                
                new_ent = ent.copy()
                new_ent['start'] = new_start
                new_ent['end'] = new_end
                
                # Check slice matching
                slice_text = sent_text[new_start:new_end]
                if slice_text != ent['text']:
                    # If there's a character alignment warning (like a trailing space or punctuation difference),
                    # we log it but keep the entity.
                    print(f"Warning: Slice mismatch for entity '{ent['text']}' at index {index} (slice: '{slice_text}')")
                
                sent_entities.append(new_ent)
                
        # Format the entities back to a JSON string or NaN if empty
        entities_val = json.dumps(sent_entities) if sent_entities else float('nan')
        
        new_rows.append({
            'PMCID': pmcid,
            'sentence': sent_text,
            'entities': entities_val,
            'data_source': data_source
        })
        
    # Check for unmatched entities in this row
    for i, ent in enumerate(entities):
        if i not in mapped_entity_indices:
            print(f"Warning: Unmatched entity '{ent['text']}' (offsets: {ent['start']}-{ent['end']}) in PMCID {pmcid}")
            unmatched_entities_count += 1

# Create the new DataFrame
exploded_df = pd.DataFrame(new_rows)

# Save to the TSV file
exploded_df.to_csv(output_path, sep='\t', index=False)

print(f"Successfully processed and exploded the TSV.")
print(f"Original row count: {len(df)}")
print(f"New exploded row count: {len(exploded_df)}")
print(f"Total entities parsed: {total_entities_count}")
print(f"Unmatched entities: {unmatched_entities_count}")
