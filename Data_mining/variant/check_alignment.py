import pandas as pd
import ast

df = pd.read_csv("variant_ner_dataset.csv")

total = 0
correct = 0
failures = []

for i, row in df.iterrows():
    if pd.isna(row['entities']) or row['entities'] == "[]":
        continue
    
    try:
        ents = ast.literal_eval(row['entities'])
    except Exception as e:
        continue
        
    sentence = str(row['sentence'])
    
    for ent in ents:
        total += 1
        start, end = ent['start'], ent['end']
        expected_text = ent['text']
        actual_text = sentence[start:end]
        
        if actual_text == expected_text:
            correct += 1
        else:
            failures.append(f"Row {i} mismatch: Exp='{expected_text}' (len {len(expected_text)}) vs Act='{actual_text}' (len {len(actual_text)}) at {start}:{end}")
            
print(f"Total entities checked: {total}")
print(f"Perfectly aligned: {correct}")
if failures:
    print(f"MISALIGNED: {len(failures)}")
    for f in failures[:5]:
        print(f)
