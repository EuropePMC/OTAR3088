import pandas as pd
import ast
import re
import os
from var_utils import HGVS, CYTOBAND, GENOME_RE, REFSNP_RE, STAR_ALLELE_RE, map_to_ascii

def run_regex_extraction(text: str) -> list[tuple[int, int, str, str]]:
    """
    Run all standard regex patterns on the input text.
    Returns a list of (start, end, text, label) tuples.
    """
    normalized_text = map_to_ascii(text)
    matches = []
    
    # Match HGVS
    for m in HGVS.finditer(normalized_text):
        matches.append((m.start(), m.end(), m.group(), "HGVSVar"))
        
    # Match Cytoband
    for m in CYTOBAND.finditer(normalized_text):
        matches.append((m.start(), m.end(), m.group(), "ISCNVar"))
        
    # Match Refgenome
    for m in GENOME_RE.finditer(normalized_text):
        matches.append((m.start(), m.end(), m.group(), "Refgenome"))
        
    # Match RefSNP
    for m in REFSNP_RE.finditer(normalized_text):
        matches.append((m.start(), m.end(), m.group(), "RefSNP"))
        
    # Match StarAllele
    for m in STAR_ALLELE_RE.finditer(normalized_text):
        matches.append((m.start(), m.end(), m.group(), "StarAllele"))
        
    return matches

def check_overlap(span1: tuple[int, int], span2: tuple[int, int]) -> bool:
    """Check if two character spans overlap."""
    s1, e1 = span1
    s2, e2 = span2
    return max(s1, s2) < min(e1, e2)

def main():
    dataset_path = "variant_ner_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return
        
    df = pd.read_csv(dataset_path)
    
    total_sme_entities = 0
    missed_sme_entities = []
    extra_regex_matches = []
    perfect_matches = 0
    overlapping_matches = 0
    
    for idx, row in df.iterrows():
        sentence = str(row['sentence'])
        if pd.isna(row['entities']) or row['entities'] == "[]":
            sme_ents = []
        else:
            try:
                sme_ents = ast.literal_eval(row['entities'])
            except Exception as e:
                print(f"Row {idx} parse error: {e}")
                continue
                
        # Run regex
        regex_matches = run_regex_extraction(sentence)
        
        # Convert SME entities to standard structure
        sme_list = []
        for ent in sme_ents:
            sme_list.append({
                'start': ent['start'],
                'end': ent['end'],
                'text': ent['text'],
                'labels': ent.get('label', ent.get('labels', []))
            })
            total_sme_entities += 1
            
        # Check coverage
        matched_sme_indices = set()
        matched_regex_indices = set()
        
        for s_idx, sme in enumerate(sme_list):
            sme_span = (sme['start'], sme['end'])
            for r_idx, reg in enumerate(regex_matches):
                reg_span = (reg[0], reg[1])
                
                if check_overlap(sme_span, reg_span):
                    matched_sme_indices.add(s_idx)
                    matched_regex_indices.add(r_idx)
                    if sme['start'] == reg[0] and sme['end'] == reg[1]:
                        perfect_matches += 1
                    else:
                        overlapping_matches += 1
                        
        # Gaps: SME entities not caught by regex
        for s_idx, sme in enumerate(sme_list):
            if s_idx not in matched_sme_indices:
                missed_sme_entities.append({
                    'row': idx,
                    'sentence': sentence,
                    'text': sme['text'],
                    'span': (sme['start'], sme['end']),
                    'label': sme['labels']
                })
                
        # Extra/Noise: Regex matches not associated with any SME entities
        for r_idx, reg in enumerate(regex_matches):
            if r_idx not in matched_regex_indices:
                extra_regex_matches.append({
                    'row': idx,
                    'sentence': sentence,
                    'text': reg[2],
                    'span': (reg[0], reg[1]),
                    'label': reg[3]
                })
                
    # Print summary
    print("=" * 60)
    print("GENETIC VARIANT REGEX COVERAGE SCREEN REPORT (WITH ALL TAGS)")
    print("=" * 60)
    print(f"Total SME Verified Entities:  {total_sme_entities}")
    print(f"Perfect Regex Matches:        {perfect_matches} ({perfect_matches/max(1, total_sme_entities)*100:.2f}%)")
    print(f"Overlapping/Partial Matches:  {overlapping_matches} ({overlapping_matches/max(1, total_sme_entities)*100:.2f}%)")
    print(f"Missed Entities (Gaps):       {len(missed_sme_entities)} ({len(missed_sme_entities)/max(1, total_sme_entities)*100:.2f}%)")
    print(f"Extra Regex Matches (Noise):  {len(extra_regex_matches)}")
    print("-" * 60)
    
    # Display top missed entity types/phrases
    if missed_sme_entities:
        print("\nTOP MISSED ENTITIES (Gaps in Regex Coverage):")
        missed_df = pd.DataFrame(missed_sme_entities)
        counts = missed_df['text'].value_counts()
        for phrase, count in counts.head(20).items():
            # Get the labels for this phrase
            ph_labels = missed_df[missed_df['text'] == phrase]['label'].iloc[0]
            print(f"  - '{phrase}' ({ph_labels}): missed {count} times")
            
        # Print a few examples of sentences with missed variants
        print("\nEXAMPLES OF MISSED VARIANTS:")
        for idx, item in enumerate(missed_sme_entities[:5]):
            print(f"  {idx+1}. Text: '{item['text']}' (Label: {item['label']})")
            print(f"     Context: {item['sentence'][:120]}...")
            print()
            
    # Save reports
    os.makedirs("output", exist_ok=True)
    if missed_sme_entities:
        pd.DataFrame(missed_sme_entities).to_csv("output/regex_gaps.csv", index=False)
        print("Detailed gaps report saved to 'output/regex_gaps.csv'")
    if extra_regex_matches:
        pd.DataFrame(extra_regex_matches).to_csv("output/regex_noise.csv", index=False)
        print("Detailed noise report saved to 'output/regex_noise.csv'")

if __name__ == "__main__":
    main()
