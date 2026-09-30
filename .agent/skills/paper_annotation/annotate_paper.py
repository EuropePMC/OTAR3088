import re
import os

input_file = '/Users/withers/GitProjects/OTAR3088/Data_mining/labelstudio_e2e/extracted_texts/PMC8809252.txt'
output_file = '/Users/withers/GitProjects/OTAR3088/Data_mining/labelstudio_e2e/extracted_texts/PMC8809252_annotated.txt'

# Entity Dictionaries based on paper content and guidelines
# Note: Case-insensitive matching will be used, but keys keep original casing for clarity if needed.
# Prioritize longer matches first (e.g. "liver sinusoidal endothelial cells" before "cells")

entity_map = {
    # CellType
    "Kupffer cells": "CellType",
    "KCs": "CellType",
    "KC": "CellType",
    "KC1s": "CellType",
    "KC1": "CellType",
    "KC2s": "CellType",
    "KC2": "CellType",
    "LAMs": "CellType",
    "LAM": "CellType",
    "lipid-associated macrophages": "CellType",
    "hepatocytes": "CellType",
    "hepatocyte": "CellType",
    "LSECs": "CellType",
    "LSEC": "CellType",
    "liver sinusoidal endothelial cells": "CellType",
    "stellate cells": "CellType",
    "HSCs": "CellType",
    "macrophages": "CellType",
    "macs": "CellType",
    "mac": "CellType",
    "monocytes": "CellType",
    "neutrophils": "CellType",
    "B cells": "CellType",
    "B cell": "CellType",
    "T cells": "CellType",
    "T cell": "CellType",
    "NK cells": "CellType",
    "cDCs": "CellType",
    "cDC": "CellType",
    "cDC1s": "CellType",
    "cDC2s": "CellType",
    "dendritic cells": "CellType",
    "cholangiocytes": "CellType",
    "stromal cells": "CellType",
    "fibroblasts": "CellType",
    "mesothelial cells": "CellType",
    "VSMCs": "CellType",
    "vascular smooth muscle cells": "CellType",
    "myeloid cells": "CellType",
    "CD4+ T cells": "CellType",
    "Th17 cells": "CellType",
    "neurons": "CellType",
    "endothelial cells": "CellType",
    "ECs": "CellType",
    "LECs": "CellType",
    "lymphatic ECs": "CellType",
    "peritoneal macs": "CellType",
    "capsule macs": "CellType",
    "CD207+ macs": "CellType",
    "moKCs": "CellType",
    "monocyte-derived KCs": "CellType",
    "transitioning monocytes": "CellType",
    
    # Tissue
    "liver": "Tissue",
    "bile duct": "Tissue",
    "bile ducts": "Tissue",
    "BDs": "Tissue",
    "BD": "Tissue",
    "portal vein": "Tissue",
    "PV": "Tissue",
    "PVs": "Tissue",
    "central vein": "Tissue",
    "CV": "Tissue",
    "CVs": "Tissue",
    "capsule": "Tissue",
    "endothelium": "Tissue",
    "stroma": "Tissue",
    "parenchyma": "Tissue",
    "hepatic artery": "Tissue",
    "spleen": "Tissue",
    "blood": "Tissue",
    "plasma": "Tissue",
    "blood vessels": "Tissue",
    "intestine": "Tissue",
    "bone marrow": "Tissue",
    "BM": "Tissue",
    "brain": "Tissue",
    "lung": "Tissue",
    "heart": "Tissue",
    "adipose tissue": "Tissue",
    "duodenum": "Tissue",
    "immune system": "Tissue",
    "retinal pigment epithelium": "Tissue",

    # CellLine
    "HeLa": "CellLine",
    "HepG2": "CellLine",
    "MCF-7": "CellLine",
    
    # Vague Terms (mapped as requested)
    "cells": "CellType",
    "cell": "CellType",
    "infected cells": "CellType",
    "control cells": "CellType",
    "tissue": "Tissue",
    "tissues": "Tissue",
    "non-lymphoid tissues": "Tissue",
    "cancer tissue": "Tissue",
    "samples": "Tissue", # Context dependent often, but mapping as vague tissue per request inference "vague... within these entity types"
    "sample": "Tissue",
    "biopsy": "Tissue", # Guideline says NO, but prompt says "include any terms deemed 'vague' ... within these entity types". 
                        # Guideline says "vague tissue" is a subtype of tissue... but also says "intended meaning... does not include... biopsy".
                        # However, strictly following "Include any terms deemed 'vague' by the annotation guidelines within these entity types too".
                        # The guideline lists "vague tissue" as a subtype to exclude from "true entity type" but for ML production. 
                        # BUT the USER PROMPT says: "Include any terms deemed 'vague' by the annotation guidelines within these entity types too". 
                        # So I will include "biopsy" and "sample" as Tissue if they appear in context of biological material.
}

# Sort entities by length descending to match longest first
sorted_entities = sorted(entity_map.keys(), key=len, reverse=True)

def tokenize(text):
    # Simple whitespace tokenization, preserving punctuation as separate tokens often required for CoNLL
    # But strictly speaking CoNLL usually splits punctuation. 
    # For this task, we'll use a regex to split keeping punctuation.
    return re.findall(r"[\w']+|[.,!?;()\[\]{}]", text)

def annotate_text(text):
    tokens = tokenize(text)
    tags = ['O'] * len(tokens)
    
    # We need to map tokens back to text positions or just iterate and match
    # Since we need IOB tags, we can iterate through the token list and look ahead.
    
    i = 0
    while i < len(tokens):
        match_found = False
        # Try to match starting at i
        # We construct strings from tokens[i:j] and see if they match an entity
        # We need to handle potential whitespace issues. 
        # A simple approach: reconstruct " ".join(tokens[i:j]) and check against lower cased dictionary.
        
        # Max entity length (in tokens) to check? Let's say 6.
        for width in range(6, 0, -1):
            if i + width > len(tokens):
                continue
            
            phrase_tokens = tokens[i : i+width]
            
            # We try joining with space. Some entities might have hyphens which tokenize logic might have split or not.
            # My tokenizer splits on non-word chars except ' 
            # Example: "lipid-associated macrophages" -> "lipid", "-", "associated", "macrophages" if I split on -
            # Current regex r"[\w']+|[.,!?;()\[\]{}]" keeps "lipid-associated" as one token if it has no spaces? 
            # No, \w usually includes alphanumeric and _. Hyphen is not \w. 
            # So "lipid-associated" becomes "lipid", "-", "associated" ?
            # Let's check regex behavior: re.findall(r"[\w']+|[.,!?;()\[\]{}-]", "lipid-associated") -> ['lipid', '-', 'associated'] matches my assumption.
            
            # So we typically join with spaces... but "lipid-associated" has no spaces.
            # This simple token match is tricky.
            # Better Approach: Find span character offsets of tokens. Find entity character offsets in raw text. Map them.
            pass
        
        if not match_found:
             i += 1
             
    # RE-STRATEGY:
    # 1. Find all entity occurences in raw text using regex (boundaries).
    # 2. Tokenize raw text and keep track of character spans of tokens.
    # 3. Assign tags to tokens that overlap with found entities.
    
    # Escape entities for regex
    # We match whole words roughly.
    escaped_entities = [re.escape(e) for e in sorted_entities]
    pattern = re.compile(r'\b(' + '|'.join(escaped_entities) + r')\b', re.IGNORECASE)
    
    # Find all matches
    matches = []
    for match in pattern.finditer(text):
        start, end = match.span()
        # Get the actual text matched to find the correct casing key if needed? 
        # actually we just need the type from the map.
        # We need to find which key it matched.
        matched_text = match.group()
        
        # Find the specific key (could be case variation)
        # matches corresponding key in entity_map matching lower case
        entity_type = None
        for key in entity_map:
            if key.lower() == matched_text.lower():
                entity_type = entity_map[key]
                break
        
        if entity_type:
            matches.append((start, end, entity_type))
            
    # Now tokenize and tag
    # Iterate tokens, check overlap
    tagged_tokens = []
    
    # We need a tokenizer that gives spans.
    token_iter = re.finditer(r"[\w']+|[^\w\s]", text)
    
    match_idx = 0
    matches.sort(key=lambda x: x[0]) # ensure sorted by start
    
    for word_match in token_iter:
        token_start, token_end = word_match.span()
        token_text = word_match.group()
        
        tag = 'O'
        
        # Check against matches
        # We might have overlapping matches? greedy longest match should prevail if we did it right.
        # But here we just take the first one or logic to picking best.
        # Simple logic: if token is inside a match.
        
        # Advance match_idx if we passed it
        while match_idx < len(matches) and matches[match_idx][1] <= token_start:
            match_idx += 1
            
        if match_idx < len(matches):
            m_start, m_end, m_type = matches[match_idx]
            if token_start >= m_start and token_end <= m_end:
                # overlap
                if token_start == m_start:
                    tag = f"B-{m_type}"
                else:
                    tag = f"I-{m_type}"
        
        tagged_tokens.append(f"{token_text} {tag}")
        
    return tagged_tokens

with open(input_file, 'r') as f:
    text = f.read()

annotated_lines = annotate_text(text)

with open(output_file, 'w') as f:
    f.write("\n".join(annotated_lines))

print(f"Generated {len(annotated_lines)} tokens to {output_file}")
