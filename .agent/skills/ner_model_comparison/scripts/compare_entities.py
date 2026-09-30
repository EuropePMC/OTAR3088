import pandas as pd
import os
import argparse
from dataclasses import dataclass

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int

@dataclass
class SectionResult:
    pmcid: str
    heading: str
    text: str
    entities: list[Entity]

def get_context(text: str, ent: Entity, win: int = 5) -> str:
    before = text[:ent.start].split()[-win:]
    after = text[ent.end:].split()[:win]
    return " ".join(before) + " [" + ent.text + "] " + " ".join(after)

def spans_overlap(a: Entity, b: Entity) -> bool:
    return a.start < b.end and b.start < a.end

def compare_models(section_results_a: list[SectionResult],
                   section_results_b: list[SectionResult],
                   model_a_name: str,
                   model_b_name: str,
                   context_words: int = 5) -> pd.DataFrame:
    rows = []
    b_index = {(s.pmcid, s.heading): s for s in section_results_b}
    
    for sect_a in section_results_a:
        key = (sect_a.pmcid, sect_a.heading)
        sect_b = b_index.get(key)
        b_entities = sect_b.entities if sect_b else []
        matched_b = set()
        
        for ent_a in sect_a.entities:
            context_a = get_context(sect_a.text, ent_a, context_words)
            overlaps = [(j, ent_b) for j, ent_b in enumerate(b_entities) if spans_overlap(ent_a, ent_b)]
            
            if not overlaps:
                rows.append({
                    "Identifier": sect_a.pmcid,
                    f"{model_a_name} annotation": ent_a.text,
                    f"{model_a_name} entity type": ent_a.label,
                    f"{model_b_name} annotation": "—",
                    f"{model_b_name} entity type": "—",
                    "Match": "no",
                    "Context span": context_a
                })
            else:
                for j, ent_b in overlaps:
                    matched_b.add(j)
                    match_type = "yes" if ent_a.text == ent_b.text and ent_a.label == ent_b.label else "partial"
                    rows.append({
                        "Identifier": sect_a.pmcid,
                        f"{model_a_name} annotation": ent_a.text,
                        f"{model_a_name} entity type": ent_a.label,
                        f"{model_b_name} annotation": ent_b.text,
                        f"{model_b_name} entity type": ent_b.label,
                        "Match": match_type,
                        "Context span": context_a
                    })
        
        for j, ent_b in enumerate(b_entities):
            if j not in matched_b:
                context_b = get_context(sect_b.text, ent_b, context_words)
                rows.append({
                    "Identifier": sect_b.pmcid,
                    f"{model_a_name} annotation": "—",
                    f"{model_a_name} entity type": "—",
                    f"{model_b_name} annotation": ent_b.text,
                    f"{model_b_name} entity type": ent_b.label,
                    "Match": "no",
                    "Context span": context_b
                })

    df = pd.DataFrame(rows)
    cols = [
        "Identifier",
        f"{model_a_name} annotation", f"{model_a_name} entity type",
        f"{model_b_name} annotation", f"{model_b_name} entity type",
        "Match", "Context span"
    ]
    return df[cols]

def find_entities(text: str, ent_list: list[tuple[str, str]]) -> list[Entity]:
    found = []
    current_idx = 0
    for e_text, e_label in ent_list:
        idx = text.find(e_text, current_idx)
        if idx != -1:
            found.append(Entity(e_text, e_label, idx, idx + len(e_text)))
            current_idx = idx + len(e_text)
        else:
            idx2 = text.find(e_text)
            if idx2 != -1:
                found.append(Entity(e_text, e_label, idx2, idx2 + len(e_text)))
    return found

# Usage Template:
# Define text variables and ents_a / ents_b lists, then call:
# df = compare_models([sect1_a], [sect1_b], "CeLLaTe3.0_Base_with_vague_new", "CeLLaTe3.0_Base_no_vague_new")
# df.to_csv(os.path.expanduser("~/Downloads/<ID>.tsv"), sep="\t", index=False)
