# Utility functions relating to variant model investigations
import pandas as pd
import re
import requests
import torch
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import Dict, List, Tuple

# --- Class declarations ----------------------------------------------------------------
@dataclass
class Entity:
    text:  str
    label: str           # e.g. "mutant", "residue"
    start: int           # char offset in the section text
    end:   int
@dataclass
class SectionResult:
    pmcid:    str
    heading:  str
    text:     str
    entities: list[Entity] = field(default_factory=list) # Default empty []

# ── ePMC loading / parsing functions ───────────────────────────────────────────────────
def query_epmc(query: str, page_size: int = 10) -> List:
    """
    Search for 'x' paper results on passing a query string
    E.g. output_list = query_epmc(query='HeLa cells', page_size=5)
    """
    query = f"{query} HAS_FT:Y AND OPEN_ACCESS:Y"
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": query,
        "format": "json",
        "pageSize": page_size
    }

    resp = requests.get(url, params=params)
    resp_json = resp.json()

    if 'resultList' in resp_json:
        # pprint(resp_json)
        pmcids = [i['pmcid'] for i in resp_json['resultList']['result']]
        return pmcids
    else:
        return []

def get_epmc_full_text(pmcid: str) -> str:
    """
    E.g. xml_out = get_epmc_fulltext_xml('PMC2231364')
    """
    base_url = 'https://www.ebi.ac.uk/europepmc/webservices/rest'
    url = f'{base_url}/{pmcid}/fullTextXML'    
    response = requests.get(url, params={})
        
    # Check if request was successful
    if response.status_code == 200:
        return response.text
    elif response.status_code == 404:
        print(f'Article {pmcid} not found or full-text XML not available')
    else:
        response.raise_for_status()

@dataclass
class ParsedPaper:
    pmcid: str
    title: str
    abstract: str
    sections: list[dict]  # [{"heading": str, "text": str}, ...]
    def to_plain_text(self, max_chars: int = None) -> str:
        """Flatten to a single string, suitable for a transformer input."""
        parts = []
        if self.title:
            parts.append(self.title)
        if self.abstract:
            parts.append(self.abstract)
        for sec in self.sections:
            if sec["heading"]:
                parts.append(sec["heading"])
            parts.append(sec["text"])
        text = "\n\n".join(parts)
        return text[:max_chars] if max_chars else text

def _extract_text(element) -> str:
    """Recursively extract all text from an element, ignoring tags."""
    return " ".join(element.itertext()).strip()

def _extract_sections(body_element, depth: int = 0) -> list[dict]:
    """Recursively extract <sec> elements in document order."""
    sections = []
    for sec in body_element.findall("sec"):
        title = sec.find("title")
        heading = _extract_text(title).strip() if title is not None else ""
        # Gather paragraph in section
        para_texts = [
            _extract_text(p)
            for p in sec.findall("p")
            if _extract_text(p)
        ]
        body_text = " ".join(para_texts)
        if body_text:
            sections.append({"heading": heading, "text": body_text})
        # Recurse into nested sections
        sections.extend(_extract_sections(sec, depth + 1))
    return sections

def parse_epmc_xml(pmcid: str, xml_text: str) -> ParsedPaper:
    """
    Parse a full-text XML string into a ParsedPaper.
    """
    root = ET.fromstring(xml_text)
    # --- Title ---
    title = ""
    title_elem = root.find(".//article-title")
    if title_elem is not None:
        title = _extract_text(title_elem)
    # --- Abstract ---
    abstract_parts = []
    for abs_elem in root.findall(".//abstract"):
        abstract_parts.append(_extract_text(abs_elem))
    abstract = " ".join(abstract_parts).strip()
    # --- Body sections ---
    body_elem = root.find(".//body")
    sections = _extract_sections(body_elem) if body_elem is not None else []
    return ParsedPaper(pmcid=pmcid, title=title, abstract=abstract, sections=sections)

# ── NER setup / run ──────────────────────────────────────────────────────────
def load_model(model_name: str) -> Tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    id2label = model.config.id2label
    return tokenizer, model, id2label 

def run_model(input_text: str, tokenizer, model,
              id2label, stride: int = 64, max_length: int = 512) -> list[Entity]:
    """
    Run NER over input_text, handling texts longer than max_length via a
    sliding window with stride to avoid missing entities at chunk boundaries.
    Returns a deduplicated list of Entity objects.
    """
    # Tokenise with overflow (sliding window)
    encodings = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,   # char offsets for each token
        padding=True,
    )

    offset_mapping = encodings.pop("offset_mapping") # Metadata - unhelpful for model()
    encodings.pop("overflow_to_sample_mapping", None) # ^^
    with torch.no_grad(): 
        outputs = model(**encodings)

    logits = outputs.logits   # (num_chunks, seq_len, num_labels)
    predictions = torch.argmax(logits, dim=-1)   # (num_chunks, seq_len)
    all_entities: list[Entity] = [] # Entity: text, label, start, end
    for chunk_idx in range(predictions.shape[0]):
        chunk_preds   = predictions[chunk_idx]       # (seq_len,)
        chunk_offsets = offset_mapping[chunk_idx]    # (seq_len, 2)
        current_entity_tokens: list[tuple[int, int, str]] = []  # (start, end, bio_label)
        for token_idx, (pred_id, offsets) in enumerate(
            zip(chunk_preds.tolist(), chunk_offsets.tolist())
        ):
            start, end = offsets
            # Skip special tokens ([CLS], [SEP], padding)
            if start == 0 and end == 0:
                continue
            bio_label = id2label[pred_id]
            if bio_label.startswith("B-"):
                # Flush previous entity
                _flush_entity(current_entity_tokens, input_text, all_entities)
                current_entity_tokens = [(start, end, bio_label[2:])]
            elif bio_label.startswith("I-") and current_entity_tokens:
                label_type = bio_label[2:]
                # Only continue if same type as current entity
                if current_entity_tokens[-1][2] == label_type:
                    current_entity_tokens.append((start, end, label_type))
                else:
                    _flush_entity(current_entity_tokens, input_text, all_entities)
                    current_entity_tokens = []
            else:
                # "O" label — flush any open entity
                _flush_entity(current_entity_tokens, input_text, all_entities)
                current_entity_tokens = []
        _flush_entity(current_entity_tokens, input_text, all_entities)
    return _deduplicate(all_entities)

def _flush_entity(tokens: list[tuple[int, int, str]],
                  source_text: str,
                  accumulator: list[Entity]) -> None:
    """
    Convert a run of BIO tokens into a single Entity and append it.
    - - - 
    Called when chain of 'B-' 'I-' terms of same label_type ends
    OR when 'O' label encountered.
    """
    if not tokens:
        return
    start = tokens[0][0]
    end   = tokens[-1][1]
    label = tokens[0][2]
    span  = source_text[start:end].strip()
    if span:
        accumulator.append(Entity(text=span, label=label, start=start, end=end))

def _deduplicate(entities: list[Entity]) -> list[Entity]:
    """Remove duplicate entities that arise from overlapping sliding windows."""
    seen: set[tuple[int, int, str]] = set()
    unique = []
    for ent in entities:
        key = (ent.start, ent.end, ent.label)
        if key not in seen:
            seen.add(key)
            unique.append(ent)
    return unique

# Pipeline: run over all ParsedPaper sections
def run_ner_pipeline(parsed_paper: "ParsedPaper", tokenizer, model, id2label) -> list[SectionResult]:
    """
    Run NER over every section of a ParsedPaper.
    Returns one SectionResult per section.
    """
    results = []
    # If there is one, annotate the abstract
    if parsed_paper.abstract:
        entities = run_model(parsed_paper.abstract, tokenizer, model, id2label)
        results.append(SectionResult(
            pmcid   = parsed_paper.pmcid,
            heading = "Abstract",
            text    = parsed_paper.abstract,
            entities = entities,
        ))
    for sec in parsed_paper.sections:
        if not sec["text"].strip():
            continue
        entities = run_model(sec["text"], tokenizer, model, id2label)
        results.append(SectionResult(
            pmcid   = parsed_paper.pmcid,
            heading = sec["heading"],
            text    = sec["text"],
            entities = entities,
        ))
    return results

def get_context(text: str, entity: Entity, window: int = 5) -> str:
    """
    Return up to `window` words either side of the entity span,
    with the entity highlighted as *** ... ***.
    """
    # Split on whitespace, tracking char offsets for each word
    words, offsets = [], []
    pos = 0
    for word in text.split():
        start = text.index(word, pos)
        end   = start + len(word)
        words.append(word)
        offsets.append((start, end))
        pos = end
    # Find which word indices overlap the entity span
    entity_word_idxs = [
        i for i, (ws, we) in enumerate(offsets)
        if ws < entity.end and we > entity.start
    ]
    if not entity_word_idxs:
        return entity.text  # fallback
    first, last = entity_word_idxs[0], entity_word_idxs[-1]
    left  = words[max(0, first - window) : first]
    mid   = words[first : last + 1]
    right = words[last + 1 : last + 1 + window]
    return f"…{' '.join(left)} ***{' '.join(mid)}*** {' '.join(right)}…"

# ── Regex-related funcs ──────────────────────────────────────────────────────────

# Map look-alike characters -> ASCII equivalents
HOMOGLYPH_MAP = str.maketrans({
    0x0441: 'c',   # Cyrillic с → c
    0x0440: 'p',   # Cyrillic р → p
    0x0435: 'e',   # Cyrillic е → e
    0x0430: 'a',   # Cyrillic а → a
    0x2013: '-',   # en-dash → hyphen
    0x2212: '-',   # minus sign → hyphen
    0x2014: '-',   # em-dash → hyphen
})

def map_to_ascii(text: str) -> str:
    """ 
    Replace homoglyph characters with ASCII equivalents.
    Catches chars which may block terms from being caught by regex.
    E.g. 'с.181T >G', wherein 'с' is a Cyrillic character U+0441.
    This can be confused for ASCII 'c', character U+0063
    """
    return text.translate(HOMOGLYPH_MAP)

"""
ePMC patterns file defines database cross-references based on regex patterns
Those useful for variant capture include:
- refsnp (dbSNP)
- gca (INSDC genome assembly accessions)
"""

## Rough estimate of mapped genome assembly references
GENOME_ASSEMBLY_ALIASES = {
    # Human
    "GRCh38": ["hg38", "GRCh38", "GRCh38.p13", "GRCh38.p14", "hg38/GRCh38", "NCBIBuild38"],
    "GRCh37": ["hg19", "GRCh37", "GRCh37.p13", "b37", "hs37d5", "NCBIBuild37", "NCBI37"],
    "NCBI36": ["hg18", "NCBI36", "NCBIBuild36"],
    # Mouse
    "GRCm39": ["mm39", "GRCm39"],
    "GRCm38": ["mm10", "GRCm38", "mm10/GRCm38"],
    "GRCm37": ["mm9",  "NCBI37 (mouse)"],
}
# Invert {} for string-searching
SYNONYM_TO_TERM = {
    syn: term
    for term, syns in GENOME_ASSEMBLY_ALIASES.items()
    for syn in syns
}
# Regex setup
_ASSEMBLY_TERMS = sorted(SYNONYM_TO_TERM, key=len, reverse=True)
GENOME_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(t) for t in _ASSEMBLY_TERMS) + r')\b',
    re.IGNORECASE
)

## Compilation of patterns for HGVS statements
# ── Amino acid codes ────────────────────────────────────────────────────────
_AA3 = r'(?:Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val|Ter|Sec)'
_AA1 = r'[ACDEFGHIKLMNPQRSTVWY]'

# ── Nucleotide change types (shared by g. c. n. r.) ─────────────────────────
# _NUC_POS   = r'[0-9]+(?:[ \t]*[+\-][ \t]*[0-9]+)?'           # e.g. 76, 1799+2
_NUC_POS = r'(?:\*|-)?[0-9]+(?:[ \t]*[+\-][ \t]*[0-9]+)?'
_NUC_RANGE = r'[0-9]+(?:[ \t]*[+\-][ \t]*[0-9]+)?[ \t]*_[ \t]*[0-9]+(?:[ \t]*[+\-][ \t]*[0-9]+)?' # e.g. 76_78
_NUC_BASE  = r'[ACGTUacgtu]'
_SUBST = rf'{_NUC_POS}\s*{_NUC_BASE}\s*>\s*{_NUC_BASE}' # substitution:  A>T, considerate of () // whitespace

_NUC_CHANGE = (
    r'(?:'
    rf'{_SUBST}'
    rf'|(?:{_NUC_RANGE}|{_NUC_POS})[ \t]*delins[ \t]*{_NUC_BASE}+' # indel:    76_78delinsAT
    rf'|(?:{_NUC_RANGE}|{_NUC_POS})[ \t]*del[ \t]*{_NUC_BASE}*'    # deletion: 76del / 76_78del
    rf'|(?:{_NUC_RANGE}|{_NUC_POS})[ \t]*dup[ \t]*{_NUC_BASE}*'    # duplication
    rf'|{_NUC_RANGE}[ \t]*ins[ \t]*{_NUC_BASE}+'                   # insertion: 76_77insA
    rf'|{_NUC_RANGE}[ \t]*inv'                                     # inversion
    r')'
)

# ── Per-prefix patterns ──────────────────────────────────────────────────────
HGVS_GENOMIC  = rf'g\.{_NUC_CHANGE}'                # g.140453136A>T
HGVS_CODING   = rf'c\.{_NUC_CHANGE}'                # c.1799T>A
HGVS_NCRNA    = rf'n\.{_NUC_CHANGE}'                # n.45G>C
HGVS_RNA      = rf'r\.{_NUC_CHANGE}'                # r.76a>u

HGVS_PROTEIN  = (
    r'\(?p\.\(?(?:'                                  # (p.X) and p.(X) handled
    rf'{_AA3}[0-9]+(?:'
        rf'{_AA3}(?:fs(?:Ter[0-9]+)?)?'              # p.Val600Glu / p.Val600GlufsTer5
        rf'|\*'                                      # p.Arg213*  (stop gained)
        rf'|='                                       # p.Val600=  (synonymous)
        rf'|del'                                     # p.Val600del
        rf'|dup'                                     # p.Val600dup
    r')'
    rf'|{_AA1}[0-9]+[{_AA1[1:-1]}*=]'              # p.V600E / p.R213*  (1-letter)
    r')'
)

# ── Optional accession prefix: NM_004333.6: or ENST00000288135.6: ───────────
_ACCESSION = r'(?:\(?[ \t]*(?:NM|NR|NP|NC|NG|XM|XR|ENST|ENSP)_?[0-9]+(?:\.[0-9]+)?[ \t]*\)?[ \t]*:)?'

# ── Combined pattern ─────────────────────────────────────────────────────────
HGVS = re.compile(
    _ACCESSION +
    rf'(?:{HGVS_GENOMIC}|{HGVS_CODING}|{HGVS_NCRNA}|{HGVS_RNA}|{HGVS_PROTEIN})'
)

# ISCN / cytoband notation pattern
CYTOBAND = re.compile(
    r'\b(?:chr)?([0-9]{1,2}|X|Y)'   # chromosome
    r'([pq][0-9]+(?:\.[0-9]+)?)'     # start band
    r'(?:-[pq]?[0-9]+(?:\.[0-9]+)?)?' # optional end band
    r'\s*(?:deletion|duplication|del|dup|inv)\b',
    re.IGNORECASE
)

## Star allele suffix — follows a gene name
_STAR_SUFFIX = re.compile(
    r'\s*\*\s*[0-9]+[A-Za-z]?(?::[0-9]+[A-Za-z]?)*'          # *13, *02:01
    r'(?:\s*\+\s*\*\s*[0-9]+[A-Za-z]?(?::[0-9]+[A-Za-z]?)*)*' # optional: + *2
)

def find_star_alleles(text: str, gene_spans: list[tuple]) -> list[tuple]:
    """
    For identified geneprotein spans, check if star allele notation
    immediately follows. Returns (text, start, end, label) tuples.
    gene_spans: list of (word, start, end, label)
    """
    results = []
    for word, g_start, g_end, label in gene_spans:
        m = _STAR_SUFFIX.match(text, g_end)   # anchored at gene end position
        if m:
            full_span = text[g_start:m.end()]
            results.append((full_span, g_start, m.end(), "StarAllele"))
        else:
            results.append((word, g_start, g_end, label))
    return results