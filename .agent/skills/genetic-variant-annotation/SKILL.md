---
name: genetic-variant-annotation
description: Annotate genetic variant mentions (HGVS, rsIDs, star alleles, cytobands, reference assemblies, legacy notation) in biomedical text. Use when extracting or labelling variant entities, or reviewing variant NER annotations.
---

# Genetic Variant Annotation

Identify, extract and label mentions of genetic variants and reference assemblies in biomedical text.

## Files

- `ANNOTATION_GUIDELINES.md`: the rules. Label definitions, decision order, span boundaries, exclusions and examples. **Read this before annotating.**
- `regex_patterns.md`: optional helper regexes for finding standard notation. Executable versions are in `Data_mining/variant/var_utils.py`.

## Labels

`HGVSVar`, `RefSNP`, `CosmicID`, `ClinVarID`, `DbVarID`, `StarAllele`, `ISCNVar`, `Refgenome`, `Other`. See the guidelines for how to choose between them; `Other` is a last resort.

## Workflow: draft, review, refine

1. **Draft**: scan the text and extract standard variant mentions (the regexes help here). Assign labels following the decision order.
2. **Review (self-audit)**: re-read the text for mentions the patterns miss:
   - prefix-less substitutions or indels (`V600E`, `185delAG`), mitochondrial `m.` variants, SPDI/VCF-style coordinates
   - complex star alleles with `+` or `/`, `NMa/` sub-alleles, HLA alleles
   - cytogenetic notation: derivative chromosomes, translocations, micro-deletions/duplications, full karyotypes
   - database identifiers: COSMIC, ClinVar and dbVar accessions
   - exon-level changes and verbal descriptions of mutations
   - encoding problems: Cyrillic lookalikes, odd dashes, thin spaces inside notation
   - repeated bare re-mentions of an already annotated variant
3. **Refine**: add the missed mentions and apply the span rules. Check that every start and end offset matches the original text exactly.

## Output schema

For each mention, return:

- **`text`**: the exact substring as it appears in the text.
- **`start`**: character index of the start (0-indexed).
- **`end`**: character index of the end (0-indexed, exclusive).
- **`label`**: one of `HGVSVar`, `RefSNP`, `CosmicID`, `ClinVarID`, `DbVarID`, `StarAllele`, `ISCNVar`, `Refgenome`, `Other`.
