---
name: genetic-variant-annotation
description: Guidelines and regex-based patterns for annotating genetic variant mentions in biomedical text.
---

# Genetic Variant Annotation Guidelines

This document provides instructions for Large Language Models (LLMs) to identify, extract, and label mentions of genetic variants and reference assemblies in biomedical texts.

You should use the defined regex patterns as **starting points** (anchors for standard notation), but you must also screen the surrounding text for any **gaps in coverage**—meaning genetic variant mentions that do not strictly match the regex patterns but refer to mutations, variations, cytobands, or alleles.

---

## 1. Anchoring Regex Patterns & Tags

Use the following patterns to locate standard nomenclature and classify them using the exact tags specified below:

### A. `HGVSVar` (HGVS Standard Formats)
*   **Genomic (`g.`)**: DNA level genomic variants.
    *   *Regex*: `g\.(?:[0-9]+[ACGTU]>[ACGTU]|(?:[0-9]+_[0-9]+|[0-9]+)(?:delins[ACGTU]+|del|dup|ins[ACGTU]+|inv))`
    *   *Examples*: `g.140453136A>T`, `g.117199646_117199647insA`
*   **Coding (`c.`)**: Coding DNA sequence level variants.
    *   *Regex*: `c\.(?:\*|-)?(?:[0-9]+(?:[+-][0-9]+)?)(?:[ACGTU]>[ACGTU]|delins[ACGTU]+|del|dup|ins[ACGTU]+|inv)`
    *   *Examples*: `c.1799T>A`, `c.5266dupC`, `c.68_69delAG`
*   **Non-Coding / RNA (`n.` and `r.`)**: Non-coding DNA and RNA transcripts.
    *   *Regex*: `[nr]\.(?:[0-9]+(?:[+-][0-9]+)?)(?:[ACGTUacgtu]>[ACGTUacgtu]|delins[ACGTUacgtu]+|del|dup|ins[ACGTUacgtu]+|inv)`
    *   *Examples*: `n.45G>C`, `r.76a>u`
*   **Protein (`p.`)**: Amino acid changes using 3-letter or 1-letter codes.
    *   *Regex*: Handles forms like `p.Val600Glu`, `p.(Val600Glu)`, `p.V600E`, `p.Arg213*` (stop-gain), `p.Val600=` (synonymous), `p.Val600del` (deletion), and frameshifts (e.g. `fs`, `fsTer`).
    *   *Examples*: `p.Val600Glu`, `p.V600E`, `p.Arg213*`, `p.Val600GlufsTer5`

### B. `RefSNP` (RefSNP IDs)
*   **Pattern**: Matches specific single nucleotide polymorphisms (SNPs) from dbSNP.
*   **Regex Structure**: `\brs[0-9]+\b`
*   **Examples**: `rs4845618`, `rs4845374`, `rs4453032`, `rs4379670`

### C. `StarAllele` (Pharmacogenetic Star Alleles)
*   **Pattern**: Matches star allele nomenclature following a gene name.
*   **Regex Structure**: `\s*\*\s*[0-9]+[A-Za-z]?(?::[0-9]+[A-Za-z]?)*(?:\s*\+\s*\*\s*[0-9]+[A-Za-z]?)*`
*   **Examples**: `CYP2C19*17`, `CYP2D6*2`, `CYP2C9*1/*3`, `CYP2C19*37`, `CYP2D6*68`, `CYP2D6*68  +  *4`, `CYP2D6*13  +  *2`, `CYP2C19*1/*1`

### D. `ISCNVar` (Cytogenetic Bands)
*   **Pattern**: Matches chromosomal cytogenetic band deletions, duplications, or inversions.
*   **Regex Structure**: `(?:chr)?([0-9]{1,2}|X|Y)([pq][0-9]+(?:\.[0-9]+)?)(?:-[pq]?[0-9]+(?:\.[0-9]+)?)?\s*(?:deletion|duplication|del|dup|inv)`
*   **Examples**: `1q21 deletion`, `chrXp21.1 duplication`

### E. `Refgenome` (Reference Genome Assemblies)
*   **Pattern**: Matches standard reference assemblies and build versions.
*   **Regex Structure**: Case-insensitive aliases for human/mouse builds.
*   **Examples**: `GRCh38`, `hg38`, `GRCh37`, `hg19`, `NCBIBuild37`, `GRCm39`

### F. `Other` (Non-standard / Legacy / Descriptive Variants)
*   **Pattern**: Mentions that describe variants or genomic changes but do not conform to standard HGVS prefix syntax or standard build patterns.
*   **Prefix-less insertions/deletions**: `185delAG`, `2080delA`, `3819del5`, `3875del4`, `4153delA`, `5382insC`, `6174delT`
*   **Prefix-less protein changes**: `T1815M`, `V600E`, `L2487V`, `R1239H`, `R776W`
*   **Descriptive mutations / genomic changes**: "T-to-G substitution at nucleotide 181", "deletion of codon 600", "BRCA1 exon 11 mutation"
*   **Cytogenetic Translocations / Derivative Chromosomes**: `der(9)t(9;11)(p24.1;q24.1)`, `inv(Y)(p11.2p11.2)`, `der(X)(Ypter_Yp11.2)...`
*   **Microdeletions and Microduplications**: `microduplication at 16q22.3`, `microdeletion at 4q34.1`
*   **Non-standard Star Alleles (containing arbitrary words/sub-alleles)**: `CYP2D6 NMa/*68  +  *4`, `CYP2D6 NMa/*4`, `NMa/*41 `

---


## 2. Screening for Gaps in Coverage (Beyond Regex)

You must explicitly screen the text to identify variant mentions that do **not** conform to the standard patterns in Section 1. 

Look out for:
1.  **Non-standard character encoding / Homoglyphs**: Cyrillic 'с' instead of Latin 'c' (e.g., `с.181T>G`), or fancy dashes/minus signs (e.g., `c.1799T−>A`). Map these to `HGVSVar`.
2.  **Encoding and Spacing Irregularities**: Space gaps in HGVS strings (e.g. `c.181T >G` or thin spaces like `c.181T > G`). Map these to `HGVSVar`.

---

## 3. Annotation Workflow: Draft, Review, Refine

To maximize annotation coverage and eliminate false negatives, you must follow this three-step process:

1.  **Drafting Phase**: 
    Scan the text and extract standard genetic variant mentions based on the rules in Section 1. Assign the appropriate labels (`HGVSVar`, `RefSNP`, `StarAllele`, `ISCNVar`, `Refgenome`).
2.  **Review Phase (Self-Audit)**:
    Re-evaluate the source text and search specifically for any genetic variations, mutations, or reference builds that were missed because they did not match the strict regex formats. Specifically check for:
    *   Prefix-less deletions, insertions, or amino acid changes (e.g., `185delAG`, `T1815M`, `V600E`).
    *   Complex star allele configurations with operators (e.g., `+`, `/`).
    *   Textual descriptions of mutations (e.g., "mutation in exon 11").
    *   Char encoding issues (Cyrillic lookalikes) or spacing gaps.
3.  **Refine Phase**:
    Incorporate the missed variants into your final list, mapping them to the `Other` category (or standard categories if they were simply malformed). Double-check that all start and end character offsets correspond precisely to the original text.

---

## 4. Output Schema

For each genetic variant mention identified, return the entity span using the following schema:
- **`text`**: The exact substring of the variant mention as it appears in the text.
- **`start`**: Character index of the start of the mention (0-indexed).
- **`end`**: Character index of the end of the mention (0-indexed, exclusive).
- **`label`**: Classification category (MUST be one of: `HGVSVar`, `RefSNP`, `StarAllele`, `ISCNVar`, `Refgenome`, `Other`).

