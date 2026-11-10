# Genetic Variant Annotation Guidelines

Annotation rules for genetic variant mentions in biomedical text. These are the definition of each label.
Regexes (see `regex_patterns.md`) are only a helper for finding candidates and never override these rules.

## 1. Labels and decision order

Apply the first rule that matches.

| # | If the mention is… | Label | Examples |
|---|---|---|---|
| 1 | A reference assembly | `Refgenome` | `GRCh38`, `hg19`, `GRCm39`, `NCBIBuild37` |
| 2 | A dbSNP ID | `RefSNP` | `rs4845618` |
| 3 | A COSMIC identifier | `CosmicID` | `COSM476`, `COSV52765161` |
| 4 | A ClinVar accession | `ClinVarID` | `VCV000012375`, `RCV000013510` |
| 5 | A dbVar structural-variant accession | `DbVarID` | `nsv929778`, `nssv1614481` |
| 6 | A gene-anchored star-allele-style name: pharmacogenetic star alleles (single, diplotype or `+` combination), `NMa/` sub-alleles, and HLA alleles | `StarAllele` | `CYP2D6*4`, `CYP2C9*1/*3`, `CYP2D6*68 + *4`, `NMa/*41`, `HLA-B*57:01` |
| 7 | Cytogenetic notation: cytoband gain/loss, derivative chromosomes, translocations, inversions, micro-deletions/duplications at a band, full ISCN karyotypes | `ISCNVar` | `9p24.3-p24.1 deletion`, `der(9)t(9;11)(p24.1;q24.1)`, `microdeletion at 4q34.1`, `46,XX,t(9;22)(q34;q11.2)` |
| 8 | A sequence-level variant written in HGVS, HGVS-style or coordinate notation (§3) | `HGVSVar` | `c.1799T>A`, `m.3243A>G`, `V600E`, `185delAG`, `chr7:140453136:A:T` |
| 9 | Any other specific variant mention that fits none of the above (§4) | `Other` | `exon 2-5 deletion`, "deletion of codon 600" |

Malformed HGVS (homoglyphs, stray spaces, `T−>A`) is still `HGVSVar`.
`Other` is a last resort: before using it, check that none of rules 1–8 applies.

## 2. Span rules (all labels)

1. **Tight span**: no leading or trailing whitespace.
2. **Do not normalise the text**: odd characters stay inside the span exactly as written. This covers thin spaces,
   `c.181T > G`, `*68  +  *4`, fancy minus signs and Cyrillic lookalikes (`с.181T>G` is `HGVSVar`).
3. **Gene name**: included only for `StarAllele` (`CYP2D6*4`, `HLA-A*02:01`). Never for other labels
   (`BRCA1 c.68_69delAG` → span is `c.68_69delAG`).
4. **Accession prefix** (`NM_004333.6:c.1799T>A`) is part of the `HGVSVar` span.
5. **Parentheses**: include a balanced `(`…`)` that belongs to the notation (`p.(Thr117Met)`).
   Exclude brackets that wrap the notation in prose (`(p.T1815M)` → `p.T1815M`).
6. **Trailing type word** `deletion` / `duplication` is part of the `ISCNVar` span when it follows a cytoband.
7. **Repeated mentions**: annotate every occurrence, including bare re-mentions
   (`T1815M is located…` after `p.T1815M`).
8. **Coordinates without a change type** (`c.942_944`): tag as `HGVSVar` if the prefix is standard.
9. **Identifiers** (`rs…`, `COSM…`, `VCV…`, `nsv…`): span is the identifier only, with no surrounding words.
10. **Offsets** are 0-indexed, end-exclusive, and must match the original text exactly.

## 3. `HGVSVar` scope

| Form | Examples |
|---|---|
| HGVS with prefix `g.` `c.` `n.` `r.` `p.` | `g.140453136A>T`, `c.1799T>A`, `p.(Val600Glu)`, `p.Arg213*` |
| Mitochondrial `m.` | `m.3243A>G` |
| Prefix-less substitution (one-letter protein change) | `V600E`, `T1815M`, `R776W` |
| Prefix-less indel | `185delAG`, `2080delA`, `5382insC`, `3819del5` |
| SPDI or VCF-style coordinate notation | `NC_000007.14:140453135:A:T`, `chr7:140453136:A:T` |

Prefix-less forms count as `HGVSVar` only when they name a specific change at a position.
A bare residue with no substitution (`T1815`) is not a variant mention.

## 4. What goes in `Other`

`Other` is for specific variant mentions that are not sequence notation, not cytogenetic and not an identifier.

| Sub-type | Examples | Span |
|---|---|---|
| Exon-level change | `exon 2-5 deletion`, `del_exon1-5` | `exon` phrase plus type word; exclude the gene name |
| Verbal description of a specific change | "T-to-G substitution at nucleotide 181", "deletion of codon 600" | whole phrase |
| Tool- or study-specific structural variant names | `DEL_CHR10_28B50744` | whole token |

## 5. Do NOT annotate

- Gene or protein names alone (`BRCA1`, `MMP13 mutant`).
- Generic words ("mutation", "variant", "deletion") with no specific notation. For "BRCA1 exon 11 mutation",
  tag `exon 11 mutation` as `Other`; tag nothing if no exon is named.
- Positions or ranges with no change (`chr9:8,856,403-8,869,328`, breakpoints).
- Tokens that look like substitutions but are not variants: antibody clones, cell lines (`HEK293`),
  instrument codes, or a residue with no substitution (`T1815`).
- Figure and table callouts (`Fig. 5E–H`).
- If a sentence contains no variant notation, the correct annotation is an empty list.

## 6. Boundary examples

| Text | Annotate | Why |
|---|---|---|
| `BRCA1 c.68_69delAG` | `c.68_69delAG` → HGVSVar | rule 3 |
| `BRCA1 185delAG` | `185delAG` → HGVSVar | prefix-less indel (§3) |
| `CYP2D6 *4` / `CYP2D6 * 124` | whole string → StarAllele | spaces allowed |
| `CYP2D6*13  +  *2` | whole string → StarAllele | a combination is one mention |
| `CYP2D6 NMa/*68  +  *4` | whole string → StarAllele | non-standard sub-allele names are still star alleles |
| `*68 + *4` (gene named earlier in the sentence) | `*68 + *4` → StarAllele | gene need not be adjacent |
| `HLA-B*57:01` | whole string → StarAllele | HLA allele, gene included |
| `c.5444C>T (p.T1815M)` | `c.5444C>T`, `p.T1815M` | rule 5 |
| `T1815M is located…` | `T1815M` → HGVSVar | rule 7 |
| `ogm[GRCh38] der(9)t(9;11)(p24.1;q24.1)` | `GRCh38` → Refgenome, `der(9)t(9;11)(p24.1;q24.1)` → ISCNVar | `ogm[]` brackets excluded |
| `a 16q22.3 microduplication` | `16q22.3 microduplication` → ISCNVar | cytoband-anchored gain |
| `karyotype 46,XX,t(9;22)(q34;q11.2)` | `46,XX,t(9;22)(q34;q11.2)` → ISCNVar | full karyotype |
| `ClinVar RCV000013510` | `RCV000013510` → ClinVarID | identifier only |
| `exon 2-5 deletion in CYP2D6` | `exon 2-5 deletion` → Other | not sequence notation; gene excluded |

## 7. Entity linking targets (reference)

Each label maps to the resource a mention would later be linked to. This is the reason the identifier labels are separate.

| Label | Linking targets |
|---|---|
| `HGVSVar` | ClinVar, dbSNP via Ensembl VEP, HGMD, UniProt variants; MITOMAP for `m.`; COSMIC, BRCAExchange after normalisation |
| `RefSNP` | dbSNP (the rsID is the identifier) |
| `CosmicID` | COSMIC (the ID is the identifier) |
| `ClinVarID` | ClinVar (the accession is the identifier) |
| `DbVarID` | dbVar (the accession is the identifier) |
| `StarAllele` | PharmGKB, CPIC; IMGT/HLA for HLA alleles |
| `ISCNVar` | dbVar, DECIPHER, ClinGen dosage map; Mitelman for translocations |
| `Refgenome` | NCBI Assembly, Ensembl release mapping |
| `Other` | none |
