# Candidate-finding regex patterns

Helper patterns for locating standard notation. They only propose candidates: label and span decisions
follow `ANNOTATION_GUIDELINES.md`. The authoritative, executable versions live in
`Data_mining/variant/var_utils.py`; update that file first if a pattern changes.

Patterns marked *(new)* are not yet implemented in `var_utils.py`.

`Other` has no regex. Find it by reading for variants these patterns miss.

## `HGVSVar`

| Prefix | Regex | Examples |
|---|---|---|
| Genomic `g.` | `g\.(?:[0-9]+[ACGTU]>[ACGTU]\|(?:[0-9]+_[0-9]+\|[0-9]+)(?:delins[ACGTU]+\|del\|dup\|ins[ACGTU]+\|inv))` | `g.140453136A>T`, `g.117199646_117199647insA` |
| Coding `c.` | `c\.(?:\*\|-)?(?:[0-9]+(?:[+-][0-9]+)?)(?:[ACGTU]>[ACGTU]\|delins[ACGTU]+\|del\|dup\|ins[ACGTU]+\|inv)` | `c.1799T>A`, `c.5266dupC`, `c.68_69delAG` |
| Non-coding / RNA `n.` `r.` | `[nr]\.(?:[0-9]+(?:[+-][0-9]+)?)(?:[ACGTUacgtu]>[ACGTUacgtu]\|delins[ACGTUacgtu]+\|del\|dup\|ins[ACGTUacgtu]+\|inv)` | `n.45G>C`, `r.76a>u` |
| Protein `p.` | 3- or 1-letter codes; also `p.(…)`, stop-gain `*`, synonymous `=`, `del`, `dup`, `fs` / `fsTer` | `p.Val600Glu`, `p.V600E`, `p.Arg213*`, `p.Val600GlufsTer5` |

An optional accession prefix (`NM_`, `NR_`, `NP_`, `NC_`, `NG_`, `XM_`, `XR_`, `ENST`, `ENSP`) may precede any of these.

## `RefSNP`

`\brs[0-9]+\b`: `rs4845618`, `rs4845374`

## `StarAllele`

Matches the allele suffix; it must follow a gene name.

`\s*\*\s*[0-9]+[A-Za-z]?(?::[0-9]+[A-Za-z]?)*(?:\s*\+\s*\*\s*[0-9]+[A-Za-z]?(?::[0-9]+[A-Za-z]?)*)*`

Examples: `CYP2C19*17`, `CYP2C9*1/*3`, `CYP2D6*68  +  *4`

## `ISCNVar`

`(?:chr)?([0-9]{1,2}|X|Y)([pq][0-9]+(?:\.[0-9]+)?)(?:-[pq]?[0-9]+(?:\.[0-9]+)?)?\s*(?:deletion|duplication|del|dup|inv)` (case-insensitive)

Examples: `1q21 deletion`, `chrXp21.1 duplication`

## `Refgenome`

Case-insensitive aliases for human and mouse builds: `GRCh38`, `hg38`, `GRCh37`, `hg19`, `NCBIBuild37`, `GRCm39`

## Additional patterns *(new)*

| Label | Form | Regex | Examples |
|---|---|---|---|
| `HGVSVar` | Mitochondrial `m.` | same body as the `c.` pattern with prefix `m\.` | `m.3243A>G` |
| `HGVSVar` | Prefix-less substitution | `\b[ACDEFGHIKLMNPQRSTVWY][0-9]{1,5}[ACDEFGHIKLMNPQRSTVWY*]\b` (many false positives; confirm context) | `V600E`, `R776W` |
| `HGVSVar` | Prefix-less indel | `\b[0-9]+(?:del\|ins\|dup)[ACGT0-9]+\b` | `185delAG`, `5382insC` |
| `HGVSVar` | VCF / SPDI | `(?:chr)?[0-9XYM]{1,2}:[0-9]+:[ACGT]+:[ACGT]+` and `NC_[0-9]+\.[0-9]+:[0-9]+:[ACGT]*:[ACGT]*` | `chr7:140453136:A:T` |
| `StarAllele` | HLA allele | `HLA-[A-Z0-9]+\*[0-9]{2,3}(?::[0-9]{2,3}){0,3}[A-Z]?` | `HLA-B*57:01` |
| `ISCNVar` | Derivative / translocation / inversion | `\b(?:der\|t\|inv\|dup\|del)\([0-9XY;]+\)(?:\([^)]*\))*` | `der(9)t(9;11)(p24.1;q24.1)` |
| `ISCNVar` | Full karyotype | `\b[0-9]{2},X[XY],[^\s;]+(?:\([^)]*\))+` | `46,XX,t(9;22)(q34;q11.2)` |
| `ISCNVar` | Micro-deletion / duplication at a band | `micro(?:deletion\|duplication)\s+at\s+[0-9XY]{1,2}[pq][0-9.]+` | `microdeletion at 4q34.1` |
| `CosmicID` | COSMIC | `\bCOS[MV][0-9]+\b` | `COSM476`, `COSV52765161` |
| `ClinVarID` | ClinVar | `\b[VR]CV[0-9]{9}(?:\.[0-9]+)?\b` | `VCV000012375`, `RCV000013510` |
| `DbVarID` | dbVar | `\b(?:nsv\|nssv\|esv\|essv)[0-9]+\b` | `nsv929778`, `nssv1614481` |
