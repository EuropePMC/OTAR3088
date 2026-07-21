# Genetic Variant Annotation Benchmarking: Session Summary (2026-06-23)

This document captures the context, design decisions, and status of the project at the end of the session on June 23, 2026.

---

## 1. Project Context & Objectives
The goal of this project is to benchmark the performance of Large Language Models (LLMs)—specifically **Gemini**, **GPT**, **Claude**, and select open-source models—on the task of annotating genetic variant mentions in literature retrieved from Europe PMC.
*   **Evaluation Engine**: [Karenina](https://github.com/biocypher/karenina) benchmarking suite.
*   **Evaluation Evaluator/Judge**: **Gemini** (configured as the Judge LLM).
*   **Ground Truth Dataset**: [variant_ner_dataset.csv](file:///Users/withers/GitProjects/OTAR3088/Data_mining/variant/variant_ner_dataset.csv) (SME-verified annotations).

The benchmarking consists of two comparative runs:
1.  **Run 1 (Zero-Shot)**: Models are left to extract annotations without explicit constraints.
2.  **Run 2 (Skill-Guided)**: Models are guided by a `SKILL.md` prompt instructions document.

---

## 2. Completed Tasks & File Summary

During this session, we completed the groundwork for **Phase 2 (Skill Compilation & Audit)**:

### Core Code Modifications
*   [var_utils.py](file:///Users/withers/GitProjects/OTAR3088/Data_mining/variant/var_utils.py):
    *   Consolidated and appended definitions for RefSNP (`REFSNP_RE` matching `rs...`) and Star Alleles (`STAR_ALLELE_RE`) into the central utility library.
*   [find_coverage_gaps.py](file:///Users/withers/GitProjects/OTAR3088/Data_mining/variant/find_coverage_gaps.py):
    *   Created this multi-use auditing script to check heuristic regex coverage against `variant_ner_dataset.csv`.
    *   Imports all patterns from `var_utils.py` and isolates false negatives (gaps) and false positives (noise), saving details to `output/regex_gaps.csv` and `output/regex_noise.csv`.
*   [SKILL.md](file:///Users/withers/GitProjects/OTAR3088/Data_mining/variant/SKILL.md):
    *   Created the prompt guidance document for LLM Run 2.
    *   Configured separate tags: `HGVSVar`, `RefSNP`, `StarAllele`, `ISCNVar`, `Refgenome`, and `Other`.
    *   Instructed models to look for known gaps (e.g. prefix-less mutations, legacy formatting, and character encoding issues).
    *   MANDATED the **Draft, Review, Refine** annotation workflow.

### Project Artifacts (Saved in `.gemini/antigravity-ide/brain/c52eebe8-b6fc-4c4b-a2c5-92403db1253f/`)
*   `project_plan.md`: The roadmap detailing setup, pipeline mechanics, and comparison metrics.
*   `walkthrough.md`: Summarizes initial screening baseline stats (42.83% missed coverage before RefSNP/StarAllele rules integration) and refactoring notes.

---

## 3. The Annotation Workflow: Draft, Review, Refine
To minimize missed entities, the LLM will follow this structured self-audit loop:
1.  **Draft**: Extract standard variants matching strict regex rules.
2.  **Review (Self-Audit)**: Screen the text again to catch prefix-less entries (e.g. `185delAG`, `T1815M`, `V600E`) or descriptive texts (e.g. "mutation in exon 11").
3.  **Refine**: Consolidate and output final JSON annotations using the designated separate tags.

---

## 4. Next Steps
When resuming the project:
1.  **Run updated baseline screen**: Execute `python find_coverage_gaps.py` in this directory to evaluate the coverage level including the newly integrated RefSNP and StarAllele rules.
2.  **Phase 3: Karenina Configuration**:
    *   Set up the Karenina backend and server environments.
    *   Configure Gemini API details for the Judge evaluator.
3.  **Phase 4: Run Execution**: Execute Run 1 (Zero-Shot) and Run 2 (Skill-Guided + Self-Audit loop) prompts on target model APIs.

---

## 5. Metric Evaluation Logic (How Regex Coverage is Judged)
The audit run calculates metrics by comparing standard regex matches against target SME-annotated spans in `variant_ner_dataset.csv` using:
*   **Total SME Verified**: The count of all annotations in the dataset.
*   **Perfect Matches**: Regex spans that *exactly* match the SME coordinates.
*   **Overlapping/Partial Matches**: Spans where the regex and SME overlap, but do not share identical start/end offsets (e.g., `(p.Val600Glu)` vs `p.Val600Glu`).
*   **Missed Entities (Gaps)**: SME annotations with zero overlapping regex matches.
*   **Extra Regex Matches (Noise)**: Regex extractions that do not overlap with any SME annotations in that sentence.

---

## 6. Bugs & Issues Resolved

During Phase 3 and Phase 4 setup, several critical blockers were resolved:
1.  **Hatch Build Crash**: Comments/duplicate directives in `karenina/pyproject.toml` caused Hatch to crash due to a duplicate include (`adele/data/AS.txt`). This was bypassed by commenting out the duplicate configuration.
2.  **Rate Limiting & Token Exhaustion (HTTP 429)**: Standard model free-tiers hit strict limits on parallel requests. We modified [run_karenina_benchmarks.py](file:///Users/withers/GitProjects/OTAR3088/Data_mining/variant/run_karenina_benchmarks.py) to run sequentially (`async_enabled=False`, `async_max_workers=1`) and monkeypatched the LangChain `ChatGoogleGenerativeAI` and `ChatOpenAI` invoke methods to introduce a `4.0` second sleep between API calls.
3.  **Pydantic Deserialization (Dict Coercion)**: Custom Karenina template schema metadata returned gold-standard objects as raw dictionaries rather than instantiated Pydantic models. We modified the `verify()` and `verify_granular()` code to dynamically coerce raw dictionaries back to `Entity` objects before comparing sets.
4.  **LangChain Mocking Structure**: Mocking `with_structured_output(...)` for offline testing required stubbing the structured output model's `.ainvoke(...)` to return an instantiated Pydantic class (`Answer`) instead of a raw message, avoiding Pydantic validation failures.
5.  **Pandas KeyError for 'granular_score'**: Karenina's `to_dataframe()` output did not include the custom `verify_granular_result` property. We resolved this by mapping `verify_granular_result` of each result back to the `'granular_score'` column using `question_id`.
6.  **Overall Pass Rate Printing Format Error**: Formatting the dictionary returned by `template_results.aggregate_pass_rate()` directly with `:.2%` resulted in an `unsupported format string passed to dict.__format__` error. We resolved this by computing the average float percentage from the dictionary values before printing.

---

## 7. Current Project Status & Completed Mock Tests

We added a comprehensive unit test suite in [test_benchmarks_offline.py](file:///Users/withers/GitProjects/OTAR3088/Data_mining/variant/test_benchmarks_offline.py) that covers:
*   **F1 Scoring Logic**: Boundary conditions (exact matches, partial matches, and empty predictions/labels).
*   **Orchestration Pipeline**: End-to-end mock execution using mocked LangChain structured outputs, verifying correct template extraction and resolving the pandas KeyError on `'granular_score'`.
*   **Helper Regex & Extraction Utilities**: Unit tests validating homoglyph mapping (`map_to_ascii`), dbSNP formatting (`REFSNP_RE`), cytobands (`CYTOBAND`), genome assemblies (`GENOME_RE`), star alleles (`STAR_ALLELE_RE`), and gene-level suffix extraction (`find_star_alleles`).

All 11 unit tests pass, and the pipeline is fully prepared for executing live benchmarks.


