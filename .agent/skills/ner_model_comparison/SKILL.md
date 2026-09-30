---
name: ner_model_comparison
description: A workflow to manually compare the entity annotations of different NER models from uploaded screenshots, and generate TSV comparison files.
---

# Objective
The objective of this skill is to systematically extract, align, and compare Named Entity Recognition (NER) annotations from screenshots of different model outputs to the same text. 
Names of the models and the Document Identifier (e.g., PMCID) should be provided by the user. If they are not provided in the initial request, you MUST ask the user to clarify them before proceeding with generating any scripts or TSVs. The results are then compiled into structured TSV reports.

# Process

## Step 1: Image Analysis & Extraction
When the user uploads a sequence of images, only regard the pieces of the image you need to. Do not get bogged down in parsing the whole image. You will:
1. Identify which model corresponds to which image, by inspecting the 'Model Description' cell in the UI screenshot.
2. For each model, carefully transcribe the exact raw text block visible in the corresponding images, in the 'Tagged entities' cell in the UI screenshot.
3. For each model, parse the exact spans of text that have been highlighted and note their corresponding entity types (e.g., `CELLTYPE`, `TISSUE`, `CELLLINE`). Pay strict attention to spacing, punctuation, bounding box boundaries, and whether single terms are split into multiple adjacent annotations.
## Step 2: Script Generation
Generate a Python script (typically stored in `/tmp/gen_tsv.py`) containing the alignment and comparison logic. Instead of writing the overlap logic from scratch, **use the boilerplate template provided in:**
`/Users/withers/GitProjects/OTAR3088/.agent/skills/ner_model_comparison/scripts/compare_entities.py`.

1. Read the contents of that exact template script and copy it into your new `/tmp/gen_tsv.py` execution script.
2. Initialize `text` variables populated with the transcriptions you gathered.
3. Define the lists of extracted entities (e.g., `ents_a = [("T cells", "CELLTYPE"), ("senescent", "CELLTYPE")]`) representing the annotations from the models.
4. Construct the `SectionResult` instances using the script's handy `find_entities()` function to convert string lists into positional boundary spans. Make sure to use the Document Identifier (e.g., PMCID) provided by the user for the `pmcid` field.
5. Generate an aligned output data frame using `compare_models([sect_a...], [sect_b...], "model_x", "model_y")`.

## Step 3: Execution & TSV Export
1. Execute the python script.
2. The script must output the resulting DataFrame to a TSV file in the user's Downloads folder at: `~/Downloads/<Identifier>.tsv`. Ensure `sep="\t"` and `index=False` are configured.
3. Configure the script to simultaneously export a markdown representation of the table to a temporary file (e.g. `/tmp/out.md`).

## Step 4: User Reporting
1. Read the temporary markdown file containing the comparison table.
2. Present the markdown table directly to the user in the chat interface so they can immediately see the discrepancies.
3. Provide a concluding list of **High-Level Takeaways** summarizing the behaviors observed in each model (e.g., successful dropping of vague terms like "cells"/"tissues", unintended span fragmentation, state/prefix inclusion, or interesting tokenization artifacts).
