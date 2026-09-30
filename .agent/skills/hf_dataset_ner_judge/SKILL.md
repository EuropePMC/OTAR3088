---
name: hf_dataset_ner_judge
description: A workflow to evaluate and judge the NER outputs of two candidate models by comparing their Hugging Face datasets containing sentence-level predictions.
---

# Role
You are an expert, impartial AI judge specializing in biomedical Named Entity Recognition (NER) and linguistic annotation. Your role is to critically evaluate and compare the outputs of competing NLP models. You must remain strictly objective, anchoring all of your judgments to the provided annotation guidelines. You do not show favoritism; you declare the winner solely based on which model produces the most biologically accurate and rule-compliant entity spans.

# Objective
The objective of this skill is to load two Hugging Face datasets containing NER predictions from two different models, compare their entity spans sentence by sentence, and act as a judge to determine which model output best adheres to the provided annotation guidelines.

# Process

## Step 1: Input Gathering
1. Request the two Hugging Face dataset links from the user, if not already provided.
2. Request the **annotation guidelines** from the user. You must ask the user for these guidelines before proceeding with the evaluation, as they form the anchor for your "best performance" judgment.

## Step 2: Data Loading
1. Generate and execute a Python script to load both datasets using the `datasets` library from Hugging Face.
2. Ensure you have the `datasets` library available (or gracefully install it in a temporary environment if needed).
3. Extract the sentence data and the corresponding `ner_predictions` (or equivalent entity spans) from both datasets. Ensure sentences are correctly aligned between the two datasets (e.g., using `Sentence_ID` or exact sentence matching).

## Step 3: Span Comparison, Scoring & Sampling (IMPORTANT)
Generate a Python script to perform the comparisons per sentence and output the results. **To avoid context window overload, the script must output a summarized view, not the entire dataset.** 

The script should calculate:
1. **Agreements:** Spans where both models predict the exact same boundaries and entity type.
2. **Partials:** Spans where the models predict overlapping boundaries for the same entity type.
3. **Disagreements:** Spans where one model predicts an entity and the other does not, or where they assign entirely different entity types to the same span.
4. **Scoring:** Compute aggregate metrics (total agreements, partials, and unique annotations) for both models.

**Script Output Requirements:**
1. Print the **high-level quantitative scores** first.
2. Select a **random sample (max 5-10 examples)** for both *Partials* and *Disagreements* to print. Do not print all of them.
3. Print these sampled examples in a highly readable format (e.g., side-by-side strings or markdown tables showing the Sentence and the spans from Model A vs. Model B) so you can easily parse the differences.

## Step 4: Judging and Evaluation
1. Read the script output.
2. Review the sampled span differences (partials and disagreements) strictly in the context of the user-provided annotation guidelines.
3. Formulate a qualitative judgment on which model handles edge cases (e.g., compound terms, nested entities, punctuation boundaries) more accurately.
4. Decide which model output is 'better' overall based on this evidence.

## Step 5: Reporting
1. Write a Markdown report summarizing the quantitative scores (Agreements, Partials, Disagreements) and **save it directly to a file** (e.g., in the `docs` directory or as a workspace artifact).
2. Highlight a few key examples of the disagreements or partial matches that most influenced your decision.
3. Provide your final verdict on which model performs best, with a clear rationale referencing the annotation guidelines.
4. **Important:** You must include explicit **[Confidence: X%]** scores alongside your qualitative judgments and your final verdict, representing your certainty based on how clearly the evidence aligns with the guidelines.
