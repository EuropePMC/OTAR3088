---
name: paper_annotation
description: A workflow to annotate papers with specified entity types
---

# Paper Annotation Skill

This is a workflow to annotate papers with specified entity types. The specified entity types will be provided by the user. You will refer to a description of the entity types, in the form of annotation guideline.

## Instructions

1. Source the annotation guideline from the user, usually provided via a link to a github md file
2. Understand the context which defines these entity types
3. Using a provided input file, produce an annotated file, in the IOB CoNLL format, with the following columns:
    - Token
    - Tag
4. Save the output file in the same directory as the input file, with the suffix "_annotated" before the file extension