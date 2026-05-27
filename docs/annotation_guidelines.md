Annotation Guidelines
=====================

* [index](./index.md)

## Entity type definitions for NLP recognition

To ensure high-precision term grounding and biological data extraction, we have defined separate annotation categories for "tissues", "cell types" and "cell lines". These categories distinguish between different levels of biological organization and their experimental contexts. This structural division is intended to allow for more accurate semantic mapping and prevents the conflation of lexically similar but biologically distinct systems and concepts during text mining.


### Definition of the annotated entity type "tissue"

The entity type "tissue" is defined as supra-cellular anatomical entities — encompassing anatomical structures, body substances, boundaries, and spatial regions — derived from organisms within the taxon Metazoa (NCBITaxon:33208). This definition is intentionally broad, capturing all levels of anatomical organisation including tissues, organs, organ systems, and other anatomical structures or body compartments. The supra-cellular scope distinguishes tissue entities from constituent cell type entities, as they represent a higher order of biological organisation comprising cells and connective tissues collectively.

Critically, "tissue" in this context primarily refers to the anatomical site of origin rather than to physical specimens.

When annotating tissue entities, the full descriptive phrase identifying or distinguishing a set of anatomical entities should be captured, including relevant adjectives and modifiers. The rationale for this is to help distinguish one group of anatomical entities from another similar group.

Certain lexical entities that include anatomical terms are explicitly excluded from this entity type. This is the case when they form part of a compositionally distinct entity. Examples include but not limited to anatomical terms embedded within disease names (e.g., "lung cancer", "lung adenocarcinoma"), biological process names (e.g., "heart contraction"), or named resources and databases (e.g., "The Human Lung Cell Atlas"). These instances are not to be annotated as "tissue" entities.

#### Anatomical entity terminologies

Terminologies, vocabularies and ontologies of anatomical entities include, but are not limited to the following list of resources. Any anatomical term referencing supra-cellular entities, anatomical volumes or spaces would fall under our definition of "tissue".

* Anatomy textbooks, anatomy atlases
* Uberon multi-species anatomy ontology [(Uberon)](https://www.ebi.ac.uk/ols4/ontologies/uberon)
    * Mungall CJ, Torniai C, Gkoutos GV, Lewis SE, Haendel MA. Uberon, an integrative multi-species anatomy ontology. Genome Biology. 2012 Jan;13(1):R5. DOI: 10.1186/gb-2012-13-1-r5. PMID: 22293552; PMCID: PMC3334586.
    * Haendel MA, Balhoff JP, Bastian FB, et al. Unification of multi-species vertebrate anatomy ontologies for comparative biology in Uberon. Journal of Biomedical Semantics. 2014 ;5:21. DOI: 10.1186/2041-1480-5-21. PMID: 25009735; PMCID: PMC4089931.
* Experimental Factor Ontology (EFO) - [EFO:0000786 anatomy basic component](https://www.ebi.ac.uk/ols4/ontologies/efo/classes/http%253A%252F%252Fwww.ebi.ac.uk%252Fefo%252FEFO_0000786) branch
* [Medical Subject Headings](https://meshb.nlm.nih.gov) (MeSH Concepts and synonyms, MeSH TopicalDescriptors)
* [Terminologia Anatomica](https://ta2viewer.openanatomy.org/) (TA; a standard by the International Federation of Associations of Anatomists)
* NCI Thesaurus - [NCIT:C12219 Anatomic Structure, System, or Substance](https://www.ebi.ac.uk/ols4/ontologies/ncit/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FNCIT_C12219?lang=en)
* Foundational Model of Anatomy Ontology [(FMA)](https://www.ebi.ac.uk/ols4/ontologies/fma)
* wikipedia - [Anatomy](https://en.wikipedia.org/wiki/Anatomy)
* [SNOMED CT](https://www.ebi.ac.uk/ols4/ontologies/snomed)

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

### Definition of the annotated entity type "cell type"

The entity type "cell type" is defined as in vivo cell types constituting the anatomical entities of Metazoan organisms (NCBITaxon:33208), classifiable as subclasses of "CL:0000000 cell" in the Cell Ontology. This category encompasses any classification scheme applied to the physical entities biologists refer to as cells, including classifications based on morphology, size, histological staining properties, tissue of origin, developmental stage, or physiological- transcriptional-, or other cell states.
Cell states — defined here as cells at a particular stage of one or more biological processes, including those characterised by a distinct transcriptional profile or gene expression signature — are treated as subcategories of cell types for the purposes of entity recognition. When annotating cell type entities, the full descriptive phrase identifying or distinguishing a population of cells should be captured, including all relevant adjectives and modifiers. Representative annotation examples include: "CD4+ T cells", "Th17 cells", "T cells", "neutrophils", "neurons".


#### Cell type references
* Cell Ontology [(CL)](https://www.ebi.ac.uk/ols4/ontologies/cl)
    * Tan SZK, Puig-Barbe A, Goutte-Gattat D, Eastwood C, Aevermann B, Avola A, Balhoff JP, Bayindir IU, Belfiore J, Caron AR, Fischer DS, George N, Gyori BM, Haendel MA, Hoyt CT, Kir H, Lubiana T, Matentzoglu N, Overton JA, Peng B, Peters B, Quardokus EM, Ray PL, Roncaglia P, Rivera AD, Stefancsik R, Teh WK, Toro S, Vasilevsky N, Xu C, Zhang Y, Scheuermann RH, Mungall CJ, Diehl AD, Osumi-Sutherland D. The Cell Ontology in the age of single-cell omics. Sci Data. 2026 Apr 24. doi: 10.1038/s41597-026-07173-8. Epub ahead of print. PMID: 42031777.
    * Meehan TF, Masci AM, Abdulla A, et al. Logical development of the cell ontology. BMC Bioinformatics. 2011 Jan;12:6. DOI: 10.1186/1471-2105-12-6. PMID: 21208450; PMCID: PMC3024222.
* Experimental Factor Ontology (EFO) - [CL:0000000 cell](https://www.ebi.ac.uk/ols4/ontologies/efo/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FCL_0000000)
* Provisional Cell Ontology [(PCL)](https://www.ebi.ac.uk/ols4/ontologies/pcl)
* [Medical Subject Headings](https://meshb.nlm.nih.gov) (MeSH Concepts and synonyms, MeSH TopicalDescriptors)
* NCI Thesaurus - [NCIT:C12508 Cell](https://www.ebi.ac.uk/ols4/ontologies/ncit/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FNCIT_C12508)
* wikipedia - [Cel type](https://en.wikipedia.org/wiki/Cell_type)
SNOMED CT ("cell")

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

### Definition of the annotated entity type "cell line"

The entity type "cell line" is defined as immortalised cultured cell lines (CLO:0000019) derived from Metazoan organisms (NCBITaxon:33208), corresponding to cultured cells (CL:0000010) both with an immortal phenotype allowing unlimited number of passages within some practical limits, but also other types of cultured cells, e.g. primary cells or cell  lines, if the authors refer to those as such . When annotating cell line entities, the full descriptive phrase identifying or distinguishing a given cell line should be captured, including relevant adjectives and modifiers. Representative annotation examples include: "MCF-7", "HeLa", and "HepG2".

#### Cell line terminologies

Catalogues, terminologies, vocabularies and ontologies of cell lines include, but are not limited to the following list of resources. Any term referencing a cell line in the following, or similar, resources would fall under our definition of "cell line".

* American Type Culture Collection [(ATCC)](https://www.atcc.org/)
* [Cellosaurus](https://www.cellosaurus.org/) - Cell line encyclopedia
    * Bairoch A. The Cellosaurus, a Cell-Line Knowledge Resource. J Biomol Tech. 2018 Jul;29(2):25-38. [doi: 10.7171/jbt.18-2902-002](https://doi.org/10.7171/jbt.18-2902-002). Epub 2018 May 10. PMID: 29805321; PMCID: PMC5945021.
* Cell Line Ontology [(CLO)](https://www.ebi.ac.uk/ols4/ontologies/clo/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FCLO_0000031?lang=en)
    * Sarntivijai S, Lin Y, Xiang Z, et al. CLO: The cell line ontology. Journal of Biomedical Semantics. 2014 ;5:37. [DOI: 10.1186/2041-1480-5-37](https://doi.org/10.1186/2041-1480-5-37). PMID: 25852852; PMCID: PMC4387853.


* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

#### Vague or ambiguous entities

Scientific text can sometimes contain vague or ambiguous mentions of cell populations that broadly correspond to cell or tissue types, or an arbitrary combination of specific types and a very generic one. Greater specificity in cell and tissue types allows for the extraction of more precise information during text mining.
Some descriptors of terms overlapping the definitions of "cell type" and "tissue type" above could be deemed too vague to carry sufficiently specific information on tissue or cell-type entities. The text spans corresponding to these "vague" types can optionally be filtered out if deemed necessary or useful for particular applications of our annotated corpus. However, these annotations can also aid in understanding our annotation logic and allow for the production of data that is more usable for machine learning (ML).
* "vague tissue": A subtype of "tissue" (as defined above) that encompasses too wide a range of tissues, anatomical entities, or structures to be useful for the annotation of experimental data used in ML model production or in knowledge bases. It may or may not be clear which specific anatomical entity a "vague tissue" mention refers to within the broader context of the source text; regardless, outside of the specific context of the source text, these terms remain highly unspecific.
    * Examples of identified text spans of this type include: "non-lymphoid tissues", "cancer tissue".
* "vague cell type": A cell type (as defined above) that maps to an overly broad category, making it too non-specific to be used for the annotation of experimental data used in ML model production or in knowledge bases. It may or may not be clear which specific cell category a "vague cell type" refers to within the broader context of the source text; regardless, outside of the specific context of the source text, these terms remain highly unspecific.
    * Examples of identified text spans of this type include: "cells", "infected cells", "control cells".
* "vague cell line": A cell line (as defined above) that maps to an overly broad category, making it too non-specific to be used for the annotation of experimental data used in ML model production or in knowledge bases. It may or may not be clear which specific cell line a "vague cell type" refers to within the broader context of the source text; regardless, outside the specific context of the source text, these terms remain highly unspecific.
    * Examples of identified text spans of this type include: "cell lines", "healthy cell lines", "patient cell lines".

Additional examples of the above types corresponding to vague or ambiguous entities can be found in the table [vague_entity_examples.tsv](./vague_entity_examples.tsv).

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
