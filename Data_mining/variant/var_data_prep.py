import argparse 
import json
import sys, os
import textwrap
import uuid
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm
from var_utils import *

# Adding ePMC pattern tools to path

HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '../../../epmc-tools/europmc_dev_tool')))
from spacy_patterns import patterns

## Papers sourced via Claude search in ePMC, selected for having a variety of styles of genetic variant mentions
# PMC7334197,PMC12713268,PMC12465344,PMC12859152,PMC12874668,PMC4560075,PMC11354791,PMC8254301

# - - - - - - - - - - - - - - - -Argument parsing - - - - - - - - - - - - - - - - - - - - - - - - - - 
def list_of_strings(arg):
    return [item.strip() for item in arg.split(",")]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\
        Script ingests a list PMCIDs, parses full-text where available 
        & annotates these texts with variant regex patterns, outputting
        in json format.

        E.g. 
            python var_data_prep.py 
                --pmcids PMC7334197,PMC12713268 [REQUIRED]
                --outfile ./output/genevar_outputs/variants_ls.json (default = './var_out.json')
        '''))
parser.add_argument("--pmcids", "-p", type=list_of_strings, required=True, help="Comma-separated list of PMCIDs")
parser.add_argument("--outfile", "-o", type=str, help="Path to .json output file", default='./var_out.json')
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

args = parser.parse_args()
papers = args.pmcids
output_path = args.outfile

all_tasks = [] # All LabelStudio tasks (i.e. sections of text)

# BioCODGE-GO, labels multiple entity types, filtered here for GeneProtein
ner_gene = pipeline(
    "ner",
    model="OTAR3088/BioCODGE-GO",
    aggregation_strategy="simple"   # merges subword tokens into full spans
)
# Load the same tokenizer the pipeline uses
tokenizer = AutoTokenizer.from_pretrained("OTAR3088/BioCODGE-GO")

# 'patterns' imported here sourced from ePMC
label_list = ["refsnp", "gca"] # relevant patterns
var_patterns = [x for x in patterns if x['label'] in label_list]

for pmcid in tqdm(papers):
    print(f"\nParsing full-text for {pmcid}.\n")
    xml_out = get_epmc_full_text(pmcid=pmcid)
    if xml_out is None:
        continue
    parsed  = parse_epmc_xml(pmcid=pmcid, xml_text=xml_out)

    for sec in parsed.sections:
        ls_tasks = [] # tasks (sections) for paper 'pmcid'
        epmc_matches = []

        text = '\n'.join(sec['text'].split(sep='. ')) # prepare so as to not mess span numbers
        text = map_to_ascii(text) # catch non-ASCII chars

        for p in var_patterns:
            pattern = p['pattern']
            epmc_matches.extend([(m.group(), m.start(), m.end(), "ePMCVar") for m in re.finditer(pattern, text)])
        matches = [(m.group(), m.start(), m.end(), "HGVSVar") for m in HGVS.finditer(text)]
        genome = [(m.group(), m.start(), m.end(), "Refgenome") for m in GENOME_RE.finditer(text)]
        iscn = [(m.group(), m.start(), m.end(), "ISCNVar") for m in CYTOBAND.finditer(text)]
        
        # Gathering matches
        matches.extend(epmc_matches)
        matches.extend(genome)
        matches.extend(iscn)

        # Add geneprotein matches
        bert_results = run_gene_ner_chunked(tokenizer=tokenizer, text=text, pipe=ner_gene)
        # Keeping only 'GP' annotations
        bert_results = [(ent['word'], ent['start'], ent['end'], "GeneProtein") for ent in bert_results if ent.get('entity_group') == "GP"]


        # Catch star allele descriptions if present as 'Variant', otherwise keep as 'GeneProtein'
        star_checked = find_star_alleles(text, bert_results)
        matches.extend(star_checked)

        # Add to LabelStudio-compliant json
        for m in matches:
            ls_tasks.append({
                "value": {
                    "start": m[1],
                    "end":   m[2],
                    "text":  m[0],
                    "labels": [m[3]]
                },
                "id":        str(uuid.uuid4())[:10],
                "from_name": "label",
                "to_name":   "text",
                "type":      "labels",
                "origin":    "prediction"  # marks as pre-annotation, not human
            })

        all_tasks.append({
            "data": {
                "text":    text,
                "heading": sec["heading"] or "Body",
                "pmcid":   pmcid,
            },
            "annotations": [
                {
                    "result":       ls_tasks,
                    "was_cancelled": False,
                    "ground_truth": False,
                }
            ]
        })

# output_path = f"./output/genevar_outputs/variants_ls.json"
with open(output_path, "w") as f:
    json.dump(all_tasks, f, indent=2)
print(f"Written {len(all_tasks)} tasks to {output_path}")

