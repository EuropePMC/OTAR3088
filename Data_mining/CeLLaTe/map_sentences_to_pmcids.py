import os
import glob
import re
import pandas as pd

def load_single_cell_lookup():
    """
    Loads raw Single_Cell texts and creates a dictionary mapping PMCID to the normalized full-text string.
    """
    single_cell_dir = '/Users/withers/Downloads/All_annotations/Single_Cell/'
    sc_files = glob.glob(os.path.join(single_cell_dir, '*.tsv'))
    lookup = {}
    print(f"Loading {len(sc_files)} raw Single_Cell TSVs...")
    for f in sc_files:
        pmcid = os.path.basename(f).replace('.tsv', '')
        df_sc = pd.read_csv(f, sep='\t')
        full_text = ' '.join(df_sc['text'].dropna().astype(str).tolist())
        lookup[pmcid] = ' '.join(full_text.split())
    return lookup

def load_chembl_lookup():
    """
    Loads raw ChEMBL texts from ChEMBL_assay_desc.tsv and maps PMCID to the normalized full-text string.
    """
    chembl_file = '/Users/withers/Downloads/All_annotations/ChEMBL_assay_desc.tsv'
    print(f"Loading raw ChEMBL texts from {chembl_file}...")
    chembl_src = pd.read_csv(chembl_file, sep='\t')
    lookup = {}
    for idx, row in chembl_src.iterrows():
        pmcid = row['PMCID']
        text = str(row['text'])
        lookup[pmcid] = ' '.join(text.split())
    return lookup

def load_cellfinder_lookup():
    """
    Loads raw CellFinder texts from cellfinder1_brat text files, and maps PMCIDs using the PMID-to-PMCID mapping.
    """
    # PMID to PMCID mapping resolved from Europe PMC web service
    pmid_to_pmcid = {
        '15971941': 'PMC1160574',
        '16316465': 'PMC1315352',
        '16623949': 'PMC1462997',
        '16672070': 'PMC1523200',
        '17288595': 'PMC1802744',
        '17381551': 'PMC2063610',
        '17389645': 'PMC1885650',
        '17967047': 'PMC2041973',
        '18162134': 'PMC2211323',
        '18286199': 'PMC2238795'
    }
    cf_dir = '/Users/withers/Downloads/cellfinder1_brat/'
    cf_files = glob.glob(os.path.join(cf_dir, '*.txt'))
    lookup = {}
    print(f"Loading {len(cf_files)} raw CellFinder txt files...")
    for f in cf_files:
        pmid = os.path.basename(f).replace('.txt', '')
        pmcid = pmid_to_pmcid.get(pmid)
        if pmcid:
            with open(f, 'r', encoding='utf-8') as file_obj:
                content = file_obj.read()
            lookup[pmcid] = ' '.join(content.split())
    return lookup

def map_pmcid_to_df(df_path, sc_lookup, chembl_lookup, cf_lookup):
    """
    Reads the target dataset TSV, maps each sentence to a PMCID using normalized substring matching,
    and returns a DataFrame with 'PMCID' as the first column.
    """
    df = pd.read_csv(df_path, sep='\t')
    pmcids = []
    
    for idx, row in df.iterrows():
        sentence = str(row['sentence'])
        # Remove any metadata prefix/suffix if present
        if '\nSource paper:' in sentence:
            sentence = sentence.split('\nSource paper:')[0]
            
        clean_sent = ' '.join(sentence.split())
        data_source = row['data_source']
        found_pmcid = None
        
        if data_source == 'Single_Cell':
            for pmcid, text in sc_lookup.items():
                if clean_sent in text:
                    found_pmcid = pmcid
                    break
        elif data_source == 'Chembl_V2':
            for pmcid, text in chembl_lookup.items():
                if clean_sent in text:
                    found_pmcid = pmcid
                    break
        elif data_source == 'CellFinder':
            for pmcid, text in cf_lookup.items():
                if clean_sent in text:
                    found_pmcid = pmcid
                    break
                    
        pmcids.append(found_pmcid)
        
    df['PMCID'] = pmcids
    
    # Rearrange columns
    cols = ['PMCID', 'sentence', 'entities', 'data_source']
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df[cols]

def main():
    # Directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load raw source lookups
    sc_lookup = load_single_cell_lookup()
    chembl_lookup = load_chembl_lookup()
    cf_lookup = load_cellfinder_lookup()
    
    # Define inputs and outputs
    inputs = {
        'cellate_final.tsv': 'cellate_final_with_pmcid.tsv',
        'cellate_final_with_vague.tsv': 'cellate_final_with_vague_with_pmcid.tsv'
    }
    
    for in_name, out_name in inputs.items():
        in_path = os.path.join(current_dir, in_name)
        out_path = os.path.join(current_dir, out_name)
        
        if os.path.exists(in_path):
            print(f"\nProcessing {in_name}...")
            mapped_df = map_pmcid_to_df(in_path, sc_lookup, chembl_lookup, cf_lookup)
            
            # Count matches
            total = len(mapped_df)
            matched = mapped_df['PMCID'].notna().sum()
            print(f"Matched {matched} out of {total} sentences ({matched/total*100:.2f}%)")
            
            # Save results
            mapped_df.to_csv(out_path, sep='\t', index=False)
            print(f"Saved mapped dataset to {out_path}")
        else:
            print(f"\nError: Input file {in_path} does not exist.")

if __name__ == '__main__':
    main()
