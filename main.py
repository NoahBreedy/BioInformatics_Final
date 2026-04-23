import os
import re
import time
import requests
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import CountVectorizer
from Bio import Entrez, SeqIO
from io import StringIO

Entrez.email = "hokansk@sunypoly.edu"  # NCBI requires this

def get_ucp_id(query, retmax=1000, retries=3):
    """Fetch protein IDs with retry logic for NCBI API failures."""
    for attempt in range(retries):
        try:
            handle = Entrez.esearch(db="protein", term=query, retmax=retmax)
            record = Entrez.read(handle)
            handle.close()
            return record.get("IdList", [])
        except RuntimeError as e:
            if attempt < retries - 1:
                wait_time = (attempt + 1) * 5 #5s, 10s, 15s
                print(f"  NCBI API error (attempt {attempt + 1}/{retries}): {e}")
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {retries} attempts. Skipping query: {query}")
                return []

def get_ucp_sequence(id_list):
    ids = ",".join(id_list)
    handle = Entrez.efetch(db="protein", id=ids, rettype="fasta", retmode="text")
    records = list(SeqIO.parse(handle, "fasta"))
    handle.close()
    return records




# UNIPROT FUNCTIONS 

UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"
 
def fetch_uniprot_sequences(gene_name: str, taxonomy_id: str,
                             max_results: int = 200,
                             reviewed_only: bool = False,
                             retries: int = 3) -> list:
    """
    Fetch protein sequences from UniProt REST API for a given gene name
    and NCBI taxonomy ID.
 
    Args:
        gene_name:     Gene symbol, e.g. "UCP1"
        taxonomy_id:   NCBI taxon ID string, e.g. "50557" (Insecta)
        max_results:   Cap on sequences returned (UniProt can return thousands)
        reviewed_only: If True, restrict to Swiss-Prot (manually reviewed) entries.
                       Useful for Mammalia where quality > quantity. For sparse
                       clades, keep False to include TrEMBL entries too.
        retries:       Number of retry attempts on network errors.
 
    Returns:
        List of Bio.SeqRecord objects, same format as get_ucp_sequence().
    """
    # Build query string
    # UniProt query syntax: gene_exact:"UCP1" AND taxonomy_id:50557
    query_parts = [f'gene_exact:"{gene_name}"', f"taxonomy_id:{taxonomy_id}"]
    if reviewed_only:
        query_parts.append("reviewed:true")
    query = " AND ".join(query_parts)
 
    params = {
        "query":  query,
        "format": "fasta",
        "size":   max_results,
    }
 
    for attempt in range(retries):
        try:
            response = requests.get(UNIPROT_API, params=params, timeout=30)
            response.raise_for_status()
 
            fasta_text = response.text.strip()
            if not fasta_text:
                return []
 
            records = list(SeqIO.parse(StringIO(fasta_text), "fasta"))
            return records
 
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"  UniProt API error (attempt {attempt + 1}/{retries}): {e}")
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  UniProt failed after {retries} attempts. Skipping {gene_name} / taxon {taxonomy_id}")
                return []
 
    return []
 
 
def write_uniprot_fasta(records: list, filepath: str):
    """Append UniProt records to an existing FASTA file (or create it)."""
    if records:
        mode = "a" if os.path.exists(filepath) else "w"
        with open(filepath, mode) as f:
            SeqIO.write(records, f, "fasta")


#==========================

def create_dir(directory_name):
    try:
        os.mkdir(directory_name)
    except FileExistsError:
        return
    except FileNotFoundError:
        return
    
def load_fasta_to_dataframe(base_path="."):
    """
    Crawls the kingdom/class directory structure and parses FASTA files 
    into a labeled Pandas DataFrame.
    """
    data_list = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".fasta"):
                # Path: ./Kingdom/Class/Class_Protein.fasta
                parts = root.split(os.sep)
                if len(parts) >= 2:
                    kingdom = parts[-2]
                    tax_class = parts[-1]
                    
                    file_path = os.path.join(root, file)
                    
                    # Parse each sequence in the FASTA file
                    for record in SeqIO.parse(file_path, "fasta"):
                        # Extract organism name from header: "UCP1 [Mus musculus]" -> "Mus musculus"
                        org_match = re.search(r'\[(.*?)\]', record.description)
                        organism = org_match.group(1) if org_match else "Unknown"
                        
                        data_list.append({
                            "Kingdom": kingdom,
                            "Class": tax_class,
                            "Protein_Type": file.replace(".fasta", "").split("_")[-1],
                            "Organism": organism,
                            "Sequence": str(record.seq),
                            "Seq_Length": len(record.seq)
                        })
    
    return pd.DataFrame(data_list)

 # Define protein IDs for each kingdom/class
KINGDOMS = {
    "Animalia": {
        # 
        "Mammalia": ["UCP1", "UCP2", "UCP3", "UCP4", "UCP5"],
        # 
        "Actinopterygii": ["UCP1", "UCP2", "UCP3", "UCP4", "UCP5"],
        #
        "Insecta": ["DmUCP4A", "DmUCP4B", "DmUCP4C", "DmUCP5"],
        # 
        "Sauropsida": ["UCP1", "UCP2", "UCP3", "UCP4", "UCP5"],
        # 
        # "Cnidaria": ["UCP1", "UCP2", "UCP4", "UCP5"],
        # # 
        # "Nematoda": ["CPT6", "Y71H2AM.1", "C02F5.1"],
        # # 
        # "Echinodermata": ["UCP1", "UCP2", "UCP4", "UCP5"]
    },
    # "Plantae": {
    #     # 
    #     "Eudicots": ["AtUCP1", "AtUCP2"],
    #     #
    #     "Monocots": ["OsUCP1", "OsUCP2"],
    #     #
    #     "Bryophytes": ["PpUCP1", "PpUCP2"]
    # },
    "Fungi": {
        "Ascomycota": ["YOR157C", "AUC1", "FUN26"]
    },
    # "Protists": {
    #     "Amoebozoa": ["UCP", "AAC", "MCP1"]
    # }
}

UNIPROT_SUPPLEMENT = {
    "Animalia": {
        # "Mammalia": {
        #     "ncbi_taxon_id": "40674",
        #     "proteins":      ["UCP4", "UCP5"],
        #     "reviewed_only": False,
        #     "max_results":   150,
        #     "kingdom":       "Animalia",
        # },
        # "Actinopterygii": {
        #     "ncbi_taxon_id": "7898",
        #     "proteins":      ["UCP3", "UCP4", "UCP5"],
        #     "reviewed_only": False,
        #     "max_results":   150,
        #     "kingdom":       "Animalia",
        # },
        "Insecta": {
            "ncbi_taxon_id": "50557",
            "proteins":      ["UCP4", "UCP5"],   # broader names for UniProt
            "reviewed_only": False,
            "max_results":   150,
            "kingdom":       "Animalia",
        },
        "Sauropsida": {
            "ncbi_taxon_id": "8457",             # covers reptiles + birds
            "proteins":      ["UCP1", "UCP2", "UCP3"],
            "reviewed_only": False,
            "max_results":   150,
            "kingdom":       "Animalia",
        },
        # "Cnidaria": {
        #     "ncbi_taxon_id": "6073",
        #     "proteins":      ["UCP", "AAC"],     # UCP homologs in cnidarians
        #     "reviewed_only": False,
        #     "max_results":   100,
        #     "kingdom":       "Animalia",
        # },
        # "Nematoda": {
        #     "ncbi_taxon_id": "6231",
        #     "proteins":      ["UCP", "AAC"],
        #     "reviewed_only": False,
        #     "max_results":   150,
        #     "kingdom":       "Animalia",
        # },
        # "Echinodermata": {
        #     "ncbi_taxon_id": "7586",
        #     "proteins":      ["UCP", "AAC"],
        #     "reviewed_only": False,
        #     "max_results":   150,
        #     "kingdom":       "Animalia",
        # },
    },
    # "Plantae": {
    #     "Eudicots": {
    #         "ncbi_taxon_id": "71240",
    #         "proteins":      ["UCP1", "UCP2"],
    #         "reviewed_only": False,
    #         "max_results":   150,
    #         "kingdom":       "Plantae",
    #     },
    #     "Monocots": {
    #         "ncbi_taxon_id": "4447",
    #         "proteins":      ["UCP1", "UCP2"],
    #         "reviewed_only": False,
    #         "max_results":   150,
    #         "kingdom":       "Plantae",
    #     },
    #     "Bryophytes": {
    #         "ncbi_taxon_id": "3208",
    #         "proteins":      ["UCP1", "UCP2"],
    #         "reviewed_only": False,
    #         "max_results":   150,
    #         "kingdom":       "Plantae",
    #     },
    # },
    "Fungi": {
        "Ascomycota": {
            "ncbi_taxon_id": "4890",
            "proteins":      ["UCP", "AAC"],
            "reviewed_only": False,
            "max_results":   150,
            "kingdom":       "Fungi",
        },
    },
    # "Protists": {
    #     "Amoebozoa": {
    #         "ncbi_taxon_id": "554915",
    #         "proteins":      ["UCP", "AAC", "MCP"],
    #         "reviewed_only": False,
    #         "max_results":   150,
    #         "kingdom":       "Protists",
    #     },
    # },
}

def build_kmer_matrix(df, k=3):
    seqs = df["Sequence"].astype(str).tolist()

    vect = CountVectorizer(
        analyzer="char",
        ngram_range=(k, k)
    )

    X = vect.fit_transform(seqs)
    return X.toarray()

def compute_kmer_distance(df, k=3):
    X = build_kmer_matrix(df, k)
    dist = squareform(pdist(X, metric="euclidean"))
    return dist

# ----------------------------------------------------------
# 3. Build taxonomy distance matrix
#
# same class      = 0
# same kingdom    = 1
# different group = 2
# ----------------------------------------------------------
def compute_taxonomy_distance(df):
    n = len(df)
    tax = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                tax[i, j] = 0
            elif df.iloc[i]["Class"] == df.iloc[j]["Class"]:
                tax[i, j] = 0
            elif df.iloc[i]["Kingdom"] == df.iloc[j]["Kingdom"]:
                tax[i, j] = 1
            else:
                tax[i, j] = 2

    return tax


def mantel_test(mat1, mat2, perms=999):
    """
    Simple Mantel test using Pearson correlation
    """

    # upper triangle only
    idx = np.triu_indices_from(mat1, k=1)

    x = mat1[idx]
    y = mat2[idx]

    obs_r, _ = pearsonr(x, y)

    count = 0
    n = mat1.shape[0]

    for _ in range(perms):
        perm = np.random.permutation(n)
        permuted = mat2[perm][:, perm]
        y_perm = permuted[idx]

        r_perm, _ = pearsonr(x, y_perm)

        if abs(r_perm) >= abs(obs_r):
            count += 1

    pval = (count + 1) / (perms + 1)

    return obs_r, pval

def main():
   
    # NCBI fetch
    for kingdom in KINGDOMS.keys():
        create_dir(kingdom)
        for clas in KINGDOMS[kingdom].keys():
            create_dir(f"{kingdom}/{clas}")
            for protein in KINGDOMS[kingdom][clas]:  # Iterate over protein names
                if (clas == "Ascomycota"):
                    query = f"{protein}[Gene Name] AND {clas}[Organism]"
                else:
                    query = f"{protein}[Gene Name] AND {clas}[Organism] AND RefSeq[Filter]" 
                
                print(f"Fetching: {protein} for {clas} in {kingdom}...", end=" ")
                ids = get_ucp_id(query)
                
                if not ids:
                    print("(no results)")
                    continue
                    
                print(f"({len(ids)} sequences found)")
                sequences = get_ucp_sequence(ids)
                SeqIO.write(sequences, f"{kingdom}/{clas}/{clas}_{protein}.fasta", "fasta")
                print(f"  → File Written: {kingdom}/{clas}/{clas}_{protein}.fasta")
                time.sleep(1)  # NCBI servers timing
                
    #UniProt supplement
    for kingdom, classes in UNIPROT_SUPPLEMENT.items():
            for clas, config in classes.items():
                taxon_id     = config["ncbi_taxon_id"]
                proteins     = config["proteins"]
                reviewed     = config["reviewed_only"]
                max_results  = config["max_results"]
                fasta_dir    = f"{kingdom}/{clas}"
    
                create_dir(kingdom)
                create_dir(fasta_dir)
    
                for protein in proteins:
                    print(f"Fetching UniProt: {protein} for {clas} (taxon {taxon_id})...", end=" ")
                    records = fetch_uniprot_sequences(
                        gene_name=protein,
                        taxonomy_id=taxon_id,
                        max_results=max_results,
                        reviewed_only=reviewed,
                    )
    
                    if not records:
                        print("(no results)")
                        continue
    
                    print(f"({len(records)} sequences)")
                    # Append into the same per-class FASTA files so
                    # load_fasta_to_dataframe() picks them up automatically.
                    fasta_path = f"{fasta_dir}/{clas}_{protein}.fasta"
                    write_uniprot_fasta(records, fasta_path)
                    print(f"  → Written/appended: {fasta_path}")
                    time.sleep(0.5)

    df = load_fasta_to_dataframe()
    df = df.drop_duplicates(subset=["Sequence"])
    df.to_csv("ucp_ml_dataset.csv", index=False)
    
    print(f"\nTotal sequences after deduplication: {len(df)}")
    print("\nBreakdown by Kingdom / Class:")
    print(df.groupby(['Kingdom', 'Class']).size())

    print("\nRunning Mantel test...")

    kmer_dist = compute_kmer_distance(df, k=3)
    tax_dist  = compute_taxonomy_distance(df)

    r, p = mantel_test(kmer_dist, tax_dist, perms=999)

    print(f"Mantel correlation r = {r:.4f}")
    print(f"P-value              = {p:.4f}")

    if p < 0.05:
        print("Significant result: k-mer composition tracks phylogeny.")
    else:
        print("No significant relationship detected.")

    return

if __name__ == "__main__":
    main()