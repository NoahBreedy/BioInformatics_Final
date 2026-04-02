import os
import re
import time
import pandas as pd
from Bio import Entrez, SeqIO

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

def main():
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
            "Cnidaria": ["UCP1", "UCP2", "UCP4", "UCP5"],
            # 
            "Nematoda": ["CPT6", "Y71H2AM.1", "C02F5.1"],
            # 
            "Echinodermata": ["UCP1", "UCP2", "UCP4", "UCP5"]
        },
        "Plantae": {
            # 
            "Eudicots": ["AtUCP1", "AtUCP2"],
            #
            "Monocots": ["OsUCP1", "OsUCP2"],
            #
            "Bryophytes": ["PpUCP1", "PpUCP2"]
        },
        "Fungi": {
            #
            "Ascomycota": ["YOR157C", "AUC1", "FUN26"]
        },
        "Protists": {
            #
            "Amoebozoa": ["UCP", "AAC", "MCP1"]
        }
    }
    
    # Get Sequence Ids
    for kingdom in KINGDOMS.keys():
        create_dir(kingdom)
        for clas in KINGDOMS[kingdom].keys():
            create_dir(f"{kingdom}/{clas}")
            for protein in KINGDOMS[kingdom][clas]:  # Iterate over protein names
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
                
                
    df = load_fasta_to_dataframe()
    
    df = df.drop_duplicates(subset=["Sequence"])
    
    df.to_csv("ucp_ml_dataset.csv", index=False)
    
    print(df.groupby(['Kingdom', 'Class']).size())
            
    return

if __name__ == "__main__":
    main()