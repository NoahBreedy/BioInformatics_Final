import os
from Bio import Entrez, SeqIO

Entrez.email = "hokansk@sunypoly.edu"  # NCBI requires this

def get_ucp_id(query, retmax=1000):
    handle = Entrez.esearch(db="protein", term=query, retmax=retmax)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

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

def main():
    PROTEIN_IDS = [
                   ["UCP1","UCP2","UCP3","UCP4","UCP5"],        # for animalia kingdom
                   ["DmUCP4A", "DmUCP4B", "DmUCP4C", "DmUCP5"], # for Insecta class
                   ["AtUCP1", "AtUCP2"]                         # for plantae kingdom
    ]

    CLASSES = [
            {
               "Mammalia": PROTEIN_IDS[0], 
               "Actinopterygii":PROTEIN_IDS[0],
               "Insecta": PROTEIN_IDS[1],
               "Sauropsida": PROTEIN_IDS[0]
            },
            {
               "Eudicots": PROTEIN_IDS[2], 
               "Monocots":PROTEIN_IDS[2],
            }
    ]

    KINGDOMS = {
        "Animalia": CLASSES[0],
        "Plantae":  CLASSES[1]
    }
    
    # Get Sequence Ids
    for kingdom in KINGDOMS.keys():
        create_dir(kingdom)
        for clas in KINGDOMS[kingdom].keys():
            create_dir(f"{kingdom}/{clas}")
            for protein in range(0,len(KINGDOMS[kingdom])):
                query = f"{KINGDOMS[kingdom][clas][protein]}[Gene Name] AND {clas}[Organism] AND RefSeq[Filter]"
                ids = get_ucp_id(query)
                sequences = get_ucp_sequence(ids)
                SeqIO.write(sequences, f"{kingdom}/{clas}/{clas}_{KINGDOMS[kingdom][clas][protein]}.fasta", "fasta")
                print(f"File Written: {kingdom}/{clas}/{clas}_{KINGDOMS[kingdom][clas][protein]}.fasta")
            
    return

if __name__ == "__main__":
    main()