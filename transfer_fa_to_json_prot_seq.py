

import json

protein_data = {}
current_key = None
sequence = ""

with open("/raid/home/yoyowu/MicrobiomeMeta/Data/HMDB/protein.fasta", "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            # If we encounter a new protein, we save the current one and reset the key and sequence
            if current_key and sequence:
                protein_data[current_key] = sequence
                sequence = ""
            # Extract the CHEMBL ID
            #current_key = line.split()[1]  # for chembl
            current_key = line.split()[0].lstrip(">") # for hmdb
            check = 1
        else:
            sequence += line

    # Save the last protein sequence
    if current_key and sequence:
        protein_data[current_key] = sequence

# Save the data to a JSON file
with open("/raid/home/yoyowu/transformerCPI/dataset/chembl/10042023_hmdb_prot_seq.json", "w") as f:
    json.dump(protein_data, f)

print("Data saved to output_file.json")
