import torch
from torch.utils.data import Dataset
from mol_featurizer import *
from word2vec import seq_to_kmers, get_protein_embedding
from gensim.models import Word2Vec
import json
import random

def load_data_map(smiles_mapping_path, protein_seq_mapping_path, chembl_uniprot_mapping_path):
    # Load mappings...
    # Load SMILES mapping
    smiles_mapping = {}
    # if the file is tsv, use the following code
    if smiles_mapping_path.endswith(".tsv"):
        with open(smiles_mapping_path, "r") as f:
            for line in f:
                _,chembl_id, smiles = line.strip().split('\t')
                smiles_mapping[chembl_id] = smiles
    else:
        # if the file is json, use the following code
        with open(smiles_mapping_path, "r") as f:
            smiles_mapping = json.load(f)
    # Load protein sequence mapping     
    with open(protein_seq_mapping_path, "r") as f:
        protein_seq_mapping = json.load(f)

    # Load Chembl to Uniprot mapping
    with open(chembl_uniprot_mapping_path, "r") as f:
        chembl_uniprot_mapping = json.load(f)
    
    return  smiles_mapping, protein_seq_mapping, chembl_uniprot_mapping


class ProteinCompoundDataset(Dataset):
    def __init__(self, data, smiles_mapping, protein_seq_mapping, chembl_uniprot_mapping, unlabeled=False):
        self.data = data
        self.smiles_mapping = smiles_mapping
        self.protein_seq_mapping = protein_seq_mapping
        self.chembl_uniprot_mapping = chembl_uniprot_mapping
        self.word_model = Word2Vec.load("/raid/home/yoyowu/TransformerCPI/GPCR/word2vec_30.model")
        self.unlabeled = unlabeled

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.unlabeled:
            compound_id, uniprot_id  = self.data[idx]
        else:
            compound_id, uniprot_id, label = self.data[idx]
            # convert label to float if it is string 
            if isinstance(label, str):
                label = float(label)
        
        smiles = self.smiles_mapping.get(compound_id, None)
        
        chembl_id_for_protein = next((k for k, v in self.chembl_uniprot_mapping.items() if v == uniprot_id), None)

        protein_seq = self.protein_seq_mapping.get(chembl_id_for_protein, None)
        if protein_seq is not None:
            protein_embedding = torch.tensor(get_protein_embedding(self.word_model, seq_to_kmers(protein_seq)))
        else:
            protein_embedding = None
        
        if smiles is None or protein_seq is None or chembl_id_for_protein is None:
            # Handle missing data
            return None
        
        atom_feature, adj = mol_features(smiles)
        if self.unlabeled:
            return torch.tensor(atom_feature), torch.tensor(adj), protein_embedding
        else:
            return torch.tensor(atom_feature), torch.tensor(adj), protein_embedding, torch.tensor(int(label))



def pack(atoms, adjs, proteins, labels):
    atoms_len = 0
    proteins_len = 0

    N = len(atoms)

    atom_num = torch.zeros((N, 1))
    i = 0
    for atom in atoms:
        atom_num[i] = atom.shape[0]
        i += 1
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]

    protein_num = torch.zeros((N, 1))
    i = 0
    for protein in proteins:
        protein_num[i] = protein.shape[0]
        i += 1
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]

    atoms_new = torch.zeros((N, atoms_len, 34))
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, :a_len, :] = atom
        i += 1

    adjs_new = torch.zeros((N, atoms_len, atoms_len))
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        adjs_new[i, :a_len, :a_len] = adj
        i += 1

    proteins_new = torch.zeros((N, proteins_len, 100))
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        proteins_new[i, :a_len, :] = protein
        i += 1

    labels_new = torch.zeros(N)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1
    return atoms_new, adjs_new, proteins_new, labels_new, atom_num, protein_num


def collate_fn(batch):
    """
    Args batch: list of data, each atom, adj, protein, label = data
    """
    atoms, adjs, proteins, labels = zip(*batch)
    return pack(atoms, adjs, proteins, labels)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)