import rdkit
from rdkit import Chem

def smile_to_molecule(smile: str):
    return Chem.MolFromSmiles(smile, sanitize=True)