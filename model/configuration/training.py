import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataset.download_elements import download_periodic
import csv

def atoms():
    filename = download_periodic()
    symbols = set()
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            symbols.add(str(row[1]))
    return {
        "element": symbols,
        "valence_electrons": {0, 1, 2, 3, 4, 5, 6},
        "hydrogen_bonds": {0, 1, 2, 3, 4},
        "orbital_hybridization": {"s", "sp", "sp2", "sp3"},
    }

def bonds():
    return {
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugation_state": {True, False},
    }