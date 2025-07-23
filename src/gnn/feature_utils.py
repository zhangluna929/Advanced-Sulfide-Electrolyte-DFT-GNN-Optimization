"""特征工具模块"""
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Tuple

ATOM_TYPES = [
    "H", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"
]

ATOM_PROPERTIES = {
    "H": {"atomic_number": 1, "atomic_mass": 1.008, "electronegativity": 2.20, "covalent_radius": 0.31},
    "Li": {"atomic_number": 3, "atomic_mass": 6.941, "electronegativity": 0.98, "covalent_radius": 1.28},
    "P": {"atomic_number": 15, "atomic_mass": 30.974, "electronegativity": 2.19, "covalent_radius": 1.07},
    "S": {"atomic_number": 16, "atomic_mass": 32.065, "electronegativity": 2.58, "covalent_radius": 1.05},
    "Cl": {"atomic_number": 17, "atomic_mass": 35.453, "electronegativity": 3.16, "covalent_radius": 1.02},
    "Br": {"atomic_number": 35, "atomic_mass": 79.904, "electronegativity": 2.96, "covalent_radius": 1.14},
    "I": {"atomic_number": 53, "atomic_mass": 126.904, "electronegativity": 2.66, "covalent_radius": 1.33},
}

def get_atom_features(atom_type: str) -> torch.Tensor:
    if atom_type in ATOM_PROPERTIES:
        props = ATOM_PROPERTIES[atom_type]
        features = [
            props["atomic_number"] / 100.0,
            props["atomic_mass"] / 300.0,
            props["electronegativity"] / 4.0,
            props["covalent_radius"] / 2.0,
        ]
    else:
        features = [0.0, 0.0, 0.0, 0.0]
    
    return torch.tensor(features, dtype=torch.float)

def get_edge_features(distance: float) -> torch.Tensor:
    features = [
        distance,
        1.0 / (distance + 1e-6),
        np.exp(-distance),
        np.exp(-distance / 2.0),
    ]
    return torch.tensor(features, dtype=torch.float)

def encode_atom_type(atom_type: str) -> int:
    return ATOM_TYPES.index(atom_type) if atom_type in ATOM_TYPES else 0

def get_bond_features(atom1: str, atom2: str, distance: float) -> torch.Tensor:
    atom1_features = get_atom_features(atom1)
    atom2_features = get_atom_features(atom2)
    edge_features = get_edge_features(distance)
    
    bond_features = torch.cat([atom1_features, atom2_features, edge_features])
    return bond_features 