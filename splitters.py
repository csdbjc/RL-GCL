import torch
import random
import numpy as np
from itertools import compress
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from tqdm import tqdm


def generate_scaffold(mol, include_chirality=False):
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def scaffold_to_smiles(mols, use_indices=False):
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)
    return scaffolds


def scaffold_split(dataset, smiles_list, balanced=False, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    # Split
    train_size, val_size, test_size = frac_train * len(dataset), frac_valid * len(dataset), frac_test * len(dataset)
    train_idx, val_idx, test_idx = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(smiles_list, use_indices=True)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)
    for index_set in index_sets:
        if len(train_idx) + len(index_set) <= train_size:
            train_idx += index_set
            train_scaffold_count += 1
        elif len(val_idx) + len(index_set) <= val_size:
            val_idx += index_set
            val_scaffold_count += 1
        else:
            test_idx += index_set
            test_scaffold_count += 1

    assert len(set(train_idx).intersection(set(val_idx))) == 0
    assert len(set(test_idx).intersection(set(val_idx))) == 0
    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(val_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    return train_dataset, valid_dataset, test_dataset


def random_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):  # split the dataset randomly
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    if seed is not None:
        random.seed(seed)
    index = list(range(len(dataset)))
    random.shuffle(index)
    train_size = int(frac_train * len(dataset))
    val_size = int(frac_valid * len(dataset))
    train_dataset = dataset[index[:train_size]]
    val_dataset = dataset[index[train_size:train_size + val_size]]
    test_dataset = dataset[index[train_size + val_size:]]

    return train_dataset, val_dataset, test_dataset
