from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools


def tanimoto_similarity(fp1, fp2):  # Tanimoto similarity
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def AVALON_fingerprint(smiles):  # get AVALON fingerprint
    molecule = Chem.MolFromSmiles(smiles)
    fp = pyAvalonTools.GetAvalonFP(molecule)
    return fp


def get_similarity(original_molecule, transform_molecule):  # Get molecule-to-molecule similarity
    similarity = lambda m1, m2: tanimoto_similarity(m1, m2)
    m1 = AVALON_fingerprint(original_molecule)
    m2 = AVALON_fingerprint(transform_molecule)
    sim_score = similarity(m1, m2)
    return sim_score
