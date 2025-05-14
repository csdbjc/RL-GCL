import pandas as pd
import torch

from loader import MoleculeDataset
from torch_geometric.loader import DataLoader, DataListLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import GRU
from torch_geometric.nn import MessagePassing, NNConv
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.inits import glorot, zeros
from ogb.graphproppred.mol_encoder import BondEncoder, AtomEncoder

from model import GNN
import torch
import torch.nn as nn
import torch.nn.functional as F

dataset = MoleculeDataset("data/" + 'C3F6', device=0, dataset_name='C3F6')
loader = DataLoader(dataset, batch_size=128, shuffle=False)
model = GNN(3, 300, 26, JK="last", drop_ratio=0.,
            graph_pooling="mean", gnn_type="gin")
model.predict_f = False
model.pretrain = False
model.load_state_dict(torch.load('gcn_model.pth'))
file_name = 'smiles_gcn.csv'
smiles = []
embs = []
for batch in loader:
    smiles += batch.smiles[0]
    emb = model(batch)
    embs.append(emb)
embedding = torch.cat(embs, dim=0)
print(embedding.shape)
torch.save(embedding, 'gcn_emb.pt')
df = pd.DataFrame(smiles, columns=['smiles'])
df.to_csv(file_name, index=False)