import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import GRU
from torch_geometric.nn import MessagePassing, NNConv
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.inits import glorot, zeros
from ogb.graphproppred.mol_encoder import BondEncoder, AtomEncoder


import torch
import torch.nn as nn
import torch.nn.functional as F

class MPNNLayer(MessagePassing):
    def __init__(self, in_node, in_edge, out_node):
        super(MPNNLayer, self).__init__(aggr='add')  # 或 'mean' / 'max'
        self.node_mlp = nn.Linear(in_node, out_node)
        self.edge_mlp = nn.Linear(in_edge, out_node)
        self.update_mlp = nn.Sequential(
            nn.Linear(out_node + in_node, out_node),
            nn.ReLU(),
            nn.Linear(out_node, out_node)
        )

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, in_node]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, in_edge]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: 源节点的特征，edge_attr: 边特征
        msg = self.node_mlp(x_j.float()) + self.edge_mlp(edge_attr.float())
        return F.relu(msg)

    def update(self, aggr_out, x):
        # aggr_out: 聚合后的消息，x: 原始节点特征
        combined = torch.cat([aggr_out, x], dim=1)
        return self.update_mlp(combined)

class MPNN(nn.Module):
    def __init__(self, in_node, in_edge, hidden_dim, out_dim, num_layers=3):
        super(MPNN, self).__init__()
        self.node_proj = nn.Linear(in_node, hidden_dim)
        self.layers = nn.ModuleList([
            MPNNLayer(hidden_dim, in_edge, hidden_dim)
            for _ in range(num_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        x = self.node_proj(x.float())
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # 图级表示
        if batch is not None:
            from torch_geometric.nn import global_add_pool
            graph_repr = global_add_pool(x, batch)
        else:
            graph_repr = torch.sum(x, dim=0, keepdim=True)

        return self.readout(graph_repr)



class WeaveModule(nn.Module):
    def __init__(self, in_node, in_edge, out_node, out_edge):
        super(WeaveModule, self).__init__()
        self.node_to_node = nn.Linear(in_node, out_node)
        self.edge_to_node = nn.Linear(in_edge, out_node)

        self.node_pair_to_edge = nn.Linear(2 * in_node, out_edge)
        self.edge_to_edge = nn.Linear(in_edge, out_edge)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index  # row: source node idx, col: target node idx

        # === 更新节点特征 ===
        edge_msg = self.edge_to_node(edge_attr.float())  # edge -> node messages
        agg_msg = torch.zeros(x.size(0), edge_msg.size(1), device=x.device, dtype=edge_msg.dtype)         # 聚合边的消息到节点
        agg_msg.index_add_(0, row, edge_msg)    # 每条边的 source 节点聚合对应边特征
        node_out = self.node_to_node(x.float()) + agg_msg
        node_out = F.relu(node_out)

        # === 更新边特征 ===
        src_node = x[row]
        tgt_node = x[col]
        node_pair = torch.cat([src_node, tgt_node], dim=-1)
        edge_out = self.node_pair_to_edge(node_pair.float()) + self.edge_to_edge(edge_attr.float())
        edge_out = F.relu(edge_out)

        return node_out, edge_out

class WeaveModel(nn.Module):
    def __init__(self, in_node, in_edge, hidden_dim, n_layers=2, out_dim=128):
        super(WeaveModel, self).__init__()
        self.weave_layers = nn.ModuleList()
        for i in range(n_layers):
            self.weave_layers.append(
                WeaveModule(
                    in_node if i == 0 else hidden_dim,
                    in_edge if i == 0 else hidden_dim,
                    hidden_dim,
                    hidden_dim
                )
            )
        self.emb_dim = hidden_dim
        self.num_tasks = out_dim
        ffn_hidden_size = int(self.emb_dim / 2)
        ffn_num_layers = 2
        dropout = nn.Dropout(0.0)
        activation = nn.ReLU()
        if ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(self.emb_dim, self.num_tasks)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(self.emb_dim, ffn_hidden_size)
            ]
            for _ in range(ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(ffn_hidden_size, ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(ffn_hidden_size, self.num_tasks),
            ])

        # Create FFN model
        self.pred = nn.Sequential(*ffn)

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        for weave in self.weave_layers:
            x, edge_attr = weave(x, edge_index, edge_attr)

        # 如果提供了 batch 信息（比如通过 DataLoader），做全局池化
        if batch is not None:
            from torch_geometric.nn import global_add_pool
            graph_repr = global_add_pool(x, batch)
        else:
            # 否则默认对所有节点求和（单图）
            graph_repr = torch.sum(x, dim=0, keepdim=True)
        return self.pred(graph_repr)


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr = "add")
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    # def message(self, x_j, edge_attr):
    #     return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

# GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNN_Node(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, drop_ratio=0.5, JK="last", residual=False, gnn_type='gin'):
        super(GNN_Node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        # add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.gnn_type = gnn_type

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        # computing input node embedding
        h_list = [self.atom_encoder(x)]

        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0., JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### computing input node embedding

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


class GNN(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, residual = False, proj_dim=128, JK = "last", drop_ratio = 0., graph_pooling = "mean", gnn_type = "gin", pretrain=False):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.pretrain = pretrain
        self.predict_f = False

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)

        self.proj_head = nn.Sequential(nn.Linear(emb_dim, proj_dim), nn.ReLU(inplace=True), nn.Linear(proj_dim, proj_dim))
        ffn_hidden_size = int(self.emb_dim / 2)
        ffn_num_layers = 2
        dropout = nn.Dropout(0.0)
        activation = nn.ReLU()
        if ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(self.emb_dim, self.num_tasks)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(self.emb_dim, ffn_hidden_size)
            ]
            for _ in range(ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(ffn_hidden_size, ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(ffn_hidden_size, self.num_tasks),
            ])

        # Create FFN model
        self.pred = nn.Sequential(*ffn)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

    def save_pretrained(self, file):
        torch.save(self.state_dict(), file)

    def forward(self, batch_data):
        node_representation = self.gnn(batch_data)
        graph_representation = self.pool(node_representation, batch_data.batch)
        if self.predict_f:
            return self.pred(graph_representation)
        if self.pretrain:
            return F.normalize(self.proj_head(graph_representation), dim=1)
            # return graph_representation
        else:
            return graph_representation


class VertexEmbedding(nn.Module):
    def __init__(self, num_atom_types, emb_dim):
        super().__init__()
        self.atom_embedding = nn.Linear(num_atom_types, emb_dim)

    def forward(self, x):
        return self.atom_embedding(x)  # x 是原子类型的索引张量 [num_nodes]


def extract_n_grams(edge_index, num_nodes, n=3, num_walks=100):
    import random
    from collections import defaultdict

    adj = defaultdict(list)
    row, col = edge_index
    for r, c in zip(row.tolist(), col.tolist()):
        adj[r].append(c)

    walks = []
    for node in range(num_nodes):
        for _ in range(num_walks):
            walk = [node]
            for _ in range(n - 1):
                curr = walk[-1]
                if adj[curr]:
                    walk.append(random.choice(adj[curr]))
                else:
                    break
            if len(walk) == n:
                walks.append(walk)
    return walks  # List of lists, each of length n
class NGramGraphEncoder(nn.Module):
    def __init__(self, emb_dim, n):
        super().__init__()
        self.n = n
        self.linear = nn.Linear(n * emb_dim, emb_dim)

    def forward(self, x, walks):
        n_gram_embeddings = []
        for walk in walks:
            gram = torch.cat([x[idx] for idx in walk], dim=0)
            n_gram_embeddings.append(gram)
        n_gram_embeddings = torch.stack(n_gram_embeddings)  # [num_walks, n * emb_dim]
        # graph_emb = self.linear(n_gram_embeddings).mean(dim=0)  # mean pooling
        return n_gram_embeddings

class NGramGraphModel(nn.Module):
    def __init__(self, num_atom_types, emb_dim, n=3, num_walks=3, out_dim=1):
        super().__init__()
        self.emb_dim = emb_dim
        self.vertex_embedding =  AtomEncoder(emb_dim)
        self.encoder = NGramGraphEncoder(emb_dim, n)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, out_dim)
        )
        self.n = n
        self.num_walks = num_walks

    def forward(self, data):
        x = self.vertex_embedding(data.x)  # [num_nodes, emb_dim]
        walks = extract_n_grams(data.edge_index, data.x.shape[0], n=self.n, num_walks=self.num_walks)
        graph_emb = self.encoder(x, walks)  # [emb_dim]
        out = self.classifier(graph_emb)  # [1, out_dim]
        return out
