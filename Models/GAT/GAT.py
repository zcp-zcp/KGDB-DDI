import torch
import sys
sys.path.append("../..")
from KG import Generate_kg
from DATASet import DataSet
from torch_geometric.nn import GATConv
from sklearn.decomposition import PCA
import torch.nn as nn
import numpy as np

class TwoLayerGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels,heads=8, concat=True, dropout=0.6, bias=True):
        super(TwoLayerGAT, self).__init__()
        self.conv1 = GATConv(in_channels, 128, heads=heads, concat=concat,dropout=dropout,  bias=bias,edge_dim=32)
        first_layer_out_channels = 128 * heads if concat else 128
        self.dropout2 = nn.Dropout(p=dropout)
        self.conv2 = GATConv(first_layer_out_channels, out_channels, heads=heads, concat=concat, dropout=dropout,bias=bias,edge_dim=32)
        self.conv3 = GATConv(first_layer_out_channels, out_channels, heads=heads, concat=False, dropout=dropout,
                             bias=bias, edge_dim=32)
        self.conv2_1 = GATConv(first_layer_out_channels, out_channels, heads=heads, concat=False, dropout=dropout,
                             bias=bias, edge_dim=32)


    def forward(self, data, dataset):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.dropout2(x)
        if dataset == 'KEGG_DRUG':
            x = self.conv2(x, edge_index, edge_attr)
            x = torch.relu(x)
            x = self.dropout2(x)
            x = self.conv3(x, edge_index, edge_attr)
        if dataset == 'Drugbank':
            x = self.conv2_1(x, edge_index, edge_attr)

        return x


def all_drugs_fea(dataset):
    G = Generate_kg.Generate_kg(dataset=dataset)
    data, node_map = Generate_kg.Generate_features_index(G)
    out_channels = 128

    model = TwoLayerGAT(in_channels=data.x.size(1), out_channels=out_channels)
    outputs = model(data, dataset=dataset)

    x_numpy = outputs.detach().numpy()
    pca = PCA(n_components=64)
    pca_representations = pca.fit_transform(x_numpy)
    pca_representations_tensor = torch.from_numpy(pca_representations)
    return pca_representations_tensor,node_map,64


