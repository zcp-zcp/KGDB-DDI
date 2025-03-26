import networkx as nx
from tqdm import tqdm
import torch
from torch_geometric.data import Data

def read_triples1(file_path):
    with open(file_path, 'r') as file:
        triples = [tuple(line.strip().split(',')) for line in file]
    return triples


def Generate_kg(dataset):
    file_path = ''
    if dataset == 'KEGG_DRUG':
        file_path = '../../Data/KEGG_DRUG/a_triple.txt'
    if dataset == 'Drugbank':
        file_path = '../../Data/Drugbank/triple.txt'
    G = nx.Graph()
    triples = read_triples1(file_path)
    for Head_entity, Tail_entity, Relationship in triples:
        if not G.has_node(Head_entity):
            G.add_node(Head_entity)
        if not G.has_node(Tail_entity):
            G.add_node(Tail_entity)
        G.add_edge(Head_entity, Tail_entity, label=Relationship)

    return G

def get_edge_label(G, edge):
    u, v = edge
    return G.edges[u, v]['label']


def Generate_features_index(G):

    num_nodes = G.number_of_nodes()
    num_features = 512
    x = torch.ones(num_nodes, num_features)  #

    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    edges = [(node_map[u], node_map[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    edge_attr = torch.zeros(len(edges), 32)

    for i, (u, v) in enumerate(edges):

        original_u = [node for node, idx in node_map.items() if idx == u][0]
        original_v = [node for node, idx in node_map.items() if idx == v][0]
        label = float(get_edge_label(G, (original_u, original_v)))
        edge_attr[i] = torch.tensor([label] * 32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data,node_map


