from torch.utils.data import Dataset, DataLoader
import torch
import sys
sys.path.append("../")
from Text import noise_reduction
# from models.Attention import attention
def get_node_text_features(node, output, node_map):
    output = output.detach()
    idx = node_map[node]
    if idx is not None:
        return output[idx]

class InteractionDataset(Dataset):
    def __init__(self, file_path,output, node_map,dataset):
        reduced_embeddings_dict = noise_reduction.noice_reduction_pca(dataset=dataset)
        # self.data = []
        self.labels = []
        self.node1_features = []
        self.node2_features = []
        self.drug1_text_emb = []
        self.drug2_text_emb = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                node1_features = get_node_text_features(parts[0],output, node_map)
                node2_features = get_node_text_features(parts[1], output, node_map)
                drug1_text_emb = reduced_embeddings_dict[int(parts[0])]
                drug2_text_emb = reduced_embeddings_dict[int(parts[1])]
                if dataset == "KEGG_DRUG":
                    self.node1_features.append(node1_features)
                    self.node1_features.append(node2_features)

                    self.drug1_text_emb.append(drug1_text_emb)
                    self.drug1_text_emb.append(drug2_text_emb)

                    self.node2_features.append(node2_features)
                    self.node2_features.append(node1_features)

                    self.drug2_text_emb.append(drug2_text_emb)
                    self.drug2_text_emb.append(drug1_text_emb)

                    self.labels.append(int(parts[2]))
                    self.labels.append(int(parts[2]))
                else:
                    self.node1_features.append(node1_features)
                    self.drug1_text_emb.append(drug1_text_emb)
                    self.node2_features.append(node2_features)
                    self.drug2_text_emb.append(drug2_text_emb)
                    self.labels.append(int(parts[2]))

        self.node1_features = torch.stack(self.node1_features)
        self.node2_features = torch.stack(self.node2_features)
        self.drug1_text_emb = torch.stack(self.drug1_text_emb)
        self.drug2_text_emb = torch.stack(self.drug2_text_emb)
        self.labels = torch.tensor(self.labels,dtype=torch.float)

    def __len__(self):
        return len(self.node1_features)

    def __getitem__(self, idx):
        return self.node1_features[idx], self.node2_features[idx], self.drug1_text_emb[idx], self.drug2_text_emb[idx],self.labels[idx]


