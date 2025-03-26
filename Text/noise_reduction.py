import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from ast import literal_eval
import torch

def load_text_embeddings(csv_file):
    df = pd.read_csv(csv_file)
    embeddings = {}
    for index, row in df.iterrows():
        drug_id = row['id']
        embedding_str = row['embedding']
        embeddings[drug_id] = embedding_str

    return embeddings

def noice_reduction_pca(dataset):
    embeddings_dict = ''
    if dataset == 'KEGG_DRUG':
        embeddings_dict = load_text_embeddings('../../Data/KEGG_DRUG/a_lex_summary_text_embeddings.csv')
    if dataset == 'Drugbank':
        embeddings_dict = load_text_embeddings('../../Data/Drugbank/lex_summary_text_embeddings.csv')

    embeddings_np = {}

    for key, embedding_str in embeddings_dict.items():
        embedding_list = literal_eval(embedding_str)
        embedding_array = np.array(embedding_list, dtype=float)
        embeddings_np[key] = embedding_array

    embeddings_array = np.array([embedding for embedding in embeddings_np.values()])
    pca = PCA(n_components=64)
    reduced_embeddings = pca.fit_transform(embeddings_array)
    reduced_embeddings = torch.from_numpy(reduced_embeddings)
    reduced_embeddings_dict = {key: reduced_embedding for key, reduced_embedding in
                               zip(embeddings_np.keys(), reduced_embeddings)}

    return reduced_embeddings_dict
