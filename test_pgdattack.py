import torch
import argparse
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.linalg import eigh
from codes.utils import *
from codes.embedding import *
from codes.perturbation_attack import *
from codes.testModel import *
import time
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import PGDAttack
from sklearn.model_selection import train_test_split

_,adj,gul = getAdj(0)
adj = standardize(adj)
u = gul.u_nodes
v = gul.v_nodes
time_start=time.time()

labels = np.zeros((u+v))
labels[u:] = 1

val_size = 0.1
test_size = 0.8
train_size = 1 - test_size - val_size

def get_train_val_test(idx, train_size, val_size, test_size, stratify):

    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test
    
def preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=False, sparse=False):
    if preprocess_adj == True:
        adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))

    if preprocess_feature:
        features = normalize_f(features)

    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = sparse_mx_to_torch_sparse_tensor(features)
        labels = sparse_mx_to_torch_sparse_tensor(labels)
    else:
        labels = torch.LongTensor(np.array(labels))
        features = torch.FloatTensor(np.array(features.todense()))
        adj = torch.FloatTensor(adj.todense())

    return adj, features, labels

embedding_u = deepwalk_skipgram(adj_matrix=adj[:u,u:]@adj[u:,:u], embedding_dim=32, window_size=5)
embedding_v = deepwalk_skipgram(adj_matrix=adj[u:,:u]@adj[:u,u:], embedding_dim=32, window_size=2)
embedding_imp = np.row_stack((embedding_u,embedding_v))
embedding_exp = deepwalk_skipgram(adj_matrix=adj, embedding_dim=32, window_size=5)
features = (embedding_imp+embedding_exp)/2
features = sp.csr_matrix(features)
    
idx = np.arange(adj.shape[0])
idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=labels)
idx_unlabeled = np.union1d(idx_val, idx_test)
perturbations = 5 #Perturbation percentage

device = 'cpu'
surrogate = GCN(nfeat=32, nclass=2,
            nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)

model = PGDAttack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
    attack_structure=True, attack_features=False, device=device).to(device)
    
model.attack(features.todense(), adj.todense(), labels, idx_train, n_perturbations=perturbations, ll_constraint=False)
adj_matrix_flipped = sp.csr_matrix(model.modified_adj)

dim = 64 
window_size = 5 
n_node_pairs = 100000 
seed=0
data=open("pgd.txt",'w+')
for _ in range(5):
    print(_)
    print(_,file=data)  
    u_node_pairs = np.random.randint(0, u-1, [n_node_pairs*2, 1])
    v_node_pairs = np.random.randint(u, u+v-1, [n_node_pairs*2, 1])
    node_pairs = np.column_stack((u_node_pairs,v_node_pairs))

    adj_matrix_flipped[:u,:u]=0
    adj_matrix_flipped[u:,u:]=0

    #svd
    embedding_u, _, _, _ = deepwalk_svd(adj_matrix_flipped[:u,u:]@adj_matrix_flipped[u:,:u], window_size, dim)
    embedding_v, _, _, _ = deepwalk_svd(adj_matrix_flipped[u:,:u]@adj_matrix_flipped[:u,u:], window_size, dim)
    embedding_imp = np.row_stack((embedding_u,embedding_v))
    embedding_exp, _, _, _ = deepwalk_svd(adj_matrix_flipped, window_size, dim)
    embedding = (embedding_imp+embedding_exp)/2
    auc_score = evaluate_embedding_link_prediction(
        adj_matrix=adj_matrix_flipped, 
        node_pairs=node_pairs, 
        embedding_matrix=embedding
    )
    print('svd pgd auc:{}'.format(auc_score))
    print('svd pgd auc:{}'.format(auc_score),file=data)
time_end=time.time()
print('pgdattack') 
print('pgdattack',file=data) 
print(time_end-time_start)
print(time_end-time_start,file=data)     
data.close()