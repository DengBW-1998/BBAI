from deeprobust.graph.data import Dataset
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
import deeprobust.graph.defense as dr
from sklearn.model_selection import train_test_split
from BGNN.bgnn_adv import *
from BGNN.bgnn_mlp import *
from memory_profiler import profile

n_flips = -1 #Number of perturbations per iteration
dim = 64 #Embedding vector dimension
window_size = 5 #Window size
n_node_pairs = 100000 #Number of test edges
threshold = 5 #Implicit relationship threshold
batch_size=64
dataset='wiki'
rate=1
train_model='netmf'
ptb_rate=10#rate of perturbed edges
file_name="HA_"+dataset+"_"+str(ptb_rate)+"_"+train_model+".txt"
data_file=open(file_name, 'w+')

if dataset=='dblp':
    n_flips=int(1800*ptb_rate/100)
    rate=5
if dataset=='wiki':
    n_flips=int(3600*ptb_rate/100)
    rate=10
if dataset == 'citeseer':
    n_flips = int(2840 / 2 * ptb_rate / 100)
if dataset == 'pubmed':
    n_flips = int(38782 / 2 * ptb_rate / 100)
n_candidates=10*n_flips #Number of candidate perturbed edges

seed=2
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

adj_nn,adj,u,v,test_labels = getAdj(threshold,dataset,rate)
emb0_u,emb0_v,dim_u,dim_v = getAttribut(u,v,dataset)  
adj = standardize(adj)


vlabels = pd.read_csv('./data/' + dataset + '_' + 'vlabels.dat', sep=' ', header=None).to_numpy().astype(int)
vlabels = np.squeeze(vlabels)[:u+v]

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

val_size = 0.1
test_size = 0.8
train_size = 1 - test_size - val_size

idx = np.arange(adj.shape[0])
idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=vlabels)
idx_unlabeled = np.union1d(idx_val, idx_test)

features = np.ones((adj.shape[0],32))
features = sp.csr_matrix(features)

device = 'cpu'

adj = adj.todense()

idx_unlabeled = np.union1d(idx_val, idx_test)
idx_unlabeled = np.union1d(idx_val, idx_test)

    
def compute_lambda(adj, idx_train, idx_test):
    num_all = adj.sum().item() / 2
    train_train = adj[idx_train][:, idx_train].sum().item() / 2
    test_test = adj[idx_test][:, idx_test].sum().item() / 2
    train_test = num_all - train_train - test_test
    return train_train / num_all, train_test / num_all, test_test / num_all

@profile
def heuristic_attack(adj, n_perturbations, idx_train, idx_unlabeled, lambda_1, lambda_2):
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_unlabeled)
    degree = adj.sum(dim=1).to('cpu')
    canditate_train_idx = idx_train[degree[idx_train] < (int(degree.mean()) + 1)]
    candidate_test_idx = idx_test[degree[idx_test] < (int(degree.mean()) + 1)]
    #     candidate_test_idx = idx_test
    perturbed_adj = adj.clone()
    cnt = 0
    train_ratio = lambda_1 / (lambda_1 + lambda_2)
    n_train = int(n_perturbations * train_ratio)
    n_test = n_perturbations - n_train
    while cnt < n_train:
        node_1 = np.random.choice(canditate_train_idx, 1)
        node_2 = np.random.choice(canditate_train_idx, 1)
        if vlabels[node_1] != vlabels[node_2] and adj[node_1, node_2] == 0:
            perturbed_adj[node_1, node_2] = 1
            perturbed_adj[node_2, node_1] = 1
            cnt += 1

    cnt = 0
    while cnt < n_test:
        node_1 = np.random.choice(canditate_train_idx, 1)
        node_2 = np.random.choice(candidate_test_idx, 1)
        if vlabels[node_1] != vlabels[node_2] and perturbed_adj[node_1, node_2] == 0:
            perturbed_adj[node_1, node_2] = 1
            perturbed_adj[node_2, node_1] = 1
            cnt += 1
    return perturbed_adj


# Generate perturbations
lambda_1, lambda_2, lambda_3 = compute_lambda(adj, idx_train, idx_unlabeled)
time_start=time.time()
adj_matrix_flipped = heuristic_attack(torch.from_numpy(adj).float(), n_flips, idx_train, idx_unlabeled, lambda_1, lambda_2)
time_end=time.time()
adj_matrix_flipped = sp.csr_matrix(adj_matrix_flipped)

for _ in range(5):
    u_node_pairs = np.random.randint(0, u-1, [n_node_pairs*2, 1])
    v_node_pairs = np.random.randint(u, u+v-1, [n_node_pairs*2, 1])
    node_pairs = np.column_stack((u_node_pairs,v_node_pairs))

    adj_matrix_flipped[:u,:u]=0
    adj_matrix_flipped[u:,u:]=0

    if train_model=='netmf':
        embedding_u, _, _, _ = deepwalk_svd(adj_matrix_flipped[:u,u:]@adj_matrix_flipped[u:,:u], window_size, dim)
        embedding_v, _, _, _ = deepwalk_svd(adj_matrix_flipped[u:,:u]@adj_matrix_flipped[:u,u:], window_size, dim)
        embedding_imp = np.row_stack((embedding_u,embedding_v))
        embedding_exp, _, _, _ = deepwalk_svd(adj_matrix_flipped, window_size, dim)
        embedding = (embedding_imp+embedding_exp)/2
    if train_model=='bgnn':
        bgnn = BGNNAdversarial(u,v,batch_size,adj_matrix_flipped[:u,u:],adj_matrix_flipped[u:,:u],emb0_u,emb0_v, dim_u,dim_v, dataset)
        embedding = bgnn.adversarial_learning()
    if dataset == 'dblp' or dataset == 'wiki':
        auc_score = evaluate_embedding_link_prediction(
            adj_matrix=adj_matrix_flipped,
            node_pairs=node_pairs,
            embedding_matrix=embedding
        )
        print('HA auc:{:.5f}'.format(auc_score))
        print('HA auc:{:.5f}'.format(auc_score), file=data_file)
    else:
        f1_scores_mean, _ = evaluate_embedding_node_classification(embedding, test_labels)
        print('HA, F1: {:.5f} {:.5f}'.format(f1_scores_mean[0], f1_scores_mean[1]))
        print('HA, F1: {:.5f} {:.5f}'.format(f1_scores_mean[0], f1_scores_mean[1]), file=data_file)

print(train_model)
print(train_model, file=data_file)
print('HA') 
print('HA', file=data_file)
print(dataset)
print(dataset, file=data_file)
print(time_end-time_start)
print(time_end - time_start, file=data_file)
data_file.close()


