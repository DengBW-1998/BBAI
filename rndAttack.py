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
from Dice.dice import DICE
from sklearn.model_selection import train_test_split
from BGNN.bgnn_adv import *
from BGNN.bgnn_mlp import *


dim = 64
window_size = 5
n_node_pairs = 100000
seed=2
threshold = 5 #Implicit relationship threshold
dataset='dblp'
rate=1
train_model='netmf'
n_flips = -1 #Perturbation Number
batch_size=64
ptb_rate=5 #rate of perturbed edges
file_name="dice_"+dataset+"_"+str(ptb_rate)+"_"+train_model+".txt"
data_file=open(file_name, 'w+')

if dataset=='dblp':
    n_flips=90
    rate=5
    nclass=5
if dataset=='wiki':
    n_flips=180
    rate=10
    nclass = 5
if dataset == 'citeseer':
    n_flips = int(2840 / 2 * ptb_rate / 100)
    nclass = 6
if dataset == 'pubmed':
    n_flips = int(38782 / 2 * ptb_rate / 100)
    nclass = 3
n_candidates=10*n_flips #Number of candidate perturbed edges

adj_nn,adj,u,v,test_labels = getAdj(threshold,dataset,rate)
adj = standardize(adj)
emb0_u,emb0_v,dim_u,dim_v = getAttribut(u,v,dataset)
time_start=time.time()

vlabels=pd.read_csv('./data/'+dataset+'_'+'vlabels.dat',sep=' ',header=None).to_numpy().astype(int)
vlabels=np.squeeze(vlabels)[:u+v]

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

    
idx = np.arange(adj.shape[0])
idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=vlabels)
idx_unlabeled = np.union1d(idx_val, idx_test)

features = np.ones((adj.shape[0],32))
features = sp.csr_matrix(features)

device = 'cpu'
surrogate = dr.GCN(nfeat=32, nclass=nclass,
            nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
surrogate.fit(features, adj, vlabels, idx_train, idx_val, patience=30)

model = DICE(surrogate, nnodes=adj.shape[0],
    attack_structure=True, attack_features=False, device=device).to(device)
    
model.attack(adj, vlabels, n_perturbations=n_flips)
time_end=time.time()
print(time_end-time_start)

print(time_end - time_start, file=data_file)

adj_matrix_flipped = sp.csr_matrix(model.modified_adj)

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
        print('svd dice auc:{:.5f}'.format(auc_score))
        print('svd dice auc:{:.5f}'.format(auc_score), file=data_file)
    else:
        f1_scores_mean, _ = evaluate_embedding_node_classification(embedding, test_labels)
        print('dice, F1: {:.5f} {:.5f}'.format(f1_scores_mean[0], f1_scores_mean[1]))
        print('dice, F1: {:.5f} {:.5f}'.format(f1_scores_mean[0], f1_scores_mean[1]), file=data_file)

print(train_model)
print(train_model, file=data_file)
print('dice') 
print('dice', file=data_file)
print(dataset)
print(dataset, file=data_file)
data_file.close()
