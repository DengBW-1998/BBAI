import torch
from Metattack.utils import *
import argparse
import numpy as np
from Metattack.metattack import Metattack
from Metattack.gcn import GCN
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

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default='dblp',
                    choices=['dblp'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--model', type=str, default='Meta-Self', choices=['A-Meta-Self', 'Meta-Self'], help='model variant')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

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

features = np.ones((adj.shape[0],32))
features = sp.csr_matrix(features)

idx = np.arange(adj.shape[0])
idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=labels)
idx_unlabeled = np.union1d(idx_val, idx_test)
perturbations = 180 #Perturbation number
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5

model = Metattack(nfeat=32, hidden_sizes=[args.hidden],
                   nnodes=adj.shape[0], nclass=2, dropout=0.5,
                   train_iters=20, attack_features=False, lambda_=lambda_)

def main():
    data=open("metattack.txt",'w+')
    
    dim = 64 
    window_size = 5 
    n_node_pairs = 100000 
    seed=0
    
    modified_adj = model(features, adj, labels, idx_unlabeled,
                         idx_unlabeled, perturbations, ll_constraint=True)
    adj_matrix_flipped = sp.csr_matrix(modified_adj.detach())
    
    adj_matrix_flipped[:u,:u]=0
    adj_matrix_flipped[u:,u:]=0
    for _ in range(5):
        print(_)
        print(_,file=data)
        u_node_pairs = np.random.randint(0, u-1, [n_node_pairs*2, 1])
        v_node_pairs = np.random.randint(u, u+v-1, [n_node_pairs*2, 1])
        node_pairs = np.column_stack((u_node_pairs,v_node_pairs))
 
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
        print('svd meta auc:{}'.format(auc_score))
        print('svd meta auc:{}'.format(auc_score),file=data)
        
    time_end=time.time()
    print('metattack') 
    print('metattack',file=data) 
    print(time_end-time_start)
    print(time_end-time_start,file=data) 
    data.close() 
if __name__ == '__main__':
    main()

