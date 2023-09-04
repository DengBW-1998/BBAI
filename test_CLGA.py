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
from CLGA.args import CLGAargs
from CLGA.data import Data
from CLGA.CLGA import Metacl

n_flips = -1 #Perturbation number
dim = 64 
window_size = 5 
n_node_pairs = 100000 
seed=2
threshold = 5 #Implicit relationship threshold
dataset='dblp'
rate=-1
train_model='netmf'
batch_size=64
data=open("CLGA_dblp.txt",'w+')

if dataset=='dblp':
    n_flips=90
    rate=5
if dataset=='wiki':
    n_flips=180
    rate=10
if dataset=='pubmed':
    n_flips=70
      
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

adj_nn,adj,u,v = getAdj(threshold,dataset,rate)
adj = standardize(adj)
time_start=time.time()

time_start=time.time()
args=CLGAargs(dataset)
data_info=Data(adj,u,v,dim)
model = Metacl(dataset, args.param,data_info)
adj_matrix_flipped = model.attack(n_flips,dim)
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
        
    auc_score = evaluate_embedding_link_prediction(
        adj_matrix=adj_matrix_flipped, 
        node_pairs=node_pairs, 
        embedding_matrix=embedding
    )
    print('svd CLGA auc:{}'.format(auc_score))
    print('svd CLGA auc:{}'.format(auc_score),file=data)

print(train_model)
print(train_model,file=data)
print('CLGA') 
print('CLGA',file=data)
print(dataset)
print(dataset,file=data) 
print(time_end-time_start)
print(time_end-time_start,file=data)     
data.close()


