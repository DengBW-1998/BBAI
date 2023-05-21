import numpy as np
from scipy.linalg import eigh
from codes.utils import *
from codes.embedding import *
from codes.perturbation_attack import *
from codes.testModel import *
import time

adj_nn,adj,gul = getAdj(5)
adj = standardize(adj)
adj_nn = standardize(adj_nn)
u = gul.u_nodes
v = gul.v_nodes

n_flips=1
dim = 64
window_size = 5 
n_candidates=10*n_flips 
n_node_pairs = 100000 
iteration=2


import warnings
warnings.filterwarnings("ignore")
data=open("cln.txt",'w+')
seed=0
time_start=time.time()

adj_matrix_flipped=adj
for ite in range(iteration):
    print("\n",file=data)
    
    ### Link Prediction
    for _ in range(5):
        u_node_pairs = np.random.randint(0, u-1, [n_node_pairs*2, 1])
        v_node_pairs = np.random.randint(u, u+v-1, [n_node_pairs*2, 1])
        node_pairs = np.column_stack((u_node_pairs,v_node_pairs))       
        print(_)
        print(_,file=data)
   
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
        print('svd cln auc:{}'.format(auc_score))
        print('svd cln auc:{}'.format(auc_score),file=data)

time_end=time.time()
print('cln') 
print('cln',file=data)    
print(time_end-time_start)
print(time_end-time_start,file=data)       
data.close()        