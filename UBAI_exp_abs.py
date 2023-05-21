import numpy as np
from scipy.linalg import eigh
from codes.utils import *
from codes.embedding import *
from codes.perturbation_attack import *
from codes.testModel import *
import time

adj_nn,adj,gul = getAdj()
adj = standardize(adj)
adj_nn = standardize(adj_nn)
u = gul.u_nodes
v = gul.v_nodes

n_flips = 1900 
dim = 64 
window_size = 5 
n_candidates=10*n_flips 
n_node_pairs = 100000 
iteration=2 
emp=0.5 

import warnings
warnings.filterwarnings("ignore")
data=open("UBAI_exp_abs.txt",'w+')
seed=0
np.random.seed(seed)
time_start=time.time()

for ite in range(iteration):
    print("\n",file=data)
    candidates = generate_candidates_addition(adj_matrix=adj, n_candidates=n_candidates,u=u,v=v,seed=seed)
    flips,vals_org, vecs_org = perturbation_top_flips(adj_matrix=adj, candidates=candidates, n_flips=n_flips, dim=dim, window_size=window_size)

    ### Link Prediction
    for _ in range(1):
        u_node_pairs = np.random.randint(0, u-1, [n_node_pairs*2, 1])
        v_node_pairs = np.random.randint(u, u+v-1, [n_node_pairs*2, 1])
        node_pairs = np.column_stack((u_node_pairs,v_node_pairs))
        
        print(_)
        print(_,file=data)
        adj_matrix_flipped = flip_candidates(adj, flips)
        adj_matrix_flipped[:u,:u]=0
        adj_matrix_flipped[u:,u:]=0
        
        if ite==iteration-1:
            print(ite)
            print(ite,file=data)

            #svd
            embedding_u, _, _, _ = deepwalk_svd(adj_matrix_flipped[:u,u:]@adj_matrix_flipped[u:,:u], window_size, dim)
            embedding_v, _, _, _ = deepwalk_svd(adj_matrix_flipped[u:,:u]@adj_matrix_flipped[:u,u:], window_size, dim)
            embedding_imp = np.row_stack((embedding_u,embedding_v))
            embedding_exp, _, _, _ = deepwalk_svd(adj_matrix_flipped, window_size, dim)
            embedding = (1-emp)*embedding_imp+emp*embedding_exp
            auc_score = evaluate_embedding_link_prediction(
                adj_matrix=adj_matrix_flipped, 
                node_pairs=node_pairs, 
                embedding_matrix=embedding
            )
            print('svd our auc:{}'.format(auc_score))
            print('svd our auc:{}'.format(auc_score),file=data)

        adj=adj_matrix_flipped.copy()
        adj=standardize(adj)

time_end=time.time()
print('UBAI_exp_abs') 
print('UBAI_exp_abs',file=data) 
print(time_end-time_start)
print(time_end-time_start,file=data)          
data.close()        