import numpy as np
from scipy.linalg import eigh
from codes.utils import *
from codes.embedding import *
from codes.perturbation_attack import *
from codes.testModel import *
import time

n_flips = 1900 #Number of perturbations per iteration
dim = 64 #Embedding vector dimension
window_size = 5 #Window size
n_candidates=10*n_flips #Number of candidate perturbed edges
n_node_pairs = 100000 #Number of test edges
iteration= 2 #Iteration rounds
emp=0.5 #Explicit embedding ratio
threshold = 5 #Implicit relationship threshold

adj_nn,adj,gul = getAdj(threshold)
adj = standardize(adj)#Adjacency matrix without implicit relation
adj_nn = standardize(adj_nn)#Adjacency matrix with implicit relation
u = gul.u_nodes
v = gul.v_nodes


import warnings
warnings.filterwarnings("ignore")
data=open("UBAI.txt",'w+')
seed=2
np.random.seed(seed)
time_start=time.time()

for ite in range(iteration):
    
    candidates_nn = generate_candidates_addition(adj_matrix=adj_nn, n_candidates=n_candidates,u=u,v=v,seed=seed)
    if ite==0:
        flips,vals_est, vecs_org = perturbation_top_flips(adj_matrix=adj_nn, candidates=candidates_nn, n_flips=n_flips, dim=dim, window_size=window_size)
    else:
        flips,vals_est, vecs_org = increment_perturbation_top_flips(
        adj_matrix=adj_nn, 
        candidates=candidates_nn, 
        n_flips=n_flips, 
        dim=dim, 
        window_size=window_size,
        vals_org=vals_org, 
        vecs_org=vecs_org,
        flips_org=flips
        )
    ### Link Prediction
    for _ in range(5):
        u_node_pairs = np.random.randint(0, u-1, [n_node_pairs*2, 1])
        v_node_pairs = np.random.randint(u, u+v-1, [n_node_pairs*2, 1])
        node_pairs = np.column_stack((u_node_pairs,v_node_pairs))
        
        print(_)
        print(_,file=data)
        adj_matrix_flipped = flip_candidates(adj_nn, flips)
        adj_matrix_flipped[:u,:u]=0
        adj_matrix_flipped[u:,u:]=0
        
        if ite==iteration-1:
            print(ite)
            print(ite,file=data)
            
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
            print('svd adjnnIncr auc:{}'.format(auc_score))
            print('svd adjnnIncr auc:{}'.format(auc_score),file=data)
            
        adj_nn=adjsp_2_adjnn(adj_matrix_flipped,gul,threshold)
        adj_nn=standardize(adj_nn)
    
    vals_org=vals_est.copy()

time_end=time.time()
print('UBAI') 
print('UBAI',file=data)
print(time_end-time_start)
print(time_end-time_start,file=data)
print(threshold) 
data.close()        