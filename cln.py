import numpy as np
from scipy.linalg import eigh
from codes.utils import *
from codes.embedding import *
from codes.perturbation_attack import *
from codes.testModel import *
import time
from BGNN.bgnn_adv import *
from BGNN.bgnn_mlp import *
from memory_profiler import profile
import warnings
warnings.filterwarnings("ignore")

n_flips = -1 #Number of perturbations per iteration
dim = 64 #Embedding vector dimension
window_size = 5 #Window size
n_node_pairs = 100000 #Number of test edges
iteration= 1 #Iteration rounds
threshold = 5 #Implicit relationship threshold
batch_size=64
dataset='pubmed'
rate=1
train_model='bgnn'
ptb_rate=5 #rate of perturbed edges
file_name="cln_"+dataset+"_"+str(ptb_rate)+"_"+train_model+".txt"
data_file=open(file_name, 'w+')

if dataset=='dblp':
    rate=5
if dataset=='wiki':
    rate=10
if dataset == 'citeseer':
    n_flips = int(2840 / 2 * ptb_rate / 100/iteration)
if dataset == 'pubmed':
    n_flips = int(38782 / 2 * ptb_rate / 100/iteration)
n_candidates=10*n_flips #Number of candidate perturbed edges

adj_nn,adj,u,v,labels = getAdj(threshold,dataset,rate)
adj = standardize(adj)

emb0_u,emb0_v,dim_u,dim_v = getAttribut(u,v,dataset)
#print(type(emb0_u)) #np

time_start=time.time()

adj_matrix_flipped=adj
for ite in range(iteration):

    ### Link Prediction
    for _ in range(5):
        u_node_pairs = np.random.randint(0, u-1, [n_node_pairs*2, 1])
        v_node_pairs = np.random.randint(u, u+v-1, [n_node_pairs*2, 1])
        node_pairs = np.column_stack((u_node_pairs,v_node_pairs))
        
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
            print('cln auc:{:.5f}'.format(auc_score))
            print('cln auc:{:.5f}'.format(auc_score), file=data_file)
        else:
            f1_scores_mean, _ = evaluate_embedding_node_classification(embedding, labels)
            print('cln, F1: {:.5f} {:.5f}'.format(f1_scores_mean[0], f1_scores_mean[1]))
            print('cln, F1: {:.5f} {:.5f}'.format(f1_scores_mean[0], f1_scores_mean[1]), file=data_file)

time_end=time.time()
print(train_model)
print(train_model, file=data_file)
print(time_end-time_start)
print(time_end - time_start, file=data_file)
print(dataset)
print(dataset, file=data_file)
data_file.close()
