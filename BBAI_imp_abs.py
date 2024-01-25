import numpy as np
from scipy.linalg import eigh
from codes.utils import *
from codes.embedding import *
from codes.perturbation_attack import *
from codes.testModel import *
import time
from BGNN.bgnn_adv import *
from BGNN.bgnn_mlp import *

import warnings
warnings.filterwarnings("ignore")

n_flips = -1 #Number of perturbations per iteration
dim = 64 #Embedding vector dimension
window_size = 5 #Window size
n_node_pairs = 100000 #Number of test edges
iteration= 2 #Iteration rounds
threshold = 5 #Implicit relationship threshold
batch_size=64
dataset='wiki'
rate=1
train_model='netmf'
ptb_rate=5 #rate of perturbed edges
file_name="BBAI_imp_abs_"+dataset+"_"+str(ptb_rate)+"_"+train_model+".txt"
read_dir=False #reading ptb_matrix directly
linkpre_abl=True #It is True when the dowmstream task is link prediction in the ablation experience

data_file=open(file_name, 'w+')

if dataset=='dblp' and linkpre_abl==False:
    n_flips=int(1800*ptb_rate/100/iteration)
    rate=5
if dataset=='dblp' and linkpre_abl:
    n_flips=int(72000*ptb_rate/100/iteration)
    rate=1    
if dataset=='wiki' and linkpre_abl==False:
    n_flips=int(3600*ptb_rate/100/iteration)
    rate=10
if dataset=='wiki' and linkpre_abl:
    n_flips=int(76000*ptb_rate/100/iteration)
    rate=2
if dataset == 'citeseer':
    n_flips = int(2840 / 2 * ptb_rate / 100/iteration)
if dataset == 'pubmed':
    n_flips = int(38782 / 2 * ptb_rate / 100/iteration)

n_candidates=10*n_flips #Number of candidate perturbed edges

adj_nn,adj,u,v,labels = getAdj(threshold,dataset,rate)
adj = standardize(adj)
adj_nn = standardize(adj_nn)

#emb0_u,emb0_v,dim_u,dim_v = getAttribut(u,v,dataset)
seed=0
np.random.seed(seed)
time_start=time.time()

if not read_dir:
    for ite in range(iteration):     
        candidates_nn = generate_candidates_addition(adj_matrix=adj_nn, n_candidates=n_candidates,u=u,v=v,seed=seed)
        flips,vals_org, vecs_org = perturbation_top_flips(adj_matrix=adj_nn, candidates=candidates_nn, n_flips=n_flips, dim=dim, window_size=window_size)

        ### Link Prediction
        for _ in range(1):
            u_node_pairs = np.random.randint(0, u-1, [n_node_pairs*2, 1])
            v_node_pairs = np.random.randint(u, u+v-1, [n_node_pairs*2, 1])
            node_pairs = np.column_stack((u_node_pairs,v_node_pairs))

            adj_matrix_flipped = flip_candidates(adj_nn, flips)
            np.savetxt('./ptb_matrix/BBAI_imp_abs_ptb_'+dataset+'_'+str(ptb_rate)+'.dat',adj_matrix_flipped.copy().toarray(),fmt='%.2f',delimiter=' ')
                
            adj_matrix_flipped[:u,:u]=0
            adj_matrix_flipped[u:,u:]=0
            
            if ite==iteration-1:
                print(ite)
                print(ite, file=data_file)

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
                    print('svd BBAI_imp_abs auc:{:.5f}'.format(auc_score))
                    print('svd BBAI_imp_abs auc:{:.5f}'.format(auc_score), file=data_file)
                    
                if (dataset == 'cora' or dataset == 'citeseer' or dataset == 'pubmed'):
                    f1_scores_mean, _ = evaluate_embedding_node_classification(embedding, labels)
                    print('BBAI_imp_abs, F1: {:.5f} {:.5f}'.format(f1_scores_mean[0], f1_scores_mean[1]))
                    print('BBAI_imp_abs, F1: {:.5f} {:.5f}'.format(f1_scores_mean[0], f1_scores_mean[1]), file=data_file)

            adj_nn=adjsp_2_adjnn(adj_matrix_flipped,u,v,threshold)
            adj_nn=standardize(adj_nn)

else:
    adj_matrix_flipped=pd.read_csv('./ptb_matrix/BBAI_imp_abs_ptb_'+dataset+'_'+str(ptb_rate)+'.dat',sep=' ',header=None)
    adj_matrix_flipped=sp.csr_matrix(torch.tensor(np.array(adj_matrix_flipped)))
    adj_matrix_flipped[:u,:u]=0
    adj_matrix_flipped[u:,u:]=0
    
    print(iteration-1)
    print(iteration-1, file=data_file)

    for _ in range(5):
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
            print('svd BBAI_imp_abs auc:{:.5f}'.format(auc_score))
            print('svd BBAI_imp_abs auc:{:.5f}'.format(auc_score), file=data_file)
            
        if (dataset == 'cora' or dataset == 'citeseer' or dataset == 'pubmed'):
            f1_scores_mean, _ = evaluate_embedding_node_classification(embedding, labels)
            print('BBAI_imp_abs, F1: {:.5f} {:.5f}'.format(f1_scores_mean[0], f1_scores_mean[1]))
            print('BBAI_imp_abs, F1: {:.5f} {:.5f}'.format(f1_scores_mean[0], f1_scores_mean[1]), file=data_file)

time_end=time.time()
print(train_model)
print(train_model, file=data_file)
print(time_end-time_start)
print(time_end - time_start, file=data_file)
#print(bgnn.epochs)
#print(bgnn.epochs,file=data)           
data_file.close()
