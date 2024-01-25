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
from BGNN.bgnn_adv import *
from BGNN.bgnn_mlp import *
from deeprobust.graph.data import PrePtbDataset

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
ptb_rate=10 #rate of perturbed edges
file_name="meta_"+dataset+"_"+str(ptb_rate)+"_"+train_model+".txt"
read_dir=False #reading ptb_matrix directly
data_file=open(file_name, 'w+')

if dataset=='dblp':
    n_flips=int(1800*ptb_rate/100)
    rate=5
    nclass = 5
if dataset=='wiki':
    n_flips=int(3600*ptb_rate/100)
    rate=10
    nclass = 5
if dataset == 'citeseer':
    nclass = 6
if dataset == 'pubmed':
    nclass = 3

adj_nn,adj,u,v,test_labels = getAdj(threshold,dataset,rate)
adj = standardize(adj)

emb0_u,emb0_v,dim_u,dim_v = getAttribut(u,v,dataset)
time_start=time.time()

vlabels = pd.read_csv('./data/' + dataset + '_' + 'vlabels.dat', sep=' ', header=None).to_numpy().astype(int)
vlabels = np.squeeze(vlabels)[:u+v]

val_size = 0.1
test_size = 0.8
train_size = 1 - test_size - val_size

features = np.ones((adj.shape[0],32))
features = sp.csr_matrix(features)

idx = np.arange(adj.shape[0])
idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=vlabels)
idx_unlabeled = np.union1d(idx_val, idx_test)
adj, features, vlabels = preprocess(adj, features, vlabels, preprocess_adj=False)

if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5

if(dataset=='dblp' or dataset=='wiki') and read_dir==False:
    model = Metattack(nfeat=32, hidden_sizes=[args.hidden],
                       nnodes=adj.shape[0], nclass=nclass, dropout=0.5,
                       train_iters=20, attack_features=False, lambda_=lambda_)

    modified_adj = model(features, adj, vlabels, idx_unlabeled,idx_unlabeled, n_flips, ll_constraint=True)
    adj_matrix_flipped = sp.csr_matrix(modified_adj.detach())
    np.savetxt('./ptb_matrix/meta_ptb_' + dataset + '_' + str(ptb_rate) + '.dat', adj_matrix_flipped.copy().toarray(),
               fmt='%.2f', delimiter=' ')

if(dataset=='dblp' or dataset=='wiki') and read_dir:
    adj_matrix_flipped = pd.read_csv('./ptb_matrix/meta_ptb_' + dataset + '_' + str(ptb_rate) + '.dat', sep=' ',
                                     header=None)
    adj_matrix_flipped = sp.csr_matrix(torch.tensor(np.array(adj_matrix_flipped)))


if (dataset == 'cora' or dataset == 'citeseer' or dataset == 'pubmed'):
    # 加载攻击后的数据集
    perturbed_data = PrePtbDataset(root='./data/meta/',
                                   name=dataset,
                                   attack_method='meta',
                                   ptb_rate=ptb_rate/100)
    modified_adj = perturbed_data.adj
    modified_adj[:u,:u]=0
    modified_adj[u:,u:]=0
    adj_matrix_flipped = modified_adj.copy()
#np.savetxt('metattack_modified_adj_pubmed70.dat',modified_adj.detach().numpy(),fmt='%.2f',delimiter=' ')

#modified_adj=pd.read_csv('metattack_modified_adj_wiki10.dat',sep=' ',header=None)
#modified_adj=torch.tensor(np.array(modified_adj))


adj_matrix_flipped[:u,:u]=0
adj_matrix_flipped[u:,u:]=0
#print((adj_matrix_flipped).shape)

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

    if (dataset == 'dblp' or dataset == 'wiki'):
        auc_score = evaluate_embedding_link_prediction(
            adj_matrix=adj_matrix_flipped,
            node_pairs=node_pairs,
            embedding_matrix=embedding
        )
        print('svd meta auc:{:.5f}'.format(auc_score))
        print('svd meta auc:{:.5f}'.format(auc_score), file=data_file)
    else:
        f1_scores_mean, _ = evaluate_embedding_node_classification(embedding, test_labels)
        print('meta, F1: {:.5f} {:.5f}'.format(f1_scores_mean[0], f1_scores_mean[1]))
        print('meta, F1: {:.5f} {:.5f}'.format(f1_scores_mean[0], f1_scores_mean[1]), file=data_file)

time_end=time.time()
print(train_model)
print(train_model, file=data_file)
print(time_end-time_start)
print(time_end - time_start, file=data_file)
print(dataset)
print(dataset, file=data_file)

data_file.close()

