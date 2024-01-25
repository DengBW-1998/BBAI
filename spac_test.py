import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os
import argparse
import pickle as pkl
import copy
import networkx as nx

from spac.gnns import GCN as spacGCN
from spac.data_loader import Dataset
from spac.utils import calc_acc, save_all, save_utility

from spac.global_attack import PGDAttack, MinMax
from spac.global_attack import MetaApprox, Metattack
from spac.global_attack import Random, DICE

from codes.utils import *
from codes.embedding import *
from codes.perturbation_attack import *
from codes.testModel import *
import time
from BGNN.bgnn_adv import *
from BGNN.bgnn_mlp import *
from sklearn.model_selection import train_test_split
import scipy.sparse as sp


# print the victim model's performance given graph info
def check_victim_model_performance(victim_model, features, adj, labels, idx_test, idx_train):
    output = victim_model.predict(features, adj)
    loss_test_clean, acc_test_clean = calc_acc(output, labels, idx_test)
    loss_train_clean, acc_train_clean = calc_acc(output, labels, idx_train)

    log = 'train loss: {:.4f}, train acc: {:.4f}, train misacc: {:.4f}'
    print(log.format(loss_train_clean, acc_train_clean, 1-acc_train_clean))
    log = 'test loss: {:.4f}, test acc: {:.4f}, test misacc: {:.4f}'
    print(log.format(loss_test_clean, acc_test_clean, 1-acc_test_clean))

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
    
# Set the random seed so things involved torch.randn are repetable
def set_random_seed(seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if device != 'cpu':
        torch.cuda.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0, help='cuda')
parser.add_argument('--seed', type=int, default=123, help='Random seed for model')
parser.add_argument('--data_seed', type=int, default=123,help='Random seed for data split')
parser.add_argument('--dataset', type=str, default='cora', help='dataset')
parser.add_argument('--attacker', type=str, default='minmax', help='attacker variant')  # ['minmax', 'Meta-Self', 'A-Meta-Self', 'Meta-Train', 'A-Meta-Train', 'random']
parser.add_argument('--loss_type', type=str, default='CE', help='loss type')
parser.add_argument('--att_lr', type=float, default=50, help='Initial learning rate')
parser.add_argument('--perturb_epochs', type=int, default=50, help='Number of epochs to poisoning loop')
parser.add_argument('--ptb_rate', type=float, default=0.05, help='pertubation rate')
parser.add_argument('--loss_weight', type=float, default=1.0, help='loss weight')
parser.add_argument('--spac_weight', type=float, default=1.0, help='spac weight')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters)')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability)')
parser.add_argument('--data_dir', type=str, default='./log/dataset', help='Directory to download dataset')
parser.add_argument('--target_node', type=str, default='train', help='target node set')
parser.add_argument('--sanitycheck', type=str, default='no', help='whether store the intermediate results')
parser.add_argument('--distance_type', type=str, default='l2', help='distance type')
parser.add_argument('--opt_type', type=str, default='max', help='optimization type')
parser.add_argument('--sample_type', type=str, default='sample', help='sample type')

args = parser.parse_args()
args.device = 'cpu'

dim = 64 
window_size = 5 
n_node_pairs = 100000 
seed=2
threshold = 5 #Implicit relationship threshold
dataset='pubmed'
rate=1
train_model='netmf'
n_flips = -1 #Perturbation Number
batch_size=64
ptb_rate=5 #rate of perturbed edges
file_name="spac_"+dataset+"_"+str(ptb_rate)+"_"+train_model+".txt"
read_dir=False #reading ptb_matrix directly
data_file=open(file_name, 'w+')

args.ptb_rate=ptb_rate
args.seed=seed
args.dataset=dataset

torch.set_num_threads(1) # limit cpu use   
set_random_seed(int(time.time()), args.device)

#########################################################
# Load data for node classification task
if dataset=='dblp':
    n_flips=int(1800*ptb_rate/100)
    rate=5
    nclass = 5
if dataset=='wiki':
    n_flips=int(3600*ptb_rate/100)
    rate=10
    nclass = 5
if dataset == 'citeseer':
    n_flips = int(2840 / 2 * ptb_rate / 100)
    nclass = 6
if dataset == 'pubmed':
    n_flips = int(38782 / 2 * ptb_rate / 100)
    nclass = 3
    args.att_lr=10
    

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


#########################################################
weight_decay = 0 if args.dataset == 'polblogs' else args.weight_decay
surrogate_model = spacGCN(
    nfeat=features.shape[1], 
    nclass=nclass, 
    nhid=args.hidden,
    dropout=args.dropout, 
    weight_decay=weight_decay,
    device=args.device)
surrogate_model = surrogate_model.to(args.device)

print('==== Initial Surrogate Model on Clean Graph ====')
surrogate_model.eval()
check_victim_model_performance(surrogate_model, features, adj, vlabels, idx_test, idx_train)

time_start=time.time()
#########################################################
# Setup attacker
if dataset=='pubmed':
    args.attacker = 'random'
if args.attacker == 'minmax':
    attacker = MinMax(
        model=surrogate_model, 
        nnodes=adj.shape[0], 
        loss_type=args.loss_type,
        loss_weight=args.loss_weight,
        spac_weight=args.spac_weight,
        device=args.device)
    attacker = attacker.to(args.device)
elif 'Meta' in args.attacker:  # 'Meta-Self', 'A-Meta-Self', 'Meta-Train', 'A-Meta-Train'
    if 'Self' in args.attacker:
        lambda_ = 0
    if 'Train' in args.attacker:
        lambda_ = 1
    if 'Both' in args.attacker:
        lambda_ = 0.5

    if 'A' in args.attacker:
        attacker = MetaApprox(model=surrogate_model, 
                        nnodes=adj.shape[0], 
                        attack_structure=True, 
                        attack_features=False, 
                        spac_weight=args.spac_weight,
                        device=args.device, 
                        lambda_=lambda_)
    else:
        attacker = Metattack(model=surrogate_model, 
                        nnodes=adj.shape[0], 
                        attack_structure=True, 
                        attack_features=False, 
                        spac_weight=args.spac_weight,
                        device=args.device, 
                        lambda_=lambda_)
    attacker = attacker.to(args.device)
elif args.attacker == 'random':
    attacker = Random()
elif args.moattackerdel == 'dice':
    attacker = DICE(model=surrogate_model, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=args.device)
else:
    raise AssertionError (f'Attack {args.attacker} not found!')

#########################################################
if not read_dir:
    perturbations = n_flips
    nat_adj = copy.deepcopy(adj)

    # switch target node set for minmax attack
    if args.target_node == 'test':
        idx_target = idx_test
    elif args.target_node == 'train':
        idx_target = idx_train
    else:
        idx_target = np.hstack((idx_test, idx_train)).astype(np.int)

    # Start attack
    if args.attacker == 'random':
        attacker.attack(nat_adj, perturbations, 'flip')
    elif args.attacker == 'dice':
        attacker.attack(nat_adj, vlabels, perturbations)
    elif 'Meta' in args.attacker:
        attacker.attack(
            features, 
            nat_adj, 
            vlabels, 
            idx_train, 
            idx_unlabeled, 
            perturbations, 
            ll_constraint=False, 
            verbose=True)
    else:
        attacker.attack(
            features, 
            nat_adj, 
            vlabels, 
            idx_target, 
            perturbations, 
            att_lr=args.att_lr, 
            epochs=args.perturb_epochs,
            distance_type=args.distance_type,
            sample_type=args.sample_type,
            opt_type=args.opt_type,
            idx_test=idx_test,
            idx_train=idx_train)
    time_end=time.time()
    adj_matrix_flipped = sp.csr_matrix(attacker.modified_adj.numpy())
    np.savetxt('./ptb_matrix/spac_ptb_' + dataset + '_' + str(ptb_rate) + '.dat', adj_matrix_flipped.copy().toarray(),
                   fmt='%.2f', delimiter=' ')
else:
    adj_matrix_flipped = pd.read_csv('./ptb_matrix/spac_ptb_' + dataset + '_' + str(ptb_rate) + '.dat', sep=' ',
                                     header=None)
    adj_matrix_flipped = sp.csr_matrix(torch.tensor(np.array(adj_matrix_flipped)))
    time_end=time.time()

for _ in range(5):
    print(_)
    print(_, file=data_file)
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
        print('spac auc:{:.5f}'.format(auc_score))
        print('spac auc:{:.5f}'.format(auc_score), file=data_file)
    else:
        f1_scores_mean, _ = evaluate_embedding_node_classification(embedding, test_labels)
        print('spac, F1: {:.5f} {:.5f}'.format(f1_scores_mean[0], f1_scores_mean[1]))
        print('spac, F1: {:.5f} {:.5f}'.format(f1_scores_mean[0], f1_scores_mean[1]), file=data_file)

print(train_model)
print(train_model, file=data_file)
print(time_end-time_start)
print(time_end - time_start, file=data_file)
print(dataset)
print(dataset, file=data_file)
data_file.close()
