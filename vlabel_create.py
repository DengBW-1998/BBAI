#this .py file is to create virtual labels of every nodes
from codes.utils import *
from codes.embedding import *
from codes.perturbation_attack import *
from codes.testModel import *
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans

threshold=5
rate=1
window_size=5
dim=64

dataset = 'dblp'
if dataset=='dblp':
    class_num = 5
    rate=5
if dataset=='wiki':
    class_num = 5
    rate=10
if dataset=='citeseer':
    class_num=6
if dataset=='pubmed':
    class_num=3

if dataset=='wiki':
    adj_nn, adj, u, v, test_labels = getAdj(threshold, dataset, rate)
    adj_matrix_flipped = standardize(adj)
    embedding_u, _, _, _ = deepwalk_svd(adj_matrix_flipped[:u, u:] @ adj_matrix_flipped[u:, :u], window_size, dim)
    embedding_v, _, _, _ = deepwalk_svd(adj_matrix_flipped[u:, :u] @ adj_matrix_flipped[:u, u:], window_size, dim)
    embedding_imp = np.row_stack((embedding_u, embedding_v))
    embedding_exp, _, _, _ = deepwalk_svd(adj_matrix_flipped, window_size, dim)
    embedding = (embedding_imp + embedding_exp) / 2
else:
    adj_nn,adj,u,v,test_labels = getAdj(threshold,dataset,rate)
    adj_matrix_flipped = standardize(adj)
    embedding_u, _, _, _ = deepwalk_svd(adj_matrix_flipped[:u, u:] @ adj_matrix_flipped[u:, :u], window_size, dim)
    embedding_v, _, _, _ = deepwalk_svd(adj_matrix_flipped[u:, :u] @ adj_matrix_flipped[:u, u:], window_size, dim)
    embedding_imp = np.row_stack((embedding_u, embedding_v))
    embedding_exp, _, _, _ = deepwalk_svd(adj_matrix_flipped, window_size, dim)
    embedding = (embedding_imp + embedding_exp) / 2


# 假设节点的嵌入为embeddings，所需分类数为num_classes
def kmeans(embeddings, num_classes):
    kmeans = KMeans(n_clusters=num_classes)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    return labels

vlabels = kmeans(embedding, class_num)
#print(type(vlabels))
#print(vlabels[:10])
#确保存入的值是numpy类型
np.savetxt('./data/'+dataset+'_'+'vlabels.dat',vlabels,fmt='%.2f',delimiter=' ')


