import numpy as np
from scipy.linalg import eigh
from codes.utils import *
from codes.embedding import *
from codes.perturbation_attack import *
from codes.testModel import *
import warnings
import torch
import scipy.sparse as sp
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

u = 4
v = 5
dim = 4
window_size = 3
imp = True
attack = True

Bu = np.array([[1, 1, 1, 0, 1], [0, 1, 1, 1, 1], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0]])
Bv = Bu.T

adj = np.zeros((u + v, u + v))
adj[:u, u:] = Bu
adj[u:, :u] = Bv
if imp:
    adj[0, 1] = adj[1, 0] = 1

adj = sp.csr_matrix(adj)

candidates = []
for i in range(u):
    for j in range(u, u + v):
        if adj[i, j] == 0:
            candidates.append([i, j])
candidates = np.array(candidates)

adj_matrix = adj
n_nodes = u + v
delta_w = 1 - 2 * adj_matrix[candidates[:, 0], candidates[:, 1]].A1
deg_matrix = np.diag(adj_matrix.sum(1).A1)
deg_matrix = deg_matrix + 0.001 * np.identity(n_nodes)
vals_org, vecs_org = spl.eigh(adj_matrix.toarray(), deg_matrix)
loss_for_candidates = estimate_loss_with_delta_eigenvals(candidates, delta_w, vals_org, vecs_org, n_nodes, dim,
                                                         window_size)
print(loss_for_candidates)
flips = candidates[loss_for_candidates.argsort()[-2:]]
print(flips)

vals_est = vals_org
if imp:
    adj[0, 1] = adj[1, 0] = 0

if attack:
    for flip in flips:
        adj[flip[0], flip[1]] = 1
        adj[flip[1], flip[0]] = 1
embedding, _, _, _ = deepwalk_svd(adj, window_size, dim)


def kmeans(embeddings, num_classes):
    kmeans = KMeans(n_clusters=num_classes)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    return labels


prediction = kmeans(embedding, 2)
print(prediction)
'''
The results without impliment realations:
The perturb scores of each edges:[0.48569211 0.26338152 0.43107848 0.39561032 0.39561032 0.39954862
 0.43107848 0.39561032 0.39561032 0.39954862]
The perturbed edges are [[2 4] [0 7]]
The clustering results of each nodes are [0 0 1 1 1 0 0 1 0]

The results with impliment realations:
The perturb scores of each edges:[0.39151191 0.18312303 0.47331974 0.48200666 0.57295743 0.48924207
 0.47331974 0.44970019 0.51040016 0.43120993]
The perturbed edge are [[3 6] [2 6]]
The clustering results of each nodes are [1 0 0 0 1 1 0 0 1]

If we don't perturb the graph, the clustering results of each nodes are [1 1 0 0 1 1 1 0 1]
'''
