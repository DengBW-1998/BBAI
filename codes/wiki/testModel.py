import sys
import numpy as np
from sklearn import preprocessing
from codes.data_utils import DataUtils
from codes.graph_utils import GraphUtils
import random
import math
import os
import scipy.sparse as sp

class BineModel(object):
    def __init__(self):        
        self.train_data=r'data/wiki/rating_train.dat'
        self.test_data=r'data/wiki/rating_test.dat'
        self.model_name='wiki'
        self.vectors_u=r'data/wiki/vectors_u.dat'
        self.vectors_v=r'data/wiki/vectors_v.dat'
        self.case_train=r'data/wiki/case_train.dat'
        self.case_test=r'data/wiki/case_test.dat' 
        self.dataset='wiki'

def adj_2_adjnn(adj,gul,th):
    u,v,n=gul.u_nodes,gul.v_nodes,gul.n_nodes
    adj_uv=adj[:u,u:]
    adj_vu=adj[u:,:u]
    adj_uv=np.array(adj_uv)
    adj_vu=np.array(adj_vu)

    adj_uv=sp.csr_matrix(adj_uv)
    adj_vu=sp.csr_matrix(adj_vu)
    adj=sp.csr_matrix(adj)
    adj_uu=adj_uv.dot(adj_vu)
    adj_vv=adj_vu.dot(adj_uv)
    
    
    adj_uu=adj_uu.todense()
    adj_vv=adj_vv.todense()
    adj_uu-=th
    adj_uu[adj_uu>=0]=1
    adj_uu[adj_uu<0]=0
    adj_vv-=th
    adj_vv[adj_vv>=0]=1
    adj_vv[adj_vv<0]=0

    adj_nn=adj.copy()
    adj_nn[:u,:u]=adj_uu
    adj_nn[u:,u:]=adj_vv 
    return adj_nn

def adjsp_2_adjnn(adj,gul,th):
    u,v,n=gul.u_nodes,gul.v_nodes,gul.n_nodes
    adj_uv=adj[:u,u:]
    adj_vu=adj[u:,:u]
    
    adj_uu=adj_uv.dot(adj_vu)
    adj_vv=adj_vu.dot(adj_uv)
        
    adj_uu=adj_uu.todense()
    adj_vv=adj_vv.todense()
    adj_uu-=th
    adj_uu[adj_uu>=0]=1
    adj_uu[adj_uu<0]=0
    adj_vv-=th
    adj_vv[adj_vv>=0]=1
    adj_vv[adj_vv<0]=0

    adj_nn=adj.copy()
    adj_nn[:u,:u]=adj_uu
    adj_nn[u:,u:]=adj_vv   
    return adj_nn

def getAdj(th):
    
    bine=BineModel()

    model_path = os.path.join('./', bine.model_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)

    dul = DataUtils(model_path)
    gul = GraphUtils(model_path)
    gul.construct_training_graph(bine.train_data)

    rate = 2 #(1/rate) represents the proportion of nodes in the selected subgraph
    gul.u_nodes = gul.u_nodes//rate
    gul.v_nodes = gul.v_nodes//rate
    gul.n_nodes = gul.n_nodes//rate
    u,v,n=gul.u_nodes,gul.v_nodes,gul.n_nodes
    adj=[[0]*n for _ in range(n)] 

    for i in range(len(gul.edge_list)):
        u_index = (int)(gul.edge_list[i][0][1:])
        v_index = (int)(gul.edge_list[i][1][1:])
        if u>u_index and v>v_index:
            adj[u_index][v_index+u]=1
            adj[v_index+u][u_index]=1
    adj=np.array(adj)
  
    adj_nn=adj_2_adjnn(adj,gul,th)
    adj=sp.csr_matrix(adj)

    return adj_nn,adj,gul

