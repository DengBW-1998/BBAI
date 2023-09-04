from __future__ import division
from __future__ import print_function

import logging
import os

import numpy as np
import torch

from BGNN.mlp.models import MLPLearning
from BGNN.gcn.models import GCN
import BGNN.conf
from BGNN.conf import (MODEL, BATCH_SIZE, EPOCHS, LEARNING_RATE,
                        WEIGHT_DECAY, DROPOUT, HIDDEN_DIMENSIONS, GCN_OUTPUT_DIM, ENCODER_HIDDEN_DIMENSIONS,
                        DECODER_HIDDEN_DIMENSIONS, MLP_HIDDEN_DIMENSIONS, LATENT_DIMENSIONS)



class BGNNMLP(object):
    def __init__(self, u,v,batch_size,adj_u,adj_v,emb0_u,emb0_v, dim,dataset,device='cpu', layer_depth=3, rank=-1):
        self.rank = rank
        self.dataset = dataset
        self.epochs = EPOCHS
        self.dis_hidden_dim = HIDDEN_DIMENSIONS
        self.learning_rate = LEARNING_RATE
        self.weight_decay = WEIGHT_DECAY
        self.dropout = DROPOUT
        self.device = device
        self.u_attr_dimensions=dim
        self.v_attr_dimensions=dim
        self.u_attr=emb0_u
        self.v_attr=emb0_v

        self.layer_depth = layer_depth
        self.batch_size = batch_size


        self.batch_num_u = int(u/batch_size)+1
        self.batch_num_v = int(v/batch_size)+1
        self.u_adj = adj_u
        self.v_adj = adj_v
        self.u_num = u
        self.v_num = v
        self.u_list=[]
        self.v_list=[]
        for i in range(u):
            self.u_list.append(i)
        for i in range(u,u+v):
            self.v_list.append(i)
        self.gcn_output_dim = GCN_OUTPUT_DIM
        self.decoder_hidfeat = DECODER_HIDDEN_DIMENSIONS

        self.gcn_explicit = GCN(self.v_attr_dimensions, self.gcn_output_dim)
        self.gcn_implicit = GCN(self.u_attr_dimensions, self.gcn_output_dim)
        self.gcn_merge = GCN(self.v_attr_dimensions, self.gcn_output_dim)
        self.gcn_opposite = GCN(self.u_attr_dimensions, self.gcn_output_dim)
        
        self.mlp_explicit = MLPLearning(self.gcn_explicit, self.gcn_output_dim, self.u_attr_dimensions, self.decoder_hidfeat,
                                        self.learning_rate, self.weight_decay, self.dropout, self.device)
        self.mlp_implicit = MLPLearning(self.gcn_implicit, self.gcn_output_dim, self.v_attr_dimensions, self.decoder_hidfeat,
                                        self.learning_rate, self.weight_decay, self.dropout, self.device)
        self.mlp_merge = MLPLearning(self.gcn_merge, self.gcn_output_dim, self.u_attr_dimensions, self.decoder_hidfeat,
                                     self.learning_rate, self.weight_decay, self.dropout, self.device)
        self.mlp_opposite = MLPLearning(self.gcn_opposite, self.gcn_output_dim, self.v_attr_dimensions, self.decoder_hidfeat,
                                        self.learning_rate, self.weight_decay, self.dropout, self.device)

        

    def __sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def relation_learning(self):
        # depth 1
        logging.info('### Depth 1 starts!\n')
        for i in range(self.epochs):
            for iter in range(self.batch_num_u):
                start_index = self.batch_size * iter
                end_index = self.batch_size * (iter + 1)
                if iter == self.batch_num_u - 1:
                    end_index = self.u_num
                u_attr_batch = self.u_attr[start_index:end_index]
                u_adj_batch = self.u_adj[start_index:end_index]

                # prepare data to the tensor
                u_attr_tensor = torch.FloatTensor(u_attr_batch)
                u_adj_tensor = self.__sparse_mx_to_torch_sparse_tensor(u_adj_batch).to(device=self.device)

                # training
                gcn_explicit_output = self.gcn_explicit(torch.FloatTensor(self.v_attr), u_adj_tensor)
                #print(u_attr_tensor.shape)
                #print(gcn_explicit_output.shape)
                self.mlp_explicit.forward_backward(u_attr_tensor.clone(), gcn_explicit_output.clone(), step=1, epoch=i, iter=iter)

        u_explicit_attr = torch.FloatTensor([]).to(self.device)
        for iter in range(self.batch_num_u):
            start_index = self.batch_size * iter
            end_index = self.batch_size * (iter + 1)
            if iter == self.batch_num_u - 1:
                end_index = self.u_num
            u_adj_batch = self.u_adj[start_index:end_index]

            # prepare data to the tensor
            u_adj_tensor = self.__sparse_mx_to_torch_sparse_tensor(u_adj_batch).to(device=self.device)

            # inference
            gcn_explicit_output = self.gcn_explicit(torch.as_tensor(self.v_attr, device=self.device), u_adj_tensor)
            decoder_explicit_output = self.mlp_explicit.forward(gcn_explicit_output.detach())
            u_explicit_attr = torch.cat((u_explicit_attr, decoder_explicit_output.detach()), 0)

        # depth 2
        logging.info('### Depth 2 starts!\n')
        for i in range(self.epochs):
            for iter in range(self.batch_num_v):
                start_index = self.batch_size * iter
                end_index = self.batch_size * (iter + 1)
                if iter == self.batch_num_v - 1:
                    end_index = self.v_num
                v_attr_batch = self.v_attr[start_index:end_index]
                v_adj_batch = self.v_adj[start_index:end_index]

                # prepare the data to the tensor
                v_attr_tensor = torch.as_tensor(v_attr_batch, dtype=torch.float, device=self.device)
                v_adj_tensor = self.__sparse_mx_to_torch_sparse_tensor(v_adj_batch).to(device=self.device)

                # training
                gcn_implicit_output = self.gcn_implicit(u_explicit_attr, v_adj_tensor)
                self.mlp_implicit.forward_backward(v_attr_tensor, gcn_implicit_output, step=2, epoch=i, iter=iter)

        v_implicit_attr = torch.FloatTensor([]).to(self.device)
        for iter in range(self.batch_num_v):
            start_index = self.batch_size * iter
            end_index = self.batch_size * (iter + 1)
            if iter == self.batch_num_v - 1:
                end_index = self.v_num
            v_adj_batch = self.v_adj[start_index:end_index]

            # prepare the data to the tensor
            v_adj_tensor = self.__sparse_mx_to_torch_sparse_tensor(v_adj_batch).to(device=self.device)

            # inference
            gcn_implicit_output = self.gcn_implicit(u_explicit_attr, v_adj_tensor)
            decoder_implicit_output = self.mlp_implicit.forward(gcn_implicit_output.detach())
            v_implicit_attr = torch.cat((v_implicit_attr, decoder_implicit_output.detach()), 0)

        # merge
        logging.info('### Depth 3 starts!\n')
        for i in range(self.epochs):
            for iter in range(self.batch_num_u):
                start_index = self.batch_size * iter
                end_index = self.batch_size * (iter + 1)
                if iter == self.batch_num_u - 1:
                    end_index = self.u_num
                u_adj_batch = self.u_adj[start_index:end_index]

                # prepare the data to the tensor
                u_adj_tensor = self.__sparse_mx_to_torch_sparse_tensor(u_adj_batch).to(device=self.device)

                # training
                gcn_merge_output = self.gcn_merge(v_implicit_attr, u_adj_tensor)
                u_input = u_explicit_attr[start_index:end_index]
                self.mlp_merge.forward_backward(u_input, gcn_merge_output, step=3, epoch=i, iter=iter)

        u_merge_attr = torch.FloatTensor([]).to(self.device)
        for iter in range(self.batch_num_u):
            start_index = self.batch_size * iter
            end_index = self.batch_size * (iter + 1)
            if iter == self.batch_num_u - 1:
                end_index = self.u_num
            u_adj_batch = self.u_adj[start_index:end_index]

            # prepare the data to the tensor
            u_adj_tensor = self.__sparse_mx_to_torch_sparse_tensor(u_adj_batch).to(device=self.device)

            # inference
            gcn_merge_output = self.gcn_merge(v_implicit_attr, u_adj_tensor)
            decoder_merge_output = self.mlp_merge.forward(gcn_merge_output.detach())
            u_merge_attr = torch.cat((u_merge_attr, decoder_merge_output.detach()), 0)

        self.__save_embedding_to_file(u_merge_attr.cpu().numpy(), self.bipartite_graph_data_loader.get_u_list())

    def __save_embedding_to_file(self, gcn_merge_output, node_id_list):
        """ embedding file format:
            line1: number of the node, dimension of the embedding vector
            line2: node_id, embedding vector
            line3: ...
            lineN: node_id, embedding vector

        :param gcn_merge_output:
        :param node_id_list:
        :return:
        """
        logging.info("Start to save embedding file\n")
        node_num = gcn_merge_output.shape[0]
        logging.info("node_num = %s" % node_num)
        dimension_embedding = gcn_merge_output.shape[1]
        logging.info("dimension_embedding = %s" % dimension_embedding)
        output_folder = "./out/bgnn-mlp/" + str(self.dataset)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        f_emb = open(output_folder + '/bgnn.emb', 'w')
        f_node_list = open(output_folder + '/node_list', 'w')

        str_first_line = str(node_num) + " " + str(dimension_embedding) + "\n"
        f_emb.write(str_first_line)
        for n_idx in range(node_num):
            f_emb.write(str(node_id_list[n_idx]) + ' ')
            f_node_list.write(str(node_id_list[n_idx]))
            emb_vec = gcn_merge_output[n_idx]
            for d_idx in range(dimension_embedding):
                if d_idx != dimension_embedding - 1:
                    f_emb.write(str(emb_vec[d_idx]) + ' ')
                else:
                    f_emb.write(str(emb_vec[d_idx]))
            if n_idx != node_num - 1:
                f_emb.write('\n')
                f_node_list.write('\n')
        f_emb.close()
        f_node_list.close()
        logging.info("Saved embedding file")
