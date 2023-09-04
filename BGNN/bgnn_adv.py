#from __future__ import division
#from __future__ import print_function

import logging
import os

import numpy as np
import torch

import BGNN.conf
from BGNN.conf import (MODEL, BATCH_SIZE, EPOCHS, LEARNING_RATE,
                        WEIGHT_DECAY, DROPOUT, HIDDEN_DIMENSIONS, GCN_OUTPUT_DIM, ENCODER_HIDDEN_DIMENSIONS,
                        DECODER_HIDDEN_DIMENSIONS, MLP_HIDDEN_DIMENSIONS, LATENT_DIMENSIONS)
from BGNN.adversarial.models import AdversarialLearning
from BGNN.gcn.models import GCN, twoLayersGCN

class BGNNAdversarial(object):
    def __init__(self, u,v,batch_size,adj_u,adj_v,emb0_u,emb0_v, dim_u,dim_v,dataset,device='cpu', layer_depth=3, rank=-1):
        self.rank = rank
        self.dataset = dataset
        self.epochs = EPOCHS
        self.dis_hidden_dim = HIDDEN_DIMENSIONS
        self.learning_rate = LEARNING_RATE
        self.weight_decay = WEIGHT_DECAY
        self.dropout = DROPOUT
        self.device = device
        self.u_attr_dimensions=dim_u
        self.v_attr_dimensions=dim_v
        self.emb0_u=emb0_u.copy()
        #print(emb0_u.shape)
        #print(self.emb0_u.shape)
        self.emb0_v=emb0_v.copy()

        self.layer_depth = layer_depth
        self.batch_size = batch_size


        self.batch_num_u = int(u/batch_size)+1
        self.batch_num_v = int(v/batch_size)+1
        self.adj_u = adj_u
        self.adj_v = adj_v
        self.u_num = u
        self.v_num = v
        self.u_list=[]
        self.v_list=[]
        for i in range(u):
            self.u_list.append(i)
        for i in range(u,u+v):
            self.v_list.append(i)

        self.gcn_explicit = GCN(dim_v, dim_u)
        self.gcn_implicit = GCN(dim_u, dim_v)
        self.gcn_merge = GCN(dim_v,dim_u)

        self.learning_type = 'inference'

    # initialize the layers, start with v as input
    def __layer_initialize(self):
        gcn_layers = []
        adversarial_layers = []
        for i in range(self.layer_depth):
            if i % 2 == 0:
                one_gcn_layer = GCN(self.v_attr_dimensions, self.u_attr_dimensions).to(self.device)
                gcn_layers.append(one_gcn_layer)
                adversarial_layers.append(
                    AdversarialLearning(one_gcn_layer, self.u_attr_dimensions, self.v_attr_dimensions, self.dis_hidden_dim, self.learning_rate,
                                        self.weight_decay, self.dropout, self.device, outfeat=1))
            else:
                one_gcn_layer = GCN(self.u_attr_dimensions, self.v_attr_dimensions).to(self.device)
                gcn_layers.append(one_gcn_layer)
                adversarial_layers.append(
                    AdversarialLearning(one_gcn_layer, self.v_attr_dimensions, self.u_attr_dimensions, self.dis_hidden_dim, self.learning_rate,
                                        self.weight_decay, self.dropout, self.device, outfeat=1))
        return gcn_layers, adversarial_layers

    # end to end learning
    def __layer__gcn(self, real_embedding, real_adj, fake_embedding, fake_adj, step):
        gcn = twoLayersGCN(self.v_attr_dimensions, self.u_attr_dimensions).to(self.device)
        adversarial = AdversarialLearning(gcn, self.u_attr_dimensions, self.v_attr_dimensions, self.dis_hidden_dim, self.learning_rate,
                                           self.weight_decay, self.dropout, self.device, outfeat=1)
        real_adj = self.__sparse_mx_to_torch_sparse_tensor(real_adj)
        fake_adj = self.__sparse_mx_to_torch_sparse_tensor(fake_adj)
        for i in range(self.epochs):
            gc3_output, gc4_output = gcn(real_embedding, real_adj, fake_embedding, fake_adj)

            lossD, lossG = adversarial.two_layers_forward_backward(real_embedding, fake_embedding, gc3_output, gc4_output, step=step, epoch=i, iter=iter)

        new_real_embedding, _ = gcn(real_embedding, real_adj, fake_embedding, fake_adj)
        return new_real_embedding.detach()

    # run the layer-wise inference
    #将fake集合聚合到real集合
    def __layer_inference(self, gcn, adversarial, real_batch_num, real_num, real_embedding, real_adj, fake_embedding,
                          step):
        for i in range(self.epochs):
            for iter in range(real_batch_num):
                start_index = self.batch_size * iter
                end_index = self.batch_size * (iter + 1)
                if iter == real_batch_num - 1:
                    end_index = real_num
                attr_batch = real_embedding[start_index:end_index]
                #print(real_adj.shape)
                adj_batch_temp = real_adj[start_index:end_index]
                #print(adj_batch_temp.shape)
                adj_batch = self.__sparse_mx_to_torch_sparse_tensor(adj_batch_temp)
                
                #print(fake_embedding.shape) #(320,64)
                #print(adj_batch.shape) #(64,1200)
                gcn_output = gcn(fake_embedding, adj_batch)
                #torch.Tensor(attr_batch)：原本的真实嵌入，gcn_output：域间传播得到的嵌入
                lossD, lossG = adversarial.forward_backward(torch.Tensor(attr_batch), gcn_output, step=step, epoch=i, iter=iter)
                #self.f_loss.write("%s %s\n" % (lossD, lossG))

        new_real_embedding = torch.FloatTensor([]).to(self.device)
        for iter in range(real_batch_num):
            start_index = self.batch_size * iter
            end_index = self.batch_size * (iter + 1)
            if iter == real_batch_num - 1:
                end_index = real_num
            adj_batch_temp = real_adj[start_index:end_index]
            adj_batch = self.__sparse_mx_to_torch_sparse_tensor(adj_batch_temp).to(device=self.device)
            gcn_output = gcn(torch.as_tensor(fake_embedding, device=self.device), adj_batch)
            new_real_embedding = torch.cat((new_real_embedding, gcn_output.detach()), 0)
        #self.f_loss.write("###Depth finished!\n")

        return new_real_embedding

    def adversarial_learning(self):
        # default start with V as input
        gcn_layers, adversarial_layers = self.__layer_initialize()
        logging.info('adversarial_train')
        u_previous_embedding = self.emb0_u.copy()
        #print(u_previous_embedding.shape)
        v_previous_embedding = self.emb0_v.copy()
        if self.learning_type == 'inference':
            for i in range(self.layer_depth):
                if i % 2 == 0:
                    #从v聚合到u
                    u_previous_embedding = self.__layer_inference(gcn_layers[i], adversarial_layers[i],
                                                                  self.batch_num_u,
                                                                  self.u_num, u_previous_embedding, self.adj_u,
                                                                  v_previous_embedding, i)
                else:
                    #从u聚合到v
                    v_previous_embedding = self.__layer_inference(gcn_layers[i], adversarial_layers[i],
                                                                  self.batch_num_v,
                                                                  self.v_num, v_previous_embedding, self.adj_v,
                                                                  u_previous_embedding, i)
                                                                  
        elif self.learning_type == 'end2end':
            u_previous_embedding = self.__layer__gcn(u_previous_embedding, self.adj_u, v_previous_embedding, self.adj_v, 0)            

        #self.__save_embedding_to_file(u_previous_embedding.numpy(), self.u_list)
        embedding = np.row_stack((u_previous_embedding,v_previous_embedding))
        return embedding

    def __sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

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
        logging.info("Start to save embedding file")
        # print(gcn_merge_output)
        node_num = gcn_merge_output.shape[0]
        logging.info("node_num = %s" % node_num)
        dimension_embedding = gcn_merge_output.shape[1]
        logging.info("dimension_embedding = %s" % dimension_embedding)
        output_folder = "./out/bgnn-adv/" + str(self.dataset)
        if self.rank != -1:
            output_folder = "/mnt/shared/home/bipartite-graph-learning/out/bgnn-adv/" + self.dataset + "/" + str(
                self.rank)

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
