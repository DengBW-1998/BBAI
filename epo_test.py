import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from EpoAtk.arg_setting import get_args
from EpoAtk.utils import *
from EpoAtk.models import Model, ChebModel
from EpoAtk.metrics import *
from numpy import random
from torch import autograd
from collections import OrderedDict
import scipy.sparse as sp
import random
from BGNN.bgnn_adv import *
from BGNN.bgnn_mlp import *
from codes.utils import *
from codes.embedding import *
from codes.perturbation_attack import *
from codes.testModel import *
from memory_profiler import profile


def method_add(x, y, new_adj_tensor_cuda, new_surrogate_model, _N, _X_cuda, pre_all_labels_cuda, extra_idx_cuda,
               nclass):
    ori_adj_cuda = new_adj_tensor_cuda.clone()
    find = 0
    loss_test = None
    ori_adj_cuda[x, y] = 1
    ori_adj_cuda[y, x] = 1
    adj_selfloops = torch.add(ori_adj_cuda, torch.eye(_N))
    inv_degrees = torch.pow(torch.sum(adj_selfloops, dim=0, keepdim=True), -0.5)
    adj_norm_tensor_cuda = adj_selfloops * inv_degrees * inv_degrees.transpose(0, 1)
    # print('add edge (%d, %d)' % (x, y))
    find = 1
    new_surrogate_model.model_test(_X_cuda, adj_norm_tensor_cuda, pre_all_labels_cuda, extra_idx_cuda, nclass,
                                   use_relu=False)
    loss_test = -new_surrogate_model.loss_test
    return find, loss_test


def method_del(x, y, new_adj_tensor_cuda, new_surrogate_model, _N, _X_cuda, pre_all_labels_cuda, extra_idx_cuda,
               nclass):
    ori_adj_cuda = new_adj_tensor_cuda.clone()
    ori_adj_cuda[x, y] = 0
    ori_adj_cuda[y, x] = 0
    print('delete edge (%d, %d)' % (x, y))
    find = 1
    loss_test = None
    adj_selfloops = torch.add(ori_adj_cuda, torch.eye(_N))
    inv_degrees = torch.pow(torch.sum(adj_selfloops, dim=0, keepdim=True), -0.5)
    adj_norm_tensor_cuda = adj_selfloops * inv_degrees * inv_degrees.transpose(0, 1)
    new_surrogate_model.model_test(_X_cuda, adj_norm_tensor_cuda, pre_all_labels_cuda, extra_idx_cuda, nclass,
                                   use_relu=False)
    loss_test = -new_surrogate_model.loss_test
    return find, loss_test


def get_greedy_list(ori_adj_cuda, Greedy_edges, change_edges, _N, _K, _F, args):
    new_adj_tensor_cuda = ori_adj_cuda.clone()
    adj_selfloops = torch.add(new_adj_tensor_cuda, torch.eye(_N))
    inv_degrees = torch.pow(torch.sum(adj_selfloops, dim=0, keepdim=True), -0.5)
    new_adj_norm_tensor_cuda = adj_selfloops * inv_degrees * inv_degrees.transpose(0, 1)
    # new_adj_norm_tensor_cuda.requires_grad = True

    new_surrogate_model = Model(_F, args.tar_hidden, _K)
    new_surrogate_optimizer = optim.Adam(new_surrogate_model.parameters(), lr=args.tar_lr,
                                         weight_decay=args.tar_weight_decay)
    new_surrogate_model.model_train(new_surrogate_optimizer, args.tar_epochs, _X_cuda, new_adj_norm_tensor_cuda,
                                    _z_cuda, idx_train_cuda, idx_val_cuda, use_relu=False, drop_rate=args.drop_rate)

    new_surrogate_model.zero_grad()
    new_adj_norm_tensor_cuda.requires_grad = True

    outputs = new_surrogate_model(_X_cuda, new_adj_norm_tensor_cuda, False, drop_rate=args.drop_rate)
    loss = F.nll_loss(outputs[idx_train_cuda], _z_cuda[idx_train_cuda])

    loss = -loss
    loss.backward()

    grad = -(new_adj_norm_tensor_cuda.grad.data.cpu().numpy().flatten())
    grad_abs = -(np.abs(grad))

    idxes = np.argsort(grad_abs)
    find = 0
    acc = None

    for p in idxes:
        if (len(Greedy_edges) < args.greedy_edges):
            x = p // _N
            y = p % _N
            if (x, y) in change_edges or (y, x) in change_edges:
                continue

            # add edge
            if grad[p] > 0:
                signal = 1
                if x == y or x in onehops_dict[y] or y in onehops_dict[x]:
                    continue
                else:
                    find, acc = method_add(x, y, new_adj_tensor_cuda, new_surrogate_model)
                    # ori_adj_cuda = new_adj_tensor_cuda.clone()
            # delete edge
            else:
                signal = 0
                if x == y or not x in onehops_dict[y] or not y in onehops_dict[x]:
                    continue
                else:
                    find, acc = method_del(x, y, new_adj_tensor_cuda, new_surrogate_model)
            if find == 1:
                edge_oper = (x, y, signal)
                acc = acc.item()
                Greedy_edges[edge_oper] = acc
                print('Greedy edge number', len(Greedy_edges))
        else:
            break
    Greedy_list = sorted(Greedy_edges.items(), key=lambda x: x[1])

    return Greedy_list


def crossover(fir_edge, sec_edge, adj, changes):
    co_list = []
    fitness_list = []
    co_list.append(fir_edge)
    co_list.append(sec_edge)
    fir_x, fir_y, fir_signal = fir_edge
    sec_x, sec_y, sec_signal = sec_edge
    signal = adj[fir_x, sec_y]
    if signal > 0:
        third_signal = 0
    else:
        third_signal = 1
    third_edge = (fir_x, sec_y, third_signal)
    signal = adj[0][sec_x, fir_y]
    if signal > 0:
        four_signal = 0
    else:
        four_signal = 1
    four_edge = (sec_x, fir_y, four_signal)
    co_list.append(third_edge)
    co_list.append(four_edge)

    for i in range(len(co_list)):
        x, y, signal = co_list[i]
        new_adj = adj.clone()
        if (x, y) in changes or (y, x) in changes:
            fitness_list.append(sys.maxsize)
            continue
        else:
            if signal == 1:
                new_adj[x, y] = 1.0
                new_adj[y, x] = 1.0
            if signal == 0:
                new_adj[x, y] = 0.0
                new_adj[x, y] = 0.0

            adj_selfloops = torch.add(new_adj, torch.eye(_N))
            inv_degrees = torch.pow(torch.sum(adj_selfloops, dim=0, keepdim=True), -0.5)
            adj_norm_tensor_cuda = adj_selfloops * inv_degrees * inv_degrees.transpose(0, 1)

            new_model = Model(_F, args.tar_hidden, _K)
            new_optimizer = optim.Adam(new_model.parameters(), lr=args.tar_lr,
                                       weight_decay=args.tar_weight_decay)
            new_model.model_train(new_optimizer, args.tar_epochs, _X_cuda, adj_norm_tensor_cuda, _z_cuda,
                                  idx_train_cuda, idx_val_cuda, use_relu=False, drop_rate=args.drop_rate)
            new_model.model_test(_X_cuda, adj_norm_tensor_cuda, pre_all_labels_cuda, extra_idx_cuda,
                                 use_relu=False)
            loss_test = -new_model.loss_test
            fitness_list.append(loss_test)

    fitness_idx = sorted(range(len(fitness_list)), key=lambda k: fitness_list[k])
    index = fitness_idx[0]
    return co_list[index]


@profile
def main():
    dim = 64
    window_size = 5
    n_node_pairs = 100000
    threshold = 5  # Implicit relationship threshold
    dataset = 'pubmed'
    rate = 1
    train_model = 'netmf'
    n_flips = -1  # Perturbation Number
    batch_size = 64
    ptb_rate = 5  # rate of perturbed edges
    file_name = "epo_" + dataset + "_" + str(ptb_rate) + "_" + train_model + ".txt"
    read_dir = True  # reading ptb_matrix directly
    data_file = open(file_name, 'w+')

    args = get_args()
    args.seed = 2
    args.dataset = dataset
    # setting random seeds
    set_seed(3, 'cpu')

    # Load data for node classification task
    if dataset == 'dblp':
        n_flips = int(1800 * ptb_rate / 100)
        rate = 5
        nclass = 5
    if dataset == 'wiki':
        n_flips = int(3600 * ptb_rate / 100)
        rate = 10
        nclass = 5
    if dataset == 'citeseer':
        n_flips = int(2840 / 2 * ptb_rate / 100)
        nclass = 6
    if dataset == 'pubmed':
        n_flips = int(38782 / 2 * ptb_rate / 100)
        nclass = 3
    adj_nn, adj, u, v, test_labels = getAdj(threshold, dataset, rate)
    adj = standardize(adj)
    emb0_u, emb0_v, dim_u, dim_v = getAttribut(u, v, dataset)
    time_start = time.time()
    vlabels = pd.read_csv('./data/' + dataset + '_' + 'vlabels.dat', sep=' ', header=None).to_numpy().astype(int)
    vlabels = np.squeeze(vlabels)[:u + v]
    features = np.ones((adj.shape[0], 32))
    features = sp.csr_matrix(features)

    # adj_matrix, attr_matrix, labels
    A_obs = adj
    _X_obs = features
    _z_obs = vlabels
    '''
    <class 'scipy.sparse.csr.csr_matrix'>
    <class 'scipy.sparse.csr.csr_matrix'>
    <class 'numpy.ndarray'>
    (2995, 2995)
    (2995, 2879)
    (2995,)
    '''

    if _X_obs is None:
        _X_obs = sp.eye(A_obs.shape[0]).tocsr()

    _A_obs = A_obs + A_obs.T
    _A_obs[_A_obs > 1] = 1
    # lcc = largest_connected_components(_A_obs)
    # _A_obs = _A_obs[lcc][:,lcc]
    _A_obs.setdiag(0)
    _A_obs = _A_obs.astype("float32")
    _A_obs.eliminate_zeros()
    _X_obs = _X_obs.astype("float32")

    # node numbers cora--2485
    _N = u + v
    # node classes cora--7
    _K = nclass
    # node feature dim cora--1433
    _F = 32
    _Z_obs = np.eye(_K)[_z_obs]
    # print("node number: %d; node class: %d; node feature: %d" % (_N, _K, _F))
    # onehops_dict, twohops_dict, threehops_dict = sparse_mx_to_khopsgraph(_A_obs)
    # normalized adj sparse matrix

    unlabeled_share = 0.8
    val_share = 0.1
    train_share = 1 - unlabeled_share - val_share
    split_train, split_val, split_test = train_val_test_split_tabular(np.arange(_N),
                                                                      train_size=train_share,
                                                                      val_size=val_share,
                                                                      test_size=unlabeled_share,
                                                                      stratify=_z_obs)

    split_unlabeled = np.union1d(split_val, split_test)
    share_perturbations = args.fake_ratio

    mod_adj_number = n_flips

    ori_adj_tensor = torch.tensor(_A_obs.toarray(), dtype=torch.float32, requires_grad=False)
    ori_adj_tensor_cuda = ori_adj_tensor
    inv_degrees = torch.pow(torch.sum(ori_adj_tensor_cuda, dim=0, keepdim=True), -0.5)
    adj_norm_tensor_cuda = ori_adj_tensor_cuda * inv_degrees * inv_degrees.transpose(0, 1)

    adj_selfloops = torch.add(ori_adj_tensor_cuda, torch.eye(_N))
    target_inv_degrees = torch.pow(torch.sum(adj_selfloops, dim=0, keepdim=True), -0.5)
    target_adj_norm_tensor_cuda = adj_selfloops * target_inv_degrees * target_inv_degrees.transpose(0, 1)

    _X_cuda, _z_cuda, idx_train_cuda, idx_val_cuda, idx_test_cuda = convert_to_Tensor(
        [_X_obs, _Z_obs, split_train, split_val, split_test])

    all_idx_cuda = torch.cat((idx_train_cuda, idx_val_cuda, idx_test_cuda))
    extra_idx_cuda = torch.cat((idx_val_cuda, idx_test_cuda))

    if not read_dir:
        time_start = time.time()
        surrogate_model = Model(_F, args.tar_hidden, _K)

        surrogate_optimizer = optim.Adam(surrogate_model.parameters(), lr=args.tar_lr,
                                         weight_decay=args.tar_weight_decay)

        surrogate_model.model_train(surrogate_optimizer, args.tar_epochs, _X_cuda, target_adj_norm_tensor_cuda, _z_cuda,
                                    idx_train_cuda, idx_val_cuda, use_relu=False, drop_rate=args.drop_rate)

        target_model = Model(_F, args.tar_hidden, _K)
        target_optimizer = optim.Adam(target_model.parameters(), lr=args.tar_lr, weight_decay=args.tar_weight_decay)
        target_model.model_train(target_optimizer, args.tar_epochs, _X_cuda, target_adj_norm_tensor_cuda, _z_cuda,
                                 idx_train_cuda,
                                 idx_val_cuda, use_relu=True, drop_rate=0)
        target_model.model_test(_X_cuda, target_adj_norm_tensor_cuda, _z_cuda, idx_test_cuda, nclass, use_relu=True)
        print('------------------------------------------------------')

        change_edges_list = [[] for i in range(args.init_alive_numbers)]
        changing_adj_list = [ori_adj_tensor_cuda for i in range(args.init_alive_numbers)]
        Greedy_edges_list = [{} for i in range(args.init_alive_numbers)]

        onehops_dict, twohops_dict, threehops_dict, fourhops_dict, degree_distrib = sparse_mx_to_khopsgraph(_A_obs)

        # results of valid and test data
        surrogate_outputs = surrogate_model(_X_cuda, target_adj_norm_tensor_cuda, False, drop_rate=args.drop_rate)
        pre_labels = surrogate_outputs[idx_test_cuda]
        _, predict_test_labels = torch.max(pre_labels, 1)
        predict_test_labels = predict_test_labels.cpu().numpy()
        real_train_labels = _z_obs[split_train]
        real_valid_labels = _z_obs[split_val]
        pre_all_labels = get_all_labels(split_train, real_train_labels, split_val, real_valid_labels, split_test,
                                        predict_test_labels)
        pre_all_labels_cuda = torch.LongTensor(pre_all_labels)

        begin_mutation_rate = 1.0
        end_mutation_rate = args.re_rate
        mutation_step = (begin_mutation_rate - end_mutation_rate) / mod_adj_number
        mutation_rate = 1.0

        for i in range(mod_adj_number):
            mutation_rate -= mutation_step
            # print('current recombination rate %.3f' % mutation_rate)
            for ii in range(len(changing_adj_list)):
                ori_adj_cuda = changing_adj_list[ii]
                Greedy_edges = Greedy_edges_list[ii]
                change_edges = change_edges_list[ii]
                new_adj_tensor_cuda = ori_adj_cuda.clone()
                adj_selfloops = torch.add(new_adj_tensor_cuda, torch.eye(_N))
                inv_degrees = torch.pow(torch.sum(adj_selfloops, dim=0, keepdim=True), -0.5)
                new_adj_norm_tensor_cuda = adj_selfloops * inv_degrees * inv_degrees.transpose(0, 1)
                # new_adj_norm_tensor_cuda.requires_grad = True

                new_surrogate_model = Model(_F, args.tar_hidden, _K)
                new_surrogate_optimizer = optim.Adam(new_surrogate_model.parameters(), lr=args.tar_lr,
                                                     weight_decay=args.tar_weight_decay)
                new_surrogate_model.model_train(new_surrogate_optimizer, args.tar_epochs, _X_cuda,
                                                new_adj_norm_tensor_cuda,
                                                _z_cuda, idx_train_cuda, idx_val_cuda, use_relu=False,
                                                drop_rate=args.drop_rate)

                new_surrogate_model.zero_grad()
                new_adj_norm_tensor_cuda.requires_grad = True

                outputs = new_surrogate_model(_X_cuda, new_adj_norm_tensor_cuda, False, drop_rate=args.drop_rate)
                loss = F.nll_loss(outputs[idx_train_cuda], _z_cuda[idx_train_cuda])

                loss = -loss
                loss.backward()

                grad = -(new_adj_norm_tensor_cuda.grad.data.cpu().numpy().flatten())
                grad_abs = -(np.abs(grad))

                idxes = np.argsort(grad_abs)
                find = 0
                acc = None

                for p in idxes:
                    if (len(Greedy_edges) < args.greedy_edges):
                        x = p // _N
                        y = p % _N
                        if (x, y) in change_edges or (y, x) in change_edges:
                            continue

                        # add edge
                        if grad[p] > 0:
                            signal = 1
                            if x == y or x in onehops_dict[y] or y in onehops_dict[x]:
                                continue
                            else:
                                find, acc = method_add(x, y, new_adj_tensor_cuda, new_surrogate_model, _N, _X_cuda,
                                                       pre_all_labels_cuda, extra_idx_cuda, nclass)
                                # ori_adj_cuda = new_adj_tensor_cuda.clone()
                        # delete edge
                        else:
                            signal = 0
                            if x == y or not x in onehops_dict[y] or not y in onehops_dict[x]:
                                continue
                            else:
                                find, acc = method_del(x, y, new_adj_tensor_cuda, new_surrogate_model, _N, _X_cuda,
                                                       pre_all_labels_cuda, extra_idx_cuda, nclass)
                        if find == 1:
                            edge_oper = (x, y, signal)
                            acc = acc.item()
                            Greedy_edges[edge_oper] = acc
                            # print('Greedy edge number', len(Greedy_edges))
                    else:
                        break
                Greedy_list = sorted(Greedy_edges.items(), key=lambda x: x[1])
                bi_prob = np.random.binomial(1, mutation_rate, 1)[0]
                if bi_prob:
                    selected_edge, cur_acc = Greedy_list[0]
                    x, y, signal = selected_edge
                    change_edges_list[ii].append((x, y))
                    change_edges_list[ii].append((y, x))
                    if signal > 0:
                        changing_adj_list[ii][x, y] = 1
                        changing_adj_list[ii][y, x] = 1
                    else:
                        changing_adj_list[ii][x, y] = 0
                        changing_adj_list[ii][y, x] = 0
                    # print('selected edge: ', x, y, signal, i, ii)
                    Greedy_edges_list[ii].clear()

                else:
                    # print('recombination!------')
                    fir_edge, _ = Greedy_list[0]
                    mu_Greedy_list = Greedy_list[1:]
                    inverse_ranks = [1 / i for i in range(1, len(mu_Greedy_list) + 1)]
                    dis_prob = [i[1] for i in mu_Greedy_list]
                    # dis_counts = sum(dis_values)
                    # dis_prob = [i / dis_counts for i in dis_values]
                    # print(dis_prob)
                    new_dis_prob = [dis_prob[i] * inverse_ranks[i] for i in range(len(dis_prob))]
                    new_dis_prob = torch.FloatTensor(new_dis_prob)
                    new_dis_prob = torch.unsqueeze(new_dis_prob, 0)
                    index = F.gumbel_softmax(new_dis_prob, tau=0.5, hard=True).nonzero()[-1][-1].item()
                    # index_1, index_2 = np.random.choice(len(Greedy_list), 2, replace=False, p=dis_prob)
                    # fir_edge, _ = Greedy_list[index_1]
                    sec_edge, _ = mu_Greedy_list[index]

                    adj = changing_adj_list[ii],
                    changes = change_edges_list[ii]
                    co_list = []
                    fitness_list = []
                    co_list.append(fir_edge)
                    co_list.append(sec_edge)
                    fir_x, fir_y, fir_signal = fir_edge
                    sec_x, sec_y, sec_signal = sec_edge
                    signal = adj[0][fir_x, sec_y]
                    if signal > 0:
                        third_signal = 0
                    else:
                        third_signal = 1
                    third_edge = (fir_x, sec_y, third_signal)
                    signal = adj[0][sec_x, fir_y]
                    if signal > 0:
                        four_signal = 0
                    else:
                        four_signal = 1
                    four_edge = (sec_x, fir_y, four_signal)
                    co_list.append(third_edge)
                    co_list.append(four_edge)

                    for i in range(len(co_list)):
                        x, y, signal = co_list[i]
                        new_adj = adj[0]
                        if (x, y) in changes or (y, x) in changes:
                            fitness_list.append(sys.maxsize)
                            continue
                        else:
                            if signal == 1:
                                new_adj[x, y] = 1.0
                                new_adj[y, x] = 1.0
                            if signal == 0:
                                new_adj[x, y] = 0.0
                                new_adj[x, y] = 0.0

                            adj_selfloops = torch.add(new_adj, torch.eye(_N))
                            inv_degrees = torch.pow(torch.sum(adj_selfloops, dim=0, keepdim=True), -0.5)
                            adj_norm_tensor_cuda = adj_selfloops * inv_degrees * inv_degrees.transpose(0, 1)

                            new_model = Model(_F, args.tar_hidden, _K)
                            new_optimizer = optim.Adam(new_model.parameters(), lr=args.tar_lr,
                                                       weight_decay=args.tar_weight_decay)
                            new_model.model_train(new_optimizer, args.tar_epochs, _X_cuda, adj_norm_tensor_cuda,
                                                  _z_cuda,
                                                  idx_train_cuda, idx_val_cuda, use_relu=False,
                                                  drop_rate=args.drop_rate)
                            new_model.model_test(_X_cuda, adj_norm_tensor_cuda, pre_all_labels_cuda, extra_idx_cuda,
                                                 nclass,
                                                 use_relu=False)
                            loss_test = -new_model.loss_test
                            fitness_list.append(loss_test)

                    fitness_idx = sorted(range(len(fitness_list)), key=lambda k: fitness_list[k])
                    index = fitness_idx[0]

                    selected_edge = co_list[index]
                    x, y, signal = selected_edge
                    change_edges_list[ii].append((x, y))
                    change_edges_list[ii].append((y, x))
                    if signal > 0:
                        changing_adj_list[ii][x, y] = 1
                        changing_adj_list[ii][y, x] = 1
                    else:
                        changing_adj_list[ii][x, y] = 0
                        changing_adj_list[ii][y, x] = 0
                    # print('selected edge: ', x, y, signal, i, ii)
                    Greedy_edges_list[ii].clear()

        for i in range(len(changing_adj_list)):
            accuracies_atk = []
            final_adj_cuda = changing_adj_list[i].clone()
            save_adj = final_adj_cuda.numpy()
            # np.save('modified_graph/' + args.modified_graph_filename + str(i + 1), save_adj)

            adj_selfloops = torch.add(final_adj_cuda, torch.eye(_N))
            inv_degrees = torch.pow(torch.sum(adj_selfloops, dim=0, keepdim=True), -0.5)
            lp_adj_norm_tensor_cuda = adj_selfloops * inv_degrees * inv_degrees.transpose(0, 1)
            inv_degrees = torch.pow(torch.sum(final_adj_cuda, dim=0, keepdim=True), -0.5)
            # perturbed adj
            nolp_adj_norm_tensor_cuda = final_adj_cuda * inv_degrees * inv_degrees.transpose(0, 1)
            # print('The %dth graph results:-----------------------------------------------------------' % (i + 1))
            time_end = time.time()
            # print(type(nolp_adj_norm_tensor_cuda))
            nolp_adj_norm_tensor_cuda = torch.nan_to_num(nolp_adj_norm_tensor_cuda)
            # print(nolp_adj_norm_tensor_cuda)
            nolp_adj_norm_tensor_cuda = nolp_adj_norm_tensor_cuda.numpy()
            nolp_adj_norm_tensor_cuda[nolp_adj_norm_tensor_cuda != 0] = 1
            nolp_adj_norm_tensor_cuda = torch.Tensor(nolp_adj_norm_tensor_cuda)
            adj_matrix_flipped = sp.csr_matrix(nolp_adj_norm_tensor_cuda)
            np.savetxt('./ptb_matrix/epo_ptb_' + dataset + '_' + str(ptb_rate) + '.dat',
                       adj_matrix_flipped.copy().toarray(),
                       fmt='%.2f', delimiter=' ')
    else:
        adj_matrix_flipped = pd.read_csv('./ptb_matrix/epo_ptb_' + dataset + '_' + str(ptb_rate) + '.dat', sep=' ',
                                         header=None)
        adj_matrix_flipped = sp.csr_matrix(torch.tensor(np.array(adj_matrix_flipped)))
        time_end = time.time()

    for _ in range(5):
        print(_)
        print(_, file=data_file)
        u_node_pairs = np.random.randint(0, u - 1, [n_node_pairs * 2, 1])
        v_node_pairs = np.random.randint(u, u + v - 1, [n_node_pairs * 2, 1])
        node_pairs = np.column_stack((u_node_pairs, v_node_pairs))

        adj_matrix_flipped[:u, :u] = 0
        adj_matrix_flipped[u:, u:] = 0
        if train_model == 'netmf':
            embedding_u, _, _, _ = deepwalk_svd(adj_matrix_flipped[:u, u:] @ adj_matrix_flipped[u:, :u], window_size,
                                                dim)
            embedding_v, _, _, _ = deepwalk_svd(adj_matrix_flipped[u:, :u] @ adj_matrix_flipped[:u, u:], window_size,
                                                dim)
            embedding_imp = np.row_stack((embedding_u, embedding_v))
            embedding_exp, _, _, _ = deepwalk_svd(adj_matrix_flipped, window_size, dim)
            embedding = (embedding_imp + embedding_exp) / 2
        if train_model == 'bgnn':
            bgnn = BGNNAdversarial(u, v, batch_size, adj_matrix_flipped[:u, u:], adj_matrix_flipped[u:, :u], emb0_u,
                                   emb0_v, dim_u, dim_v, dataset)
            embedding = bgnn.adversarial_learning()

        if dataset == 'dblp' or dataset == 'wiki':
            auc_score = evaluate_embedding_link_prediction(
                adj_matrix=adj_matrix_flipped,
                node_pairs=node_pairs,
                embedding_matrix=embedding
            )
            print('epo auc:{:.5f}'.format(auc_score))
            print('epo auc:{:.5f}'.format(auc_score), file=data_file)
        else:
            f1_scores_mean, _ = evaluate_embedding_node_classification(embedding, test_labels)
            print('epo, F1: {:.5f} {:.5f}'.format(f1_scores_mean[0], f1_scores_mean[1]))
            print('epo, F1: {:.5f} {:.5f}'.format(f1_scores_mean[0], f1_scores_mean[1]), file=data_file)

    print(train_model)
    print(train_model, file=data_file)
    print(time_end - time_start)
    print(time_end - time_start, file=data_file)
    print(dataset)
    print(dataset, file=data_file)
    data_file.close()


if __name__ == '__main__':
    main()
