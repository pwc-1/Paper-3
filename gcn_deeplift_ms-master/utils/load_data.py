import numpy as np
import scipy.sparse as sp
#import torch
from sklearn import metrics
import pickle
import os.path as osp
from scipy.special import logsumexp
#from torch_geometric.data import Data,InMemoryDataset, DataLoader
import os
import networkx as nx
import time,datetime
import pandas as pd
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common import dtype as mstype


def read_data(tvt, urp, city_name, train_ratio, val_ratio):
    #    tvt = 'train' / 'val' / 'test'
    #    urp = 'user' / 'review' / 'prod'
    test_ratio = str(round(100 * (1 - train_ratio - val_ratio)))
    train_ratio = str(int(100 * train_ratio))
    val_ratio = str(int(100 * val_ratio))
    with open(f'data/{city_name}/' + tvt + '_' + urp + '_' + city_name + '_' + train_ratio + val_ratio + test_ratio, 'rb') as f:
        #          str(int(100*train_ratio)) +
        #          str(int(100*val_ratio)) +
        #          str(round(100*test_ratio)), 'rb') as f:
        nodelist = pickle.load(f)
    return nodelist
def read_user_prod(review_list):
    user_list =[]
    prod_list =[]
    # for x in review_list:
    #     if x[0] not in user_list:
    #         user_list.append(x[0])
    #     if x[1] not in prod_list:
    #         prod_list.append(x[1])
    user_list = list(set([x[0] for x in review_list]))
    prod_list = list(set([x[1] for x in review_list]))
    user_list=sorted(user_list)
    prod_list=sorted(prod_list)
    return user_list, prod_list
def seperate_r_u(features, idx_list, l_idx, l_fea, l_nums, temp):
    r_idx = []
    r_fea = []
    u_idx = []
    u_fea = []
    for idx in idx_list:
        # print('idx',idx)
        if isinstance(idx, tuple):
            r_idx.append(idx)
            r_fea.append(features[idx])
        elif idx[0] == 'u':
            u_idx.append(idx)
            u_fea.append(features[idx])
    l_idx += (r_idx + u_idx)
    l_fea += (r_fea + u_fea)
    l_nums.append([len(r_idx) + temp,
                   len(r_idx) + len(u_idx) + temp])
    temp += len(r_idx) + len(u_idx)
    return l_idx, l_fea, l_nums, temp
def feature_matrix(features, p_train, p_val, p_test):
    l_idx = []
    l_fea = []
    l_nums = []

    temp = 0
    l_idx, l_fea, l_nums, temp = seperate_r_u(features, p_train, l_idx, l_fea, l_nums, temp)
    l_idx, l_fea, l_nums, temp = seperate_r_u(features, p_val, l_idx, l_fea, l_nums, temp)
    l_idx, l_fea, l_nums, temp = seperate_r_u(features, p_test, l_idx, l_fea, l_nums, temp)

    # features_list=list(features.keys())
    # print(features_list)
    # prod_idx=[]
    # for idx in features_list:
    #     if idx not in p_train and idx not in p_val and idx not in p_test:
    #         prod_idx.append(idx)
    prod_idx = list(set(list(features.keys())) - set(p_train) - set(p_val) - set(p_test))
    prod_idx=sorted(prod_idx)
    # print('prod_idx',prod_idx)
    prod_fea = []
    for idx in prod_idx:
        prod_fea.append(features[idx])

    l_idx += prod_idx
    l_fea += prod_fea
    l_nums.append([len(p_train),
                   len(p_train) + len(p_val),
                   len(p_train) + len(p_val) + len(p_test),
                   len(p_train) + len(p_val) + len(p_test) + len(prod_idx)])
    return l_idx, l_fea, l_nums
def onehot_label(ground_truth, list_idx):
    labels = np.zeros((len(list_idx), 2))

    gt = {}
    user_gt = {}
    for k, v in ground_truth.items():
        u = k[0]
        p = k[1]
        if u not in gt.keys():
            gt[u] = v
            user_gt[u] = v
        else:
            gt[u] |= v #update
            user_gt[u] |= v
        if p not in gt.keys():
            gt[p] = v
        else:
            gt[p] |= v
    ground_truth = {**ground_truth, **gt}

    for it, k in enumerate(list_idx):
        labels[it][ground_truth[k]] = 1
    return labels, user_gt
def construct_edge(ground_truth, idx_map, labels, rev_time, time1, time2, flag):
    edges = [[], []]

    # print(ground_truth.keys())
    keys_list = list(ground_truth.keys())
    if flag == 'month':
        for it, r_id in enumerate(ground_truth.keys()):
            if rev_time[r_id][1] >= time1 and rev_time[r_id][1] < time2:
                edges[0].append(idx_map[r_id])
                edges[1].append(idx_map[r_id[0]])
                edges[0].append(idx_map[r_id])
                edges[1].append(idx_map[r_id[1]])
                edges[0].append(idx_map[r_id[1]])
                edges[1].append(idx_map[r_id])
                edges[0].append(idx_map[r_id[0]])
                edges[1].append(idx_map[r_id])
                # edges.append((idx_map[r_id], idx_map[r_id[0]]))
                # edges.append((idx_map[r_id], idx_map[r_id[1]]))
                #
                # edges.append((idx_map[r_id[0]], idx_map[r_id]))
                # edges.append((idx_map[r_id[1]], idx_map[r_id]))
    if flag == 'week':
        for it, r_id in enumerate(ground_truth.keys()):
            # print('r_id',r_id)
            if rev_time[r_id][2] < time2 and rev_time[r_id][2] >= time1:
                edges[0].append(idx_map[r_id])
                edges[1].append(idx_map[r_id[0]])
                edges[0].append(idx_map[r_id])
                edges[1].append(idx_map[r_id[1]])
                edges[0].append(idx_map[r_id[1]])
                edges[1].append(idx_map[r_id])
                edges[0].append(idx_map[r_id[0]])
                edges[1].append(idx_map[r_id])
        # for i in range(0, len(keys_list)):
        #     r_id = keys_list[i]
        #     if  rev_time[r_id][2] >= time1 and rev_time[r_id][2]<time2 :
        #         edges.append((idx_map[r_id], idx_map[r_id[0]]))
        #         edges.append((idx_map[r_id], idx_map[r_id[1]]))
        #
        #         edges.append((idx_map[r_id[0]], idx_map[r_id]))
        #         edges.append((idx_map[r_id[1]], idx_map[r_id]))
    if flag == 'year':
        # for i in range(0, len(keys_list)):
        #     r_id = keys_list[i]
        #     if rev_time[r_id][0] < time2 and rev_time[r_id][0] >= time1:
        #         edges.append((idx_map[r_id], idx_map[r_id[0]]))
        #         edges.append((idx_map[r_id], idx_map[r_id[1]]))
        #
        #         edges.append((idx_map[r_id[0]], idx_map[r_id]))
        #         edges.append((idx_map[r_id[1]], idx_map[r_id]))
        for it, r_id in enumerate(ground_truth.keys()):
            # print('r_id',r_id)
            if rev_time[r_id][0] < time2 and rev_time[r_id][0] >= time1:
                edges[0].append(idx_map[r_id])
                edges[1].append(idx_map[r_id[0]])
                edges[0].append(idx_map[r_id])
                edges[1].append(idx_map[r_id[1]])
                edges[0].append(idx_map[r_id[1]])
                edges[1].append(idx_map[r_id])
                edges[0].append(idx_map[r_id[0]])
                edges[1].append(idx_map[r_id])
    # for i in range(0,len(keys_list)):
    #     r_id=keys_list[i]
    #     if rev_time[r_id]< year2 and rev_time[r_id]>=year1:
    #         edges.append((idx_map[r_id], idx_map[r_id[0]]))
    #         edges.append((idx_map[r_id], idx_map[r_id[1]]))
    #
    #         edges.append((idx_map[r_id[0]], idx_map[r_id]))
    #         edges.append((idx_map[r_id[1]], idx_map[r_id]))
    edgeitself = list(range(labels.shape[0]))
    for it, edge in enumerate(edgeitself):
        edges[0].append(edge)
        edges[1].append(edge)

    return edges
def construct_adj_matrix(ground_truth, idx_map, labels,rev_time,time1,time2,flag):
    edges = []
    # print(ground_truth.keys())
    keys_list=list(ground_truth.keys())
    if flag=='month':
        for it, r_id in enumerate(ground_truth.keys()):
            if rev_time[r_id][1] >= time1 and rev_time[r_id][1]<time2 :
                edges.append((idx_map[r_id], idx_map[r_id[0]]))
                edges.append((idx_map[r_id], idx_map[r_id[1]]))

                edges.append((idx_map[r_id[0]], idx_map[r_id]))
                edges.append((idx_map[r_id[1]], idx_map[r_id]))
    if flag == 'week':
        for it, r_id in enumerate(ground_truth.keys()):
            # print('r_id',r_id)
            if rev_time[r_id][2] < time2 and rev_time[r_id][2] >= time1:
                edges.append((idx_map[r_id], idx_map[r_id[0]]))
                edges.append((idx_map[r_id], idx_map[r_id[1]]))

                edges.append((idx_map[r_id[0]], idx_map[r_id]))
                edges.append((idx_map[r_id[1]], idx_map[r_id]))
        # for i in range(0, len(keys_list)):
        #     r_id = keys_list[i]
        #     if  rev_time[r_id][2] >= time1 and rev_time[r_id][2]<time2 :
        #         edges.append((idx_map[r_id], idx_map[r_id[0]]))
        #         edges.append((idx_map[r_id], idx_map[r_id[1]]))
        #
        #         edges.append((idx_map[r_id[0]], idx_map[r_id]))
        #         edges.append((idx_map[r_id[1]], idx_map[r_id]))
    if flag == 'year':
        # for i in range(0, len(keys_list)):
        #     r_id = keys_list[i]
        #     if rev_time[r_id][0] < time2 and rev_time[r_id][0] >= time1:
        #         edges.append((idx_map[r_id], idx_map[r_id[0]]))
        #         edges.append((idx_map[r_id], idx_map[r_id[1]]))
        #
        #         edges.append((idx_map[r_id[0]], idx_map[r_id]))
        #         edges.append((idx_map[r_id[1]], idx_map[r_id]))
        for it, r_id in enumerate(ground_truth.keys()):
            # print('r_id',r_id)
            if rev_time[r_id][0]< time2 and rev_time[r_id][0]>=time1:
                edges.append((idx_map[r_id], idx_map[r_id[0]]))
                edges.append((idx_map[r_id], idx_map[r_id[1]]))

                edges.append((idx_map[r_id[0]], idx_map[r_id]))
                edges.append((idx_map[r_id[1]], idx_map[r_id]))
    # for i in range(0,len(keys_list)):
    #     r_id=keys_list[i]
    #     if rev_time[r_id]< year2 and rev_time[r_id]>=year1:
    #         edges.append((idx_map[r_id], idx_map[r_id[0]]))
    #         edges.append((idx_map[r_id], idx_map[r_id[1]]))
    #
    #         edges.append((idx_map[r_id[0]], idx_map[r_id]))
    #         edges.append((idx_map[r_id[1]], idx_map[r_id]))




    edges = np.array(edges)
    # print('edges',edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj
def difference(edgeindex1,edgeindex2): #tensor
    edgedict1=dict()
    edgedict2 = dict()
    in1_not2=[]
    in2_not1=[]
    for idx,node in enumerate(edgeindex1[0]):
        # print(node)
        # print(edgeindex1[1][idx])
        edgedict1[node,edgeindex1[1][idx]]=idx
    for idx,node in enumerate(edgeindex2[0]):
        edgedict2[node,edgeindex2[1][idx]]=idx
    for key in edgedict1.keys():
        if key not in edgedict2.keys():
            in1_not2.append(key)
    for key in edgedict2.keys():
        if key not in edgedict1.keys():
            in2_not1.append(key)
    # print('in1_not2',in1_not2)
    # print('in2_not1', in2_not1)
    # print(len(in1_not2))
    # print(len(in2_not1))
    return in1_not2,in2_not1
def clear(edges):
    edge_clear=[]
    for idx,edge in enumerate(edges):
        if idx%1000==0:
            print('idx',idx)
        if [edge[0],edge[1]] not in edge_clear and [edge[1],edge[0]] not in edge_clear:
            edge_clear.append([edge[0],edge[1]])
    return edge_clear
def matrixtodict(nonzero): # Convert adjacency matrix into dictionary form
    a = []
    graph = dict()
    for i in range(0, len(nonzero[1])):
        if i != len(nonzero[1]) - 1:
            if nonzero[0][i] == nonzero[0][i + 1]:
                a.append(nonzero[1][i])
            if nonzero[0][i] != nonzero[0][i + 1]:
                a.append(nonzero[1][i])
                graph[nonzero[0][i]] = a
                a = []
        if i == len(nonzero[1]) - 1:
            a.append(nonzero[1][i])
        graph[nonzero[0][len(nonzero[1]) - 1]] = a
    return graph
#def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#    """Convert a scipy sparse matrix to a torch sparse tensor."""
#    sparse_mx = sparse_mx.tocoo().astype(np.float32)
#    indices = torch.from_numpy(
#        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#    values = torch.from_numpy(sparse_mx.data)
#   shape = torch.Size(sparse_mx.shape)
#    return torch.sparse.FloatTensor(indices, values, shape)
def sparse_mx_to_mindspore_coo_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a MindSpore COOTensor."""
    # 将稀疏矩阵转换为COO格式，并将数据类型转换为float32
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # 从COO格式矩阵中提取行索引、列索引和值
    indices = np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)  # COOTensor需要64位整型索引
    values = sparse_mx.data
    # 将Numpy数组转换为MindSpore Tensor
    indices_tensor = ms.Tensor(indices, dtype=mstype.int64)
    values_tensor = ms.Tensor(values, dtype=mstype.float32)
    # 创建并返回MindSpore COOTensor
    shape = sparse_mx.shape
    coo_tensor = ms.COOTensor(indices_tensor, values_tensor, shape=shape)
    return coo_tensor
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
def auc_score(output, ground_truth, list_idx, idx_range, u_or_r):
    prob = torch.exp(output[:, 1]).detach().numpy()
    prob_dic = {}
    for it, idx in enumerate(list_idx):
        prob_dic[idx] = prob[it]
    sub_list = [list_idx[x] for x in idx_range]
    sub_true = []
    sub_prob = []
    if u_or_r == 'r':
        for x in sub_list:
            if isinstance(x, tuple):
                sub_true.append(ground_truth[x])
                sub_prob.append(prob_dic[x])
    elif u_or_r == 'u':
        for x in sub_list:
            if isinstance(x, str) and x[0] == 'u':
                sub_true.append(ground_truth[x])
                sub_prob.append(prob_dic[x])
    fpr, tpr, thre = metrics.roc_curve(sub_true, sub_prob)

    #    sub_prob = prob[idx_range].detach().numpy()
    #    if u_or_r == 'r':
    #        sub_true = [ground_truth[x] for x in sub_list if isinstance(x, tuple)]
    #    elif u_or_r == 'u':
    #        sub_true = [ground_truth[x] for x in sub_list if x[0]=='u']
    #    fpr, tpr, thre = metrics.roc_curve(sub_true, sub_prob)

    #    if u_or_r == 'r':
    #        review_prob = prob[len(output)-len(ground_truth):].detach().numpy()
    #        review_true = np.zeros(len(review_prob))
    #        for r_id, num in idx_map.items():
    #            if isinstance(r_id, tuple):
    #                review_true[num + len(ground_truth) - len(output)] = ground_truth[r_id]
    #        fpr, tpr, thre = metrics.roc_curve(review_true, review_prob)
    #
    #    elif u_or_r == 'u':
    #        user_prob = prob[:len(ground_truth)].detach().numpy()
    #        user_true = np.zeros(len(user_prob))
    #        for u_id, num in idx_map.items():
    #            if isinstance(u_id, str) and u_id[0] == 'u':
    #                user_true[num] = ground_truth[u_id]
    #        fpr, tpr, thre = metrics.roc_curve(user_true, user_prob)

    return metrics.auc(fpr, tpr)
def rumor_construct_adj_matrix(edges_index,x):
    edges = []
    # print(ground_truth.keys())
    for idx,node in enumerate(edges_index[0]):
        edges.append((node,edges_index[1][idx]))


    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(x, x),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj
def link_load_data(path):
    # data_dir = 'data'
    # data_csv = 'bitcoinotc.csv'
    # filename = os.path.join(data_dir, data_csv)
    df = pd.read_csv(path)
    # print(df)
    Graphtype = nx.DiGraph()
    G = nx.from_pandas_edgelist(df, source='SOURCE', target='TARGET', edge_attr='RATING', create_using=Graphtype)

    mapping = {}
    count = 0
    for node in list(G.nodes):
        mapping[node] = count
        count = count + 1
    G = nx.relabel_nodes(G, mapping)

    rating = nx.get_edge_attributes(G, 'RATING')
    # print('rating',rating)
    max_rating = rating[max(rating, key=rating.get)]
    degree_sequence_in = [d for n, d in G.in_degree()]
    dmax_in = max(degree_sequence_in)
    degree_sequence_out = [d for n, d in G.out_degree()]
    dmax_out = max(degree_sequence_out)
    # print(A)
    # if (6002,6000) in G.edges():
    #     print('yes')
    # else:
    #     print('false')

    # print(len(G.edges()))
    # print(len(edges_index[0]))

    feat_dict = {}
    feature_length = 8
    for node in list(G.nodes):
        out_edges_list = G.out_edges(node)
        # print('out_edges_list',out_edges_list)

        if len(out_edges_list) == 0:
            features = np.ones(feature_length, dtype=float) / 1000
            feat_dict[node] = {'feat': features}
        else:
            features = np.zeros(feature_length, dtype=float)
            w_pos = 0
            w_neg = 0
            for (_, target) in out_edges_list:
                w = G.get_edge_data(node, target)['RATING']
                if w >= 0:
                    w_pos = w_pos + w
                else:
                    w_neg = w_neg - w

            abstotal = (w_pos + w_neg)
            average = (w_pos - w_neg) / len(out_edges_list) / max_rating

            features[0] = w_pos / max_rating / len(out_edges_list)  # average positive vote
            features[1] = w_neg / max_rating / len(out_edges_list)  # average negative vote
            features[2] = w_pos / abstotal
            features[3] = average
            features[4] = features[0] * G.in_degree(node) / dmax_in
            features[5] = features[1] * G.in_degree(node) / dmax_in
            features[6] = features[0] * G.out_degree(node) / dmax_out
            features[7] = features[1] * G.out_degree(node) / dmax_out

            features = features / 1.01 + 0.001

            feat_dict[node] = {'feat': features}
    nx.set_node_attributes(G, feat_dict)
    G = G.to_undirected()
    # print(G.edges())
    A = nx.adjacency_matrix(G).todense()
    X = np.asarray([G.nodes[node]['feat'] for node in list(G.nodes)])
    edges_index = [[], []]
    for edge in G.edges():
        edges_index[0].append(edge[0])
        edges_index[1].append(edge[1])
        edges_index[1].append(edge[0])
        edges_index[0].append(edge[1])
    # for i in range(max(edges_index[1])+1):
    #     edges_index[0].append(i)
    #     edges_index[1].append(i)

    time_dict = dict()
    df = df.values
    for i in range(0, df.shape[0]):
        edge_0 = df[i][0]
        edge_1 = df[i][1]
        t1 = datetime.datetime.utcfromtimestamp(df[i][3])
        time_dict[(mapping[edge_0], mapping[edge_1])] = t1
    # print(A.shape)
    # print(X)
    # print(X.shape)
    return edges_index,X,mapping,time_dict #边 节点特征
def link_read_data(folder: str, prefix):
    # path=os.path.join(folder, f"{prefix}.npz")
    # data_csv = 'bitcoinotc.csv'
    path = os.path.join(folder, f"{prefix}.csv")
    print(path)
    edges_index,X ,mapping,time_dict= link_load_data(path)

    # x = torch.from_numpy(features).float()
    # y = torch.from_numpy(labels)
    # print('y',y)
    features=torch.DoubleTensor(X)

    edge_index = torch.LongTensor(edges_index)
    print('ed',edge_index)
    data = Data(x=features,  edge_index=edge_index,node_map=mapping,time_dict=time_dict)
    # node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    # node_mask= torch.zeros(adj.shape[0], dtype=torch.bool)
    # train_mask=node_mask.clone()
    # val_mask = node_mask.clone()
    # test_mask = node_mask.clone()
    # print('node_mask',node_mask)
    # for i in range(0,adj.shape[0]):
    #     if i in idx_train:
    #         # print('train')
    #         train_mask[i]=True
    #     if i in idx_test:
    #         # print('test')
    #         test_mask[i]=True
    #     if i in idx_val:
    #         # print('val')
    #         val_mask[i]=True
    #
    # data.train_mask = train_mask
    # data.val_mask = val_mask
    # data.test_mask = test_mask
    return data
# class SynGraphDataset(InMemoryDataset):
#     def __init__(self, root, name, transform=None, pre_transform=None):
#         self.name = name
#         super(SynGraphDataset, self).__init__(root, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])
#
#     @property
#     def raw_dir(self):
#         return osp.join(self.root, self.name, 'raw')
#
#     @property
#     def processed_dir(self):
#         return osp.join(self.root, self.name, 'processed')
#
#     @property
#     def raw_file_names(self):
#         return [f"{self.name}.csv"]
#
#     @property
#     def processed_file_names(self):
#         return ['data.pt']
#
#     def process(self):
#         # Read data into huge `Data` list.
#         data = link_read_data(self.root, self.name)
#         data = data if self.pre_transform is None else self.pre_transform(data)
#         torch.save(self.collate([data]), self.processed_paths[0])
def clear_time(time_dict):
    edge_time = dict()
    for key, value in time_dict.items():
        month = (value.year - 2010) * 12 + value.month
        week = (value.year - 2010) * 52 + value.isocalendar()[1]
        edge_time[key]=(value.year,month,week)
    clear_time = dict()
    for key, value in edge_time.items():
        if (key[1], key[0]) in edge_time.keys():
            clear_time[key] = value
            clear_time[(key[1], key[0])] = value
        if (key[1], key[0]) not in edge_time.keys():
            clear_time[key] = value
            clear_time[(key[1], key[0])] = value
    return clear_time
def clear_time_UCI(time_dict):
    edge_time = dict()
    for key, value in time_dict.items():
        month = (value.year - 2004) * 12 + value.month
        week = (value.year - 2004) * 52 + value.isocalendar()[1]
        edge_time[key]=(value.year,month,week)
    clear_time = dict()
    for key, value in edge_time.items():
        if (key[1], key[0]) in edge_time.keys():
            clear_time[key] = value
            clear_time[(key[1], key[0])] = value
        if (key[1], key[0]) not in edge_time.keys():
            clear_time[key] = value
            clear_time[(key[1], key[0])] = value
    return clear_time
def split_edge(start,end,flag,clear_time):
    edge_index = [[], []]
    if flag == 'year':
        for key, value in clear_time.items():
            if value[0] >= start and value[0] < end:
                edge_index[0].append(key[0])
                edge_index[1].append(key[1])
    if flag == 'month':
        for key, value in clear_time.items():
            if value[1] >= start and value[1] < end:
                edge_index[0].append(key[0])
                edge_index[1].append(key[1])

    if flag=='week':
        for key, value in clear_time.items():
            if value[2] >= start and value[2] < end:
                edge_index[0].append(key[0])
                edge_index[1].append(key[1])
    return edge_index