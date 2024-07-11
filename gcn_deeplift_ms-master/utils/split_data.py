import pickle
from .load_data import *
from .utils_deeplift import *
import random
from mindspore_gl import BatchedGraph, GraphField, Graph
import numpy as np

class gen_graph_data():
    def __init__(self,dataset,index,addedgenum,removeedgenum):
        self.dataset=dataset
        self.index=index
        self.addedgenum=addedgenum
        self.removeedgenum=removeedgenum
    def gen_original_edge(self):
        data = self.dataset[self.index]
        x, edge_index = ms.Tensor(self.dataset.graph_node_feat(self.index),dtype=ms.float32), data.adj_coo
        edge_index_list = data.adj_coo.tolist()
        for i in range(0, x.shape[0]):
            edge_index_list[0].append(i)
            edge_index_list[1].append(i)
        edge_index = ms.tensor(edge_index_list)
        adj_old = rumor_construct_adj_matrix(edge_index_list, x.shape[0])
        adj_old_nonzero = adj_old.nonzero()
        graph_old = matrixtodict(adj_old_nonzero)
        edges_dict_old = dict()
        for i, node in enumerate(edge_index_list[0]):
            edges_dict_old[(node, edge_index_list[1][i])] = i

        in_deg = np.zeros(shape=data.node_count, dtype=np.int32)
        out_deg = np.zeros(shape=data.node_count, dtype=np.int32)
        for r in data.adj_coo[0]:
            out_deg[r] += 1
        for r in data.adj_coo[1]:
            in_deg[r] += 1
        in_deg = ms.Tensor(in_deg, ms.int32)
        out_deg = ms.Tensor(out_deg, ms.int32)
        g = GraphField(ms.Tensor(data.adj_coo[0], dtype=ms.int32), ms.Tensor(data.adj_coo[1], dtype=ms.int32), int(data.node_count), int(data.edge_count))
        return x,edge_index,graph_old,edges_dict_old,adj_old, in_deg, out_deg, g

    def random_edges(self,edge_index,graph_old,edges_dict_old):
        edge_index_list = edge_index.numpy().tolist()
        edge_index_list_new = copy.deepcopy(edge_index_list)
        random.seed()
        removeedgelist = []
        removeedgeindex = []
        removeedgesave = []
        addedgelist = []

        addedgesave = []
        data = self.dataset[self.index]
        x=self.dataset.graph_node_feat(self.index)
        # addedgelist = [[13,6]]
        # removeedgelist=[[12,13]]
        # for path in removeedgelist:
        #     removeedgeindex.append(edges_dict_old[(path[0], path[1])])
        #     removeedgeindex.append(edges_dict_old[(path[1], path[0])])


        for i in range(0, self.removeedgenum):
            a = random.choice(list(range(x.shape[0])))
            # print('a',a)
            # print(graph_old[a])
            list1 = copy.deepcopy(graph_old[a])
            list1.remove(a)

            if list1 != []:
                b = random.choice(list1)
                # print(retD)
                if [a,b] not in removeedgelist and [b,a] not in removeedgelist:
                    removeedgelist.append([a, b])
                    # removeedgesave.append((str(a), str(b)))
                    removeedgeindex.append(edges_dict_old[(a, b)])
                    removeedgeindex.append(edges_dict_old[(b, a)])


        for i in range(0,self.addedgenum):

            c = random.choice(list(range(x.shape[0])))
            retD = list(set(range(0, x.shape[0])).difference(set(graph_old[c])))
            # print(retD)
            if retD != []:
                d = random.choice(retD)
                if [c,d] not in addedgelist and [d,c] not in addedgelist:
                    addedgelist.append([c, d])
                    addedgesave.append((str(c), str(d)))



        # print(len(removeedgeindex))
        # print('removeedgelist', removeedgelist)
        removeedgeindex = list(set(removeedgeindex))
        removeedgeindex = sorted(removeedgeindex)


        for j in reversed(removeedgeindex):
            # if [edge_index_list_new[0][j], edge_index_list_new[1][j]] in removeedgelist or [edge_index_list_new[1][j], edge_index_list_new[0][j]] in removeedgelist:
            #     print('yes')
            # else:
            #     print('false')
            # print((edge_index_list_new[0][j], edge_index_list_new[1][j]))
            del edge_index_list_new[0][j]
            del edge_index_list_new[1][j]
        # print('len edge',len(edge_index_list_new[0]))
        for addpath in addedgelist:
            edge_index_list_new[0].append(addpath[0])
            edge_index_list_new[1].append(addpath[1])
            edge_index_list_new[1].append(addpath[0])
            edge_index_list_new[0].append(addpath[1])
        adj_new = rumor_construct_adj_matrix(edge_index_list_new, x.shape[0])
        # print(adj_old)
        adj_new_nonzero = adj_new.nonzero()
        graph_new = matrixtodict(adj_new_nonzero)

        edge_index_tensor_new = ms.Tensor(edge_index_list_new)
        #==========================
        row=edge_index_list_new[0]
        col=edge_index_list_new[1]
        in_deg = np.zeros(shape=data.node_count, dtype=np.int32)
        out_deg = np.zeros(shape=data.node_count, dtype=np.int32)
        for r in row:
            out_deg[r] += 1
        for r in col:
            in_deg[r] += 1
        in_deg = ms.Tensor(in_deg, ms.int32)
        out_deg = ms.Tensor(out_deg, ms.int32)
        g = GraphField(ms.Tensor(row, dtype=ms.int32), ms.Tensor(col, dtype=ms.int32),
                       int(data.node_count), int(data.edge_count-self.removeedgenum+self.addedgenum))

        return addedgelist,removeedgelist,edge_index_tensor_new,graph_new,adj_new, in_deg, out_deg, g

    def gen_parameters(self,model,in_deg,out_deg,g):
        model.set_train(False)
        feature = self.dataset.graph_node_feat(self.index)
        W = dict()
        #W[0] = feature.detach().numpy()
        W[0] = feature
        for name, param in model.parameters_and_names():
            if name == 'layer0.fc.weight':
                W[1] = param.asnumpy().T
            if name == 'layer1.fc.weight':
                W[2] = param.asnumpy().T

        Hold = dict()
        Hold[0] = feature
        x = ms.Tensor(feature, dtype=ms.float32)
        Hold[1], Hold[2], Hold[3] = model.back(x, in_deg, out_deg, g)
        Hold[1] = Hold[1].asnumpy()
        Hold[2] = Hold[2].asnumpy()
        Hold[3] = Hold[3].asnumpy()
        return W,Hold,feature

    # def guding_edges(self,edge_index,graph_old,edges_dict_old,addedgelist, removeedgelist ):
    #     edge_index_list = edge_index.numpy().tolist()
    #     edge_index_list_new = copy.deepcopy(edge_index_list)
    #     # random.seed()
    #     removeedgeindex = []
    #     removeedgesave = []
    #
    #
    #     addedgesave = []
    #     data = self.dataset[self.index]
    #     x = data.x
    #
    #
    #     for path in removeedgelist:
    #         removeedgeindex.append(edges_dict_old[(path[0], path[1])])
    #         removeedgeindex.append(edges_dict_old[(path[1], path[0])])
    #
    #     removeedgeindex = list(set(removeedgeindex))
    #     removeedgeindex = sorted(removeedgeindex)
    #
    #     for j in reversed(removeedgeindex):
    #         # if [edge_index_list_new[0][j], edge_index_list_new[1][j]] in removeedgelist or [edge_index_list_new[1][j], edge_index_list_new[0][j]] in removeedgelist:
    #         #     print('yes')
    #         # else:
    #         #     print('false')
    #         # print((edge_index_list_new[0][j], edge_index_list_new[1][j]))
    #         del edge_index_list_new[0][j]
    #         del edge_index_list_new[1][j]
    #     # print('len edge',len(edge_index_list_new[0]))
    #     for addpath in addedgelist:
    #         edge_index_list_new[0].append(addpath[0])
    #         edge_index_list_new[1].append(addpath[1])
    #         edge_index_list_new[1].append(addpath[0])
    #         edge_index_list_new[0].append(addpath[1])
    #     adj_new = rumor_construct_adj_matrix(edge_index_list_new, x.shape[0])
    #     # print(adj_old)
    #     adj_new_nonzero = adj_new.nonzero()
    #     graph_new = matrixtodict(adj_new_nonzero)
    #
    #     edge_index_tensor_new = torch.tensor(edge_index_list_new)
    #     return addedgelist, removeedgelist, edge_index_tensor_new, graph_new, adj_new
    # def gen_case_edge(self):
    #     data = self.dataset[self.index]
    #     # print(data)
    #     #
    #     # print(len(data.edge_index[0]))
    #
    #     x, edge_index = data.x, data.edge_index
    #
    #     edge_index_list = data.edge_index.numpy().tolist()
    #     # for i in range(0, x.shape[0]):
    #     #     edge_index_list[0].append(i)
    #     #     edge_index_list[1].append(i)
    #     edge_index = torch.tensor(edge_index_list)
    #     adj_old = rumor_construct_adj_matrix(edge_index_list, x.shape[0])
    #     # print(adj_old)
    #     adj_old_nonzero = adj_old.nonzero()
    #     graph_old = matrixtodict(adj_old_nonzero)
    #     edges_dict_old = dict()
    #     for i, node in enumerate(edge_index_list[0]):
    #         edges_dict_old[(node, edge_index_list[1][i])] = i
    #
    #     return x,edge_index,graph_old,edges_dict_old,adj_old





























