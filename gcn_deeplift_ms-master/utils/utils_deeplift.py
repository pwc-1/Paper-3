import numpy as np
import cvxpy as cvx
#import torch
import mindspore as ms
import copy
#from torch import Tensor
from mindspore import Tensor
import mindspore.ops as ops
import scipy.sparse as sp
from mindspore.common import dtype as mstype




def dfs(start,index,end,graph,length,path=[],paths=[]):#Find all paths from a given start point to the given end point
    path.append(index)
    if len(path)==length:
        if path[-1]==end:
            paths.append(path.copy())
            path.pop()
        else:
            path.pop()

    else:
        for item in graph[index]:
            # if item not in path:
                dfs(start,item,end,graph,length,path,paths)
        path.pop()
    return paths
def dfs2(start,index,graph,length,path=[],paths=[]):#Given a length and a starting point, find all paths starting from the starting point with a given length
    path.append(index)
    # print('index',index)
    # if length==0:
    #     return paths
    if len(path)==length:
        paths.append(path.copy())
        path.pop()
    else:
        for item in graph[index]:
            # if item not in path:
                dfs2(start,item,graph,length,path,paths)
        path.pop()

    return paths
def findnewpath(addedgelist,graph,layernumbers,goal): # Find all changed paths.
    resultpath=[]
    for edge in addedgelist:
        # print(edge)
        if edge[0] == goal:
            pt5 = dfs2(edge[1], edge[1], graph, layernumbers, [], [])
            # print(pt5)
            for i1 in pt5:
                # print(i1)
                i1.pop(0)
                if [goal, edge[1]] + i1 not in resultpath:
                    resultpath.append([goal, edge[1]] + i1)

        if edge[1] == goal:
            pt6 = dfs2(edge[0], edge[0], graph, layernumbers, [], [])
            for i1 in pt6:
                i1.pop(0)
                if [goal, edge[0]] + i1 not in resultpath:
                    resultpath.append([goal, edge[0]] + i1)

        for i in range(0, layernumbers - 1):
            # print('i', i)
            pt1 = dfs(goal, goal, edge[0], graph, i + 2, [], [])
            pt2 = dfs2(edge[1], edge[1], graph, layernumbers - i - 1, [], [])
            # print('pt1', pt1)
            # print('pt2', pt2)
            if pt2 != [] or pt1 != []:
                for i1 in pt1:
                    for j1 in pt2:
                        # print(i1 + j1)
                        if i1 + j1 not in resultpath:
                            resultpath.append(i1 + j1)

            # print(edge[1])
            # print(i)
            pt3 = dfs(goal, goal, edge[1], graph, i + 2, [], [])
            pt4 = dfs2(edge[0], edge[0], graph, layernumbers - i - 1, [], [])
            # print('pt3', pt3)
            # print('pt4', pt4)
            if pt3 != [] or pt4 != []:
                for i1 in pt3:
                    for j1 in pt4:
                        # print(i1 + j1)
                        if i1 + j1 not in resultpath:
                            resultpath.append(i1 + j1)
    return resultpath
def cal_contribution(goal,label_index,layernumbers,goalpaths,Hnew,Hold,edgelist,W,feature): #有relu层
    result=dict()
    sumcon=0

    for path in goalpaths:
        R=dict()
        index = layernumbers + 2
        for edge in edgelist:
            for i in range(0, len(path) - 1):
                if path[i] == edge[0] and path[i + 1] == edge[1]:
                    if i < index:
                        index = i
                if path[i] == edge[1] and path[i + 1] == edge[0]:
                    if i < index:
                        index = i
        # print(path, index)
        if index==0:
            a = np.zeros((1, Hnew[layernumbers*2 - 2].shape[1]))
            for i in range(0, Hnew[layernumbers*2 - 2].shape[1]):
                a[0][i] = W[layernumbers][i][label_index]

            R[layernumbers - 1] = a
            # print(R)
            for i in range(layernumbers - 1, 0, -1):
                a = np.zeros((R[i].shape[1], W[i - 1].shape[1]))
                row = 0

                for j in range(0, R[i].shape[1]):
                    x = W[i][:, j]
                    if Hnew[2 * i - 1][path[layernumbers - i]][j] != 0:
                        y = Hnew[2*i][path[layernumbers - i]][j] * x / Hnew[2 * i - 1][path[layernumbers - i]][j]
                        for m in range(0, W[i].shape[0]):
                            a[row][m] = y[m]
                            # print(a[row])
                    row = row + 1
                # print(a)
                R[i - 1] = a

        if index != 0:
            a = np.zeros((1, Hnew[layernumbers*2 - 2].shape[1]))
            for i in range(0, Hnew[layernumbers*2 - 2].shape[1]):
                # print(Hnew[layernumbers - 1].shape[1])
                a[0][i] = W[layernumbers][i][label_index]

            R[layernumbers - 1] = a

            for i in range(layernumbers - 1, 0, -1):
                # print(path, index)
                # print('i', i)
                if i<=(layernumbers-1 )and i >(layernumbers-index):
                    a = np.zeros((R[i].shape[1], W[i - 1].shape[1]))
                    row = 0

                    for j in range(0, R[i].shape[1]):
                        x = W[i][:, j]
                        if (Hnew[2 * i - 1][path[layernumbers - i]][j] -
                            Hold[2 * i - 1][path[layernumbers - i]][
                                j]) != 0:
                            y = (Hnew[2*i][path[layernumbers - i]][j] - Hold[2*i][path[layernumbers - i]][j]) *x / (Hnew[2 * i - 1][path[layernumbers - i]][j] -
                                     Hold[2 * i - 1][path[layernumbers - i]][
                                         j])
                            for m in range(0, W[i].shape[0]):
                                a[row][m] = y[m]

                                # print(a[row])
                        row = row + 1
                    R[i - 1] = a
                if i ==(layernumbers - index) :
                    a = np.zeros((R[i].shape[1], W[i - 1].shape[1]))
                    row = 0

                    for j in range(0, R[i].shape[1]):
                        x =  W[i][:, j]
                        if (Hnew[2 * i - 1][path[layernumbers - i]][j] -
                            Hold[2 * i - 1][path[layernumbers - i]][
                                j]) != 0:
                            y = x / (
                                    Hnew[2 * i - 1][path[layernumbers - i]][j] - Hold[2 * i - 1][
                                path[layernumbers - i]][j])*(
                                    Hnew[2 * i][path[layernumbers - i]][j] - Hold[2 * i][
                                path[layernumbers - i]][j])
                            for m in range(0, W[i].shape[0]):
                                a[row][m] = y[m]

                            # print(a[row])
                        row = row + 1

                    R[i - 1] = a
                if i <(layernumbers - index) and i >0:
                    a = np.zeros((R[i].shape[1], W[i-1].shape[1]))
                    row = 0

                    for j in range(0, R[i].shape[1]):
                        x = W[i][:, j]
                        # print('x.shape',x.shape)
                        if Hnew[2 * i - 1][path[layernumbers - i]][j] == 0:
                            a[row] = 0
                        else:
                            y = x / Hnew[2 * i - 1][path[layernumbers - i]][j]*Hnew[2 * i][path[layernumbers - i]][j]
                            # print(y.shape)
                            # print('w.SHAPE',W[1].shape)
                            for m in range(0, W[i].shape[0]):
                                a[row][m] = y[m]
                                # print(a[row])
                        row = row + 1

                    # print(a)
                    R[i - 1] = a
        # print('r0.shape',R[0].shape)
        # print(R[1].shape)
        # print(R[2].shape)
        R[0]=R[0]*feature[path[-1]]
        mul = dict()
        mul[layernumbers - 2] = np.zeros((R[layernumbers - 2].shape[0], R[layernumbers - 2].shape[1]))
        for i in range(0, R[layernumbers - 1].shape[1]):
            if R[layernumbers - 1][0][i] != 0:
                mul[layernumbers - 2][i] = R[layernumbers - 1][0][i] * R[layernumbers - 2][i]
        for i in range(layernumbers - 2, 0, -1):
            mul[i - 1] = np.zeros((mul[i].shape[0] * mul[i].shape[1], R[i - 1].shape[1]))
            for j in range(0, mul[i].shape[0]):
                if (np.all(mul[i][j] == 0) == False):
                    for l in range(0, mul[i].shape[1]):
                        if mul[i][j][l] != 0:
                            mul[i - 1][j * mul[i].shape[1] + l] = mul[i][j][l] * R[i - 1][l]
        attr = mul[0]

        atr = attr.sum(0)
        atr =atr.sum(0)

        strpath = []
        for pathindex in path:
            strpath.append(str(pathindex))
        c = ','.join(strpath)
        result[c] = atr
        # if path==[0,0,4,0] or path==[0,0,3,0]:
        #     print(path,R)
        # print(R[0].sum(1).sum(0))
        sumcon = sumcon + atr.sum(0)
    return result, sumcon
def main_con(number1,number2,length,goal,ma,num_classes,Hnew,Hold,layernumbers): #Choose important paths so that if  "added" them then will make G_0 quickly close to G_1

    x = cvx.Variable(ma.shape[0])
    y=cvx.Variable(ma.shape[1])

    # y=cvx.Variable(ma.shape[0])
    # print('x',x)
    a=np.zeros((1,num_classes))
    for i in range(0,ma.shape[0]):
        a=a+ma[i]*x[i]
    #
    # c=a+Hold[layernumbers*2-1][goal]
    # print('c.shape',c.shape)
    # print("constraints is DCP:", (sum(x) == length).is_dcp())
    d=0
    for i in range(0,ma.shape[1]):
        d=d+a[i]*softmax(Hnew[layernumbers*2-1][goal] )[i]
    # e=0
    # for i in range(0,ma.shape[1]):
    #     # e=e+cvx.atoms.exp(c[i])
    #     e = e + cvx.atoms.exp(y[i])
    # print('e',e)
    # print('e.shape',e.shape)
    #
    #
    #
    #
    # # f=sum(c*softmax(Hnew[layernumbers*2-1][goal]))
    # # print(c*softmax(Hnew[layernumbers*2-1][goal]).shape)
    # # print(cvx.atoms.log_sum_exp(c).shape)
    objective = cvx.Minimize(-d+cvx.atoms.log_sum_exp(y))
    constraints = [sum(x)== length, y==a+Hold[layernumbers*2-1][goal]]

    for i in range(0,ma.shape[0]):
        constraints.append(0 <= x[i])
        constraints.append(x[i] <= 1)
    # print(constraints)
    prob = cvx.Problem(objective, constraints)
    #
    prob.solve(solver='SCS') #solver='SCS' CVXOPT MOSEKMOSEK
    # print("Optimal value", prob.solve())
    # print("Optimal var")
    # print('x.value',x.value)  # A numpy ndarray.**
    return x.value
def main_con_graph(number1,number2,length,ma,num_classes,Hnew,Hold,layernumbers): #convex

    x = cvx.Variable(ma.shape[0])
    y=cvx.Variable(ma.shape[1])

    # y=cvx.Variable(ma.shape[0])
    # print('x',x)
    a=np.zeros((1,num_classes))
    for i in range(0,ma.shape[0]):
        a=a+ma[i]*x[i]
    #
    # c=a+Hold[layernumbers*2-1][goal]
    # print('c.shape',c.shape)
    # print("constraints is DCP:", (sum(x) == length).is_dcp())
    d=0
    for i in range(0,ma.shape[1]):
        d=d+a[i]*softmax(Hnew )[i]
    # e=0
    # for i in range(0,ma.shape[1]):
    #     # e=e+cvx.atoms.exp(c[i])
    #     e = e + cvx.atoms.exp(y[i])
    # print('e',e)
    # print('e.shape',e.shape)
    #
    #
    #
    #
    # # f=sum(c*softmax(Hnew[layernumbers*2-1][goal]))
    # # print(c*softmax(Hnew[layernumbers*2-1][goal]).shape)
    # # print(cvx.atoms.log_sum_exp(c).shape)
    objective = cvx.Minimize(-d+cvx.atoms.log_sum_exp(y))
    constraints = [sum(x)== length, y==a+Hold]

    for i in range(0,ma.shape[0]):
        constraints.append(0 <= x[i])
        constraints.append(x[i] <= 1)
    # print(constraints)
    prob = cvx.Problem(objective, constraints)
    #
    prob.solve(solver='SCS') #solver='SCS' CVXOPT MOSEKMOSEK
    # print("Optimal value", prob.solve())
    # print("Optimal var")
    # print('x.value',x.value)  # A numpy ndarray.**
    return x.value
def main_con_mask(number1,number2,length,goal,ma,num_classes,Hnew,Hold,layernumbers): #Choose important paths so that if  "masked" them then will make G_1 quickly return to G_0

    x = cvx.Variable(ma.shape[0])
    y=cvx.Variable(ma.shape[1])

    # y=cvx.Variable(ma.shape[0])
    # print('x',x)
    a=np.zeros((1,num_classes))
    for i in range(0,ma.shape[0]):
        a=a+ma[i]*x[i]
    #
    # c=a+Hold[layernumbers*2-1][goal]
    # print('c.shape',c.shape)
    # print("constraints is DCP:", (sum(x) == length).is_dcp())
    d=0
    for i in range(0,ma.shape[1]):
        d=d+a[i]*softmax(Hold[layernumbers*2-1][goal] )[i]
    # e=0
    # for i in range(0,ma.shape[1]):
    #     # e=e+cvx.atoms.exp(c[i])
    #     e = e + cvx.atoms.exp(y[i])
    # print('e',e)
    # print('e.shape',e.shape)
    #
    #
    #
    #
    # # f=sum(c*softmax(Hnew[layernumbers*2-1][goal]))
    # # print(c*softmax(Hnew[layernumbers*2-1][goal]).shape)
    # # print(cvx.atoms.log_sum_exp(c).shape)
    objective = cvx.Minimize(d+cvx.atoms.log_sum_exp(y))
    constraints = [sum(x)== length, y==Hnew[layernumbers*2-1][goal]-a]

    for i in range(0,ma.shape[0]):
        constraints.append(0 <= x[i])
        constraints.append(x[i] <= 1)
    # print(constraints)
    prob = cvx.Problem(objective, constraints)
    #
    prob.solve(solver='SCS') #solver='SCS' CVXOPT MOSEKMOSEK
    # print("Optimal value", prob.solve())
    # print("Optimal var")
    # print('x.value',x.value)  # A numpy ndarray.**
    return x.value
def main_con_mask_graph(number1,number2,length,ma,num_classes,Hnew,Hold,layernumbers): #convex

    x = cvx.Variable(ma.shape[0])
    y=cvx.Variable(ma.shape[1])

    # y=cvx.Variable(ma.shape[0])
    # print('x',x)
    a=np.zeros((1,num_classes))
    for i in range(0,ma.shape[0]):
        a=a+ma[i]*x[i]
    #
    # c=a+Hold[layernumbers*2-1][goal]
    # print('c.shape',c.shape)
    # print("constraints is DCP:", (sum(x) == length).is_dcp())
    d=0
    for i in range(0,ma.shape[1]):
        d=d+a[i]*softmax(Hold )[i]
    # e=0
    # for i in range(0,ma.shape[1]):
    #     # e=e+cvx.atoms.exp(c[i])
    #     e = e + cvx.atoms.exp(y[i])
    # print('e',e)
    # print('e.shape',e.shape)
    #
    #
    #
    #
    # # f=sum(c*softmax(Hnew[layernumbers*2-1][goal]))
    # # print(c*softmax(Hnew[layernumbers*2-1][goal]).shape)
    # # print(cvx.atoms.log_sum_exp(c).shape)
    objective = cvx.Minimize(d+cvx.atoms.log_sum_exp(y))
    constraints = [sum(x)== length, y==Hnew-a]

    for i in range(0,ma.shape[0]):
        constraints.append(0 <= x[i])
        constraints.append(x[i] <= 1)
    # print(constraints)
    prob = cvx.Problem(objective, constraints)
    #
    prob.solve(solver='MOSEK') #solver='SCS' CVXOPT MOSEKMOSEK
    # print("Optimal value", prob.solve())
    # print("Optimal var")
    # print('x.value',x.value)  # A numpy ndarray.**
    return x.value
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def edge_index_both(edges_dict,pa_add,pa_remove,edges_new):
    remove_path_1=[]
    remove_index_1=[]
    remove_path_2=[]
    remove_index_2=[]
    add_path_1=[]
    add_path_2=[]
    if pa_add!=[]:
        for it, path in enumerate(pa_add):
            # print('it',it)
            # if (path[1], path[2]) not in remove_path_1:
            #     remove_path_1.append((path[1], path[2]))
            if (path[2], path[1]) not in remove_path_1:
                remove_path_1.append((path[2], path[1]))
                remove_index_1.append(edges_dict[(path[2], path[1])])
            if (path[1], path[0]) not in add_path_2:
                add_path_2.append((path[1], path[0]))
                remove_index_2.append(edges_dict[(path[1], path[0])])
    if pa_remove!=[]:
        for it, path in enumerate(pa_remove):
            if (path[2], path[1]) not in edges_dict.keys() and (path[2], path[1]) not in add_path_1:
                add_path_1.append((path[2], path[1]))
                # add_index_1.append(edges_dict[(path[2], path[1])])
            if (path[1], path[0]) not in add_path_2:
                add_path_2.append((path[1], path[0]))
            if (path[1], path[0]) not in remove_path_2 and (path[1], path[0]) in edges_dict.keys():
                remove_path_2.append((path[1], path[0]))
                remove_index_2.append(edges_dict[(path[1], path[0])])
    remove_index_1=list(set(remove_index_1))
    remove_index_1 = sorted(remove_index_1)
    remove_index_2 = list(set(remove_index_2))
    remove_index_2 = sorted(remove_index_2)

    edges_1 = copy.deepcopy(edges_new)  # h

    # print('add_path_1', add_path_1)
    # print('add_path_2',add_path_2)
    # print('remove_path_1',remove_path_1)
    # print('remove_path_2', remove_path_2)

    for i in reversed(remove_index_1):
        # print((edges_1[0][i],edges_1[1][i]))
        del edges_1[0][i]
        del edges_1[1][i]
    for path in add_path_1:
        edges_1[0].append(path[0])
        edges_1[1].append(path[1])
    edges_2 = [[], []]  # adj2
    for path in add_path_2:
        edges_2[0].append(path[0])
        edges_2[1].append(path[1])
    edges_3 = copy.deepcopy(edges_new)  # adj1
    # print('remove_index_2',remove_index_2)
    for j in reversed(remove_index_2):
        # print((edges_3[0][j], edges_3[1][j]))
        del edges_3[0][j]
        del edges_3[1][j]
    # print('add_path_1',add_path_1)
    # print('add_path_2', add_path_2)
    edges_index_1 = ms.Tensor(edges_1)
    # edges_index_2 = torch.tensor(edges_2)
    edges_index_2 = ms.Tensor(edges_2)
    edges_index_3 = ms.Tensor(edges_3)

    return edges_index_1, edges_index_2, edges_index_3

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))
def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    # node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    node_mask = ms.Tensor(np.zeros(num_nodes), dtype=ms.dtype.bool_)
    # edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
    edge_mask = ms.Tensor(np.ones(row.shape[0]), dtype=ms.dtype.bool_)

    if isinstance(node_idx, (int, list, tuple)):
        # node_idx = torch.Tensor([node_idx], device=row.device).flatten()
        node_idx = ms.Tensor([node_idx]).flatten()
    #else:
        # node_idx = node_idx.to(row.device)

    inv = None

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask = node_mask.fill(False)
        node_mask[subsets[-1]] = True
        edge_mask = ops.index_select(node_mask, 0, row)
        subsets.append(col[edge_mask])

    # subset, inv = ops.cat(subsets).unique(return_inverse=True)

    concat_op = ops.P.Concat(axis=0)  # 实例化Concat操作，设置拼接轴为0
    concatenated_tensor = concat_op(subsets)  # 拼接列表中的所有张量
    unique_op = ops.P.Unique()  # 实例化Unique操作
    subset, inv = unique_op(concatenated_tensor)

    inv = inv[:node_idx.numel()]


    node_mask = node_mask.fill(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]


    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        # node_idx = row.new_full((num_nodes, ), -1)
        initial_array = np.full((num_nodes, ), -1, dtype=int)
        node_idx = ms.Tensor(initial_array, dtype=row.dtype)
        node_idx[subset] = ops.arange(subset.shape[0])
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask
def map_edges(subedges,mapping,features,nodelist):
    map_edges=[[],[]]
    sub_array = []
    sub_dict=dict()
    subfeatures=np.zeros((len(nodelist),features.shape[1]))
    for idx,node in enumerate(subedges[0]):
        map_edges[0].append(mapping[node])
        map_edges[1].append(mapping[subedges[1][idx]])
        sub_array.append((mapping[node], mapping[subedges[1][idx]]))
        sub_dict[(mapping[node], mapping[subedges[1][idx]])]=idx
    for key,value in mapping.items():
        subfeatures[mapping[key]]=features[key]

    sub_array = np.array(sub_array)
    # print('sub_array', sub_array)
    adj = sp.coo_matrix((np.ones(sub_array.shape[0]), (sub_array[:, 0], sub_array[:, 1])),
                        shape=(len(nodelist), len(nodelist)),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return map_edges,subfeatures,adj,sub_dict
def forward_tensor(adj,layernumbers,W): #有relu
    hiddenmatrix = dict()
    # adj = torch.tensor(adj, requires_grad=True)
    # adj=sparse_mx_to_torch_sparse_tensor(adj)
    relu=ms.nn.ReLU()
    hiddenmatrix[0] = W[0]

    # 将scipy.sparse.coo_matrix对象转换为COOTensor
    values = adj.data
    row_indices = adj.row
    col_indices = adj.col
    # 将numpy数组转换为MindSpore的Tensor
    values_tensor = Tensor(values, dtype=mstype.float32)
    indices_tensor = Tensor(np.vstack((row_indices, col_indices)), dtype=mstype.int64)  # COOTensor要求indices为2D Tensor
    # 创建COOTensor
    adj = ms.COOTensor(indices_tensor, values_tensor, shape=adj.shape)

    # SparseTensorDenseMatmul函数要求输入矩阵的轶必须为2
    try:
        h = ms.nn.SparseTensorDenseMatmul(adj, W[0])
    except Exception as err:
        print(f"报错：{err}")

    hiddenmatrix[1] = ops.mm(h, W[1])
    hiddenmatrix[2]=relu(hiddenmatrix[1])
    # hiddenmatrix[1].retain_grad()
    for i in range(1, layernumbers):
        try:
            h = ms.nn.SparseTensorDenseMatmul(adj, hiddenmatrix[2*i])
        except Exception as err:
            print(f"报错：{err}")
        hiddenmatrix[2*i + 1] = ops.mm(h, W[i + 1])
        if i!=layernumbers-1:
            hiddenmatrix[2*i+2]=relu(hiddenmatrix[2*i + 1])
        # hiddenmatrix[i + 1].retain_grad()
    return hiddenmatrix

def edge_index_add(edges_dict,pa,edges_old):

    add_path_1 = []
    # add_index_1 = []
    remove_path_2 = []
    remove_index_2 = []
    add_path_2 = []

    for it, path in enumerate(pa):
        # print('it',it)
        # if (path[1], path[2]) not in add_path_1:
        #     add_path_1.append((path[1], path[2]))
        if (path[2], path[1]) not in edges_dict.keys() and (path[2], path[1]) not in add_path_1:
            add_path_1.append((path[2], path[1]))
            # add_index_1.append(edges_dict[(path[2], path[1])])
        if (path[1], path[0]) not in add_path_2:
            add_path_2.append((path[1], path[0]))
        if (path[1], path[0])  not in remove_path_2 and (path[1], path[0]) in edges_dict.keys():
            remove_path_2.append((path[1], path[0]))
            remove_index_2.append(edges_dict[(path[1], path[0])])


    remove_index_2 = sorted(remove_index_2)
    # print(add_path_1)
    # print(add_path_2)

    # delindex=0
    addedges_1 = copy.deepcopy(edges_old) #h
    # print('add_index_1',add_index_1)
    for path in add_path_1:
        addedges_1[0].append(path[0])
        addedges_1[1].append(path[1])

    deledges_2=[[],[]]  #adj2
    for path in add_path_2:
        deledges_2[0].append(path[0])
        deledges_2[1].append(path[1])
    # print('deledges_2',deledges_2)


    deledges_3 = copy.deepcopy(edges_old)  #adj1
    # print('remove_index_2',remove_index_2)
    for j in reversed(remove_index_2):
        # print((deledges_3[0][j], deledges_3[1][j]))
        del deledges_3[0][j]
        del deledges_3[1][j]

    deledges_index_1 = ms.Tensor(addedges_1)
    # deledges_index_2 = torch.tensor(deledges_2)
    deledges_index_2 = ms.Tensor(deledges_2)
    deledges_index_3 = ms.Tensor(deledges_3)

    return deledges_index_1,deledges_index_2,deledges_index_3
def smooth(arr, eps=1e-5):
    if 0 in arr:
        return abs(arr - eps)
    else:
        return arr


def KL_divergence(P, Q):
    # Input P and Q would be vector (like messages or priors)
    # Will calculate the KL-divergence D-KL(P || Q) = sum~i ( P(i) * log(Q(i)/P(i)) )
    # Refer to Wikipedia https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    P = smooth(P)
    Q = smooth(Q)
    return sum(P * np.log(P / Q))



def subadj_map(subset,edge_index):
    mapping = dict()
    for it, neighbor in enumerate(subset):
        mapping[neighbor.item()] = it
    # print(mapping)
    # print(np.array(edge_index))
    con_edges = []
    for idx, edge in enumerate(edge_index[0]):
        con_edges.append((edge.item(), edge_index[1][idx].item()))
    edges = np.array(con_edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(subset), len(subset)),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return mapping,adj
def subH(subset,mapping,H,layernumbers):
    sub_H=dict()


    for i in range(0,layernumbers*2):
        subarray=np.array(np.zeros((len(subset),H[i].shape[1])))
        for j in range(len(subset)):
            subarray[mapping[subset[j].item()]]=H[i][subset[j]]
        sub_H[i]=subarray
    return sub_H

def subh1(subset,mapping,W,layernumbers):
    h1=W[0].dot(W[1])
    subarray = np.zeros((len(subset), h1.shape[1]))
    for j in range(len(subset)):
        subarray[mapping[subset[j].item()]] = h1[subset[j]]
    return subarray
def subpath_edge(goalnewaddpath,addedgelist,submapping,edge_index):
    goalnewaddpathmap = []
    for it, path in enumerate(goalnewaddpath):
        pathmap = []
        for j in range(0, len(path)):
            pathmap.append(submapping[path[j]])
        goalnewaddpathmap.append(pathmap)
    # print(goalnewaddpathmap)
    # print(goalnewaddpath)
    addedgelistmap = []
    for it, edge in enumerate(addedgelist):
        # edge_map=[]
        # print(edge[0])
        # print(edge[1])
        if edge[0] in submapping.keys() and edge[1] in submapping.keys():
            addedgelistmap.append([submapping[edge[0]], submapping[edge[1]]])
        # if edge_map!=[]:
        #     addedgelistmap.append(edge_map)
        # if edge[0] in submapping.keys() and edge[1] in submapping.keys():
        #     edge_map.append([submapping[edge[0]],submapping[edge[1]]])
    # print(addedgelistmap)
    subedgesmap = []


    for idx, edge in enumerate(edge_index[0]):
        # print(edge[0])
        # print(edge[1])
        # print(edge[idx])
        subedgesmap.append((edge.item(), edge_index[1][idx].item()))

    return goalnewaddpathmap,addedgelistmap,subedgesmap
def subpath_goalpath(resultdict,submapping):
    submapping_revse=dict((value,key) for key,value in submapping.items())
    resultdict_goalpath=dict()
    for key,value in resultdict.items():
        deepliftpath = []
        goalpath=[]
        s1 = key.split(',')
        for j in s1:
            deepliftpath.append(submapping_revse[int(j)])
        for pathindex in deepliftpath:
           goalpath.append(str(pathindex))
        c = ','.join(goalpath)
        resultdict_goalpath[c] = value
    return resultdict_goalpath

def sparse_mx_to_mindspore_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a MindSpore dense tensor."""
    # 确保稀疏矩阵是 COO 格式，并且数据类型为 np.float32
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # 创建一个与稀疏矩阵同形状的全零密集矩阵
    dense_mx = np.zeros(sparse_mx.shape, dtype=np.float32)
    # 使用 COO 格式的行索引和列索引填充密集矩阵的相应位置
    dense_mx[sparse_mx.row, sparse_mx.col] = sparse_mx.data
    # 将密集矩阵转换为 MindSpore Tensor
    return ms.Tensor(dense_mx)
def edge_index_both_g0(edges_dict,pa_add,pa_remove,edges_old,removeedgelist): #g0出发增加重要路径
    add_path_1 = []
    add_path_2 = []
    remove_path_1 = []
    remove_index_1 = []
    remove_path_2 = []
    remove_index_2 = []
    if pa_add!=[]:
        for it, path in enumerate(pa_add):
            # print('it',it)
            # if (path[1], path[2]) not in add_path_1:
            #     add_path_1.append((path[1], path[2]))
            if (path[2], path[1]) not in edges_dict.keys() and (path[2], path[1]) not in add_path_1:
                add_path_1.append((path[2], path[1]))
                # add_index_1.append(edges_dict[(path[2], path[1])])
            if (path[1], path[0]) not in add_path_2 and (path[1], path[0]) not in edges_dict.keys():
                add_path_2.append((path[1], path[0]))
            # if (path[1], path[0])  not in remove_path_2 and (path[1], path[0]) in edges_dict.keys():
            #     remove_path_2.append((path[1], path[0]))
            #     remove_index_2.append(edges_dict[(path[1], path[0])])
        if removeedgelist!=[]:
            for remve_path in removeedgelist:
                for it,path in enumerate(add_path_1):
                    node=path[1]
                    if node==remve_path[0]:
                        if (remve_path[1],node) not in remove_path_1:
                            remove_path_1.append((remve_path[1],node))
                            remove_index_1.append(edges_dict[(remve_path[1],node)])
                    elif node==remve_path[1]:
                        if ( remve_path[0],node) not in remove_path_1:
                            remove_path_1.append((remve_path[0],node))
                            remove_index_1.append(edges_dict[(remve_path[0],node)])

    if pa_remove!=[]:
        for it, path in enumerate(pa_remove):
            # print('it',it)
            # if (path[1], path[2]) not in remove_path_1:
            #     remove_path_1.append((path[1], path[2]))
            if (path[2], path[1]) not in remove_path_1 and (path[2], path[1]) in edges_dict.keys():
                remove_path_1.append((path[2], path[1]))
                remove_index_1.append(edges_dict[(path[2], path[1])])
            # if (path[0], path[1]) not in remove_path_2:
    # print('add_path_1',add_path_1)
    # print('add_path_2',add_path_2)
    # print('remove_path_1',remove_path_1)

    remove_index_1 = sorted(remove_index_1)

    both_edges_1 = copy.deepcopy(edges_old)  # h
    # print('remove_index_1',remove_index_1)
    for i in reversed(remove_index_1):
        # print((deledges_1[0][i],deledges_1[1][i]))
        del both_edges_1[0][i]
        del both_edges_1[1][i]


    for path in add_path_1:
        both_edges_1[0].append(path[0])
        both_edges_1[1].append(path[1])

    addedges_2 = copy.deepcopy(edges_old)  # h
    for path in add_path_2:
        addedges_2[0].append(path[0])
        addedges_2[1].append(path[1])
    deledges_index_1 = ms.tensor(both_edges_1)
    deledges_index_2 = ms.tensor(addedges_2)
    # deledges_index_3 = torch.tensor(deledges_3)

    return deledges_index_1, deledges_index_2
def edge_index_both_g1(edges_dict,pa_add,pa_remove,edges_new,addedgelist):
    remove_path_1 = []
    remove_index_1 = []
    add_path_1=[]
    add_path_2=[]


    if pa_add!=[]:
        for it, path in enumerate(pa_add):
            if (path[2], path[1]) not in remove_path_1:
                remove_path_1.append((path[2], path[1]))
                remove_index_1.append(edges_dict[(path[2], path[1])])
    if pa_remove!=[]:
        for it, path in enumerate(pa_remove):
            # print('it',it)
            # if (path[1], path[2]) not in add_path_1:
            #     add_path_1.append((path[1], path[2]))
            if (path[2], path[1]) not in edges_dict.keys() and (path[2], path[1]) not in add_path_1:
                add_path_1.append((path[2], path[1]))
                # add_index_1.append(edges_dict[(path[2], path[1])])
            if (path[1], path[0]) not in add_path_2 and (path[1], path[0]) not in edges_dict.keys():
                add_path_2.append((path[1], path[0]))
        if addedgelist!=[]:
            for add_path in addedgelist:
                for it, path in enumerate(add_path_1):
                    node = path[1]
                    if node == add_path[0]:
                        if (add_path[1], node) not in remove_path_1:
                            remove_path_1.append((add_path[1], node))
                            remove_index_1.append(edges_dict[(add_path[1], node)])
                    elif node == add_path[1]:
                        if (add_path[0], node) not in remove_path_1:
                            remove_path_1.append((add_path[0], node))
                            remove_index_1.append(edges_dict[(add_path[0], node)])

    remove_index_1 = sorted(remove_index_1)

    both_edges_1 = copy.deepcopy(edges_new)  # h
    # print('remove_index_1',remove_index_1)
    for i in reversed(remove_index_1):
        # print((deledges_1[0][i],deledges_1[1][i]))
        del both_edges_1[0][i]
        del both_edges_1[1][i]

    for path in add_path_1:
        both_edges_1[0].append(path[0])
        both_edges_1[1].append(path[1])

    addedges_2 = copy.deepcopy(edges_new)  # h
    for path in add_path_2:
        addedges_2[0].append(path[0])
        addedges_2[1].append(path[1])
    deledges_index_1 = ms.tensor(both_edges_1)
    deledges_index_2 = ms.tensor(addedges_2)
    # deledges_index_3 = torch.tensor(deledges_3)

    return deledges_index_1, deledges_index_2