
import argparse
import random
import numpy as np
#import torch
#from torch_geometric.datasets import TUDataset
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from mindspore_gl.dataset import Enzymes
from mindspore_gl import GNNCell, BatchedGraph, BatchedGraphField
from utils.split_data import gen_graph_data
from utils.utils_deeplift import findnewpath,softmax,KL_divergence
from utils.constant import time_step,path_number
from utils.evalation import metrics_KL_graph
from convex import convex_graph
from src.GCN import GCNNet, GCNLossNet
from src.dataset import MultiHomoGraphDataset



class LossNet(GNNCell):
    """ LossNet definition """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = ms.nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, node_feat, edge_weight, target, g: BatchedGraph):
        predict = self.net(node_feat, edge_weight, g)
        target = ops.Squeeze()(target)
        loss = self.loss_fn(predict, target)
        loss = ops.ReduceSum()(loss * g.graph_mask)
        return loss


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mutag') #Zip NYC
    parser.add_argument('--type', type=str, default='both') #add remove both
    parser.add_argument('--snapshot_index',type=int,default=0) #0,1,2,3
    parser.add_argument('--start_time1',type=int,default=0)
    parser.add_argument('--end_time1', type=int, default=0)
    parser.add_argument('--start_time2', type=int, default=0)
    parser.add_argument('--end_time2', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='./data/ENZYMES')
    parser.add_argument('--hidden_dim_size', type=int, default=16) #In the link prediction task, the output dimension of the GNN model
    args = parser.parse_args()
    time_step=time_step(args.dataset,args.type)
    layernumbers = 2
    #nclass = 2
    num_class = 6
    delete_list = []
    goal_kl_list = []

    if args.dataset=='mutag':
        if args.type == 'both':
            addedgenum = 5
            removeedgenum = 5
        elif args.type == 'add':
            addedgenum = 5
            removeedgenum = 0
        elif args.type == 'remove':
            addedgenum = 0
            removeedgenum = 5

        root = './data/ENZYMES'
        dataset = Enzymes(root)


        model = GCNNet(data_feat_size=dataset.node_feat_size, hidden_dim_size=16, n_classes=6, dropout=0.5, activation=nn.ReLU)
        optimizer = nn.optim.Adam(model.trainable_params(), learning_rate=0.01)
        loss_net = GCNLossNet(model)
        train_net = nn.TrainOneStepCell(loss_net, optimizer)

        model.set_train(False)
        param_dict = ms.load_checkpoint("./gcn_enzymes.ckpt")
        ms.load_param_into_net(model, param_dict)

        total_node = 480
        total_node_list = random.sample(list(range(600)), total_node)
        random_num = 5
        Convex_add_KL = np.ones((total_node * random_num, 5))
        Convex_mask_KL = np.ones((total_node * random_num, 5))
        save_edge = dict()
        index_save = 0
        graph_KL_list = []
        for index in range(len(total_node_list)):

            node_index = total_node_list[index]
            old_data = gen_graph_data(dataset, node_index, addedgenum, removeedgenum)
            x, edge_index_old, graph_old, edges_dict_old, adj_old, in_deg, out_deg, graph_field = old_data.gen_original_edge()

            layernumbers = 2
            W, Hold, feature = old_data.gen_parameters(model, in_deg, out_deg, graph_field)

            for idx in range(random_num):
                print('idx', idx)
                addedgelist, removeedgelist, edge_index_new, graph_new, adj_new, in_deg_new, out_deg_new, graph_field_new = old_data.random_edges(edge_index_old, graph_old, edges_dict_old)
                _, Hnew, _ = old_data.gen_parameters(model, in_deg_new, out_deg_new, graph_field_new)


                graph_KL = KL_divergence(softmax(np.mean(Hnew[layernumbers * 2 - 1], axis=0)),
                                         softmax(np.mean(Hold[layernumbers * 2 - 1], axis=0)))
                graph_KL_list.append(graph_KL)



                print('graph_KL', graph_KL)
                if graph_KL > 0.01:
                    index_save_edge = dict()
                    index_save_edge['addedgelist'] = addedgelist
                    index_save_edge['removeedgelist'] = removeedgelist
                    save_edge[str(index_save)] = index_save_edge
                    edges_dict_new = dict()
                    for i, node in enumerate(edge_index_new[0]):
                        edges_dict_new[(node.item(), edge_index_new[1][i].item())] = i


                    layernumbers = 2
                    addgoalpath = []
                    removegoalpath = []
                    newgoalpaths = []
                    oldgoalpaths = []

                    KL_eval = metrics_KL_graph(
                        np.mean(Hold[layernumbers * 2 - 1], axis=0), np.mean(Hnew[layernumbers * 2 - 1], axis=0))

                    addgoalpath = []
                    removegoalpath = []
                    for goal in range(0, x.shape[0]):

                        addgoalpath = addgoalpath + findnewpath(addedgelist, graph_new, layernumbers, goal)

                        removegoalpath = removegoalpath + findnewpath(removeedgelist, graph_old, layernumbers, goal)

                    print('addgoalpath', len(addgoalpath))
                    print('removegoalpath', len(removegoalpath))

                    select_pathlist = path_number(args.dataset, args.type, len(addgoalpath) + len(removegoalpath))

                    convex_method = convex_graph(Hold, Hnew, W, goal, addedgelist, removeedgelist, addgoalpath,
                                           removegoalpath, \
                                           feature, layernumbers, 6, select_pathlist, \
                                           model, edge_index_new.asnumpy().tolist(), edge_index_old.asnumpy().tolist(),
                                           args.dataset,
                                           args.type)

                    contriution_value = convex_method.contribution_value()

                    # x = ms.Tensor(dataset.graph_node_feat(total_node_list[index]))
                    contriution_value = contriution_value / ms.Tensor(dataset.graph_node_feat(total_node_list[index])).shape[0]

                    true_abs = np.mean(Hnew[layernumbers * 2 - 1], axis=0) - np.mean(Hold[layernumbers * 2 - 1], axis=0)
                    pred_abs = sum(contriution_value)
                    print('true', true_abs)
                    print('pred', pred_abs)


                    select_addpaths, select_removepaths, select_addpaths_mask, select_removepaths_mask  = convex_method.select_importantpath(
                        contriution_value)
                    convex_logists_mask, convex_logists_add = convex_method.evaluate(select_addpaths,
                                                                                     select_removepaths,
                                                                                     select_addpaths_mask,
                                                                                     select_removepaths_mask)


                    convex_mask_KL, convex_add_KL = KL_eval.KL(convex_logists_add, convex_logists_mask)

                    Convex_mask_KL[index_save] = np.array(convex_mask_KL)
                    Convex_add_KL[index_save] = np.array(convex_add_KL)

                    print('ceshi index', index_save)
                    print('convex_mask_KL', convex_mask_KL)
                    print('convex_add_KL', convex_add_KL)














