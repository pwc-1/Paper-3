import time
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore_gl.dataset import Enzymes
from mindspore_gl.dataloader import RandomBatchSampler
from mindspore_gl import BatchedGraphField, BatchedGraph, GraphField
from src.dataset import GCNDataset, MultiHomoGraphDataset
from src.GCN import GCNNet, GCNLossNet




# def train():
#     model.set_train()
#     optimizer.zero_grad()
#     loss=0
#     for i in range(150):
#         data=ds[i]
#         batch=ms.tensor([0]*(data.x.shape[0]))
#         out = model(data.x, data.edge_index, batch)
#         loss += criterion(out, data.y)
#     loss.backward()
#     optimizer.step()




def acc_val():
    model.set_train(False)

    correct = 0

    for i in range(150,180):
        data=ds[i]
        batch = ms.tensor([0] * (data.x.shape[0]))
        out = model(data.x, data.edge_index, batch)  # 一次前向传播
        pred = out.argmax(dim=1)  # 使用概率最高的类别
        correct += int((pred == data.y).sum())  # 检查真实标签
    return correct/30

def acc_train():
    model.set_train(False)

    correct = 0

    for i in range(150):
        data=ds[i]
        batch = ms.tensor([0] * (data.x.shape[0]))
        out = model(data.x, data.edge_index, batch)  # 一次前向传播
        pred = out.argmax(dim=1)  # 使用概率最高的类别
        correct += int((pred == data.y).sum())  # 检查真实标签
    return correct/150





if __name__=='__main__':
    ms.context.set_context(device_target="CPU", save_graphs=True, save_graphs_path="./run/", mode=ms.context.GRAPH_MODE, enable_graph_kernel=True)
    batch_size = 32
    data_path = './data/ENZYMES'
    dataset = Enzymes(data_path)
    train_batch_sampler = RandomBatchSampler(dataset.train_graphs, batch_size=batch_size)
    train_graph_dataset = MultiHomoGraphDataset(dataset, batch_size, len(list(train_batch_sampler)))
    train_dataloader = ds.GeneratorDataset(train_graph_dataset, ['row', 'col', 'node_count', 'edge_count',
                                                                       'node_map_idx', 'edge_map_idx', 'graph_mask',
                                                                       'batched_label', 'batched_node_feat',
                                                                       'batched_edge_feat', 'in_degree', 'out_degree'],
                                           sampler=train_batch_sampler)
    test_batch_sampler = RandomBatchSampler(dataset.val_graphs, batch_size=batch_size)
    test_graph_dataset = MultiHomoGraphDataset(dataset, batch_size, len(list(test_batch_sampler)))
    test_dataloader = ds.GeneratorDataset(test_graph_dataset, ['row', 'col', 'node_count', 'edge_count',
                                                                     'node_map_idx', 'edge_map_idx', 'graph_mask',
                                                                     'batched_label', 'batched_node_feat',
                                                                     'batched_edge_feat', 'in_degree', 'out_degree'],
                                          sampler=test_batch_sampler)

    model = GCNNet(data_feat_size=dataset.node_feat_size, hidden_dim_size=16, n_classes=6, dropout=0.5, activation=nn.ReLU)

    learning_rates = nn.piecewise_constant_lr(
        [50, 100, 150, 200, 250, 300, 500], [0.01, 0.005, 0.0025, 0.00125, 0.00125, 0.00125, 0.00125])
    optimizer = nn.optim.Adam(model.trainable_params(), learning_rate=learning_rates)
    loss_net = GCNLossNet(model)
    train_net = nn.TrainOneStepCell(loss_net, optimizer)

    np_graph_mask = [1] * (batch_size + 1)
    np_graph_mask[-1] = 0

    for epoch in range(1, 30):
        start_time = time.time()
        model.set_train(True)
        train_loss, total_iter = 0, 0
        for data in train_dataloader:
            row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask, label, node_feat, edge_feat, in_degree, out_degree = data
            batch_homo = BatchedGraphField(row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask)
            train_loss += train_net(node_feat, in_degree, out_degree, graph_mask, label, *batch_homo.get_batched_graph()) / batch_size
            total_iter += 1
        train_loss /= total_iter

        model.set_train(False)
        train_count = 0.0
        for data in train_dataloader:
            row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask, label, node_feat, edge_feat, in_degree, out_degree = data
            batch_homo = BatchedGraphField(row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask)
            output = model(node_feat, in_degree, out_degree, *batch_homo.get_batched_graph()).asnumpy()
            predict = np.argmax(output, axis=1)
            #print(f"pred:{predict},label:{label}")
            train_count += np.sum(np.equal(predict, label.asnumpy()) * np_graph_mask)
        train_acc = train_count / len(list(train_batch_sampler)) / batch_size
        end_time = time.time()

        test_count = 0.0
        for data in test_dataloader:
            row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask, label, node_feat, edge_feat, in_degree, out_degree = data
            batch_homo = BatchedGraphField(row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask)
            output = model(node_feat, in_degree, out_degree, *batch_homo.get_batched_graph()).asnumpy()
            predict = np.argmax(output, axis=1)
            test_count += np.sum(np.equal(predict, label) * np_graph_mask)
        test_acc = test_count / len(list(test_batch_sampler)) / batch_size
        print('Epoch {}, Time {:.3f} s, Train loss {}, Train acc {}, Test acc {}'.format(epoch,
                                                                                                 end_time - start_time,
                                                                                                 train_loss, train_acc,
                                                                                                 test_acc))

    # 保存模型
    save_path = "./gcn_enzymes.ckpt"
    ms.save_checkpoint(train_net, save_path)
    print(f"Model saved to {save_path}")
