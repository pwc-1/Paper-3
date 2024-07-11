import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore_gl import Graph, GraphField, BatchedGraph
from mindspore_gl.nn import GNNCell
from mindspore_gl.nn import GCNConv






class GCNNet(GNNCell):
    """ GCN Net """

    def __init__(self,
                 data_feat_size: int,
                 hidden_dim_size: int,
                 n_classes: int,
                 dropout: float,
                 activation: nn.Cell = None):
        super().__init__()
        self.layer0 = GCNConv(data_feat_size, hidden_dim_size, activation(), dropout)
        self.layer1 = GCNConv(hidden_dim_size, n_classes, None, dropout)

    def construct(self, x, in_deg, out_deg, g: BatchedGraph):
        """GCN Net forward"""
        x = self.layer0(x, in_deg, out_deg, g)
        x = ops.relu(x)
        x = self.layer1(x, in_deg, out_deg, g)
        #x = global_mean_pool(x, batch)
        x = g.avg_nodes(x)
        return x

    def back(self, x, in_deg, out_deg, g: GraphField):
        x_0 = self.layer0(x, in_deg, out_deg, *g.get_graph())
        x_1 = ops.relu(x_0)
        x = self.layer1(x_1, in_deg, out_deg, *g.get_graph())
        return (x_0, x_1, x)

    def forward_pre(self, x, in_deg_1, out_deg_1, in_deg_2, out_deg_2, g1: GraphField, g2: GraphField):
        x_0 = self.layer0(x, in_deg_1, out_deg_1, *g1.get_graph())
        x_1 = ops.relu(x_0)
        x_2 = self.layer1(x_1, in_deg_2, out_deg_2, *g2.get_graph())
        return x_2

    def get_feature(self, x):
        return x



class GCNLossNet(GNNCell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        #self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, x, in_deg, out_deg, train_mask, label, g: BatchedGraph):
        pred = self.net(x, in_deg, out_deg, g)
        # print(pred)
        label = ops.Squeeze()(label)
        loss = self.loss_fn(pred, label)
        loss = loss * train_mask
        return ms.ops.ReduceSum()(loss) / ms.ops.ReduceSum()(train_mask)
