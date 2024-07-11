import numpy as np
import mindspore as ms
from mindspore_gl import BatchedGraphField
from mindspore_gl.graph import BatchHomoGraph, PadArray2d, PadHomoGraph, PadMode, PadDirection
from mindspore_gl.dataloader import Dataset


class GCNDataset(Dataset):
    def __init__(self, enzymes_dataset, length=None):
        self.dataset = enzymes_dataset
        self.graph_count = enzymes_dataset.graph_count
        self.node_feat_size = enzymes_dataset.node_feat_size
        self.num_classes = enzymes_dataset.label_dim
        self.length = length

    def __getitem__(self, idx) -> dict:
        if idx >= self.graph_count:
            raise ValueError("Index out of range")
        graph = self.dataset[idx]  # 获取MindHomoGraph对象
        x = self.dataset.graph_node_feat(idx)  # 获取节点特征
        y = self.dataset.graph_label[idx]  # 获取图标签
        #edge_index = graph.adj_coo  # 获取边索引

        # 边索引，需要将全局索引转换为局部索引
        global_edge_index = self.dataset._edge_array[:,self.dataset.graph_edges[idx]: self.dataset.graph_edges[idx + 1]]
        edge_index = global_edge_index - self.dataset.graph_nodes[idx]

        # 将边索引转换为numpy数组，如果需要
        # if not isinstance(edge_index, np.ndarray):
        #     edge_index = np.array(edge_index)

        # 由于MindHomoGraph中的边索引可能已经是零索引，所以直接使用
        # return {
        #     "x": x,
        #     "y": np.array([y], dtype=np.int64),  # 确保y是一个向量
        #     "edge_index": local_edge_index
        # }
        return x, edge_index, y

    def __len__(self):
        return self.length


class MultiHomoGraphDataset(Dataset):
    """MultiHomoGraph Dataset"""
    def __init__(self, dataset, batch_size, length, mode=PadMode.CONST, node_size=50, edge_size=350):
        self._dataset = dataset
        self._batch_size = batch_size
        self._length = length
        self.batch_fn = BatchHomoGraph()
        self.batched_edge_feat = None
        node_size *= batch_size
        edge_size *= batch_size
        if mode == PadMode.CONST:
            edge_feat_size = 1
            self.node_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.CONST, direction=PadDirection.COL,
                                               size=(node_size, dataset.node_feat_size), fill_value=0)
            self.edge_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.CONST, direction=PadDirection.COL,
                                               size=(edge_size, edge_feat_size), fill_value=0)
            'edge_feat_size→0'
            self.graph_pad_op = PadHomoGraph(n_edge=edge_size, n_node=node_size, mode=PadMode.CONST)
        else:
            self.node_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.AUTO, direction=PadDirection.COL,
                                               fill_value=0)
            self.edge_feat_pad_op = PadArray2d(dtype=np.float32, mode=PadMode.AUTO, direction=PadDirection.COL,
                                               fill_value=0)
            self.graph_pad_op = PadHomoGraph(mode=PadMode.AUTO)

        # For Padding
        self.train_mask = np.array([True] * (self._batch_size + 1))
        self.train_mask[-1] = False

    def __getitem__(self, batch_graph_idx):
        graph_list = []
        feature_list = []
        for idx in range(batch_graph_idx.shape[0]):
            graph_list.append(self._dataset[batch_graph_idx[idx]])
            feature_list.append(self._dataset.graph_node_feat(batch_graph_idx[idx]))

        # Batch Graph
        batch_graph = self.batch_fn(graph_list)

        # Pad Graph
        batch_graph = self.graph_pad_op(batch_graph)

        # Batch Node Feat
        batched_node_feat = np.concatenate(feature_list)

        # Pad NodeFeat
        batched_node_feat = self.node_feat_pad_op(batched_node_feat)
        batched_label = self._dataset.graph_label[batch_graph_idx]

        # Pad Label
        batched_label = np.append(batched_label, batched_label[-1] * 0)

        # Get Edge Feat
        if self.batched_edge_feat is None or self.batched_edge_feat.shape[0] < batch_graph.edge_count:
            del self.batched_edge_feat
            self.batched_edge_feat = np.ones([batch_graph.edge_count, 1], dtype=np.float32)

        # Trigger Node_Map_Idx/Edge_Map_Idx Computation, Because It Is Lazily Computed
        _ = batch_graph.batch_meta.node_map_idx
        _ = batch_graph.batch_meta.edge_map_idx

        np_graph_mask = [1] * (self._batch_size + 1)
        np_graph_mask[-1] = 0
        constant_graph_mask = ms.Tensor(np_graph_mask, dtype=ms.int32)
        batchedgraphfiled = self.get_batched_graph_field(batch_graph, constant_graph_mask)
        row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask = batchedgraphfiled.get_batched_graph()
        #### =================================================
        row_numpy = row.asnumpy()
        col_numpy = col.asnumpy()
        # 计算出度和入度
        out_degree = np.bincount(row_numpy, minlength=node_count)
        in_degree = np.bincount(col_numpy, minlength=node_count)
        # 将结果转换回 Tensor
        out_degree = ms.Tensor(out_degree, dtype=ms.int32)
        in_degree = ms.Tensor(in_degree, dtype=ms.int32)
        return row, col, node_count, edge_count, node_map_idx, edge_map_idx, graph_mask, batched_label,\
               batched_node_feat, self.batched_edge_feat[:batch_graph.edge_count, :], in_degree, out_degree

    def get_batched_graph_field(self, batch_graph, constant_graph_mask):
        return BatchedGraphField(
            ms.Tensor.from_numpy(batch_graph.adj_coo[0]),
            ms.Tensor.from_numpy(batch_graph.adj_coo[1]),
            ms.Tensor(batch_graph.node_count, ms.int32),
            ms.Tensor(batch_graph.edge_count, ms.int32),
            ms.Tensor.from_numpy(batch_graph.batch_meta.node_map_idx),
            ms.Tensor.from_numpy(batch_graph.batch_meta.edge_map_idx),
            constant_graph_mask
        )


    def __len__(self):
        return self._length


