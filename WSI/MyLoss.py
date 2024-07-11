import mindspore
import numpy
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

class MyLoss(nn.LossBase):
    def __init__(self):
        super(MyLoss, self).__init__()

    def construct(self, prediction, T, E):
        current_batch_len = len(prediction)
        R_matrix_train = numpy.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_matrix_train[i, j] = T[j] >= T[i]
        train_R = Tensor(R_matrix_train, dtype=mindspore.float32)
        train_ystatus = Tensor(E, dtype=mindspore.float32)
        theta = prediction.reshape(-1)
        exp_theta = ops.exp(theta)
        loss_nn = - ops.mean((exp_theta - ops.sum(exp_theta * train_R, dim=1)) * train_ystatus)
        return loss_nn