import argparse
import logging
import mindspore
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

multiply_adds = 1


def zero_ops(m, x, y):
    m.total_ops += x2ms_adapter.Tensor([int(0)])


def count_convNd(m: _ConvNd, x: (mindspore.Tensor,), y):
    x = x[0]

    kernel_ops = x2ms_adapter.tensor_api.numel(x2ms_adapter.zeros(x2ms_adapter.tensor_api.x2ms_size(m.weight)[2:]))  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops = x2ms_adapter.tensor_api.nelement(y) * (m.in_channels // m.groups * kernel_ops + bias_ops)

    m.total_ops += x2ms_adapter.Tensor([int(total_ops)])


def count_convNd_ver2(m: _ConvNd, x: (mindspore.Tensor,), y):
    x = x[0]

    # N x H x W (exclude Cout)
    output_size = x2ms_adapter.tensor_api.numel(x2ms_adapter.zeros((x2ms_adapter.tensor_api.x2ms_size(y)[:1] + x2ms_adapter.tensor_api.x2ms_size(y)[2:])))
    # Cout x Cin x Kw x Kh
    kernel_ops = x2ms_adapter.tensor_api.nelement(m.weight)
    if m.bias is not None:
        # Cout x 1
        kernel_ops += + x2ms_adapter.tensor_api.nelement(m.bias)
    # x N x H x W x Cout x (Cin x Kw x Kh + bias)
    m.total_ops += x2ms_adapter.Tensor([int(output_size * kernel_ops)])


def count_bn(m, x, y):
    x = x[0]

    nelements = x2ms_adapter.tensor_api.numel(x)
    if not m.training:
        # subtract, divide, gamma, beta
        total_ops = 2 * nelements

    m.total_ops += x2ms_adapter.Tensor([int(total_ops)])


def count_relu(m, x, y):
    x = x[0]

    nelements = x2ms_adapter.tensor_api.numel(x)

    m.total_ops += x2ms_adapter.Tensor([int(nelements)])


def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x2ms_adapter.tensor_api.x2ms_size(x)

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += x2ms_adapter.Tensor([int(total_ops)])


def count_avgpool(m, x, y):
    # total_add = torch.prod(torch.Tensor([m.kernel_size]))
    # total_div = 1
    # kernel_ops = total_add + total_div
    kernel_ops = 1
    num_elements = x2ms_adapter.tensor_api.numel(y)
    total_ops = kernel_ops * num_elements

    m.total_ops += x2ms_adapter.Tensor([int(total_ops)])


def count_adap_avgpool(m, x, y):
    kernel = x2ms_adapter.Tensor([*(x[0].shape[2:])]) // x2ms_adapter.tensor_api.squeeze(x2ms_adapter.Tensor(list((m.output_size,))))
    total_add = x2ms_adapter.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = x2ms_adapter.tensor_api.numel(y)
    total_ops = kernel_ops * num_elements

    m.total_ops += x2ms_adapter.Tensor([int(total_ops)])


# TODO: verify the accuracy
def count_upsample(m, x, y):
    if m.mode not in ("nearest", "linear", "bilinear", "bicubic",):  # "trilinear"
        logger.warning("mode %s is not implemented yet, take it a zero op" % m.mode)
        return zero_ops(m, x, y)

    if m.mode == "nearest":
        return zero_ops(m, x, y)

    x = x[0]
    if m.mode == "linear":
        total_ops = x2ms_adapter.tensor_api.nelement(y) * 5  # 2 muls + 3 add
    elif m.mode == "bilinear":
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        total_ops = x2ms_adapter.tensor_api.nelement(y) * 11  # 6 muls + 5 adds
    elif m.mode == "bicubic":
        # https://en.wikipedia.org/wiki/Bicubic_interpolation
        # Product matrix [4x4] x [4x4] x [4x4]
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        total_ops = x2ms_adapter.tensor_api.nelement(y) * (ops_solve_A + ops_solve_p)
    elif m.mode == "trilinear":
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        # can viewed as 2 bilinear + 1 linear
        total_ops = x2ms_adapter.tensor_api.nelement(y) * (13 * 2 + 5)

    m.total_ops += x2ms_adapter.Tensor([int(total_ops)])


def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    total_add += 1 if m.bias is not None else 0
    num_elements = x2ms_adapter.tensor_api.numel(y)
    total_ops = (total_mul + total_add) * num_elements

    m.total_ops += x2ms_adapter.Tensor([int(total_ops)])
