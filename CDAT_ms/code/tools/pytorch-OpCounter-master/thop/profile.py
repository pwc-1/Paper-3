import logging
import warnings

from distutils.version import LooseVersion

from .count_hooks import *
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if LooseVersion(mindspore.__version__) < LooseVersion("1.0.0"):
    logger.warning(
        "You are using an old version PyTorch {version}, which THOP is not going to support in the future.".format(
            version=mindspore.__version__))

register_hooks = {
    x2ms_nn.Conv1d: count_convNd,
    x2ms_nn.Conv2d: count_convNd,
    x2ms_nn.Conv3d: count_convNd,
    x2ms_nn.ConvTranspose1d: count_convNd,
    x2ms_nn.ConvTranspose2d: count_convNd,
    x2ms_nn.ConvTranspose3d: count_convNd,

    x2ms_nn.BatchNorm1d: count_bn,
    x2ms_nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,

    x2ms_nn.ReLU: zero_ops,
    x2ms_nn.ReLU6: zero_ops,
    x2ms_nn.LeakyReLU: count_relu,

    x2ms_nn.MaxPool1d: zero_ops,
    x2ms_nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    mindspore.nn.AdaptiveMaxPool1d: zero_ops,
    mindspore.nn.AdaptiveMaxPool2d: zero_ops,
    mindspore.nn.AdaptiveMaxPool3d: zero_ops,

    x2ms_nn.AvgPool1d: count_avgpool,
    x2ms_nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    x2ms_nn.AdaptiveAvgPool1d: count_adap_avgpool,
    x2ms_nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,

    x2ms_nn.Linear: count_linear,
    x2ms_nn.Dropout: zero_ops,

    x2ms_nn.Upsample: count_upsample,
    x2ms_nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample
}


def profile(model, inputs, custom_ops=None, verbose=True):
    handler_collection = []
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m):
        if len(list(x2ms_adapter.nn_cell.children(m))) > 0:
            return

        if hasattr(m, "total_ops") or hasattr(m, "total_params"):
            logger.warning("Either .total_ops or .total_params is already defined in %s. "
                           "Be careful, it might change your code's behavior." % str(m))

        x2ms_adapter.nn_cell.register_buffer(m, 'total_ops', x2ms_adapter.zeros(1))
        x2ms_adapter.nn_cell.register_buffer(m, 'total_params', x2ms_adapter.zeros(1))

        for p in x2ms_adapter.parameters(m):
            m.total_params += x2ms_adapter.Tensor([x2ms_adapter.tensor_api.numel(p)])

        m_type = type(m)
        fn = None
        if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]

        if fn is None:
            if verbose:
                logger.info("THOP has not implemented counting method for ", m)
        else:
            if verbose:
                logger.info("Register FLOP counter for module %s" % str(m))
            handler = x2ms_adapter.nn_cell.register_forward_hook(m, fn)
            handler_collection.append(handler)

    training = model.training

    x2ms_adapter.x2ms_eval(model)
    x2ms_adapter.nn_cell.apply(model, add_hooks)
    model(*inputs)

    total_ops = 0
    total_params = 0
    for m in x2ms_adapter.nn_cell.modules(model):
        if len(list(x2ms_adapter.nn_cell.children(m))) > 0:  # skip for non-leaf module
            continue
        total_ops += m.total_ops
        total_params += m.total_params

    total_ops = x2ms_adapter.tensor_api.item(total_ops)
    total_params = x2ms_adapter.tensor_api.item(total_params)

    # reset model to original status
    x2ms_adapter.x2ms_train(model, training)
    for handler in handler_collection:
        handler.remove()

    # remove temporal buffers
    for n, m in x2ms_adapter.named_modules(model):
        if len(list(x2ms_adapter.nn_cell.children(m))) > 0:
            continue
        if "total_ops" in m._buffers:
            m._buffers.pop("total_ops")
        if "total_params" in m._buffers:
            m._buffers.pop("total_params")

    return total_ops, total_params


def profile_2(model: nn.Cell, inputs, custom_ops=None, verbose=True):
    handler_collection = {}
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m: nn.Cell):
        # if hasattr(m, "total_ops") or hasattr(m, "total_params"):
        #     logger.warning("Either .total_ops or .total_params is already defined in %s. "
        #                    "Be careful, it might change your code's behavior." % m._get_name())
        x2ms_adapter.nn_cell.register_buffer(m, 'total_ops', x2ms_adapter.zeros(1))
        x2ms_adapter.nn_cell.register_buffer(m, 'total_params', x2ms_adapter.zeros(1))

        for p in x2ms_adapter.parameters(m):
            m.total_params += x2ms_adapter.Tensor([x2ms_adapter.tensor_api.numel(p)])

        m_type = type(m)
        fn = None

        # if defined both op maps, custom_ops takes higher priority.
        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]

        if fn is None:
            if verbose:
                logger.info("THOP has not implemented counting method for %s." % m._get_name())
        else:
            if verbose:
                logger.info("Register FLOP counter for module %s." % m._get_name())
            handler_collection[m] = x2ms_adapter.nn_cell.register_forward_hook(m, fn)

    training = model.training

    x2ms_adapter.x2ms_eval(model)
    x2ms_adapter.nn_cell.apply(model, add_hooks)
    model(*inputs)

    def dfs_count(module: nn.Cell, prefix="\t") -> (int, int):
        total_ops, total_params = 0, 0
        for m in x2ms_adapter.nn_cell.children(module):
            # if not hasattr(m, "total_ops") and not hasattr(m, "total_params"):  # and len(list(m.children())) > 0:
            #     m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
            # else:
            #     m_ops, m_params = m.total_ops, m.total_params
            if m in handler_collection:
                m_ops, m_params = m.total_ops, m.total_params
            else:
                m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
            total_ops += m_ops
            total_params += m_params
        #  print(prefix, module._get_name(), (total_ops.item(), total_params.item()))
        return total_ops, total_params

    total_ops, total_params = (x2ms_adapter.tensor_api.item(_) for _ in dfs_count(model))

    # reset model to original status
    x2ms_adapter.x2ms_train(model, training)
    for m, handler in handler_collection.items():
        handler.remove()
        m._buffers.pop("total_ops")
        m._buffers.pop("total_params")

    return total_ops, total_params
