import mindspore
from c2net.context import prepare
import glob
import os
from dataprepare import Datasetprepare
import mindspore.dataset as ds
from MyLoss import MyLoss
from model import cv
import numpy
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore as Tensor
from mindspore.train.serialization import save_checkpoint



c2net_context = prepare()
# 获取数据集路径
dataset_path = c2net_context.dataset_path
#获取预训练模型路径
test_path = c2net_context.pretrain_model_path+"/"+"test"
# 输出结果必须保存在该目录
save = c2net_context.output_path
mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target='Ascend')

# 数据加载

file_dir = dataset_path
folder_list = ['101/101', '201/201', '301/301', '401/401', '501/501', '601/601', '701/701']
all_list = []
for folder in folder_list:
    folder_path = os.path.join(file_dir, folder)
    all_list.extend(glob.glob(os.path.join(folder_path, '*.png')))

file_list = [file for file in all_list if os.path.isfile(file)]
dataset = Datasetprepare(file_list)

data_loader = ds.GeneratorDataset(dataset, column_names=["image", "status", "surv_time"])
train_dataset, test_dataset = data_loader.split([0.8,0.2])

# 训练
# 创建自定义的损失函数、模型
loss_fn = MyLoss()
model = cv(
    image_size=6144,
    patch_size=16,
    patch_size_big=384,
    num_classes=1,
    batch_size=1,
    dim=512
)

# 创建、定义优化器
optimizer = nn.Adam(params=model.trainable_params(), learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8)

# 创建列表用于存储预测结果和标签
# label_status = []
# label_survtime = []
batch_size = 32
label_status = numpy.zeros(batch_size)
label_survtime = numpy.zeros(batch_size)

# 设置训练的epoch数
num_epochs = 1

print("------------------train---------------------")
for epoch in range(num_epochs):

    # 创建列表用于存储每个epoch的损失值
    epoch_losses = []
    count = 0
    # 循环进行模型训练
    for data in train_dataset.create_dict_iterator():
        print(f'train_count {count} ', end='')
        input = data['image']

        surv_time = numpy.atleast_1d(data['surv_time'])
        status = numpy.atleast_1d(data['status'])

        # 模型前向传播
        img = ops.permute(Tensor(input, dtype=mindspore.float32), (2, 0, 1))
        output = model(img)

        # 将预测结果和标签添加到列表中
        if count == 0:
            predictions_list = output
            label_status[0] = status[0]
            label_survtime[0] = surv_time[0]
        else:
            predictions_list = ops.cat([predictions_list, output])
            label_survtime[count] = surv_time[0]
            label_status[count] = status[0]
        # label_status.append(status)
        # label_survtime.append(surv_time)
        count = count + 1
        # 当列表中有16个输出结果和对应的标签时，进行损失计算和优化
        if len(predictions_list) == batch_size:
            # loss = loss_fn(predictions_list, label_survtime, label_status)
            # 执行梯度计算
            label_survtime = Tensor(label_survtime, dtype=mindspore.float32)
            label_status = Tensor(label_status, dtype=mindspore.float32)
            dloss, grads = ops.value_and_grad(loss_fn, None, optimizer.parameters)(predictions_list, label_survtime,
                                                                                   label_status)
            epoch_losses.append(dloss.asnumpy())
            print(f'dloss {dloss} ')
            # 更新模型参数
            optimizer(grads)

            label_status = numpy.zeros(batch_size)
            label_survtime = numpy.zeros(batch_size)
            count = 0

    # 打印每个epoch的平均损失值
    print("\n")
    print(f'Epoch {epoch + 1} - Loss: {sum(epoch_losses) / len(epoch_losses)}')

from mindspore.train.serialization import save_checkpoint
ckpt = os.path.join(save,"model.ckpt")
save_checkpoint(model, ckpt)


# from mindspore.train.serialization import load_checkpoint, load_param_into_net
# ckpt_path = os.path.join(test_path, "101201checkpoint.ckpt")
# param_dict = load_checkpoint(ckpt_path)  # 加载模型参数文件
# load_param_into_net(model, param_dict)  # 将参数加载到模型中
print("------------------test---------------------")
from lifelines.utils import concordance_index
def CIndex_lifeline(hazards, labels, surv):
    return (concordance_index(surv, hazards[::-1], labels))

label_status = []
label_survtime = []
count=0
predictions_list= []

for data in test_dataset.create_dict_iterator():
    print(f'test_count {count} ', end='')

    input = data['image']
    status = data['status'].asnumpy()
    surv_time = data['surv_time'].asnumpy()

    # 模型前向传播
    img = ops.permute(Tensor(input, dtype=mindspore.float32), (2, 0, 1))
    output = model(img)
    # 将预测结果和标签添加到列表中
    predictions_list.append(output.asnumpy()[0][0])
    label_status.append(status)
    label_survtime.append(surv_time)

    count = count + 1


test_cindex= CIndex_lifeline(predictions_list, label_status, label_survtime)

print(f'\ntest_cindex {test_cindex}')