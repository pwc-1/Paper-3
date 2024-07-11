from mindspore import Tensor
from PIL import Image
import numpy
import math


class Datasetprepare:
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        idx = int(idx)
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        target_size = (6144, 6144)
        img = img.resize(target_size)
        img = numpy.array(img)
        img_transformed = Tensor(img)
        # print(img_transformed.shape)
        # 这里原来好像写反了，把status和surv_time两个互换了
        surv_time = img_path.split('/')[-1].split('.')[0].split('_')[-1]
        status = img_path.split('/')[-1].split('.')[0].split('_')[-2]
        return img_transformed, float(status), math.log10(abs(float(surv_time)))