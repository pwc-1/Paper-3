from mindspore import nn
import mindspore.ops as ops
import mindspore.numpy as np
import mindcv


class cv(nn.Cell):
    def __init__(self, image_size, patch_size, patch_size_big, num_classes, batch_size, dim):
        super(cv, self).__init__()

        self.patch_size = patch_size
        self.patch_size_big = patch_size_big
        self.batch_size = batch_size
        self.patch_size2 = int(image_size / patch_size)  # image的长除以patch的长
        self.patch_size2_big = int(image_size / patch_size_big)
        self.patches = self.patch_size2 * self.patch_size2

        self.model = mindcv.create_model('resnet34', pretrained=True)
        modules = list(self.model.cells())[:-1]
        self.model = nn.SequentialCell(*modules)
        self.model.set_train(False)

        self.conv1 = nn.Conv2d(dim, 128, kernel_size=6, stride=5, padding=1, pad_mode='pad')
        self.conv2 = nn.Conv2d(128, 64, kernel_size=5, stride=5, padding=2, pad_mode='pad')
        self.conv3 = nn.Dense(dim, 64)
            # self.pos_embedding = nn.Parameter(np.random.randn(1, patch_size * patch_size + 1, 128))
        # self.cls_token = nn.Parameter(np.random.randn(1, 1, 128))
        self.layer_norm = nn.LayerNorm((128,))
        self.mlp_head = nn.Dense(128, 1)

        self.fc6 = nn.SequentialCell(
            nn.Dense(256, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(64, 1),
        )

    def construct(self, img):
        # small
        # TODO: x = self.to_patch_embedding1(x)
        # Rearrange('b c (h p1) (w p2) -> (b h w) c p1 p2', p1 = patch_size, p2 = patch_size)
        x = ops.reshape(img,
                        (self.batch_size, -1, self.patch_size2, self.patch_size, self.patch_size2, self.patch_size))
        x = ops.permute(x, (0, 2, 4, 1, 3, 5))
        x = ops.reshape(x, (self.patches, -1, self.patch_size, self.patch_size))

        x = self.model(x)
        x = np.squeeze(x)
        # TODO: x = self.to_patch_embedding2(x)
        # Rearrange('(b h w) d -> b d h w',b=batch_size)
        x = ops.reshape(x, (self.batch_size, self.patch_size2, self.patch_size2, -1))
        x = ops.permute(x, (0, 3, 1, 2))
        x = self.conv1(x)
        x = self.conv2(x)
        # TODO: x = self.to_patch_embedding3(x)
        # Rearrange('b d h w-> b (h w) d')
        x = ops.permute(x, (0, 2, 3, 1))
        # 改了
        x = ops.reshape(x, (1, 256, 64))

        # big
        # TODO: x_b = self.to_patch_embedding4(img)
        # Rearrange('b c (h p1) (w p2) -> (b h w) c p1 p2', p1=patch_size_big, p2=patch_size_big)
        x_b = ops.reshape(img, (
        self.batch_size, -1, self.patch_size2_big, self.patch_size_big, self.patch_size2_big, self.patch_size_big))
        x_b = ops.permute(x_b, (0, 2, 4, 1, 3, 5))
        x_b = ops.reshape(x_b, (-1, x_b.shape[3], self.patch_size_big, self.patch_size_big))

        x_b = self.model(x_b)
        x_b = np.squeeze(x_b)
        # TODO: x_b = self.to_patch_embedding5(x_b)
        # Rearrange('(b c) d -> b c d',b=batch_size)
        x_b = ops.reshape(x_b, (self.batch_size, -1, x_b.shape[1]))

        x_b = self.conv3(x_b)
        x = ops.cat((x, x_b), 2)  # ->(1,256,128)

        # 以上代码没有问题

        # x=self.transformer(x) #transformer改卷积
        # x = x[:, 0]
        # x = ops.squeeze(x)
        # feature = x
        x = self.layer_norm(x)
        x = self.mlp_head(x)
        x = ops.squeeze(x, 2)
        Y_pred = self.fc6(x)
        return Y_pred