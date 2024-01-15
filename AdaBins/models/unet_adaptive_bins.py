import torch
import torch.nn as nn
import torch.nn.functional as F

from .miniViT import mViT

class UpSampleBN(nn.Module):
    '''
    通常用于在神经网络中实现特定的上采样操作
    始化方法 (__init__ 方法):
    skip_input: 输入特征的通道数，用于连接上采样输出和之前的特征。
    output_features: 上采样块的输出通道数。
    层:

    _net: 一个由卷积、批量归一化和 LeakyReLU 组成的序列。包括两个卷积层，
    每个卷积层后面跟随批量归一化和 LeakyReLU 激活函数。

    '''
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        '''
        接收两个输入张量 x 和 concat_with。
        对输入 x 进行上采样，以与 concat_with 相同的大小，使用双线性插值（bilinear interpolation）。
        将上采样的张量和 concat_with 进行通道维度上的拼接。
        将拼接后的张量传递给上采样块的 _net 序列。
        返回上采样块的输出。
        这个模块的作用是执行上采样操作，将输入 x 上采样到与 concat_with 相同的大小，然后将上采样的结果与 concat_with 进行通道拼接，
        最终通过一系列卷积、批量归一化和激活函数处理。
        '''
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048):
        super(DecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSampleBN(skip_input=features // 1 + 112 + 64, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 40 + 24, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 24 + 16, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + 16 + 8, output_features=features // 16)

        #         self.up5 = UpSample(skip_input=features // 16 + 3, output_features=features//16)
        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[
            11]

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        #         x_d5 = self.up5(x_d4, features[0])
        out = self.conv3(x_d4)
        # out = self.act_out(out)
        # if with_features:
        #     return out, features[-1]
        # elif with_intermediate:
        #     return out, [x_block0, x_block1, x_block2, x_block3, x_block4, x_d1, x_d2, x_d3, x_d4]
        return out


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class UnetAdaptiveBins(nn.Module):
    def __init__(self, backend, n_bins=100, min_val=0.1, max_val=10, norm='linear'):
        super(UnetAdaptiveBins, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.encoder = Encoder(backend)
        self.adaptive_bins_layer = mViT(128, n_query_channels=128, patch_size=16,
                                        dim_out=n_bins,
                                        embedding_dim=128, norm=norm)

        self.decoder = DecoderBN(num_classes=128)
        self.conv_out = nn.Sequential(nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))

    def forward(self, x, **kwargs):
        unet_out = self.decoder(self.encoder(x), **kwargs)
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)
        out = self.conv_out(range_attention_maps)

        # Post process
        # n, c, h, w = out.shape
        # hist = torch.sum(out.view(n, c, h * w), dim=2) / (h * w)  # not used for training

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)

        return bin_edges, pred

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.adaptive_bins_layer, self.conv_out]
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build(cls, n_bins, **kwargs):
        basemodel_name = 'tf_efficientnet_b5_ap'

        print('Loading base model ()...'.format(basemodel_name), end='')
        #local
        repo_or_dir ="/home/whuai/.cache/torch/hub/rwightman_gen-efficientnet-pytorch_master"
        basemodel = torch.hub.load(repo_or_dir, basemodel_name, source='local',pretrained=True)

        #url
        # basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        #这一行代码将模型的global_pool属性替换为一个nn.Identity()实例。
        # 将global_pool替换为nn.Identity()相当于将全局池化层去掉，即不再执行池化操作，保留原始的特征图。
        basemodel.global_pool = nn.Identity()
        #同样地，这一行代码将模型的classifier属性替换为一个nn.Identity()实例。在卷积神经网络中，通常最后的分类器层负责将经过全局池化后的特征进行分类。
        # 将classifier替换为nn.Identity()相当于去掉分类器层，保留模型的特征提取部分而不执行最终的分类操作。
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')
        m = cls(basemodel, n_bins=n_bins, **kwargs)
        print('Done.')
        return m


if __name__ == '__main__':
    model = UnetAdaptiveBins.build(100)
    x = torch.rand(2, 3, 480, 640)
    bins, pred = model(x)
    # make_dot(pred, params=dict(list(model.named_parameters()))).render("model_graph", format="png")
    print(bins.shape, pred.shape)
