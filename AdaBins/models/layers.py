import torch
import torch.nn as nn


class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=10, embedding_dim=128, num_heads=4):
        '''
        init:
        in_channels: 输入数据的通道数。
        patch_size: 用于构建嵌入的图像块的大小。
        embedding_dim: Transformer 嵌入的维度。
        num_heads: Transformer 中的注意力头的数量。
        '''
        super(PatchTransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)  # takes shape S,N,E

        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)

        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True)

    def forward(self, x):
        '''
        将输入图像 x 通过 embedding_convPxP 进行卷积，然后展平得到嵌入序列 embeddings。
        将随机初始化的位置编码加到嵌入序列中。
        将嵌入序列转换为 Transformer 所需的形状（S, N, E）。
        通过 Transformer 编码器进行编码，并返回结果。
        :param x:
        :return:
        '''
        embeddings = self.embedding_convPxP(x).flatten(2)  # .shape = n,c,s = n, embedding_dim, s
        # embeddings = nn.functional.pad(embeddings, (1,0))  # extra special token at start ?
        embeddings = embeddings + self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0)

        # change to S,N,E format required by transformer
        embeddings = embeddings.permute(2, 0, 1)
        x = self.transformer_encoder(embeddings)  # .shape = S, N, E
        return x


class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):
        '''
        接收两个输入张量 x 和 K，其中 x 是图像块的特征表示，K 是查询。
        执行像素级的点积注意力操作。
        返回点积注意力的结果，形状为 (n, cout, h, w)，其中 n 是批次大小，cout 是 K 的维度，
        h 和 w 是图像块的高度和宽度。
        :param x:
        :param K:
        :return:
        '''
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))  # .shape = n, hw, cout
        return y.permute(0, 2, 1).view(n, cout, h, w)
