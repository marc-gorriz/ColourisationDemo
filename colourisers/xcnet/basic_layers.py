import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

__all__ = ["AttentionBlock",
           "ProjectionBlock",
           "DecoderBlock",
           "HeadBlock",
           "FeaturesExtractor"]


class AttentionBlock(nn.Module):
    def __init__(self, dim, out=False):
        super(AttentionBlock, self).__init__()
        self.embed_dim = dim
        self.proj_tgt = nn.Conv2d(dim, dim // 2, kernel_size=1, padding=0)
        self.out = out
        if self.out:
            self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.proj_ref = nn.Conv2d(dim, dim // 2, kernel_size=1, padding=0)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, tgt, ref, need_weights=False):
        bs, c, h, w = tgt.shape
        att_tgt = self.activation(self.proj_tgt(tgt)).flatten(2).transpose(1, 2)
        att_ref = self.activation(self.proj_ref(ref)).flatten(2)

        ref = ref.flatten(2).transpose(1, 2)
        pre_att = torch.bmm(att_tgt, att_ref)
        att = F.softmax(pre_att, dim=-1)
        fused = torch.bmm(att, ref)
        fused = fused.transpose(1, 2).contiguous().view(bs, c, h, w)
        if self.out:
            att_out = self.proj_out(tgt)
            fused = torch.mul(fused, att_out)
        if need_weights:
            return fused, att.view(bs, h, w, h, w)
        else:
            return fused, None


class ProjectionBlock(nn.Module):
    def __init__(self, in_dim, dim=256, get_ref=True):
        super(ProjectionBlock, self).__init__()
        self.get_ref = get_ref
        self.proj_tgt = nn.Conv2d(in_dim, dim, kernel_size=1, padding=0)
        if get_ref:
            self.proj_ref = nn.Conv2d(in_dim, dim, kernel_size=1, padding=0)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, tgt, ref=None):
        tgt = self.proj_tgt(tgt)
        if self.get_ref:
            ref = self.proj_ref(ref)
            return self.activation(tgt), self.activation(ref)
        else:
            return self.activation(tgt)


class DecoderBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(dim_in + dim_out, dim_out, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def _upsample(self, x1, x2):
        x1 = self.upsample(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, src, fused, skip):
        if fused is not None:
            src = src + fused
        x = self.conv1(src)
        x = self.activation(x)
        x = self._upsample(x, skip)
        x = self.conv2(x)
        x = self.activation(x)
        return x


class HeadBlock(nn.Module):
    def __init__(self, dim=256, out_ch=2):
        super(HeadBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0)
        self.activation = nn.ReLU(inplace=True)
        self.out_activation = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.out_activation(x)
        return x



class FeaturesExtractor(nn.Module):
    def __init__(self):
        super(FeaturesExtractor, self).__init__()
        features = vgg19(pretrained=True, progress=False).features
        self.to_relu_1_1 = nn.Sequential()
        self.to_relu_2_1 = nn.Sequential()
        self.to_relu_3_1 = nn.Sequential()
        self.to_relu_4_1 = nn.Sequential()
        self.to_relu_5_1 = nn.Sequential()
        self.to_relu_5_4 = nn.Sequential()

        for x in range(2):
            self.to_relu_1_1.add_module(str(x), features[x])
        for x in range(2, 7):
            self.to_relu_2_1.add_module(str(x), features[x])
        for x in range(7, 12):
            self.to_relu_3_1.add_module(str(x), features[x])
        for x in range(12, 21):
            self.to_relu_4_1.add_module(str(x), features[x])
        for x in range(21, 30):
            self.to_relu_5_1.add_module(str(x), features[x])
        for x in range(30, 36):
            self.to_relu_5_4.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_1(x)
        h_relu_1_1 = h
        h = self.to_relu_2_1(h)
        h_relu_2_1 = h
        h = self.to_relu_3_1(h)
        h_relu_3_1 = h
        h = self.to_relu_4_1(h)
        h_relu_4_1 = h
        h = self.to_relu_5_1(h)
        h_relu_5_1 = h
        h = self.to_relu_5_4(h)
        h_relu_5_4 = h
        out = (h_relu_1_1, h_relu_2_1, h_relu_3_1, h_relu_4_1, h_relu_5_1, h_relu_5_4)
        return out
