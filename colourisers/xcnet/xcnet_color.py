import torch
from torch import nn

from .basic_layers import AttentionBlock, DecoderBlock
from .basic_layers import FeaturesExtractor as vgg19_rgb
from .basic_layers import HeadBlock

__all__ = ["XCNET_COLOR"]


class XCNET_COLOR(nn.Module):
    def __init__(self, out=False, out_ch=2):
        super(XCNET_COLOR, self).__init__()
        self.features = vgg19_rgb().eval()

        self.att3 = AttentionBlock(256, out)
        self.att4 = AttentionBlock(512, out)
        self.att5 = AttentionBlock(512, out)

        self.dec4 = DecoderBlock(512, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        self.head4 = HeadBlock(512, out_ch)
        self.head3 = HeadBlock(256, out_ch)
        self.head2 = HeadBlock(128, out_ch)
        self.head1 = HeadBlock(64, out_ch)

    def forward(self, tgt, ref, get_feat=False):
        with torch.no_grad():
            tgt_features = self.features(tgt)
            ref_features = self.features(ref)

        tgt1, tgt2, tgt3, tgt4, tgt5, tgt6 = tgt_features
        _, ref2, ref3, ref4, ref5, _ = ref_features

        fused5 = self.att5(tgt5, ref5, need_weights=True)[0]
        fused4 = self.att4(tgt4, ref4, need_weights=True)[0]
        fused3 = self.att3(tgt3, ref3, need_weights=True)[0]

        dec4 = self.dec4(tgt6, fused5, tgt4)
        dec3 = self.dec3(dec4, fused4, tgt3)
        dec2 = self.dec2(dec3, fused3, tgt2)
        dec1 = self.dec1(dec2, None, tgt1)
        pred1 = self.head1(dec1)

        if get_feat:
            return [pred1, tgt_features]
        else:
            return [pred1]
