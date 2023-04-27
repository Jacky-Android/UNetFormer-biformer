

import timm
import torch
import torch.nn as nn
from .Decoder import Decoder
class UNetFormer(nn.Module):
    def __init__(self,
                 decode_channels=32,
                 dropout=0.1,
                 backbone_name='resnet18',
                 pretrained=True,
                 window_size=8,
                 num_classes=6
                 ):
        super(UNetFormer,self).__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()
       

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

    def forward(self, x):
        
        res  = self.backbone(x)
    
        #res1, res2, res3, res4 = self.backbone(x)
        h, w = x.size()[-2:]
       
        x = self.decoder(res,h,w)
        return x
