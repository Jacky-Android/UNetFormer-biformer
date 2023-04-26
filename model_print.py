from torchinfo import summary

#from ori import UNetFormer
from models.UNetFormer import UNetFormer
'''#model = UNetFormer(
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='swsl_resnet18',
                 pretrained=False,
                 window_size=8,
                 num_classes=6
                 )'''
#model  = GlobalLocalAttention(dim=256,num_heads=16,qkv_bias=False,window_size=8)
model = UNetFormer(
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='swsl_resnet18',
                 pretrained=False,
                 window_size=8,
                 num_classes=6)

summary(model=model,input_size=(1,3,1024,1024))
