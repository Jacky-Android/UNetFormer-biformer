import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnns import Conv,ConvBN,SeparableConvBN,SeparableConvBNReLU,ConvBNReLU
device = torch.device('cuda:0')
class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        
        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.ws)
        coords_w = torch.arange(self.ws)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.ws - 1
        relative_coords[:, :, 0] *= 2 * self.ws - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)


    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x
    
    def forward(self,x):
        B, C, H, W = x.shape

        local  = self.local1(x)+self.local2(x)
        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape

        '''qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)'''#原作者这么写
        
        qkv = self.qkv(x)
        qkv = qkv.view(B, Hp // self.ws, self.ws, Wp // self.ws, self.ws, 3*C).permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.ws, self.ws, 3*C)
        q, k, v = qkv.reshape(qkv.shape[0], self.ws*self.ws,3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        dots = (q @ k.transpose(-2, -1)) * self.scale
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        dots += relative_position_bias.unsqueeze(0)
        attn = dots.softmax(dim=-1)
        attn = attn @ v
        
        '''attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)'''#原作者写法
        attn = attn.reshape(B, C, H,W)[:, :, :H, :W]
        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))
        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out