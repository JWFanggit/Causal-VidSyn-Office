import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from inspect import isfunction
import xformers
from different_topk import STVisualTokenSelection
from diffusers.models.attention_processor import Attention
import einops
import numpy as np
import random
from einops.layers.torch import Rearrange
class up_down_sampling(nn.Module):
    def __init__(self,in_dim,out_dim,):
        super(up_down_sampling,self).__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.conv=nn.Conv2d(in_channels=in_dim, out_channels=out_dim,kernel_size=1)

    def forward(self, A, B,):

        batch_frames, h_w, channels_B = B.shape
        H, W = int(A.shape[1] ** 0.5), int(A.shape[1] ** 0.5)
        h, w = int(B.shape[1] ** 0.5), int(B.shape[1] ** 0.5)
        B = B.permute(0,2,1).contiguous().reshape(batch_frames,channels_B,h,w)
        B= F.interpolate(B, size=(H, W), mode='bilinear', align_corners=False)
        B = self.conv(B)
        batch_frames, new_channels,new_w,new_h  =B.shape
        B_resized=B.permute(0,2,3,1).contiguous().reshape(batch_frames,new_w*new_h,new_channels).contiguous()
        return B_resized

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# class CrossAttention(nn.Module):
#     def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
#         super().__init__()
#         inner_dim = dim_head * heads
#         context_dim = default(context_dim, query_dim)
#
#         self.scale = dim_head**-0.5
#         self.heads = heads
#
#         self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
#         self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
#         self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
#
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
#         )
#
#     def forward(self, x, text, mask=None):
#         B, L, C = x.shape
#         q = self.to_q(x)
#         # text = default(text, x)
#         k = self.to_k(text)
#         v = self.to_v(text)
#
#         q, k, v = map(
#             lambda t: rearrange(t, "B L (H D) -> B H L D", H=self.heads), (q, k, v)
#         )  # B H L D
#         # if ATTENTION_MODE == "flash":
#         x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
#         x = einops.rearrange(x, "B H L D -> B L (H D)")
#         # elif ATTENTION_MODE == "xformers":
#         # x = xformers.ops.memory_efficient_attention(q, k, v)
#         # x = einops.rearrange(x, "B L H D -> B L (H D)", H=self.heads)
#         # elif ATTENTION_MODE == "math":
#         #     attn = (q @ k.transpose(-2, -1)) * self.scale
#         #     attn = attn.softmax(dim=-1)
#         #     attn = self.attn_drop(attn)
#         #     x = (attn @ v).transpose(1, 2).reshape(B, L, C)
#         # else:
#         #     raise NotImplemented
#         return self.to_out(x)
#

class InfoGate(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InfoGate, self).__init__()
        self.a = input_dim
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=input_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=input_dim // 2, out_channels=output_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, input):
        x = torch.cat((input[0], input[1]), 2).permute(0, 2, 1)  # 变换维度为 [batch, channels, length]
        x = self.conv1d(x).permute(0, 2, 1)  # 输入应为 [batch, input_dim, length]
        # h, w = x.shape[2], x.shape[2]  # 假设 h 和 w 相等，若不相等需调整
        # gate = Rearrange('b c h w -> b (h w) c', h=h, w=w)(x)
        gate = F.gumbel_softmax(x, tau=0.3)
        gate = input[0]*gate

        # gate = Rearrange('b (h w) c -> b c h w', h=h, w=w)(gate)
        # out = input[0] * gate[:, 0, :, :].view(-1, 1, h, w)
        # out1 = input[1] * gate[:, 1, :, :].view(-1, 1, h, w)
        return gate








class QA(nn.Module):
    # def __init__(self, query_dim, context_dim, video_frames, embed_dim, topk, non_topk, num_class):
    def __init__(self,in_dim,out_dim):
        super(QA, self).__init__()
        # self.CrossAttention=Attention(query_dim=1024,cross_attention_dim=1024,heads=8,dim_head =64,)
        self.token_selection=STVisualTokenSelection(max_frames=16)
        self.up_down=up_down_sampling(in_dim,out_dim)
        self.info_gated=InfoGate(input_dim=2048,output_dim=1024)
    def forward(self,fusion,video_noise,map):
        batch_size=fusion.shape[0]//16
        bhw,frames,channels=video_noise.shape
        hw=bhw//batch_size
        h, w = int(hw ** 0.5), int(hw ** 0.5)
        video_noise=rearrange(video_noise,'(b h w) f c -> (b f) (h w) c',b=batch_size,h=h,w=w)
        fusion=self.info_gated((fusion,map))
        # up_down=up_down_sampling(video_noise.shape[-1],fusion.shape[-1]).to(device=video_noise.device,dtype=torch.float16)
        #if not to do, may result in GPU out-of-memory.
        video_noise= self.up_down(fusion,video_noise)
        video=fusion+video_noise
        im_token, un_im_token,un_im_do_token = self.token_selection(video)
        return im_token, un_im_token, un_im_do_token





class P_QA(nn.Module):
    # def __init__(self, query_dim, context_dim, video_frames, embed_dim, topk, non_topk, num_class):
    def __init__(self,query_dim,num_class1,num_class2):
        super(P_QA, self).__init__()  # 确保这一行在构造函数中
        self.query_dim=query_dim
        # self.context_dim=context_dim
        self.CrossAttention=Attention(query_dim=1024,cross_attention_dim=1024,heads=8,dim_head =64,)
        self.max_frames=16
        # self.max_pool_topk = nn.MaxPool1d(kernel_size=video_frames*topk)
        # self.max_pool_non_topk = nn.MaxPool1d(kernel_size=video_frames * non_topk)

        self.decoder = nn.Sequential(
            nn.Linear(query_dim, query_dim // 2),
            nn.Tanh(),
            nn.Linear(query_dim // 2, num_class1))

        self.decoder1 = nn.Sequential(
            nn.Linear(query_dim, query_dim // 2),
            nn.Tanh(),
            nn.Linear(query_dim // 2, num_class2))
        self.avg_p=nn.AdaptiveAvgPool1d(1)
    # def forward(self, video,question,option,noise,answer_id,non_aff):
    def forward(self,im_token,un_im_token,un_im_do_token,question,r_option, non_option):
        B, L, D = im_token.shape
        b=B//self.max_frames
        im_token=im_token.reshape(b, self.max_frames, -1, D).reshape(b, -1, D)
        un_im_token = un_im_token.reshape(b, self.max_frames, -1, D).reshape(b, -1, D)
        un_im_do_token =un_im_do_token.reshape(b, self.max_frames, -1, D).reshape(b, -1, D)

        im_q_v1 = torch.cat([question,im_token], dim=1)
        un_im_q_v2 = torch.cat([question,un_im_token],dim=1)
        un_im_q_v3 = torch.cat([question,un_im_do_token], dim=1)
        # max_pool_topk = nn.MaxPool1d(kernel_size=im_q_v1.shape[1])
        # max_pool_non_topk = nn.MaxPool1d(kernel_size=un_im_q_v2.shape[1])
        # max_pool_do_non_topk = nn.MaxPool1d(kernel_size=un_im_q_v3.shape[1])
        im_q_out = self.decoder(self.avg_p(self.CrossAttention(im_q_v1,r_option).transpose(1, 2)).squeeze(2))
        un_im_q_out=self.decoder1(self.avg_p(self.CrossAttention(un_im_q_v2,non_option).transpose(1, 2)).squeeze(2))
        un_im_do_out=self.decoder1(self.avg_p(self.CrossAttention(un_im_q_v3,non_option).transpose(1, 2)).squeeze(2))
        return im_q_out,un_im_q_out,un_im_do_out

