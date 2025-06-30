import torch
import torch.nn as nn
from einops import rearrange
#
# class FourierEmbedder():
#     def __init__(self, num_freqs=64, temperature=100):
#         self.num_freqs = num_freqs
#         self.temperature = temperature
#         self.freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)
#
#     @torch.no_grad()
#     def __call__(self, x, cat_dim=-1):
#         "x: arbitrary shape of tensor. dim: cat dim"
#         out = []
#         for freq in self.freq_bands:
#             out.append(torch.sin(freq * x))
#             out.append(torch.cos(freq * x))
#         return torch.cat(out, cat_dim)
#
#
# class PositionNet(nn.Module):
#     def __init__(self, in_dim, out_dim, fourier_freqs=8):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#
#         self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
#         self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy
#
#         self.linears = nn.Sequential(
#             # nn.Linear( self.in_dim + self.position_dim, 512),
#             nn.Linear(self.in_dim, 512),
#             nn.SiLU(),
#             nn.Linear(512, 512),
#             nn.SiLU(),
#             nn.Linear(512, out_dim),
#         )
#
#         self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
#         self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
#
#     def forward(self, boxes,positive_embeddings):
#         # if masks == None:
#         B, N, _ = boxes.shape
#             # masks = masks.unsqueeze(-1)
#
#             # embedding position (it may includes padding as placeholder)
#         xyxy_embedding = self.fourier_embedder(boxes)  # B*N*4 --> B*N*C
#
#             # learnable null embedding
#             # positive_null = self.null_positive_feature.view(1,1,-1)
#             # xyxy_null =  self.null_position_feature.view(1,1,-1)
#             #
#             # replace padding with learnable null embedding
#             # positive_embeddings = positive_embeddings*masks + (1-masks)*positive_null
#             # xyxy_embedding = xyxy_embedding*masks + (1-masks)*xyxy_null
#
#         objs = self.linears(torch.cat([positive_embeddings, xyxy_embedding], dim=-1)  )
#
#         assert objs.shape == torch.Size([B, N, self.out_dim])
#         return objs
#
#
# if __name__ == "__main__":
#     net = PositionNet(64, 768)
#     xx = torch.randn(2, 2, 4)
#     out = net(xx, masks=None, positive_embeddings=None)
#     print(out.shape)






# class FourierEmbedder():
#     def __init__(self, num_freqs=64, temperature=100):
#
#         self.num_freqs = num_freqs
#         self.temperature = temperature
#         self.freq_bands = temperature ** ( torch.arange(num_freqs) / num_freqs )
#
#     @ torch.no_grad()
#     def __call__(self, x, cat_dim=-1):
#         "x: arbitrary shape of tensor. dim: cat dim"
#         out = []
#         for freq in self.freq_bands:
#             out.append(torch.sin(freq*x ))
#             out.append(torch.cos(freq*x ))
#         return torch.cat(out, cat_dim)
#
# class PositionNet(nn.Module):
#     def __init__(self, in_dim, out_dim, fourier_freqs=8, max_N=10):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.max_N = max_N
#
#         self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
#         self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy
#
#         self.linears = nn.Sequential(
#             nn.Linear(self.in_dim, 512),
#             nn.SiLU(),
#             nn.Linear(512, 512),
#             nn.SiLU(),
#             nn.Linear(512, out_dim),
#         )
#
#         self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
#         self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
#
#     def forward(self, boxes):
#         B, N, _ = boxes.shape
#         xyxy_embedding = self.fourier_embedder(boxes)  # B * max_N * C
#         objs = self.linears(xyxy_embedding)
#         assert objs.shape == torch.Size([B, self.max_N, self.out_dim])
#         return objs

#
class FourierEmbedder():
    def __init__(self, num_freqs=64, temperature=100):
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)

    @torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        "x: arbitrary shape of tensor. dim: cat dim"
        out = []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, cat_dim)
# #
# #
# class PositionNet(nn.Module):
#     def __init__(self, in_dim, out_dim, fourier_freqs=8, num_frames=16):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         # self.max_N = max_N
#         self.num_frames = num_frames
#
#         self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
#         self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy
#         self.linear1=nn.Linear(1024,64)
#         self.linears = nn.Sequential(
#             nn.Linear(self.in_dim+self.position_dim, 512),
#             nn.SiLU(),
#             nn.Linear(512, 512),
#             nn.SiLU(),
#             nn.Linear(512, out_dim),
#         )
#
#         # self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
#         # self.null_position_feature = torch.nn.Parameter(torch.zeros([self.num_frames, self.max_N, self.position_dim]))
#
#     def forward(self, boxes, caption):
#         B, num_frames, N, _ = boxes.shape
#         caption_embeddings = self.linear1(caption)
#         xyxy_embedding = self.fourier_embedder(boxes.view(B * num_frames, N, -1))  # (B * num_frames) * N * C
#         # xyxy_embedding = xyxy_embedding.view(B, num_frames, N, -1)
#         # xyxy_embedding = xyxy_embedding.view(B, num_frames, N, -1)
#         caption_embeddings = caption_embeddings.repeat_interleave(repeats=B, dim=0)
#         objs = self.linears(torch.cat([caption_embeddings, xyxy_embedding], dim=-1))
#         objs = objs.view(B, num_frames, N, -1)
#         # objs = self.linears(xyxy_embedding)  # B * num_frames * max_N * out_dim
#         return objs
 # encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=num_frames, dim=0)



import torch.nn.functional as F


class PositionNet(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy

        # -------------------------------------------------------------- #
        self.linears_text = nn.Sequential(
            nn.Linear(self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        # self.linears_image = nn.Sequential(
        #     nn.Linear(self.in_dim + self.position_dim, 512),
        #     nn.SiLU(),
        #     nn.Linear(512, 512),
        #     nn.SiLU(),
        #     nn.Linear(512, out_dim),
        # )

        # -------------------------------------------------------------- #
        # self.null_text_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        # self.null_image_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        # self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def forward(self, boxes, text_embeddings):
        B, F, N, _ = boxes.shape
        text_embeddings=text_embeddings.repeat_interleave(repeats=F, dim=0)

        boxes= rearrange(boxes, "b f n d -> (b f) n d",f=F)
        # init_video = rearrange(init_video, "b f c h w -> (b f) c h w")
        # masks=rearrange(masks,"b f n -> (b f) n")
        # text_masks = rearrange( text_masks, "b f n -> (b f) n")
        # image_masks = rearrange( image_masks, "b f n -> (b f) n")
        # masks = masks.unsqueeze(-1)  # B*N*1
        # text_masks = text_masks.unsqueeze(-1)  # B*N*1
        # image_masks = image_masks.unsqueeze(-1)  # B*N*1
        # text_embeddings,\
        #     image_embeddings
        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes)  # B*N*4 --> B*N*C

        # learnable null embedding
        # text_null = self.null_text_feature.view(1, 1, -1)  # 1*1*C
        # image_null = self.null_image_feature.view(1, 1, -1)  # 1*1*C
        # xyxy_null = self.null_position_feature.view(1, 1, -1)  # 1*1*C
        #
        # # replace padding with learnable null embedding
        # text_embeddings = text_embeddings * text_masks + (1 - text_masks) * text_null
        # image_embeddings = image_embeddings * image_masks + (1 - image_masks) * image_null
        # xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null

        objs_text = self.linears_text(torch.cat([text_embeddings, xyxy_embedding], dim=-1))
        # objs_image = self.linears_image(torch.cat([image_embeddings, xyxy_embedding], dim=-1))
        # objs = torch.cat([objs_text, objs_image], dim=1)

        # assert objs.shape == torch.Size([B, N * 2, self.out_dim])
        return objs_text



if __name__ == "__main__":
    net = PositionNet(64, 768)
    # Adjust max_N as needed
    xx = torch.randn(2, 16,15, 4)
    caption=torch.randn(2,77,1024)# Assume 5 objects per frame, adjust as needed
    out = net(xx,caption)
    print(out.shape)

# class PositionNet(nn.Module):
#     def __init__(self, in_dim, out_dim, fourier_freqs=8, num_frames=16):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#
#         self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
#         self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy
#         self.linear1=nn.Linear(1024,64)
#         self.linears = nn.Sequential(
#             nn.Linear(self.in_dim + self.position_dim, 512),
#             nn.SiLU(),
#             nn.Linear(512, 512),
#             nn.SiLU(),
#             nn.Linear(512, out_dim),
#         )
#
#     def forward(self, boxes, positive_embeddings):
#         B, N, _ = boxes.shape
#
#         # Embed position information (assuming boxes is (B, 4, 4))
#         xyxy_embedding = self.fourier_embedder(boxes.view(B, -1, 4))  # Reshape boxes
#         positive_embeddings=self.linear1(positive_embeddings)
#         # Concatenate positive_embeddings and position embeddings
#         objs = self.linears(torch.cat([positive_embeddings, xyxy_embedding], dim=-1))
#
#         assert objs.shape == torch.Size([B, N, self.out_dim])
#         return objs
#
# # Example usage:
# import torch
#
# # Create a sample input with your specified shapes
# boxes = torch.randn(4, 4, 4)  # Replace this with your actual input data
# positive_embeddings = torch.randn(16, 77, 1024)  # Replace this with your actual input data
#
# # Initialize the PositionNet
# in_dim = 1024  # Adjust this as needed
# out_dim = 128  # Adjust this as needed
# fourier_freqs = 8
# position_net = PositionNet(in_dim, out_dim, fourier_freqs)
#
# # Forward pass
# output = position_net(boxes, positive_embeddings)
# print(output.shape)
