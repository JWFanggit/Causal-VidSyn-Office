import torch
import numpy
import clip

from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from fvd.pytorch_i3d import InceptionI3d
import torch
from fvd.fvd import get_fvd_logits, frechet_distance
# Original code from https://github.com/piergiaj/pytorch-i3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=1e-5, momentum=0.001)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                        stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


def caculate_fvd(real_recon, fake):
    reals = real_recon
    device = real_recon.device
    i3d = InceptionI3d(400, in_channels=3).to(device)
    # filepath = download(_I3D_PRETRAINED_ID, 'i3d_pretrained_400.pt')
    filepath = r".../rgb_imagenet.pt"
    i3d.load_state_dict(torch.load(filepath, map_location=device))
    i3d.eval()
    fake_embeddings = []
    # batch['video']=torch.randn(1,3,22,224,224)
    # for i in range(0, batch['video'].shape[0], MAX_BATCH):
    #     fake = videogpt.sample(MAX_BATCH, {k: v[i:i+MAX_BATCH] for k, v in batch.items()})
    # fake=torch.randn(2,3,22,224,224)
    fake = fake.permute(0, 2, 3, 4, 1).cpu().numpy()  # BCTHW -> BTHWC
    fake = (fake * 255).astype('uint8')
    fake_embeddings.append(get_fvd_logits(fake, i3d=i3d, device=device))
    fake_embeddings = torch.cat(fake_embeddings)

    # real = batch['video'].to(device)
    real_recon_embeddings = []
    # for i in range(0, batch['video'].shape[0], MAX_BATCH):
    #     real_recon = (videogpt.get_reconstruction(batch['video'][i:i+MAX_BATCH]) + 0.5).clamp(0, 1)
    # real_recon=torch.randn(2,3,22,224,224)
    real_recon = real_recon.permute(0, 2, 3, 4, 1).cpu().numpy()
    real_recon = (real_recon * 255).astype('uint8')
    real_recon_embeddings.append(get_fvd_logits(real_recon, i3d=i3d, device=device))
    real_recon_embeddings = torch.cat(real_recon_embeddings)

    # aaaa=torch.randn(2,3,22,224,224)
    # real = aaaa+ 0.5
    real = reals + 0.5
    real = real.permute(0, 2, 3, 4, 1).cpu().numpy()  # BCTHW -> BTHWC
    real = (real * 255).astype('uint8')
    real_embeddings = get_fvd_logits(real, i3d=i3d, device=device)
    # 要求videos在（-1，1），因此可以直接使用我的数据
    # fake_embeddings = all_gather(fake_embeddings)
    # real_recon_embeddings = all_gather(real_recon_embeddings)
    # real_embeddings = all_gather(real_embeddings)

    assert fake_embeddings.shape[0] == real_recon_embeddings.shape[0] == real_embeddings.shape[0] == 3

    fvd = frechet_distance(fake_embeddings.clone(), real_embeddings)
    fvd_star = frechet_distance(fake_embeddings.clone(), real_recon_embeddings)

    print(f"FVD: {fvd.item()}, FVD*: {fvd_star.item()}")
    return fvd, fvd_star


def caculate_clip(input_data, prompt, model, preprocess):
    # 加载CLIP模型
    device = input_data.device
    # 定义文本
    text = prompt[0]
    # Truncate or handle the text if its length exceeds 77
    max_length = 77
    if len(text) > max_length:
        text = text[:max_length]
    text = clip.tokenize(text, context_length=77).to(device)
    # 假设你的文本是 long_text，指定的最大长度是 max_length
    # max_length = 77
    # if len(text) > max_length:
    #     text = text[:max_length]
    # 如果文本长度超过最大长度，则截取文本到最大长度
    # 使用 long_text 进行后续的处理
    # 定义输入数据
    # input_data = torch.randn(1, 3, 22, 224, 224).to(device)
    # 初始化变量
    total_score = 0
    batch_size = input_data.shape[0]
    # 循环处理每个视频
    for i in range(batch_size):
        # 获取当前视频的帧数
        num_frames = 16
        # print(num_frames)
        # 初始化变量
        video_score = 0
        # 循环处理每一帧
        for j in range(0, num_frames):
            # 将当前帧转换为模型所需的格式
            # print(j)
            image = input_data[:, :, j, :].to(device)
            # print(image.shape)
            # image = preprocess(frame).unsqueeze(0).to(device)
            # 计算相似度得分
            with torch.no_grad():
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                # image_features=image_features.cpu().numpy()
                text_features = model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                # text_features=text_features.cpu().numpy()
                score = (100.0 * image_features @ text_features.T)
                clip_score = torch.diag(score).sum()

            # 获取CLIP Score并累加到视频分数中
            # clip_score = similarity[0, 0].item()
            video_score += clip_score

        # 计算平均CLIP Score并累加到总分数中
        if num_frames > 0:
            avg_clip_score = video_score / num_frames
            total_score += avg_clip_score

    # 计算平均CLIP Score
    if batch_size > 0:
        avg_clip_score = total_score / batch_size
    else:
        avg_clip_score = 0.0

    print("avg_clip_score:", avg_clip_score)
    return avg_clip_score

# Function to compute cosine similarity using PyTorch
def compute_cosine_similarity(tensor1, tensor2):
    return F.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0), dim=1).item()








if __name__ == "__main__":
    #Clip_S
    models, preprocess = clip.load("ViT-B/32", device=device)
    clip_score = caculate_clip(videos, prompts, models, preprocess)
    #FVD
    fvd = caculate_fvd(video1, video2)
    #Temporal_C
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()  # Set the model to evaluation mode
    all_temporal_consistency=[]
    for batch_index in range(videos.shape[0]):
        frames = videos[batch_index]
        # Compute features for all frames in a single batch
        with torch.no_grad():
            features = model.encode_image(frames)
            features = F.normalize(features, p=2, dim=-1)  # Normalize the features
        # Calculate temporal consistency between consecutive frames
        temporal_consistency = []
        for i in range(len(features) - 1):
            similarity = compute_cosine_similarity(features[i], features[i + 1])
            temporal_consistency.append(similarity)
        average_consistency = sum(temporal_consistency) / len(temporal_consistency)
        all_temporal_consistency.append(average_consistency)
        print(average_consistency)
