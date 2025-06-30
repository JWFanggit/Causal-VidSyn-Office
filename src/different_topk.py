import torch
from torch import nn
import torch.nn.functional as F
import random

###########################################
############# differential topK ###########
###########################################
# Calculation of differential topK is based on [Top-K](https://arxiv.org/pdf/2104.03059.pdf), thanks



class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 256, sigma: float = 0.05):
        super().__init__()
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k

    def forward(self, x):
        return PerturbedTopKFunction(x, self.k, self.num_samples, self.sigma)


def PerturbedTopKFunction(x, k: int, num_samples: int = 256, sigma: float = 0.005):
    b, d = x.shape
    noise=torch.normal(mean=0.0,std=1.0,size=(b,num_samples,d)).to(dtype=x.dtype,device=x.device)
    perturbed_x = x.unsqueeze(1) + noise * sigma  # (b, nS, d)

    # Top-k selection
    topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
    indices = topk_results.indices  # (b, nS, k)
    sorted_indices = torch.sort(indices, dim=-1).values  # (b, nS, k)
    topk_perturbed_output = F.one_hot(sorted_indices, num_classes=d).float()  # (b, nS, k, d)
    topk_indicators = topk_perturbed_output.mean(dim=1)  # (b, k, d)

    # Non-top-k selection
    all_indices = torch.arange(d, device=x.device).expand(b, num_samples, d)  # (b, nS, d)
    non_topk_mask = torch.ones((b, num_samples, d), device=x.device)  # (b, nS, d)
    non_topk_mask.scatter_(2, sorted_indices, 0)  # Set topk positions to 0
    non_topk_indices = all_indices[non_topk_mask.bool()].view(b, num_samples,
                                                              d - k)  # Get non-topk indices (b, nS, d-k)
    non_topk_perturbed_output = F.one_hot(non_topk_indices, num_classes=d).float()  # (b, nS, d-k, d)
    non_topk_indicators = non_topk_perturbed_output.mean(dim=1)  # (b, d-k, d)


    return topk_indicators, non_topk_indicators










###########################################
############# differential topK ###########
###########################################

class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, embed_dim=512):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2, bias=False),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2, bias=False),
            nn.GELU(),
            # nn.Linear(embed_dim // 2, embed_dim // 4, bias=False),
            # nn.GELU(),
            nn.Linear(embed_dim // 2, 1, bias=False),
            nn.Tanh()
            # nn.Sigmoid()
            # nn.Softmax(dim=-1)
            # nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        '''
        x: shape (bs*n_length, num_tokens, hid_dim)
        '''
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:, :, :]
        global_x = x[:, :1, :]
        # print("global_x.shape: ", global_x.shape)
        x = torch.cat([local_x, global_x.expand(B, N, C)], dim=-1)
        return self.out_conv(x)


class VisualTokenSelection(nn.Module):
    def __init__(self, max_frames, embed_dim=512, topk=3):
        super().__init__()
        self.max_frames = max_frames
        self.score_predictor = PredictorLG(embed_dim=embed_dim)
        self.topk_selector = PerturbedTopK(topk)

    def forward(self, x, training=True):
        '''
        x: input embed, shape is (bs, length*Ntokens, hid_dim)
        use cls token as global representation
        prob = Tanh(MLP(x))
        '''

        B, L, D = x.shape
        N = L // self.max_frames
        x = x.reshape(B, -1, N, D)  # shape here is (bs, max_frames, n_patches, hid_dim)
        x = x.reshape(-1, N, D)  # shape here is (bs*max_frames, n_patches, hid_dim)
        pred_score = self.score_predictor(x).squeeze()  # (bs*max_frames, n_patches)

        spatial_pred_score = pred_score[:, 1:]  # seperate the cls_token (bs*max_frames, n_patches-1)
        topk_indicator = self.topk_selector(spatial_pred_score)  # (bs*max_frames, k, n_patches-1))

        # cls token as cls token
        cls_x_feature = x[:, :1, :]  # cls_token, shape here is (bs*max_frames, 1, hid_dim)
        # # avg pool of all tokens as cls token
        # cls_x_feature = torch.mean(x, dim=1, keepdim=True)

        spatial_x_feature = x[:, 1:, :]  # seperate the cls_token, shape here is (bs*max_frames, n_patches-1, hid_dim)
        selected_patch_feature = torch.einsum("bkl,bld->bkd", topk_indicator, spatial_x_feature)

        output = torch.cat((cls_x_feature, selected_patch_feature),
                           dim=1)  # shape here is (bs*max_frames, topkPatches, hid_dim)
        output = output.reshape(B, self.max_frames, -1, D).reshape(B, -1,
                                                                   D)  # shape here is (B, max_frames*topkPatches, D)

        return output


class STPredictorConv(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, embed_dim=512):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2, bias=False),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2, bias=False),
            nn.GELU(),
            # nn.Linear(embed_dim // 2, embed_dim // 4, bias=False),
            # nn.GELU(),
            nn.Linear(embed_dim // 2, 1, bias=False),
            #  nn.Tanh()
            nn.Softmax(dim=-1)
            # nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, max_frames):
        '''
        x: shape (bs*n_length, num_tokens, hid_dim)
        '''
        x = self.in_conv(x)
        B_frame, N, C = x.size()
        B = B_frame // max_frames
        local_x = x[:, :, :]

        global_x = x[:, :1, :].reshape(B, max_frames, 1, C)  # shape (bs, n_length, cls_tokens, hid_dim)
        global_x = torch.mean(global_x, 1, True).expand(B, max_frames, 1, C).reshape(B_frame, 1, C)
        # print("global_x.shape: ", global_x.shape)

        x = torch.cat([local_x, global_x.expand(B_frame, N, C)], dim=-1)
        return self.out_conv(x)


class STVisualTokenSelection(nn.Module):
    def __init__(self, max_frames):
        super().__init__()
        self.max_frames = max_frames
        self.embed_dim=None
        self.topk = None
        # self.score_predictor = STPredictorConv(self.embed_dim)
        # self.topk_selector = PerturbedTopK(self.topk)


    def forward(self, x,training=True):
        '''
        x: input embed, shape is (bs, length*Ntokens, hid_dim)
        use cls token as global representation
        prob = Tanh(MLP(x))
        '''
        BF, L, D = x.shape
        B=int(BF/self.max_frames)
        self.embed_dim=D
        self.topk=L // 4
        self.score_predictor = STPredictorConv(self.embed_dim).to(device=x.device,dtype=x.dtype)
        self.topk_selector = PerturbedTopK(self.topk).to(device=x.device,dtype=x.dtype)
        pred_score = self.score_predictor(x, self.max_frames).squeeze()  # (bs*max_frames, n_patches)
        topk_indicator,non_topk_indicator= self.topk_selector(pred_score)  # (bs*max_frames, k, n_patches-1))
        topk_indicator=topk_indicator.to(dtype=torch.float16)
        non_topk_indicator=non_topk_indicator.to(dtype=torch.float16)
        selected_patch_feature = torch.einsum("bkl,bld->bkd", topk_indicator, x)
        non_selected_patch_feature = torch.einsum("bkl,bld->bkd", non_topk_indicator, x)
        b = non_selected_patch_feature.shape[0]
        non_num_tkens=non_selected_patch_feature.shape[1]
        n = random.randint(1,non_num_tkens-1)
        # print(n)
        # 为每个样本生成随机掩码
        mask_indices = torch.rand(b, non_num_tkens).topk(n, dim=-1, largest=False).indices
        mask_indices=mask_indices.to(x.device)
        # 创建一个全为 False 的掩码张量
        mask = torch.zeros(b,non_num_tkens, dtype=x.dtype,device=x.device)

        # 将要掩盖的 token 位置设为 True
        mask.scatter_(1, mask_indices, 1)

        # mask_token1= non_selected_patch_feature[mask]
        # non_mask_token1 = non_selected_patch_feature[~mask]
        # print("mask_token1:,",mask_token1.shape)
        # print("non_mask_token1:,",non_mask_token1.shape)
        # 获取掩盖的和未掩盖的 token
        mask_token = non_selected_patch_feature*mask.unsqueeze(-1).reshape(b,non_num_tkens,-1)
        # non_mask_token=non_selected_patch_feature*(1-mask).unsqueeze(-1).reshape(b,non_num_tkens,-1)
        # print(mask_token.shape)
        # mask_token=mask_token
        # print(mask_token.shape)
        # print(non_selected_patch_feature[~mask].shape)
        # non_mask_token = non_selected_patch_feature[~mask].reshape(b, non_num_tkens - n, -1)
        # print(non_mask_token.shape)
        # print("non_selected_patch_feature",non_selected_patch_feature.shape)
        # print("mask_token:,",mask_token.shape)
        # print("non_mask_token:,", non_mask_token.shape)
        # print("mask",mask.shape)


        # feature_dim=non_selected_patch_feature.shape[2]
        # assert mask_token.shape[2]==feature_dim,"mask_token feature dim error"
        # assert non_mask_token.shape[2]==feature_dim,"non_mask_token feature dim error"
        # print("Mask Token Shape:", mask_token.shape)
        # print("Non-Mask Token Shape:", non_mask_token.shape)

        # output1 =selected_patch_feature.reshape(B, self.max_frames, -1, D).reshape(B, -1,
        #                                                            D)
        # output2 =  non_selected_patch_feature.reshape(B, self.max_frames, -1, D).reshape(B, -1,
        #                                                                            D)
        # output2 = mask_token.reshape(B, self.max_frames, -1, D).reshape(B, -1,
        #                                                                                 D)
        # output3 = non_mask_token.reshape(B, self.max_frames, -1, D).reshape(B, -1,
        #
        #                                                                                 D)
        # print("selected_patch_feature:",selected_patch_feature.shape,"mask_token:",mask_token.shape,"non_mask_token:",non_mask_token.shape)
        # mask_token=torch.randn(16,n,1024).to(x.device,x.dtype)
        # non_mask_token=torch.randn(16,192-n,1024).to(x.device,x.dtype)
        return selected_patch_feature,non_selected_patch_feature,mask_token,


class VisualTokenRandomSelection(nn.Module):
    def __init__(self, max_frames, embed_dim=512, topk=3):
        super().__init__()
        self.max_frames = max_frames
        self.topk = topk

    def forward(self, x, training=True):
        '''
        x: input embed, shape is (bs, length*Ntokens, hid_dim)
        use cls token as global representation
        prob = Tanh(MLP(x))
        '''

        B, L, D = x.shape
        N = L // self.max_frames
        x = x.reshape(B, -1, N, D)  # shape here is (bs, max_frames, n_patches, hid_dim)
        x = x.reshape(-1, N, D)  # shape here is (bs*max_frames, n_patches, hid_dim)

        # cls token as cls token
        cls_x_feature = x[:, :1, :]  # cls_token, shape here is (bs*max_frames, 1, hid_dim)
        # # avg pool of all tokens as cls token
        # cls_x_feature = torch.mean(x, dim=1, keepdim=True)

        spatial_x_feature = x[:, 1:, :]  # seperate the cls_token, shape here is (bs*max_frames, n_patches-1, hid_dim)
        patch_len = spatial_x_feature.shape[1]
        selected_indices = torch.randperm(patch_len)[:self.topk].sort()[0]
        selected_patch_feature = spatial_x_feature[:, selected_indices, :]

        output = torch.cat((cls_x_feature, selected_patch_feature),
                           dim=1)  # shape here is (bs*max_frames, topkPatches, hid_dim)
        output = output.reshape(B, self.max_frames, -1, D).reshape(B, -1,
                                                                   D)  # shape here is (B, max_frames*topkPatches, D)

        return output


class TextPredictorLG(nn.Module):
    """ Text to Patch Embedding
    """

    def __init__(self, embed_dim=512):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2, bias=False),
            nn.GELU(),
            # nn.Linear(embed_dim // 2, embed_dim // 4, bias=False),
            # nn.GELU(),
            nn.Linear(embed_dim // 2, 1, bias=False),
            # nn.Tanh()
            nn.Sigmoid()
        )

    def forward(self, x, text):
        '''
        x: shape (bs, num_tokens, hid_dim)
        '''
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:, :, :]
        global_x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].unsqueeze(1)
        x = torch.cat([local_x, global_x.expand(B, N, C)], dim=-1)
        return self.out_conv(x)


class TextTokenSelection(nn.Module):
    def __init__(self, embed_dim=512, topk=1):
        super().__init__()
        self.score_predictor = TextPredictorLG(embed_dim=embed_dim)
        self.topk_selector = PerturbedTopK(topk)

    def forward(self, x, input_ids, attention_mask, training=True):
        '''
        x: input embed, shape is (bs, max_words, hid_dim)
        input_ids: (bs, max_words) token id, cls is the max
        attention_mask: (bs, max_words)
        use cls token as global representation
        prob = Tanh(MLP(x))
        '''
        B, N, D = x.shape
        pred_score = self.score_predictor(x, input_ids).squeeze()  # (bs, max_words)

        attention_mask_new = torch.cat(
            (attention_mask[:, 1:], torch.zeros(B, 1).to(device=attention_mask.device, dtype=attention_mask.dtype)),
            dim=1)
        # print("attention_mask: ", attention_mask[0], "\nattention_mask_new: ", attention_mask_new[0])
        word_pred_score = pred_score * attention_mask_new  # seperate the cls_token (bs, n_token-1)
        # print("word_pred_score: ", word_pred_score[0])
        topk_indicator = self.topk_selector(word_pred_score)  # (bs, k, n_token-1))

        # cls token as cls token
        cls_x_feature = x[torch.arange(x.shape[0]), input_ids.argmax(dim=-1)].unsqueeze(
            1)  # cls_token, shape here is (bs, 1, hid_dim)

        selected_patch_feature = torch.einsum("bkl,bld->bkd", topk_indicator, x)

        output = torch.cat((cls_x_feature, selected_patch_feature), dim=1)  # shape here is (bs, topkPatches, hid_dim)

        return output