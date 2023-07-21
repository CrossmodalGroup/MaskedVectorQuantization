import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
import torch.nn.functional as F

# 如果mask token初始化为全0，训练的结果也是全0
class SimpleSampleTokenizer(nn.Module):
    def __init__(self, 
                 topk,
                 input_token_num,
                 input_dim,
                 output_dim,
                 z_dim,
                 patch_size,
                 score_pred_net_mode="2layer-fc",
                 apply_norm_image_features=True,
                 
                 mask_token_initialization=0, 
                 ):
        r"""
        mask_token_initialization: mask token 初始化的方式,默认为全0
        choice: 
            0 : 全0初始化, 最后训练的结果也是全0
            1 : 全1初始化
            "uniform": 正态分布初始化
        """
        super().__init__()
        self.input_token_num = input_token_num
        self.sample_num = int(topk * input_token_num)
        self.topk_num = int(topk * input_token_num)
        self.apply_norm_image_features = apply_norm_image_features
        self.z_dim = z_dim
        self.patch_size = patch_size
        self.hw = int(input_token_num**0.5)
        if score_pred_net_mode == "2layer-fc":
            self.score_pred_net = nn.Sequential(nn.Linear(input_dim, input_dim),
                                                nn.ReLU(),
                                                nn.Linear(input_dim, 1),
                                                nn.Sigmoid())
        else:
            raise ValueError
        
        if self.apply_norm_image_features:
            self.norm_feature = nn.LayerNorm(input_dim, elementwise_affine=False)

        # 更新，2022-4-4，更新mask token的初始化方式
        if mask_token_initialization == 0:
            self.mask_token = nn.Parameter(torch.zeros(1, output_dim, 1), requires_grad=True)
        elif mask_token_initialization == 1:
            self.mask_token = nn.Parameter(torch.ones(1, output_dim, 1), requires_grad=True)
        elif mask_token_initialization == "uniform":
            self.mask_token = nn.Parameter(torch.zeros(1, output_dim, 1), requires_grad=True)
            self.mask_token.data.uniform_(-1.0 / output_dim, 1.0 / output_dim)
        elif mask_token_initialization == "random":
            self.mask_token = nn.Parameter(torch.randn(1, output_dim, 1), requires_grad=True)
        else:
            raise NotImplementedError()
        
        self.mask = torch.from_numpy(np.zeros(self.input_token_num)).float()

        self.decode_dim = output_dim
        self.post_projection = torch.nn.Conv2d(self.z_dim, self.decode_dim, 1)
        self.pre_projection = torch.nn.Linear(input_dim, self.z_dim)
    
    def preforward(self, image_features):

        image_features_avg_values = image_features.mean(1)  # for rebuttal

        image_features = rearrange(image_features, "B C H W -> B (H W) C")
        batch_size, length, channel = image_features.size()
        
        pred_score = self.score_pred_net(image_features).view(batch_size, -1)
        pred_score_clone = pred_score.clone().detach()  # Note: why clone?

        sort_score, sort_order = pred_score_clone.sort(descending=True,dim=1)
        sort_topk = sort_order[:, :self.topk_num]
        sort_topk_remaining = sort_order[:, self.topk_num:]
        ## flatten for gathering
        if self.apply_norm_image_features:
            image_features = self.norm_feature(image_features)

        ## (only) sampled features multiply with score 
        image_features_sampled = image_features.gather(
            1, sort_topk[...,None].expand(-1, -1, channel)) * pred_score.gather(1, sort_topk).unsqueeze(-1)
        image_features = rearrange(self.pre_projection(image_features_sampled), "B N C -> B C N")

        # get mask
        self.mask = self.mask.to(image_features.device)
        for i in range(batch_size):
            if i == 0:
                mask = self.mask.scatter(-1, sort_topk[i], 1.).view(self.hw, self.hw).unsqueeze(0)
            else:
                mask_i = self.mask.scatter(-1, sort_topk[i], 1.).view(self.hw, self.hw).unsqueeze(0)
                mask = torch.cat([mask, mask_i], dim=0)
        squeezed_mask = mask.view(batch_size, -1)  # [batch_size, length]
        mask = F.interpolate(mask.float().unsqueeze(1), scale_factor=self.patch_size, mode="nearest")

        # normalize the score just for better visualization
        normed_score = pred_score_clone.sub(pred_score_clone.min()).div(max(pred_score_clone.max() - pred_score_clone.min(), 1e-5)).unsqueeze(-1)
        normed_score = F.interpolate(rearrange(normed_score, "b (h w) c -> b c h w", h=self.hw, w=self.hw), scale_factor=self.patch_size, mode="nearest")

        return_dict = {
            "sample_h": image_features,
            "sample_index": sort_topk,
            "remain_index": sort_topk_remaining,
            "binary_map": mask,
            "score_map": normed_score,
            "squeezed_mask": squeezed_mask,
            "sort_score": sort_score[:, :self.topk_num],
            "image_features_avg_values": image_features_avg_values,
            "predicted_score": pred_score_clone,
        }
        
        return return_dict

    def postforward(self, sample_h, sample_index, remain_index=None):
        batch_size = sample_h.size(0)
        sample_index = sample_index.cpu().numpy().tolist()
        if remain_index == None:
            remain_index = [[j for j in range(self.input_token_num) if j not in sample_index[i]] for i in range(batch_size)]
        else:
            remain_index = remain_index.cpu().numpy().tolist()
        
        decoder_embeeding = nn.Parameter(torch.zeros((batch_size, self.decode_dim, self.input_token_num))).to(sample_h.device)

        for i in range(batch_size):
            decoder_embeeding[i, :, sample_index[i]] = sample_h[i]
            decoder_embeeding[i, :, remain_index[i]] = self.mask_token

        decoder_embeeding = rearrange(decoder_embeeding, "B C (H W) -> B C H W", H=self.hw, W=self.hw)
        decoder_embeeding = self.post_projection(decoder_embeeding)
        
        return {
            "decoder_embeeding": decoder_embeeding,
        }


if __name__ == "__main__":
    model = SimpleSampleTokenizer(
        topk=0.5,
        input_token_num=256,
        input_dim=256,
        output_dim=256,
        z_dim=256,
        patch_size=16,
        score_pred_net_mode="2layer-fc",
        apply_norm_image_features=True,
    )
    x = torch.randn(10,256,16,16)
    preforward_dict = model.preforward(x)
    postforward_dict = model.postforward(
        preforward_dict["sample_h"],
        preforward_dict["sample_index"],
    )

    # print(preforward_dict["binary_map"].size())
    # print(preforward_dict["score_map"])