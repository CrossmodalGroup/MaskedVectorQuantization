import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
import torch.nn.functional as F

# first predict scores, then norm features
class VanillaMasker(nn.Module):
    def __init__(self, 
                 topk_ratio,
                 input_token_num,
                 input_dim,
                 patch_size,
                 score_pred_net_mode = "2layer-fc",
                 codebook_dim = 32,
                 ):
        super().__init__()
        self.input_token_num = input_token_num
        self.sample_num = int(topk_ratio * input_token_num)
        self.unsampled_num = input_token_num - self.sample_num
        self.topk_num = int(topk_ratio * input_token_num)
        self.patch_size = patch_size
        self.hw = int(input_token_num**0.5)
        if score_pred_net_mode == "2layer-fc":
            self.score_pred_net = nn.Sequential(nn.Linear(input_dim, input_dim),
                                                nn.ReLU(),
                                                nn.Linear(input_dim, 1),
                                                nn.Sigmoid())
        else:
            raise ValueError
        self.norm_feature = nn.LayerNorm(input_dim, elementwise_affine=False)

        self.mask = torch.from_numpy(np.zeros(self.input_token_num)).float()
        self.pre_projection = torch.nn.Linear(input_dim, codebook_dim, bias=False)
        
    
    def forward(self, image_features):
        image_features = rearrange(image_features, "B C H W -> B (H W) C")
        batch_size, length, channel = image_features.size()
        
        pred_score = self.score_pred_net(image_features).view(batch_size, -1)
        pred_score_clone = pred_score.clone().detach()

        sort_score, sort_order = pred_score_clone.sort(descending=True,dim=1)
        sort_topk = sort_order[:, :self.topk_num]
        sort_topk_remain = sort_order[:, self.topk_num:]
        ## flatten for gathering
        image_features = self.norm_feature(image_features)

        ## (only) sampled features multiply with score 
        image_features_sampled = image_features.gather(1, sort_topk[...,None].expand(-1, -1, channel)) * pred_score.gather(1, sort_topk).unsqueeze(-1)
        image_features_sampled = rearrange(self.pre_projection(image_features_sampled), "B N C -> B C N")

        # get mask
        self.mask = self.mask.to(image_features_sampled.device)
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
            "sample_features": image_features_sampled,
            "remain_features": None, 
            "sample_index": sort_topk,
            "remain_index": sort_topk_remain,
            "binary_map": mask,
            "score_map": normed_score,
            "squeezed_mask": squeezed_mask,
            "sort_score": sort_score[:, :self.topk_num],
            "sampled_length": image_features_sampled.size(-1),
        }
        
        return return_dict