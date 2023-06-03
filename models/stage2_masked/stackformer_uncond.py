# NOTE: reverseformer,分别预测value和position，先预测value，再根据value预测position
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import time
from einops import rearrange

import os, sys
sys.path.append(os.getcwd())

from utils.utils import PositionAwareSOSProvider, instantiate_from_config
from modules.masked_quantization.permuter import identity_permuter
from models.stage2.utils import learning_rate_schedule, disabled_train

class ReverseStackformer(pl.LightningModule):
    def __init__(self, 
                 transformer_config,
                 first_stage_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 pkeep=1.0,
                 sos_token=None,
                 sos_pos_token=None,
                 loss_position_weight=None,
                 monitor=None,
                 
                 position_value_permuter_config=None,
                 weight_decay=0.01,
                 warmup_epochs=0,
                 
                 height_and_weight=16,
                 add_absolute_position=True,
                 ):
        super().__init__()
        self.sos_token = sos_token
        self.sos_pos_token = sos_pos_token
        self.first_stage_key = first_stage_key
        self.cond_stage_key = first_stage_key
        self.loss_position_weight = loss_position_weight
        self.init_first_stage_from_ckpt(first_stage_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.pkeep = pkeep
        
        print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
              f"Prepending {self.sos_token} as a sos token.")
        self.cond_stage_model = PositionAwareSOSProvider(self.sos_token, self.sos_pos_token)

        if monitor is not None:
            self.monitor = monitor
            
        if position_value_permuter_config is None:
            self.position_value_permuter = identity_permuter()
        else:
            self.position_value_permuter = instantiate_from_config(position_value_permuter_config)
        
        # new hype-parameter
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        
        # additional
        self.height_and_weight = height_and_weight
        self.add_absolute_position = add_absolute_position
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model
    
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.size(1) != 3 and len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(batch, self.first_stage_key)
        c = self.get_input(batch, self.cond_stage_key)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c
    
    @torch.no_grad()
    def encode_to_c(self, c):
        quant_c, _, [_,_,indices,indices_pos] = self.cond_stage_model.encode(c)
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
            indices_pos = indices_pos.view(c.shape[0], -1)
        return quant_c, indices, indices_pos
    
    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info, masker_output = self.first_stage_model.encode(x)
        position_indices = masker_output["sample_index"]
        indices = info[2].view(quant_z.shape[0], -1)
        
        indices, position_indices = self.position_value_permuter(
            indices=indices, position_indices=position_indices, 
            sort_score=masker_output["sort_score"], images=x
        )
        
        return quant_z, indices, position_indices
    
    @torch.no_grad()
    def decode_to_img(self, index, pos_index):
        batch_size, indices_num = index.size(0), index.size(1)
        quant_z = self.first_stage_model.quantize.get_codebook_entry(index)
        quant_z = rearrange(quant_z, "b l c -> b c l")
        
        # calculate mask according to the pos_index
        masks = torch.from_numpy(np.zeros(self.height_and_weight * self.height_and_weight)).float().to(index.device)
        for i in range(batch_size):
            if i == 0:
                mask = masks.scatter(-1, pos_index[i], 1.).view(self.height_and_weight, self.height_and_weight).unsqueeze(0)
            else:
                mask_i = masks.scatter(-1, pos_index[i], 1.).view(self.height_and_weight, self.height_and_weight).unsqueeze(0)
                mask = torch.cat([mask, mask_i], dim=0)
        squeezed_mask = mask.view(batch_size, -1)  # [batch_size, length]
        
        rec = self.first_stage_model.decode(sampled_quant=quant_z, remain_quant=None, sample_index=pos_index, remain_index=None, mask=squeezed_mask)
        return rec
    
    def forward(self, x, c):
        # one step to produce the logits
        _, z_indices, position_indices = self.encode_to_z(x)
        _, c_indices, c_indices_pos = self.encode_to_c(c)
        
        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
            
            r_indices_pos = torch.randint_like(z_indices, self.transformer.config.position_size)
            a_indices_pos = mask*position_indices+(1-mask)*r_indices_pos
        else:
            a_indices = z_indices
            a_indices_pos = position_indices
        
        cz_indices = torch.cat((c_indices, a_indices), dim=1)
        cz_indices_pos = torch.cat((c_indices_pos, a_indices_pos), dim=1)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        # make the prediction
        output = self.transformer(
            cz_indices, cz_indices_pos, z_indices, position_indices
        )

        return output

    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        output = self(x, c)
        return output
    
    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch, batch_idx)
        position_loss = output["position_loss"]
        value_loss = output["value_loss"]

        total_loss = value_loss + self.loss_position_weight * position_loss

        self.log("train_value_loss", value_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train_pos_loss", position_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train_loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch, batch_idx)
        position_loss = output["position_loss"]
        value_loss = output["value_loss"]

        total_loss = value_loss + self.loss_position_weight * position_loss

        self.log("val_value_loss", value_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val_pos_loss", position_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val_loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return total_loss

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    def top_p_logits(self, probs, p):    
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_idx_remove_cond = cum_probs >= p
        
        sorted_idx_remove_cond[..., 1:] = sorted_idx_remove_cond[..., :-1].clone()
        sorted_idx_remove_cond[..., 0] = 0
        
        indices_to_remove = sorted_idx_remove_cond.scatter(-1, sorted_indices, sorted_idx_remove_cond)
        probs = probs.masked_fill(indices_to_remove, 0.0)
        norm_probs = probs / torch.sum(probs, dim=-1, keepdim=True)
        return norm_probs
    
    def avoid_repeat_sampling(self, logits, sampled_position):
        batch_size = logits.size(0)
        out = logits.clone()
        for i in range(batch_size):
            out[i, sampled_position[i]] = -float('Inf')  # 同时避免了采样到start token
        return out
    
    @torch.no_grad()
    def sample(self, x, c, x_pos, c_pos, steps, temperature=1.0, sample=False, top_k=None,
               top_k_pos=None, top_p=None, top_p_pos=None, callback=lambda k: None):
        # print(x.size(), c.size(), "steps:", steps)
        x = torch.cat((c,x),dim=1)
        x_pos = torch.cat((c_pos,x_pos),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        
        for k in range(steps):
            callback(k)
            assert x.size(1) <= block_size # make sure model can see conditioning
            
            # crop context if needed
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
            x_pos_cond = x_pos if x_pos.size(1) <= block_size else x[:, -block_size:]
            
            # print(x_cond.size(), x_pos_cond.size())

            # sample value firstly
            logits_value = self.transformer.sample_value(x_cond, x_pos_cond)
            logits_value = logits_value[:, -1, :] / temperature
            logits_value = self.avoid_repeat_sampling(logits_value, x_cond[:, :1]) # avoid sample start token
            if top_k is not None:
                logits_value = self.top_k_logits(logits_value, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits_value, dim=-1)
            if top_p is not None:
                probs = self.top_p_logits(probs, top_p)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1)

            # sample position secondly
            logits_pos = self.transformer.sample_position(x, x_pos_cond)
            logits_pos = logits_pos[:, -1, :] / temperature
            logits_pos = self.avoid_repeat_sampling(logits_pos, x_pos_cond) # avoid repeat position sampling
            if top_k_pos is not None:
                logits_pos = self.top_k_logits(logits_pos, top_k_pos)
            probs_pos = F.softmax(logits_pos, dim=-1)
            if top_p_pos is not None:
                probs_pos = self.top_p_logits(probs_pos, top_p_pos)
            if sample:
                ix_pos = torch.multinomial(probs_pos, num_samples=1)
            else:
                _, ix_pos = torch.topk(logits_pos, k=1, dim=-1)
            # ix_pos.clamp_max_(self.transformer.config.position_size - 1)
            x_pos = torch.cat((x_pos, ix_pos), dim=1)
        
        x = x[:, c.shape[1]:]
        x_pos = x_pos[:, c_pos.shape[1]:]
        
        return x, x_pos
    
    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, top_k_pos=None, 
                   top_p=None, top_p_pos=None,
                   callback=None, lr_interface=False, **kwargs):
        log = dict()

        N = 4
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)

        quant_z, z_indices, z_indices_pos = self.encode_to_z(x)
        quant_c, c_indices, c_indices_pos = self.encode_to_c(c)

        if self.current_epoch == 0:
            log["inputs"] = x
            # reconstruction
            x_rec = self.decode_to_img(z_indices, z_indices_pos)
            log["reconstructions"] = x_rec

        # sample
        z_start_indices = z_indices[:, :0]
        z_pos_start_indices = z_indices_pos[:, :0]
        
        index_sample, pos_index_sample = self.sample(
                                   z_start_indices, c_indices, z_pos_start_indices, c_indices_pos, 
                                   steps=self.first_stage_model.masker.sample_num,
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   top_k_pos=top_k_pos if top_k_pos is not None else 10,
                                   top_p=top_p if top_p is not None else 0.95,
                                   top_p_pos=top_p_pos if top_p_pos is not None else 0.95,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample, pos_index_sample)


        # det sample
        index_sample, pos_index_sample = self.sample(
                                   z_start_indices, c_indices, z_pos_start_indices, c_indices_pos, 
                                   steps=self.first_stage_model.masker.sample_num,
                                   sample=False,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img(index_sample, pos_index_sample)


        log["samples_nopix"] = x_sample_nopix
        log["samples_det"] = x_sample_det
        return log

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        if self.add_absolute_position:
            no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        
        warmup_steps = self.steps_per_epoch * self.warmup_epochs
        multipler_min = self.min_learning_rate / self.learning_rate
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                learning_rate_schedule(
                    warmup_steps=warmup_steps, max_steps=self.training_steps, multipler_min=multipler_min
                ),
            ),
            "interval": "step",
            "frequency": 1,
        }
        
        return [optimizer], [scheduler]