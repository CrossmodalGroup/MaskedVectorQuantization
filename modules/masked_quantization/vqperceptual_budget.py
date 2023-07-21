import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.losses.lpips import LPIPS
from modules.discriminator.model import NLayerDiscriminator, weights_init
from torch.distributions import Normal, kl_divergence


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def KL_loss(mu, logvar):  # KL-loss with (0,1) gaussian distribution
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def KL_loss_general(mu, logvar, target_mu, target_logvar):
    KLD_element = 1 - target_logvar + logvar - (logvar.exp() + (mu - target_mu).pow(2))/(target_logvar.exp())
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def Gaussian_Wasserstein_Distance(mu, std, target_mu, target_std):
    element_mu = (mu - target_mu).pow(2)
    element_std = (std - target_std).pow(2)
    distance = torch.mean(element_mu + element_std)
    return distance

# def univar_continue_KL_divergence(p_mean, p_logvar, q_mean, q_logvar):
#     # p is target distribution
#     return torch.log(q_logvar / p_logvar) + (p_logvar ** 2 + (p_mean - q_mean) ** 2) / (2 * q_logvar ** 2) - 0.5

class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, target_coding_ratio, target_coding_var=0.01, coding_loss_type="kl", coding_loss_weight=1.0, 
                 disc_start=0, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", disc_weight_max=None):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        assert coding_loss_type in ["mse", "kl", "wasserstein", "wasserstein-var"]
        
        self.coding_loss_type = coding_loss_type  
        self.coding_loss_weight = coding_loss_weight
        self.target_coding_ratio = torch.from_numpy(np.array(target_coding_ratio)).float()
        if coding_loss_type == "kl":
            self.target_coding_logvar = torch.from_numpy(np.array(np.log(target_coding_var))).float()
        elif coding_loss_type == "wasserstein" or coding_loss_type == "wasserstein-var":
            self.target_coding_std = torch.from_numpy(np.array(target_coding_var)).pow(1/2).float()
        
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm, ndf=disc_ndf).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        
        self.disc_weight_max = disc_weight_max
        # recommend: 1000 for [churches, bedrooms], 1 for [ffhq]
        # paper: Unleashing Transformers: Parallel Token Prediction with Discrete Absorbing Diffusion for Fast High-Resolution Image Generation from Vector-Quantized Codes

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, coding_ratio, codebook_loss, inputs, reconstructions, optimizer_idx, global_step, last_layer=None, cond=None, split="train"):
        if self.coding_loss_type == "mse":
            coding_mean = coding_ratio.mean()
            coding_ratio_loss = nn.MSELoss()(coding_mean, self.target_coding_ratio * torch.ones_like(coding_mean).to(coding_mean.device))
        elif self.coding_loss_type == "kl":
            coding_mean = coding_ratio.mean()
            coding_logvar = torch.log(coding_ratio.var())
            coding_ratio_loss = KL_loss_general(coding_mean, coding_logvar, self.target_coding_ratio, self.target_coding_logvar)
        elif self.coding_loss_type == "wasserstein":
            coding_mean = coding_ratio.mean()
            coding_std = coding_ratio.std()
            coding_ratio_loss = Gaussian_Wasserstein_Distance(coding_mean, coding_std, self.target_coding_ratio, self.target_coding_std)
        elif self.coding_loss_type == "wasserstein-var":
            coding_std = coding_ratio.std()
            coding_ratio_loss = (coding_std - self.target_coding_std).pow(2)
        else:
            raise NotImplementedError()

        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            # 增加对disc_weight最大值的限制
            if self.disc_weight_max is not None:
                d_weight.clamp_max_(self.disc_weight_max)
                
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean() + self.coding_loss_weight * coding_ratio_loss

            log = {"{}_total_loss".format(split): loss.clone().detach().mean(),
                   "{}_coding_loss".format(split): coding_ratio_loss.detach().mean(),
                   "{}_quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}_nll_loss".format(split): nll_loss.detach().mean(),
                   "{}_rec_loss".format(split): rec_loss.detach().mean(),
                   "{}_p_loss".format(split): p_loss.detach().mean(),
                   "{}_d_weight".format(split): d_weight.detach(),
                   "{}_disc_factor".format(split): torch.tensor(disc_factor),
                   "{}_g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}_disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}_logits_real".format(split): logits_real.detach().mean(),
                   "{}_logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
