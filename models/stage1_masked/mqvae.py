import torch
import pytorch_lightning as pl
from utils.utils import instantiate_from_config
from modules.masked_quantization.tools import build_score_image
from models.stage1.utils import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay

class MaskedVectorQuantizationModel(pl.LightningModule):
    def __init__(self, 
                 encoder_config,
                 decoder_config,
                 masker_config,
                 demasker_config,
                 lossconfig,
                 vqconfig,
                 ckpt_path = None,
                 ignore_keys = [],
                 image_key = "image",
                 monitor = None,
                 warmup_epochs = 0,
                 scheduler_type = "linear-warmup_cosine-decay",
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.masker = instantiate_from_config(masker_config)
        self.demasker = instantiate_from_config(demasker_config)
        self.quantize = instantiate_from_config(vqconfig)
        self.loss = instantiate_from_config(lossconfig)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if monitor is not None:
            self.monitor = monitor
        
        # for warm up
        self.warmup_epochs = warmup_epochs
        self.scheduler_type = scheduler_type
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.masker.parameters()) + 
                                  list(self.demasker.parameters()) ,
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        
        warmup_steps = self.steps_per_epoch * self.warmup_epochs

        if self.scheduler_type == "linear-warmup":
            scheduler_ae = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_ae, Scheduler_LinearWarmup(warmup_steps)), "interval": "step", "frequency": 1,
            }
            scheduler_disc = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup(warmup_steps)), "interval": "step", "frequency": 1,
            }
        elif self.scheduler_type == "linear-warmup_cosine-decay":
            multipler_min = self.min_learning_rate / self.learning_rate
            scheduler_ae = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_ae, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=self.training_steps, multipler_min=multipler_min)), "interval": "step", "frequency": 1,
            }
            scheduler_disc = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=self.training_steps, multipler_min=multipler_min)), "interval": "step", "frequency": 1,
            }
        else:
            raise NotImplementedError()

        return [opt_ae, opt_disc], [scheduler_ae, scheduler_disc]
    
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.size(1) != 3:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    def encode(self, x):
        h = self.encoder(x)
        masker_output = self.masker(h)
        quant = masker_output["sample_features"]
        quant, emb_loss, info = self.quantize(quant)
        return quant, emb_loss, info, masker_output

    def decode(self, sampled_quant, remain_quant, sample_index, remain_index, mask):
        h = self.demasker(sampled_quant, remain_quant, sample_index, remain_index, mask)
        rec = self.decoder(h)
        return rec

    def forward(self, input):
        quant, diff, (_, _, quant_idx), masker_output = self.encode(input)

        sampled_length = masker_output["sampled_length"]
        sampled_quant = quant[:, :, :sampled_length]
        remain_quant = quant[:, :, sampled_length:]
        sample_index = masker_output["sample_index"]
        remain_index = masker_output["remain_index"]
        mask = masker_output["squeezed_mask"]

        rec = self.decode(sampled_quant, remain_quant, sample_index, remain_index, mask)

        return rec, diff
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(codebook_loss=qloss, inputs=x, reconstructions=xrec, optimizer_idx=optimizer_idx, global_step=self.global_step, last_layer=self.get_last_layer(), split="train")

            self.log("train_aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            rec_loss = log_dict_ae["train_rec_loss"]
            self.log("train_rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            del log_dict_ae["train_rec_loss"]
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(codebook_loss=qloss, inputs=x, reconstructions=xrec, optimizer_idx=optimizer_idx, global_step=self.global_step, last_layer=self.get_last_layer(), split="train")
            self.log("train_discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        aeloss, log_dict_ae = self.loss(codebook_loss=qloss, inputs=x, reconstructions=xrec, optimizer_idx=0, global_step=self.global_step, last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(codebook_loss=qloss, inputs=x, reconstructions=xrec, optimizer_idx=1, global_step=self.global_step, last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val_rec_loss"]
        self.log("val_rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        del log_dict_ae["val_rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        
        quant, diff, (_, _, quant_idx), masker_output = self.encode(x)
        sampled_length = masker_output["sampled_length"]
        sampled_quant = quant[:, :, :sampled_length]
        remain_quant = quant[:, :, sampled_length:]
        sample_index = masker_output["sample_index"]
        remain_index = masker_output["remain_index"]
        mask = masker_output["squeezed_mask"]

        xrec = self.decode(sampled_quant, remain_quant, sample_index, remain_index, mask)

        log["inputs"] = x
        log["reconstructions"] = xrec
        log["binary_masked_inputs"] = x * masker_output["binary_map"]
        log["scored_inputs"] = build_score_image(x, masker_output["score_map"], scaler=0.7)

        return log