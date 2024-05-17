import torch
from lightning import pytorch as pl
import numpy as np
from src.build.from_config import instantiate_from_config
from src.metrics import compute as compute_metrics
import logging
from src.util import rank_zero_log_only
logger = logging.getLogger(__name__)


# To np for metrics
def _torch_imgs_to_np(img: torch.Tensor) -> np.array:
    img = img.detach().cpu()
    img = torch.clamp(img, -1., 1.)
    img = (img + 1.) / 2.
    img = img.permute(0, 2, 3, 1).numpy()
    return (255 * img).astype(np.uint8)


class BaseModel(pl.LightningModule):
    def __init__(self,
                 lossconfig: dict = None,
                 ignore_keys=[],
                 image_key="image",
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.ignore_keys = ignore_keys
        if monitor is not None:
            self.monitor = monitor

        self.loss = instantiate_from_config(lossconfig) if lossconfig is not None else None

        self.save_hyperparameters()

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        rank_zero_log_only(logger, f"Restored from {path}")

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx):  # , optimizer_idx: int = 0):
        x = self.get_input(batch, self.image_key)
        y = self.get_input(batch, 'target')
        bs = x.shape[0]

        xrec = self(x)

        optimizers = self.optimizers()
        optimizer_g = optimizers[0] if isinstance(optimizers, list) or isinstance(optimizers, tuple) else optimizers

        # Train generator
        self.toggle_optimizer(optimizer_g)
        g_loss, log_dict = self.loss(y, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="train")

        # self.log("train/generator_loss", generator_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=bs, sync_dist=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # Train discriminator
        #  Measure discriminator's ability to classify real from generated samples
        if self.discriminator:
            optimizer_d = optimizers[1]
            self.toggle_optimizer(optimizer_d)
            d_loss, log_dict_disc = self.loss(y, xrec, 1, self.global_step, last_layer=self.get_last_layer(),
                                              split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=bs,
                          sync_dist=True)

            self.manual_backward(d_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        y = self.get_input(batch, 'target')
        bs = x.shape[0]

        xrec = self(x)
        g_loss, log_dict = self.loss(y, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="val")

        # To np for metrics
        y_np = _torch_imgs_to_np(y)
        xrec_np = _torch_imgs_to_np(xrec)

        # Test/validation metics
        rec_metrics = {'val/psnr': 0, 'val/ssim': 0, 'val/uiqm': 0, 'val/uciqe': 0}
        for rec, gt in zip(xrec_np, y_np):
            res = compute_metrics(rec, gt)
            for k in ['psnr', 'ssim', 'uiqm', 'uciqe']:
                rec_metrics[f'val/{k}'] += res[k]

        rec_metrics = {k: torch.tensor(v / bs, device=x.device) for k, v in rec_metrics.items()}
        self.log_dict(log_dict | rec_metrics, logger=True, on_step=True, on_epoch=True, batch_size=bs, sync_dist=True)

        return self.log_dict

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        y = self.get_input(batch, 'target')
        x = x.to(self.device)
        y = y.to(self.device)
        xrec = self(x)
        log["inputs"] = x
        log["reconstructions"] = xrec
        log["targets"] = y
        return log
