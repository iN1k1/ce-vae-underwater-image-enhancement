import torch
import torch.nn as nn
from src.models.base import BaseModel
from src.modules.autoencoder import Encoder, Decoder
from src.modules.capsules import PrimaryCaps, DigitCaps


class CEVAE(BaseModel):
    def __init__(self,
                 ddconfig: dict,
                 lossconfig: dict = None,
                 embed_dim: int = 256,
                 optimizer: dict = None,
                 ckpt_path: str = None,
                 ignore_keys: list = [],
                 image_key: str = "image",
                 monitor: str = None,
                 discriminator: bool = True
                 ):
        super(CEVAE, self).__init__(lossconfig=lossconfig,
                                    image_key=image_key,
                                    ignore_keys=ignore_keys, monitor=monitor)

        self._use_capsules = ddconfig.get("use_capsules", True)
        self.encoder = Encoder(**ddconfig)
        self.primary = PrimaryCaps() if self._use_capsules else nn.Sequential()
        self.digitcaps = DigitCaps(ddconfig["z_channels"]) if self._use_capsules else nn.Sequential()
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim,
                                          1) if self._use_capsules else nn.Sequential()
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"],
                                               1) if self._use_capsules else nn.Sequential()
        self.discriminator = discriminator

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        # Parameters that we need to specify in order to initialize our model
        self.optimizer_config = {'beta1': 0.5, 'beta2': 0.9, 'learning_rate': 1e-4}
        if optimizer:
            self.optimizer_config |= dict(optimizer)
        self.automatic_optimization = False

    def encode(self, x):
        enc = self.encoder(x)
        if self._use_capsules:
            x = self.primary(enc)
            _, x = self.digitcaps(x)
            x = self.quant_conv(x)
        return enc, x

    def decode(self, enc, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(enc, quant)
        return dec

    def forward(self, x):
        enc, quant = self.encode(x)
        dec = self.decode(enc, quant)
        return dec

    def configure_optimizers(self):
        lr = self.optimizer_config['learning_rate']
        params_to_optimize = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(
            self.post_quant_conv.parameters())

        if self._use_capsules:
            params_to_optimize += list(self.primary.parameters()) + list(self.digitcaps.parameters()) + list(
                self.quant_conv.parameters())

        optimizers = [
            torch.optim.Adam(params_to_optimize,
                             lr=lr, betas=(self.optimizer_config['beta1'], self.optimizer_config['beta2']))
        ]

        if self.discriminator:
            optimizers.append(
                torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr,
                                 betas=(self.optimizer_config['beta1'], self.optimizer_config['beta2']))
            )
        schedulers = []
        return optimizers, schedulers

    def get_last_layer(self):
        return self.decoder.conv_out.weight
