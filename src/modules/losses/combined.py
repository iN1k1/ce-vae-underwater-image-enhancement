import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.image import image_gradients
from src.modules.losses.lpips import LPIPS
from src.modules.discriminator.patchgan import PatchGANDiscriminator, weights_init
from pytorch_msssim import ms_ssim
import logging
from src.util import rank_zero_log_only

logger = logging.getLogger(__name__)


def gdl_gradient_loss(gen_frames: torch.Tensor, gt_frames: torch.Tensor, alpha: int = 1) -> torch.Tensor:
    gen_dx, gen_dy = image_gradients(gen_frames)
    gt_dx, gt_dy = image_gradients(gt_frames)

    grad_diff_x = torch.abs(gt_dx - gen_dx)
    grad_diff_y = torch.abs(gt_dy - gen_dy)

    # condense into one tensor and avg
    output = torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)
    return output


def color_loss_deep_sesr(inputs: torch.Tensor, reconstructions: torch.Tensor) -> torch.Tensor:
    input_r, input_g, input_b = torch.split(inputs, 1, dim=1)
    rec_r, rec_g, rec_b = torch.split(reconstructions, 1, dim=1)
    delta_r = torch.abs(rec_r - input_r)
    delta_g = torch.abs(rec_g - input_g)
    delta_b = torch.abs(rec_b - input_b)

    d = (4 * (delta_r - delta_g) ** 2) + ((delta_r + delta_g - (2 * delta_b)) ** 2)

    # L2 norm of d over the batch
    return torch.mean(torch.norm(d, p=2, dim=(1, 2, 3)))


class ReconstructionLoss(nn.Module):
    def __init__(self,
                 pixelloss_weight: float = 1.0,
                 perceptual_weight: float = 1.0,
                 gdl_loss_weight: float = 1.0,
                 color_loss_weight: float = 0.0,
                 ssim_loss_weight: float = 0.0):
        super().__init__()
        self._pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self._perceptual_weight = perceptual_weight
        self._reconstruction_loss = nn.L1Loss()
        self._gdl_loss_weight = gdl_loss_weight
        self._color_loss_weight = color_loss_weight
        self._ssim_loss_weight = ssim_loss_weight

    # def forward(self, inputs, reconstructions, split="train"):
    def forward(self, gt, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        rec_loss = self._reconstruction_loss(gt, reconstructions)

        p_loss = torch.tensor([0.0], device=gt.device)
        if self._perceptual_weight > 0:
            p_loss = self.perceptual_loss(reconstructions, gt.detach()).mean()

        gdl_loss = gdl_gradient_loss(reconstructions, gt.detach())

        color_loss = color_loss_deep_sesr(gt, reconstructions)

        to_01 = lambda img: torch.clip((img + 1) / 2, 0., 1.)
        ssim_loss = 1 - ms_ssim(to_01(gt), to_01(reconstructions), data_range=1, size_average=True)  # return a scalar

        # reconstruction loss + perceptual loss + gdl_loss
        loss = (self._pixel_weight * rec_loss) \
               + (self._perceptual_weight * p_loss) \
               + (self._gdl_loss_weight * gdl_loss) \
               + (self._color_loss_weight * color_loss) \
               + (self._ssim_loss_weight * ssim_loss)

        log = {"{}/total_loss".format(split): loss.clone().detach(),
               "{}/rec_loss".format(split): rec_loss.detach(),
               "{}/gdl_loss".format(split): gdl_loss.detach(),
               "{}/p_loss".format(split): p_loss.detach(),
               "{}/color_loss".format(split): color_loss.detach(),
               "{}/ssim_loss".format(split): ssim_loss.detach(),
               }
        return loss, log


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


# loss GAN
class ReconstructionLossWithDiscriminator(nn.Module):
    def __init__(self, disc_enabled: bool = True,
                 disc_start: int = 10000,
                 codebook_weight=1.0,
                 disc_num_layers: int = 3, disc_in_channels: int = 3, disc_factor: float = 1.0,
                 disc_weight: float = 1.0,
                 use_actnorm: bool = False, disc_conditional: bool = False,
                 disc_ndf: int = 64, disc_loss: str = "hinge",
                 pixelloss_weight: float = 1.0,
                 perceptual_weight: float = 1.0,
                 gdl_loss_weight: float = 1.0,
                 color_loss_weight: float = 0.0,
                 ssim_loss_weight: float = 0.0):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight

        self.discriminator = PatchGANDiscriminator(input_nc=disc_in_channels,
                                                   n_layers=disc_num_layers,
                                                   use_actnorm=use_actnorm,
                                                   ndf=disc_ndf
                                                   ).apply(weights_init) if disc_enabled else None
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        rank_zero_log_only(logger, f"ReconstructionLossWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight  # discriminator factor
        self.disc_conditional = disc_conditional

        self.reconstruction_loss = ReconstructionLoss(pixelloss_weight=pixelloss_weight,
                                                      perceptual_weight=perceptual_weight,
                                                      gdl_loss_weight=gdl_loss_weight,
                                                      color_loss_weight=color_loss_weight,
                                                      ssim_loss_weight=ssim_loss_weight)

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

    def forward(self, gt, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):

        # Reconstruction loss
        rec_loss, log_dict = self.reconstruction_loss(gt, reconstructions, optimizer_idx, global_step, last_layer, cond,
                                                      split)

        # now the GAN part
        if self.discriminator is None:
            return rec_loss, log_dict
        else:
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
                    d_weight = self.calculate_adaptive_weight(rec_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0, device=gt.device)

                # Gan loss
                disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
                gan_loss = d_weight * disc_factor * g_loss

                # Total loss
                loss = rec_loss + gan_loss

                log_dict.update({"{}/total_loss".format(split): loss.clone().detach().mean(),
                                 "{}/d_weight".format(split): d_weight.detach(),
                                 "{}/disc_factor".format(split): torch.tensor(disc_factor, device=gt.device),
                                 "{}/g_loss".format(split): g_loss.detach().mean(),
                                 })
                return loss, log_dict

            if optimizer_idx == 1:
                # second pass for discriminator update
                if cond is None:
                    logits_real = self.discriminator(gt.detach())
                    logits_fake = self.discriminator(reconstructions.detach())
                else:
                    logits_real = self.discriminator(torch.cat((gt.detach(), cond), dim=1))
                    logits_fake = self.discriminator(torch.cat((reconstructions.detach(), cond), dim=1))

                disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
                d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

                log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                       "{}/logits_real".format(split): logits_real.detach().mean(),
                       "{}/logits_fake".format(split): logits_fake.detach().mean()
                       }
                return d_loss, log
