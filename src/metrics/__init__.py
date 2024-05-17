from typing import Dict
import math
import numpy as np
from skimage.metrics import structural_similarity
from skimage.transform import resize
import torch
from src.metrics.niqe import calculate_niqe as niqe
from src.metrics.uciqe import uciqe
from src.metrics.uiqm import uiqm


def _scale_images(images, new_shape) -> np.ndarray:
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)


# def _psnr(img1, img2):
#     mse = np.mean((img1 - img2) ** 2)
#     return 10 * math.log10(255.0 ** 2 / mse)


def _psnr(img1, img2):
    # Ensure the images are of the same shape
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Convert the images to float type
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Compute the Mean Squared Error between the two images
    mse = np.mean((img1 - img2) ** 2.)

    # If MSE is zero, the PSNR is infinite
    if mse == 0:
        return float('inf')

    # Compute the maximum pixel value
    max_pixel = 255.0

    # Compute the PSNR
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))

    return psnr


def _non_reference_metrics(img_rgb: np.ndarray, dict_key_suffix:str = '') -> Dict:
    uiqm_, uicm, uism, uiconm = uiqm(img_rgb)
    return {f'niqe{dict_key_suffix}': niqe(img_rgb),
            f'uiqm{dict_key_suffix}': uiqm_,
            f'uism{dict_key_suffix}': uism,
            f'uciqe{dict_key_suffix}': uciqe(img_rgb)
            }


def compute(rec_img_rgb: np.ndarray, gt_img_rgb: np.ndarray = None,
            inception=None, lpips=None, gt_metrics:bool = False) -> Dict:
    metrics = {'psnr': -1, 'ssim': -1, 'uiqm': -1, 'uciqe': -1, 'lpips': -1, 'uiqm_gt': -1, 'uciqe_gt': -1}

    if gt_img_rgb is not None:
        assert np.all(rec_img_rgb.shape == gt_img_rgb.shape)

    # IS computation
    if inception is not None:
        rec_img_rescaled_IS = _scale_images([rec_img_rgb], (299, 299, 3))
        rec_img_rescaled_IS = torch.from_numpy(rec_img_rescaled_IS.transpose((0, 3, 1, 2))).type(torch.uint8).to(inception._device)
        inception.update(rec_img_rescaled_IS)

    if gt_img_rgb is not None:

        if lpips is not None:
            d = list(lpips.parameters())[0].device
            metrics['lpips'] = lpips(torch.from_numpy(gt_img_rgb.transpose(2, 0, 1)).to(d) / 127.5 - 1.0,
                                     torch.from_numpy(rec_img_rgb.transpose(2, 0, 1)).to(d) / 127.5 - 1.0).mean().to('cpu')

        metrics['psnr'] = _psnr(gt_img_rgb, rec_img_rgb)
        metrics['ssim'] = structural_similarity(gt_img_rgb, rec_img_rgb,
                                                data_range=255.0,
                                                win_size=11,
                                                channel_axis=2)
        # Non-reference metrics for GT
        if gt_metrics:
            metrics.update(_non_reference_metrics(gt_img_rgb, dict_key_suffix='_gt'))

    # Non-reference metrics
    metrics.update(_non_reference_metrics(rec_img_rgb))

    return metrics
