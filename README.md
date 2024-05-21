# Capsule Enhanced Variational AutoEncoder for Underwater Image Reconstruction

![Teaser Image](assets/teaser.png)

This repository contains the code accompanying the scientific paper "Capsule Enhanced Variational AutoEncoder for
Underwater Image Reconstruction" by Rita Pucci and Niki Martinel. The paper is available on [arXiv](link).

## Description

Underwater image analysis is crucial for marine monitoring but presents significant challenges such as degraded visual
quality and limitations in capturing high-resolution images due to hardware constraints. Traditional methods struggle to
address these issues effectively.

We introduce a novel architecture that jointly tackles both problems by drawing inspiration from the discrete features
quantization approach of Vector Quantized Variational Autoencoder (VQ-VAE). Our model combines an encoding network,
which compresses the input into a latent representation, with two independent decoding networks. These networks enhance
and reconstruct images using only the latent representation, with one decoder focusing on spatial information and the
other leveraging capsules to capture information about entities in the image.

This approach not only improves the visual quality of underwater images but also overcomes the differentiability issues
of VQ-VAE, allowing for end-to-end training without special optimization tricks. Our capsule layers perform feature
quantization in a fully differentiable manner.

Through extensive quantitative and qualitative evaluations on six benchmark datasets, our method demonstrates superior
performance compared to existing methods, achieving about +1.4dB gain on the challenging LSUI Test-L400 dataset.
Additionally, our approach significantly reduces the amount of space needed for data storage, making it three times more
efficient.

## TL;DR;

![Pipeline Image](assets/cevae-pipeline.png)

## Installation

To set up the environment and install the required packages, follow these steps:

1. Create a Python 3.10 virtual environment:
   ```sh
   conda create -n cevae python=3.11
   conda activate cevae
   ```

2. Install PyTorch (v.2.2) and other relevant dependencies:
   ```sh
   conda install pytorch==2.2.0 torchvision==0.17.0 pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model, run:

```sh
python main.py --config [path of config]
```

#### LSUI training config example

Examples of config files can be found in the `configs` folder.
To train our model with the default configuration on the LSUI dataset, follow these step:
1. Generate the txt training and validation files for the LSUI dataset. Assuming your local system has the following structure 
   ```
   /home/user/data/LSUI
   ├── train
   │   ├── GT
   │   └── input
   └── val
       ├── GT
       └── input
   ```
   you should first run:
      ```sh
      bash scripts/generate_dataset_txt.sh /home/user/data/LSUI/ 
      ```
   to generate the training and validation paired input text files `(LSUI_train_input.txt, LSUI_train_target.txt)` and `(LSUI_val_input.txt, LSUI_val_target.txt)`.    
   These are the "default" files that we have in the LSUI training configs `./configs/cevae_*_lsui.yaml`

3. Train the CE-VAE model without the
discriminator.
   Start by downloading the ImageNet-pretrained model from [here](https://uniudamce-my.sharepoint.com/:u:/g/personal/niki_martinel_uniud_it/ESe3q_vE9EtJur7Ioda8UMoBS-P8jCZdlXbLO3gp-XUKQg?e=RBpa8x) and save it into the `data` folder. 
   Then exectute
   ```sh
   python main.py --config configs/cevae_E2E_lsui.yaml
   ```
   **Training logs, containing checkpoints, and samples of generated images are in `./training_logs`**

4. Once you have the checkpoints for the model trained without the discriminator, you can need to edit the `ckpt_path` entry in the `configs/cevae_GAN_lsui.yaml` to point it to your local pth file and then run the following command to finetune the model with the discriminator:
   ```sh
   python main.py --config configs/cevae_GAN_lsui.yaml
   ```

### Underwater Image Enanchement

To evaluate the model on a folder, run:

```sh
python test.py --config [path of config] --checkpoint [path of checkpoint] --data-path [folder path containing images to enhance] --output-path [path of output folder where enhanced images will be saved]
```

## Results

### Quantitative comparison on the LSUI-L400 dataset

| Method                                               | PSNR ↑    | SSIM ↑   | LPIPS ↓  |
|------------------------------------------------------|-----------|----------|----------|
| RGHS                           | 18.44     | 0.80     | 0.31     |
| UDCP                            | 13.24     | 0.56     | 0.39     |
| UIBLA                          | 17.75     | 0.72     | 0.36     |
| UGAN                      | 19.40     | 0.77     | 0.37     |
| Cluie-Net                      | 18.71     | 0.78     | 0.33     |
| TWIN                           | 19.84     | 0.79     | 0.33     |
| UShape-Transformer | 23.02     | 0.82     | 0.29     |
| Spectroformer         | 20.09     | 0.79     | 0.32     |
| **CE-VAE (Our Method)**                              | **24.49** | **0.84** | **0.26** |

### Qualitative comparison on the LSUI-L400 dataset

![LSUIResults Image](assets/lsui_l400_psnr.png)

## License

This project is licensed under the terms specified in the `license.txt` file.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{pucci2024capsule,
  title={Capsule Enhanced Variational AutoEncoder for Underwater Image Reconstruction},
  author={Pucci, Rita and Martinel, Niki},
  journal={arXiv preprint arXiv:[arxiv number]},
  year={2024}
}
```

