import argparse
import shutil
import sys

from torch.utils.data import Dataset, DataLoader

sys.path.append(".")

import torch
import PIL
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF

torch.backends.cudnn.benchmark = True
print_color = 'red'

import importlib
from termcolor import colored
from thop import profile, clever_format
from pathlib import Path
import yaml
from tqdm import tqdm
from omegaconf import OmegaConf


def check_is_image_file(f: Path):
    return f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']


def open_image(path):
    return PIL.Image.open(path)


def preprocess(img: Image.Image, target_image_size: int = 256):
    img = TF.resize(img, (target_image_size, target_image_size), interpolation=PIL.Image.LANCZOS)
    img = T.ToTensor()(img)
    if img.size()[1] < 3:
        img = torch.cat([img, img, img], dim=1)
    return 2. * img - 1.


class DatasetFromFolder(Dataset):
    def __init__(self, image_dir, target_image_size: int = 256):
        super(DatasetFromFolder, self).__init__()
        self.target_image_size = target_image_size
        self.list_images = list(filter(check_is_image_file, Path(image_dir).glob('*.*')))

    def __getitem__(self, index):
        img_path = self.list_images[index]
        img = open_image(img_path)
        img = preprocess(img, self.target_image_size)
        return img, str(img_path)

    def __len__(self):
        return len(self.list_images)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(colored(yaml.dump(OmegaConf.to_container(config))), print_color)
    return config


def load_cevae(config, ckpt_path=None):
    cls = get_obj_from_str(config['model']['target'])
    for delk in ['ckpt_path', 'lossconfig', 'optimizer']:
        if delk in config.model.params:
            del config.model.params[delk]
    model = cls(**config.model.params)
    if ckpt_path is not None:
        print(colored(f'Loading from checkpoint path {ckpt_path}', print_color))
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


@torch.inference_mode
def reconstruction_batch(x, model, size: int = 320):
    x_rec = model(x)
    return x_rec


# chunkify the list of images
def chunk_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def run(opt):
    # Device
    device = torch.device(opt.device)

    # If config is not set, load it using the checkpoint folder as a reference
    if opt.config is None:
        opt.config = list((Path(opt.checkpoint).parent.parent / 'configs').glob('*-project.yaml'))[0]

    # Initialize generator
    config = load_config(opt.config, display=False)
    model = load_cevae(config, ckpt_path=opt.checkpoint).to(device)
    print(colored(f'Loaded model with config {opt.config} from {opt.checkpoint} on {device}', print_color))

    print(colored(f'Working on {opt.data_path}', print_color))
    output_folder = Path(f"{opt.output_path}")
    if output_folder.exists():
        print(colored(f'Deleting pre-exising output folder: {opt.output_path}', print_color))
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True)

    # image_paths_chunks = chunk_list(list_images, opt.batch_size)
    test_dset = DatasetFromFolder(opt.data_path)
    print(colored(f'Input folder: {opt.data_path} has {len(test_dset)} image files to be processed', print_color))

    if opt.count_flops_params:
        input = torch.randn(1, 3, 256, 256).to(device)
        macs, params = profile(model, inputs=(input,))
        macs, params = clever_format([macs, params], "%.3f")
        print(colored(f'Model MACS: {macs}', print_color))
        print(colored(f'Model Params: {params}', print_color))

    testing_data_loader = DataLoader(dataset=test_dset, batch_size=opt.batch_size, shuffle=False, num_workers=8)
    print(colored(f'Reconstructing with batch size {opt.batch_size}', print_color))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []

    for iteration_test, (imgs, img_paths) in tqdm(enumerate(testing_data_loader), total=len(testing_data_loader)):
        imgs = imgs.to(model.device)
        start.record()
        x_rec = reconstruction_batch(imgs, model)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

        rec_imgs = [custom_to_pil(x) for x in x_rec]

        for ii, img in enumerate(rec_imgs):
            img_path = Path(img_paths[ii])
            img.save(str(output_folder / img_path.name))

    # Compute average time
    average_time = sum(times) / len(times)
    print(colored(f"Average elapsed time: {average_time} ms", print_color))

    flops, params = profile(model, inputs=(imgs,), verbose=False)
    print(colored(f"GFLOPs: {flops / 1e9:.2f}", print_color))


if __name__ == '__main__':
    # Testing settings
    parser = argparse.ArgumentParser(description='Underwater VC-GAN')
    parser.add_argument('--checkpoint', '--c', required=True, help='CEVAE checkpoint path')
    parser.add_argument('--data-path', '--d', required=True, help='Data folder path')
    parser.add_argument('--output-path', '--o', default='./output', required=False, help='Output folder path (default: ./output)')
    parser.add_argument('--batch-size', '--bs', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--count-flops-params', action='store_true', help='Count number of flops and parameter of model (default: false)')
    parser.add_argument('--config', '--cfg', default=None, required=False, help='Optional model config path (default None -> assume an existing *-project.yaml config in checkpoint patth/../configs)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for model execution (default: cuda:0)')

    opt = parser.parse_args()

    print(colored("Running with pars:", print_color))
    print(colored(opt, print_color))

    # run with opt
    run(opt)
