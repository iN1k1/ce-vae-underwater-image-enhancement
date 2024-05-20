import argparse, os, sys, datetime, importlib
import copy
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
import wandb
import lightning as L
from lightning import pytorch as pl
from lightning import seed_everything
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import LightningArgumentParser
from src.build.from_config import instantiate_from_config
import logging
from src.util import rank_zero_log_only

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("debug.log"),
                            logging.StreamHandler()
                        ])


def get_parser(**parser_kwargs):
    parser = LightningArgumentParser(**parser_kwargs, add_help=False, parse_as_dict=True)
    parser.add_lightning_class_args(L.Trainer, nested_key='trainer')

    parser.add_argument("-n", "--name", type=str, const=True, default="", nargs="?", help="postfix for logdir (default: empty)")
    parser.add_argument("-r", "--resume", type=str, const=True, default="", nargs="?",
                        help="resume from logdir or checkpoint in logdir", )
    parser.add_argument("-cfg", "--config", type=str, help="path to config")
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument("-s", "--seed", type=int, default=23, help="seed for seed_everything (default: 23)")
    parser.add_argument("-f", "--postfix", type=str, default="", help="post-postfix for default name (default: empty)")

    return parser


def parse_args():
    parser = get_parser()
    opt = parser.parse_args()
    return argparse.Namespace(**opt)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.logger = None

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            rank_zero_log_only(logger, "Project config")
            rank_zero_log_only(logger, OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            # Log loss config details to logger
            self.logger.log_hyperparams(self.config["model"]["params"]["lossconfig"]["params"])

            # Log data config details to logger
            self.logger.log_hyperparams(self.config["data"]["params"])

            rank_zero_log_only(logger, "Lightning config")
            rank_zero_log_only(logger, OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

            # Log lightning config details to logger
            self.logger.log_hyperparams(self.lightning_config["trainer"])


        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, train_batch_frequency, val_batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.train_batch_freq = train_batch_frequency
        self.val_batch_freq = val_batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            # pl.loggers.TestTubeLogger: self._testtube,
        }
        self.train_log_steps = [2 ** n for n in range(int(np.log2(self.train_batch_freq)) + 1)]
        self.val_log_steps = [2 ** n for n in range(int(np.log2(self.val_batch_freq)) + 1)]
        if not increase_log_steps:
            self.train_log_steps = [self.train_batch_freq]
            self.val_log_steps = [self.val_batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx, grid_rows: int = 5):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=grid_rows)

            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{:06}_e-{:06}_b-{:06}-{}.png".format(
                global_step,
                current_epoch,
                batch_idx,
                k)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx, split) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx,
                           grid_rows=self.max_images)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx, split):
        batch_freq = self.train_batch_freq if split == 'train' else self.val_batch_freq
        log_steps = self.train_log_steps if split == 'train' else self.val_log_steps
        if (batch_idx % batch_freq) == 0 or (batch_idx in log_steps):
            try:
                log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_img(pl_module, batch, batch_idx, split="val")


if __name__ == "__main__":

    setup_logging()

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    # Parse args
    opt = parse_args()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths) - paths[::-1].index("vcgan_logs") + 1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        # base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        # opt.config = base_configs +log opt.config
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("vcgan_logs") + 1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.config:
            cfg_fname = os.path.split(opt.config)[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join("./training_logs", nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    # init and save configs
    config = OmegaConf.load(opt.config)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = OmegaConf.merge(*[OmegaConf.create(vars(opt)).trainer,
                                       lightning_config.get("trainer", OmegaConf.create())])
    # default to ddp
    trainer_config["strategy"] = trainer_config.get("strategy", "ddp")
    if trainer_config['accelerator'] != "gpu":
        del trainer_config["strategy"]
        cpu = True
    else:
        gpuinfo = trainer_config['devices']
        rank_zero_log_only(logger, f"Running on {gpuinfo} GPUs")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    model = instantiate_from_config(config.model)

    # trainer and callbacks
    trainer_kwargs = dict()

    # default logger configs
    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": nowname,
                "save_dir": logdir,
                "id": nowname,
                "project": "cevae"
            }
        }
    }
    default_logger_cfg = default_logger_cfgs["wandb"]
    logger_cfg = OmegaConf.create()

    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "main.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": copy.deepcopy(config),
                "lightning_config": lightning_config
            }
        },
        "image_logger": {
            "target": "main.ImageLogger",
            "params": {
                "train_batch_frequency": 250,
                "val_batch_frequency": 5,
                "max_images": 5,
                "clamp": True
            }
        },
        "learning_rate_logger": {
            "target": "lightning.pytorch.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
            }
        },
        "checkpoint_callback": {
            "target": "lightning.pytorch.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:04}",
                "verbose": True,
                "save_last": True,
                "monitor": "val/psnr",
                "save_top_k": 3,
                "mode": "max"
            }
        }
    }

    if hasattr(model, "monitor"):
        rank_zero_log_only(logger, f"Monitoring {model.monitor} as checkpoint metric.")
        default_callbacks_cfg["checkpoint_callback"]["params"]["monitor"] = model.monitor
        default_callbacks_cfg["checkpoint_callback"]["params"]["save_top_k"] = 3

    callbacks_cfg = OmegaConf.create(default_callbacks_cfg)
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    trainer_kwargs["callbacks"][0].logger = trainer_kwargs["logger"]

    trainer = L.Trainer(**(vars(trainer_opt) | trainer_kwargs))

    # data
    data = instantiate_from_config(OmegaConf.to_object(config.data))
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()

    # configure learning rate
    bs, base_lr = config.data.params.train_batch_size, config.model.params.optimizer.base_learning_rate
    if not cpu:
        ngpu = int(trainer_config['devices'])
    else:
        ngpu = 1
    accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches or 1
    rank_zero_log_only(logger, f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    model.optimizer_config['learning_rate'] = accumulate_grad_batches * ngpu * bs * base_lr
    rank_zero_log_only(logger,
                       "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                           model.optimizer_config['learning_rate'], accumulate_grad_batches, ngpu, bs, base_lr))


    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            rank_zero_log_only(logger, "Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)


    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb
            pudb.set_trace()


    import signal

    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    # run
    try:
        trainer.fit(model, data)
    except Exception:
        melk()
        raise
