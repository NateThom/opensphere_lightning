import argparse
from glob import glob
import time
import yaml
import os.path as osp
import os

## Local
from utils.build_helper import build_from_args

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.distributed as dist

## Torchvision
import torchvision
from torchvision.datasets import STL10
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# class Backbone(torch.nn.Module):
#     def __init__(self, hidden_dim=128):
#         super().__init__()
#         self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
#         self.l2 = torch.nn.Linear(hidden_dim, 10)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = torch.relu(self.l1(x))
#         x = torch.relu(self.l2(x))
#         return x


# class LitClassifier(pl.LightningModule):
#     def __init__(self, backbone, learning_rate=1e-3):
#         super().__init__()
#         self.save_hyperparameters()
#         self.backbone = backbone

#     def forward(self, x):
#         # use forward for inference/predictions
#         embedding = self.backbone(x)
#         return embedding

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.backbone(x)
#         loss = F.cross_entropy(y_hat, y)
#         self.log('train_loss', loss, on_epoch=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.backbone(x)
#         loss = F.cross_entropy(y_hat, y)
#         self.log('valid_loss', loss, on_step=True)

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.backbone(x)
#         loss = F.cross_entropy(y_hat, y)
#         self.log('test_loss', loss)

#     def configure_optimizers(self):
#         # self.hparams available because we called self.save_hyperparameters()
#         return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         parser = ArgumentParser(parents=[parent_parser], add_help=False)
#         parser.add_argument('--learning_rate', type=float, default=0.0001)
#         return parser

def get_config_from_args():
    # get arguments
    parser = argparse.ArgumentParser(
            description='A PyTorch framework for deep metric learning.')
    parser.add_argument('--cfg_path', default='',
            help='path of training config')
    parser.add_argument('--proj_dir', default='./project',
            help='the dir to save logs and models')
    args = parser.parse_args()

    # check if resume from a project
    ckpt_paths = glob(osp.join(args.proj_dir, 'checkpoint/ckpt_*.pth'))
    if len(ckpt_paths) > 0:
        # if resume, project directory should be provided
        cfg_path = osp.join(args.proj_dir, 'config.yml')
    else:
        # if not, path to config file should be provided
        cfg_path = args.cfg_path

    # get config
    with open(cfg_path, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)

    # update config with args
    trainer_cfg = config['trainer']
    trainer_cfg['proj_dir'] = args.proj_dir

    # time to start
    # start_time = time.strptime(args.start_time, '%Y%m%d_%H%M%S')
    # while time.localtime() < start_time:
    #     print(args.start_time)
    #     time.sleep(666)
    # print('start...')

    return config

def cli_main():
    # ------------
    # args
    # ------------
    config = get_config_from_args()

    transform_objects = []
    for transform_config in config['transforms']:
        transform_objects.append(build_from_args(transform_config, 'torchvision.transforms'))
    augmentation = torchvision.transforms.Compose(transform_objects)

    # trainer_args = {'type': config['trainer']['type'], 'config': config}
    # trainer = build_from_args(trainer_args, 'opensphere')

    temp = 0

    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = LitClassifier.add_model_specific_args(parser)
    # args = parser.parse_args()

    # ------------
    # general
    # ------------
    # pl.seed_everything(1234)

    # # ------------
    # # data
    # # ------------
    # dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    # mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    # mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    # train_loader = DataLoader(mnist_train, batch_size=args.batch_size)
    # val_loader = DataLoader(mnist_val, batch_size=args.batch_size)
    # test_loader = DataLoader(mnist_test, batch_size=args.batch_size)

    # # ------------
    # # model
    # # ------------
    # model = LitClassifier(Backbone(hidden_dim=args.hidden_dim), args.learning_rate)

    # # ------------
    # # training
    # # ------------
    # trainer = pl.Trainer.from_argparse_args(args)
    # trainer.fit(model, train_loader, val_loader)

    # # ------------
    # # testing
    # # ------------
    # result = trainer.test(test_dataloaders=test_loader)
    # print(result)


if __name__ == '__main__':
    cli_main()
