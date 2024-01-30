import argparse
import configparser
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

def read_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def setup_arg_parser(config):
    parser = argparse.ArgumentParser(description='My PyTorch Lightning Project')

    # Add an argument for the configuration file path
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration file')

    # Iterate over sections and options to dynamically add arguments
    for section in config.sections():
        for option in config.options(section):
            arg_name = f'--{section}_{option}'
            arg_value = config.get(section, option)
            arg_type = type(arg_value)
            parser.add_argument(arg_name, type=arg_type, default=arg_value, help=f'{section}.{option} in the configuration file')

    # Add any additional command-line arguments you need
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')

    return parser

def cli_main():
    # ------------
    # args
    # ------------

    # Parse command-line arguments to get the configuration file path
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration file')
    args, _ = arg_parser.parse_known_args()

    # Read the configuration file
    config = read_config(args.config_path)

    # Setup the ArgumentParser based on the configuration file
    arg_parser = setup_arg_parser(config)

    # Parse the command-line arguments
    args = arg_parser.parse_args()

    # Use the values obtained from argparse
    epochs = args.epochs

    # Use these values in your PyTorch Lightning project
    print(f"Epochs: {epochs}")

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
