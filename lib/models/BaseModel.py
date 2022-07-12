#Base
import os, argparse, numpy as np, math, numbers, abc
from collections.abc import MutableMapping
import matplotlib.pyplot as plt

#Torch
import torch, pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

#TorchIO
import torchio as tio

class CommonLightningModel(pl.LightningModule, metaclass=abc.ABCMeta):
    def __init__(self, hparams):
        '''
            Base Pytorch Lightning module with added logging and augmentation
            functionality, and Adam optimizer pre-configured
        '''
        super().__init__()
        self.save_hyperparameters(hparams)
        self.plot_batch= None
        self.learning_rate= self.hparams.lr
            
        #Prepare augmentations (flip only over x)
        flip_axis= [] + ([2] if self.hparams.transform_flip_x else [])\
                      + ([1] if self.hparams.transform_flip_y else [])\
                      + ([0] if self.hparams.transform_flip_z else [])
        self.augmentation= tio.Compose([
            tio.RandomAffine(scales=self.hparams.transform_scale,
                             degrees=(self.hparams.transform_rotate_z, self.hparams.transform_rotate_y, 
                                      self.hparams.transform_rotate_x),
                             translation=self.hparams.transform_translate,
                             p=1.
                            ),
            tio.RandomFlip(axes=flip_axis, flip_probability=0.5),
            # tio.RandomGamma(log_gamma=0.3),
            # tio.RandomNoise(std=0.1),
            # tio.RandomBiasField(p=0.25),
        ])

    def on_before_batch_transfer(self, batch, batch_idx):
        #Make sure that we are training with a real batch
        #If we do not do this check, torchinfo's summary makes it crash
        #as it passes a single image batch
        #Also, make sure that augmentation is globally enabled
        if  self.trainer.training\
            and hasattr(batch, '__len__') and len(batch) == 3\
            and self.hparams.transform:
                x, y_true, meta = batch
                batch= [*self.augment(x, y_true), meta]
        return batch
        
    def augment(self, x, y):
        with torch.no_grad():
            xs, ys= [], []
            for xi, yi in zip(x, y):
                xy= self.augmentation(torch.concat([xi, yi], axis=0))
                xs.append(xy[:x.shape[1]])
                ys.append(xy[x.shape[1]:])
            x= torch.stack(xs, axis=0)
            y= torch.stack(ys, axis=0)
        return x, y

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return { "optimizer": opt }

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def _step(self, batch, batch_idx, plot=False):
        pass
    
    @abc.abstractmethod
    def _plot(*args):
        pass
    
    def _step_and_log(self, batch, batch_idx, subset, plot=False):
        metrics= self._step(batch, batch_idx, plot=plot)
        for name, metric in metrics.items():
            self.log(f'{subset}_{name}', metric)
        return metrics['loss']

    def training_step(self, batch, batch_idx):
        #By default, plot first training image only
        return self._step_and_log(batch, batch_idx, 'train', 
                                  plot= not self.global_step and self.hparams.plot)

    def validation_step(self, batch, batch_idx):
        if self.plot_batch is None: 
            self.plot_batch= batch
        return self._step_and_log(batch, batch_idx, 'val')

    def validation_epoch_end(self, outputs):
        if self.hparams.plot:
            self._step(self.plot_batch, None, plot=True)
            self.plot_batch= None

    def test_step(self, batch, batch_idx):
        return self._step_and_log(batch, batch_idx, 'test')

    @staticmethod
    def add_common_model_args():
        parser= argparse.ArgumentParser(add_help=False)
        
        #Model params
        parser.add_argument("--lr", type=float, default=1e-3, 
                            help="Learning rate (default: 0.001)")
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
        parser.add_argument("--plot", action="store_true", help="Visualize first batch of validation set")
        parser.add_argument("--load_from_checkpoint", help="optional model checkpoint to initialize with")
        parser.add_argument("--anisotropic_stages", dest='anisotropic_stages', type=int, default=0, 
                            help="Number of anistropic unet stages")
        parser.add_argument("--input_channels", type=int, default=1, help="Input channels")
        parser.add_argument("--output_classes", type=int, default=2, help="Output channels/classes")
        parser.add_argument("--encoder_channels", nargs="+", type=int, default=[64, 128, 256, 512],
            help="U-Net encoder channels (default: [64, 128, 256, 512])")
        parser.add_argument("--decoder_channels", nargs="+", type=int, default=[64, 128, 256, 512][::-1],
            help="U-Net decoder channels (default: [64, 128, 256, 512])")
        parser.add_argument("--encoder_blocks", nargs="+", type=int, default=[1,1,1,1][::-1],
            help="Number of U-Net encoder base blocks per stage (default: [1,1,1,1])")
        parser.add_argument("--bnorm", action="store_true", help="Use batch normalization")
        parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
        parser.add_argument("--net", type=str, help="CNN model to instantiate")
        
        #Augmentation params
        parser.add_argument("--transform", action="store_true",
                            help="Enable transformation of training data")
        parser.add_argument("--transform_rotate_x", type=float, default=5., 
                            help="Random rotation in degrees around x axis (default: 8)")
        parser.add_argument("--transform_rotate_y", type=float, default=5., 
                            help="Random rotation in degrees around y axis (default: 8)")
        parser.add_argument("--transform_rotate_z", type=float, default=10., 
                            help="Random rotation in degrees around z axis (default: 16)")
        parser.add_argument("--transform_translate", type=float, default=15., 
                            help="Random translation in voxels (default: 40)")
        parser.add_argument("--transform_scale", type=float, default=0.15, 
                            help="Random scale factor (default: 0.1)")
        parser.add_argument("--transform_shear", type=float, default=0.025,
                            help="Random shear factor (default: 0.025)")
        parser.add_argument( "--transform_flip_x", dest='transform_flip_x', default=True, 
                            action='store_true', help="Randomly flip along x axis")
        parser.add_argument( "--transform_flip_y", dest='transform_flip_y', default=False, 
                            action='store_true', help="Randomly flip along y axis")
        parser.add_argument( "--transform_flip_z", dest='transform_flip_z', default=False, 
                            action='store_true', help="Randomly flip along z axis")
        
        return parser