#Basics
import os, argparse, numpy as np, sys

#Torch
import torch, torch.nn as nn

#TorchIO
import torchio as tio

#Models
from .BaseModel import CommonLightningModel
from ..metrics import SoftDiceOverlap
from ..backbones import VNet, VNetLight, VNet3

#plot_lib: only needed if plotting
try:
    import sys, os
    from pathlib import Path
    sys.path.append(os.path.join(Path.home(), 'plot_lib'))
    from plot_lib import plot
except:
    print('Warning: plot_lib was not found. '\
          'If plotting is on, an error will be raised when calling `_plot`')

class SegmentationModel(CommonLightningModel):
    def __init__(self, hparams):
        #Overwrite augmentation parameters (if not loading from checkpoint)
        if hasattr(hparams, 'transform_flip_x'):
            augmentation_factor=2
            hparams.transform_flip_x= True
            hparams.transform_translate*= augmentation_factor
            hparams.transform_rotate_x*= augmentation_factor
            hparams.transform_rotate_y*= augmentation_factor
            hparams.transform_rotate_z*= augmentation_factor
            hparams.transform_scale*= augmentation_factor
        
        #Super class intializes self.net
        super().__init__(hparams)
        self.save_hyperparameters(hparams)

        #Set net
        if self.hparams.net == 'VNET2':
            self.net = VNetLight(in_channels=self.hparams.input_channels, elu=False, classes=self.hparams.output_classes)
        elif self.hparams.net == 'VNET3':
            self.net = VNet3(in_channels=self.hparams.input_channels, elu=False, classes=self.hparams.output_classes)
        elif self.hparams.net == 'VNET':
            self.net = VNet(in_channels=self.hparams.input_channels, elu=False, classes=self.hparams.output_classes)
            
        #Set final activation
        self.activation= nn.Softmax(dim=1)

        #Set metrics and losses
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.dice_loss= lambda *a: 1. - SoftDiceOverlap(
            ignore_zero_class=True)(*a)
        
        #Prepare augmentations (flip only over x)
        #TO DO: This should only be defined in the base class
        #And all agumentation configuration should be hparams
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
            tio.RandomGamma(log_gamma=0.3),
            tio.RandomNoise(std=0.1),
            tio.RandomBiasField(p=0.25),
        ])

    def forward(self, x):
        # y_pred_onehot, y_pred_raw = self.net(x)
        y_pred_raw = self.net(x)
        y_pred_onehot = self.activation(y_pred_raw)
        y_pred = torch.argmax(y_pred_onehot, dim=1, keepdim=True)
        return y_pred_onehot, y_pred, y_pred_raw

    def _step(self, batch, batch_idx, plot=False):
        #Get batch
        x, y_true, meta = batch
        
        #Predict
        y_pred, _, y_pred_raw= self.forward(x) 
        #print(y_pred_raw.shape, y_pred.shape, y_true.shape)

        #Compute loss
        #ce_loss = self.cross_entropy_loss(y_pred_raw, y_true.squeeze(1))
        dice_loss= self.dice_loss(y_true, y_pred)
        loss= dice_loss #Combine losses
        
        if plot: self._plot(x, y_true, y_pred)
        
        #Return
        return {"loss": loss}

    def _plot(self, x, y_true, y_pred):
        with torch.no_grad():
            x_np= x.cpu().numpy()
            yp_np= y_pred.cpu().numpy()
            y_np= y_true.cpu().numpy()
            plot(x_np[0,0], masks=[y_np[0,1] > 0.5, yp_np[0,1] > 0.5], title='Predicted', interactive=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
            Adds model specific command-line args
        """
        common_parser = CommonLightningModel.add_common_model_args()
        parser = argparse.ArgumentParser( parents=[common_parser, parent_parser], add_help=False)
        parser.add_argument( "--bihead", dest='bihead', default=False, action='store_true', 
            help="Use bihead unet for segmentation")
        return parser

class CustomMaskedSegmentationModel(SegmentationModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.save_hyperparameters(hparams)
        self.pos_idx = None
        
    def _step(self, batch, batch_idx, plot=False):
        #Get batch
        x, y_true, meta = batch
        
        #Predict
        y_pred, _, y_pred_raw= self.forward(x) 

        #New implementation
        x_means_pred, y_means_pred= self.mean_z_position(y_pred[:,1], eps=1e-5)
        
        #Alternative implementation (fails)
        # x_means_pred= y_pred_raw[:,0].mean(axis=(-1, -2)) + y_pred_raw.shape[-1] / 2
        # y_means_pred= y_pred_raw[:,1].mean(axis=(-1, -2)) + y_pred_raw.shape[-2] / 2
        
        x_means_true, y_means_true= self.mean_z_position(y_true[:,1], eps=0)
        valid_z_slices= torch.isfinite(x_means_true) & torch.isfinite(x_means_pred)
        distances2= torch.sqrt( (x_means_true-x_means_pred)**2 + (y_means_true-y_means_pred)**2 )
        point_loss= distances2[valid_z_slices].mean()
        
        #Extract stuff
        y_trueout= y_true[:,:-1] #Actual segmentation is contained in the first two channels
        y_mask= y_true[:,-1:] #Mask is contained in the last channels of y
        
        y_trueout[:,0]= y_trueout[:,0] * (1-y_mask[:,0]) #Simply cut all true mask around the prostate
        y_trueout[:,1]= y_trueout[:,1] * y_mask[:,0]

        dice_loss= self.dice_loss(y_pred, y_trueout)
        
        #Combine losses
        loss= dice_loss
        # if torch.isfinite(point_loss): loss= dice_loss + point_loss
        # else: loss= dice_loss
        
        if plot: self._plot(x, y_pred, y_trueout, x_means_pred, y_means_pred, x_means_true, y_means_true)
        
        #Return
        return {"loss": loss, 
                "dice_loss": dice_loss,
                "point_loss": point_loss}
    
    def mean_z_position(self, mask, eps=1e-10):
        assert len(mask.shape) == 4, 'Mask should have 4 dimensions: (batch, z, y, x)'
        if self.pos_idx == None: 
            self.pos_idx= torch.tensor(np.indices(mask.shape)).float()[-2:].to(self.device)
                
        denominator= (mask).sum(axis=(-1, -2)) + eps
        x_means= (mask*self.pos_idx[1]).sum(axis=(-1, -2))/denominator
        y_means= (mask*self.pos_idx[0]).sum(axis=(-1, -2))/denominator
        
        return x_means, y_means
    
    def _plot(self, x, y_pred, y_true, x_means_pred, y_means_pred, x_means_true, y_means_true):
        with torch.no_grad():
            x_np= x.cpu().numpy()
            yp_np= y_pred.cpu().numpy()
            y_np= y_true.cpu().numpy()
            x_mp_np= x_means_pred.cpu().numpy()
            y_mp_np= y_means_pred.cpu().numpy()
            x_mt_np= x_means_true.cpu().numpy()
            y_mt_np= y_means_true.cpu().numpy()
            
            batch_idx= 0
            
            points_true= [(x, y, z, 'o','tab:green') for z, (x,y) in enumerate(zip(x_mt_np[batch_idx], y_mt_np[batch_idx]))]
            points_pred= [(x, y, z, 'o','tab:red') for z, (x,y) in enumerate(zip(x_mp_np[batch_idx], y_mp_np[batch_idx]))]
            
            plot(x_np[batch_idx,0], masks=[y_np[batch_idx,1] > 0.5, yp_np[0,1] > 0.5], title='Prediction', interactive=False,
                 points=points_pred + points_true)