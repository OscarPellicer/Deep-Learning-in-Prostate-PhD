#Basics
import os, argparse, numpy as np, sys

#Torch
import torch

#TorchIO
import torchio as tio

#Others
from .SegmentationModel import SegmentationModel
from .layers import NonLearnableConv, SpatialTransformer, FlowPredictor
from ..metrics import NCC, MaskedNCC
from ..metrics import MutualInformation, GradNorm, MaskedMSE, MSE, SoftDiceOverlap

#plot_lib: only needed if plotting
try:
    import sys, os
    from pathlib import Path
    sys.path.append(os.path.join(Path.home(), 'plot_lib'))
    from plot_lib import plot
except:
    print('Warning: plot_lib was not found. '\
          'If plotting is on, an error will be raised when calling `_plot`')

class FeatureExtractor(torch.nn.Module):
    def __init__(self, channels, levels=3):
        '''
            Extracts spatial gradient features using Sobel operator on several
            resolution levels
            
            Parameters
            ----------
            channels: int
                Number of input channels
            levels: int, default 3
                Number of resolution levels to evaluate the Sobel operator on
        '''
        super().__init__() 
        self.levels= levels
        self.sobel= NonLearnableConv(channels, mode='sobel', padding='valid')
        self.maxpool= torch.nn.AvgPool3d(2, stride=2)
        
    def forward(self, x, plot=False):
        data= [x]
        for l in range(self.levels-1):
            x= self.maxpool(x)
            data.append(x)
        out= [self.sobel(d) for d in data]
        if plot: self._plot(out)
        return out
    
    def _plot(self, fmaps):
        for i, f in enumerate(fmaps):
            f= f.detach().cpu().numpy()
            for j in range(0, f.shape[1], 3):
                plot(f[0,j:j+3].transpose((1,2,3,0)), title=f'Feature map {i}', is_color=True, 
                     interactive=False, normalization_threshold=5.)

class DDFModel(SegmentationModel):
    def __init__(self, hparams):      
        #Overwrite augmentation parameters (if not loading from checkpoint)
        if hasattr(hparams, 'transform_flip_x'):
            augmentation_factor=.5
            hparams.transform_flip_x= False
            hparams.transform_translate*= augmentation_factor
            hparams.transform_rotate_x*= augmentation_factor
            hparams.transform_rotate_y*= augmentation_factor
            hparams.transform_rotate_z*= augmentation_factor
            hparams.transform_scale*= augmentation_factor
        
        #Super class intializes self.net
        super().__init__(hparams)
        self.save_hyperparameters(hparams)
        
        #Replace activation
        self.activation= FlowPredictor(self.hparams.output_classes)
        
        #Get transformer
        self.transformer= SpatialTransformer()
        
        #Get feature extractors
        self.feature_levels= 3
        if self.hparams.feat_weight > 0.:
            self.feature_extractor= FeatureExtractor(1, levels=self.feature_levels)
        if self.hparams.grad_weight > 0.:
            self.feature_extractor_ddf= FeatureExtractor(3, levels=self.feature_levels)

        #Set losses
        self.diffusion_reg = GradNorm()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.masked_mse_loss= MaskedMSE()
        self.mse_loss= MSE()
        self.masked_ncc_loss= MaskedNCC(window=self.hparams.ncc_win_size)
        
        #Set metrics (some must *-1 to make a loss)
        self.dice= SoftDiceOverlap(ignore_zero_class=False)
        if self.hparams.mi_weight > 0.:
            self.mi= MutualInformation(num_bins=128, sigma=0.05, value_range=[0., 1.], normalize=True)
            
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
        ])

    def forward(self, x):
        y= self.net(x)
        return self.activation(y)
        
    def _step(self, batch, batch_idx, plot=False):
        #Get batch
        x, y_true, meta = batch
                  
        #Extract inputs and outputs
        y_mr, y_us, y_mr_mask, y_us_mask= x[:,0:1], x[:,1:2], x[:,2:3], x[:,3:4]
        y_softmask= y_true[:,-1:] #Mask is contained in the last channel of y
        y_ddf= y_true[:,:-1] #DDF is contained in the first 3 channels of y

        #Predict
        x_train= x
        #x_train= torch.concat([y_mr, y_us, y_mr_mask, y_us_mask], axis=1)
        yp_ddf= self.forward(x_train)
        yp_mrus= self.transformer(y_mr, yp_ddf) #Transformed MR image
        yp_mrus_mask= self.transformer(y_mr_mask, yp_ddf)  #Transformed MR mask
        y_mrus= self.transformer(y_mr, y_ddf) #Reference (using y_ddf) transformed MR image
        y_mrus_mask= self.transformer(y_mr_mask, y_ddf) #Reference (using y_ddf) transformed MR mask
        
        #Feature extractor loss: Compares spatial gradient of US and MR
        feat_loss= 0.
        if self.hparams.feat_weight > 0.:
            y_us_feat= self.feature_extractor(y_us)
            yp_mrus_feat= self.feature_extractor(yp_mrus, plot=plot)
            for usf, mrf in zip(y_us_feat, yp_mrus_feat):
                feat_loss+= self.mse_loss(torch.abs(usf), torch.abs(mrf))
                
        #Grad normalizer loss: Computes the mean magnitude of the spatial gradient of the DDF
        grad_loss= 0.
        if self.hparams.grad_weight > 0.:
            yp_ddf_grads= self.feature_extractor_ddf(yp_ddf, plot=plot)
            for yp_ddf_grad in yp_ddf_grads:
                feat_loss+= torch.mean(torch.sum(yp_ddf_grad**2, axis=1))
                
        #UR dsc loss    
        ur_dsc_loss= 0.
        if self.hparams.ur_dsc_weight > 0.:
            y_mr_ur_mask, y_us_ur_mask= x[:,4:5], x[:,5:6]
            yp_mrus_ur_mask= self.transformer(y_mr_ur_mask, yp_ddf)  #Transformed UR MR mask
            ur_dsc_loss= 1. - self.dice(y_us_ur_mask, yp_mrus_ur_mask)
        
        #Losses
        masked_ncc_loss= self.masked_ncc_loss(y_mrus, yp_mrus, y_softmask  > 0.15) #1. #Same modality (MR and MR)
        diff_loss = self.diffusion_reg(yp_ddf)
        masked_mse_loss= self.masked_mse_loss(y_ddf, yp_ddf, y_softmask) #Base loss
        mi_loss= self.mi(y_us, yp_mrus, plot=plot) if self.hparams.mi_weight > 0. else 0.
        dsc_loss= 1. - self.dice(y_us_mask, yp_mrus_mask)
            
        loss=   (self.hparams.mse_weight * masked_mse_loss \
              + self.hparams.dsc_weight * dsc_loss \
              + self.hparams.ur_dsc_weight * ur_dsc_loss \
              + self.hparams.ncc_weight * masked_ncc_loss \
              + self.hparams.feat_weight * feat_loss \
              + self.hparams.diff_weight * diff_loss \
              + self.hparams.mi_weight * mi_loss \
              + self.hparams.grad_weight * grad_loss) # /\
            # (self.hparams.mse_weight + self.hparams.dsc_weight + self.hparams.ur_dsc_weight + self.hparams.ncc_weight + 
            #  self.hparams.feat_weight + self.hparams.diff_weight + self.hparams.mi_weight + self.hparams.grad_weight)
        
        if plot: 
            self._plot(x, y_ddf, yp_ddf, y_softmask, yp_mrus, y_mrus, yp_mrus_mask, y_mrus_mask, 
                            y_us_feat[0] if self.hparams.feat_weight > 0. else None, 
                            yp_mrus_feat[0] if self.hparams.feat_weight > 0. else None)
        
        #Return
        results= {  "loss": loss, 
                    "dsc_loss": dsc_loss, 
                    "ur_dsc_loss": ur_dsc_loss, 
                    "mse_loss": masked_mse_loss, 
                    "feat_loss": feat_loss,
                    "diff_loss": diff_loss,
                    "mi_loss": mi_loss,
                    "ncc_loss": masked_ncc_loss, 
                    "grad_loss": grad_loss }
        return results

    def _plot(self, x, y_true, y_pred, y_mask, yp_mrus, y_mrus, yp_mrus_mask, y_mrus_mask, y_us_feat, yp_mrus_feat):
        with torch.no_grad():
            #All to numpy
            x_np= x.cpu().numpy()
            y_pred_np= y_pred.cpu().numpy()
            y_true_np= y_true.cpu().numpy()
            #y_mask_np= y_mask.cpu().numpy()
            yp_mrus_np= yp_mrus.cpu().numpy()
            y_mrus_np= y_mrus.cpu().numpy()
            yp_mrus_mask_np= yp_mrus_mask.cpu().numpy()
            y_mrus_mask_np= y_mrus_mask.cpu().numpy()
            if y_us_feat is not None and yp_mrus_feat is not None:
                y_us_feat_np= y_us_feat.cpu().numpy()
                yp_mrus_feat_np= yp_mrus_feat.cpu().numpy()
            
            #Plot
            plot(x_np[0,0], masks=[x_np[0,2] > 0.5], title='MR', interactive=False)
            plot(x_np[0,1], masks=[x_np[0,3] > 0.5], title='US', interactive=False)
            if y_us_feat is not None:
                plot_composite(np.sum(y_us_feat_np[0].transpose((1,2,3,0))**2, axis=-1), 
                               np.sum(yp_mrus_feat_np[0].transpose((1,2,3,0))**2, axis=-1),
                               title='MR transformed / US spatial gradient', 
                               interactive=False, normalization_threshold=5.)
            plot(y_true_np[0].transpose((1,2,3,0)),  masks=[x_np[0,3] > 0.5], 
                 title='Actual DDF', is_color=True, interactive=False, normalization_threshold=5.)
            plot(y_pred_np[0].transpose((1,2,3,0)),  masks=[x_np[0,3] > 0.5], 
                 title='Predicted DDF', is_color=True, interactive=False, normalization_threshold=5.)
            plot(y_mrus_np[0,0], masks=[x_np[0,3] > 0.5, y_mrus_mask_np[0,0] > 0.5], 
                 title='Actual transformed MR (along with actual mask)', interactive=False)
            plot(yp_mrus_np[0,0], masks=[x_np[0,3] > 0.5, yp_mrus_mask_np[0,0] > 0.5], 
                 title='Predicted transformed MR (along with predicted mask)', interactive=False)
            
    @staticmethod
    def add_model_specific_args(parent_parser):
        '''
            Adds model specific command-line args
        '''
        common_parser = SegmentationModel.add_common_model_args()
        parser = argparse.ArgumentParser( parents=[common_parser, parent_parser], add_help=False)
        parser.add_argument("--mse_weight", default=1., help="MSE loss weight")
        parser.add_argument("--feat_weight", default=0., help="Feature loss weight")
        parser.add_argument("--dsc_weight", default=1., help="DSC loss weight")
        parser.add_argument("--ur_dsc_weight", default=0., help="UR DSC loss weight")
        parser.add_argument("--mi_weight", default=0., help="Mutual Information loss weight")
        parser.add_argument("--diff_weight", default=1., help="Diffusion regularization loss weight")
        parser.add_argument("--grad_weight", default=0., help="Gradient regularization loss weight")
        parser.add_argument("--ncc_weight", default=1., help="NCC loss weight")
        parser.add_argument("--ncc_win_size", type=int, default=9, 
                            help="Window-Size for the NCC loss function (Default: 9)")
        return parser
