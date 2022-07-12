#Torch
import torch
from torch.utils.data import Dataset

#Other common libraries
import numpy as np, SimpleITK as sitk, sys, os
from scipy.ndimage import gaussian_filter

#Custom reprocessing lib
from ..processing.preprocessing import resample_image, rescale_intensity

def greater_than_zero(m):
    return m>0

class MedicalImageDataset(Dataset):
    def __init__(self, data_dict, inputs, outputs, ids,
                 size, spacing=None,
                 normalize_inputs=True, 
                 cache=False, verbose=False,
                 combine_inputs=True, combine_outputs=True,
                 input_channels=[0], output_channels={0: greater_than_zero},
                 save_extra_metadata= False, mask_sigma= (1,1,1),
                 problem='binary_classification', class_IDs= [],
                 process_ddf=False, input_interpolator=None):
        '''
            data_dict: {'IVO_1': {'MR': 'path to MR', 'MR_mask': 'path_to MR mask', ...}, ...}
            problem: one of {'binary_classification', 'multiclass_classification', 'regression', 
            'softmasked_regression'}
        '''
        self.data_dict, self.IDs= data_dict, ids
        self.inputs, self.outputs= inputs, outputs
        self.size, self.spacing= size, spacing
        self.normalize_inputs= normalize_inputs
        self.combine_inputs, self.combine_outputs= combine_inputs, combine_outputs
        self.input_channels, self.output_channels= input_channels, output_channels
        self.problem, self.class_IDs= problem, class_IDs
        self.cache= {} if cache else None
        self.verbose= verbose
        self.mask_sigma= mask_sigma
        self.process_ddf= process_ddf
        
        self.output_type= np.long if 'classification' in problem else np.float32
        self.output_interpolator= sitk.sitkLabelGaussian if 'classification'\
                                                            in problem else sitk.sitkBSpline
        self.input_interpolator= [sitk.sitkBSpline] * len(inputs) if input_interpolator is None \
                                                                     else input_interpolator
        self.save_extra_metadata= save_extra_metadata
        
        super().__init__()

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        #Recover PID
        PID= self.IDs[idx]
        
        #Read images
        if self.cache is not None and PID in self.cache.keys():
            inputs, outputs, meta= self.cache[PID]
        else:
            inputs= [read_image(self.data_dict[PID][i]) for i in self.inputs]
            inputs= [i if i is not None else sitk.Image(inputs[0]*0) for i in inputs]
            outputs= [read_image(self.data_dict[PID][i]) for i in self.outputs]
            meta= self.data_dict[PID]['meta']
            if self.save_extra_metadata: meta['inputs'], meta['outputs']= inputs, outputs
                                 
            #Custom processing in sitk space (TO DO: pass processing as functions instead)
            if self.process_ddf:
                outputs[0]= sitk.InvertDisplacementField(outputs[0], enforceBoundaryCondition=True)

            #Get center crop
            different_spacing= any([list(i.GetSpacing()) != list(self.spacing) for i in inputs + outputs])
            different_size= any([list(i.GetSize()) != list(self.size) for i in inputs + outputs])
            if different_size and different_spacing:
                inputs= [resample_image(i, size=self.size, spacing=self.spacing, 
                                        img_interpolator=interp) for i, interp in zip(inputs, self.input_interpolator)]
                outputs= [resample_image(o, size=self.size, spacing=self.spacing, return_as_dict=True,
                                         img_interpolator=self.output_interpolator) for o in outputs]
                if self.save_extra_metadata: meta['image_meta']= [o[1] for o in outputs]
                outputs= [o[0] for o in outputs]
            elif different_size:
                raise RuntimeError(f'{PID}: Not all images are of size {self.size}. '
                                   f'Sizes: {[i.GetSize() for i in inputs + outputs]}.'
                                   f'Please provide a spacing so that resampling can be applied')
            elif different_spacing:
                raise RuntimeError(f'{PID}: Not all images are of spacing {self.spacing}. '
                                   f'Sizes: {[i.GetSpacing() for i in inputs + outputs]}.'
                                   f'Please provide a spacing so that resampling can be applied')
            else:
                pass #All good, no processing needed!
                
            #To numpy!
            inputs= [sitk.GetArrayFromImage(i) for i in inputs]
            outputs= [sitk.GetArrayFromImage(o) for o in outputs]
            if self.verbose: print('Shape after cropping:', [i.shape for i in inputs], [o.shape for o in outputs])
            
            #Custom processing in numpy space (TO DO: pass processing as functions instead)
            if self.process_ddf:
                assert self.spacing[0] == self.spacing[1] and self.spacing[1] == self.spacing[2], \
                    'Spacing must be the same across all dimensions (TO DO)'
                outputs[0]= outputs[0][..., [2,1,0]] * 1/self.spacing[2]

            #Processing of the outputs
            outputs_processed= []
            for o in outputs:
                if len(o.shape) == 4:
                    if self.output_channels is not None:
                        outputs_processed+= [f(o[...,out_c]) for out_c, f in self.output_channels.items()]
                    else:
                        outputs_processed.append(o.transpose((3, 0, 1, 2)))  # (D, H, W, C) -> (C, D, H, W)
                else:
                    outputs_processed.append(o)
            outputs= outputs_processed
                    
            if self.problem == 'binary_classification':
                assert len(outputs) == 1, f'There should be a single output, {len(outputs)} were found'
                if self.output_channels is not None: 
                    outputs= [self.output_channels[0](o) for o in outputs]
                outputs= [np.stack([1-o, o], axis=0) for o in outputs]
            elif self.problem == 'multiclass_classification': 
                assert len(outputs) == 1, f'There should be a single output, {len(outputs)} were found'
                outputs= [np.stack([o == cid for cid in self.class_IDs], axis=0) for o in outputs]
            elif self.problem == 'masked_binary_classification':
                assert len(outputs) == 2, f'There should be two outputs (output + mask), {len(outputs)} were found'
                if self.output_channels is not None: 
                    outputs= [self.output_channels[0](o) for o in outputs]
                outputs= [1-outputs[0][None], outputs[0][None], gaussian_filter(outputs[1].astype(np.float32), self.mask_sigma)[None] > 0.25]
            elif self.problem == 'regression':
                outputs= [o[None] for o in outputs]
            elif self.problem == 'masked_regression':
                assert len(outputs) == 2, f'There should be two outputs (output + mask), {len(outputs)} were found'
                outputs= [outputs[0], gaussian_filter(outputs[1].astype(np.float32), self.mask_sigma)[None]]
            else:
                raise RuntimeError(f'{PID}: Unrecognized problem type: {self.problem}')

            #Processing of the inputs
            if len(inputs[0].shape) == 4:
                inputs= [i[...,in_c] for in_c in self.input_channels for i in inputs]
                
            if self.normalize_inputs:
                if isinstance(self.normalize_inputs, list):
                    inputs= [rescale_intensity(img) if i in self.normalize_inputs else img for i, img in enumerate(inputs)]
                else:
                    inputs= [rescale_intensity(img) for img in inputs]
                
            #Add extra dimension to inputs if needed
            inputs= [i[None] if len(i.shape) == len(self.size) else i for i in inputs]

            #Combine?
            #torch.as_tensor
            if self.verbose: print('Shape after processing:', [i.shape for i in inputs], [o.shape for o in outputs])
            if self.combine_inputs:   inputs= np.concatenate(inputs, axis=0).astype(np.float32)
            else:                     inputs= [i.astype(np.float32) for i in inputs]

            if self.combine_outputs:  outputs= np.concatenate(outputs, axis=0).astype(np.float32)
            else:                     outputs= [to_torch(o).astype(np.float32) for o in outputs]
                
            #Save to cache?
            if self.cache is not None: 
                self.cache[PID]= (inputs, outputs, meta)
        
        #Return
        if self.verbose: print('Final shape:', inputs.shape, outputs.shape)
        return inputs, outputs, meta
    
def read_image(path):
    '''
        Reads a medical image file or a dicom directory using SimpleITK
        
        Parameters
        ----------
        path: str
            Path to image file or to DICOM directory
        
        Returns
        -------
        img: SimpleITK Image
    '''
    try:
        if os.path.isdir(path):
            reader = sitk.ImageSeriesReader()
            names= reader.GetGDCMSeriesFileNames(path)
            reader.SetFileNames(names)
            return reader.Execute()
        else:
            return sitk.ReadImage(path)
    except:
        return None