import sys, os, copy, glob, json
from time import time
from functools import partial
from collections import defaultdict
from typing import Optional, Tuple, Callable, Union

import numpy as np, matplotlib.pyplot as plt
import SimpleITK as sitk

from scipy import ndimage
from scipy.stats import ttest_rel, wilcoxon, normaltest
from skimage.morphology import ball

a2i= sitk.GetImageFromArray
i2a= sitk.GetArrayFromImage

def pair_test(a, b, paired_test=wilcoxon):
    assert len(a) == len(b), 'This is a paired test!'
    a, b= np.array(a), np.array(b)
    selector= ~np.isnan(a) & ~np.isnan(b)
    a, b= a[selector], b[selector]
    try: return paired_test(a, b)[1]
    except: return -1

def info(image, show='all'):
    '''
        Prints information about a SimpleITK or a numpy image
        
        Parameters
        ----------
        image: SimpleITK Image or array
            Image to obtain information about. SimpleITK images show much more information
        show: str, any of ['all', 'size', 'origin', 'spacing', 'direction', 
            'channels', 'type', 'range'], default 'all'
            What information to show ('all') to show all
    '''
    if not isinstance(image, (np.ndarray, np.generic) ):
        if show in ['all']: print('SITK image info:')
        if show in ['all', 'size']: print(' - Size:', np.array(image.GetSize()))
        if show in ['all', 'origin']: print(' - Origin:', np.array(image.GetOrigin()))
        if show in ['all', 'spacing']: print(' - Spacing:', np.array(image.GetSpacing()))
        if show in ['all', 'direction']: print(' - Direction:', np.array(image.GetDirection()))
        if show in ['all', 'channels']: print(' - Components per pixel:', 
                                              np.array(image.GetNumberOfComponentsPerPixel()))
        if show in ['all', 'type']: print(' - Pixel type:', image.GetPixelIDTypeAsString())
        img_arr= sitk.GetArrayViewFromImage(image)
    else:
        print('Numpy image info:')
        print(' - Shape:', image.shape)
        print(' - Pixel type:', image.dtype)
        img_arr= image
        
    unique= np.unique(img_arr)
    if show in ['all', 'range']: print(' - Min/max:', np.min(img_arr), np.max(img_arr))
    if len(unique) < 50 and show in ['all', 'unique']: print(' - Unique values:', unique)

def timer(func):
    '''
        Wrapper function to time + show kwarg of function
        To use on `f(a,b)`, call `f` like: `timer(f)(a,b)`
        Or add `@timer` decorator to `f` definiton
        
        Paramters
        ---------
        func: callable
        
        Returns
        -------
        wrap_func: wrapped callable
    '''
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r}\n - Executed in {(t2-t1):.4f}s\n - KWargs: {kwargs}')
        return result
    return wrap_func

def urethra_available(prostate_mask, urethra_mask):
    prostate= sitk.GetArrayFromImage(prostate_mask) > 0.5
    urethra= sitk.GetArrayFromImage(urethra_mask)  > 0.05
    urethra= ndimage.binary_dilation(urethra, ball(2))
    idx= prostate.shape[0]//2
    start_available= np.any(~prostate[:idx]&urethra[:idx])
    end_available= np.any(~prostate[-idx:]&urethra[-idx:])

    return start_available, end_available

def read_fiducials(fiducials_path, check_fiducials=None):
    'Reads Slicer-generated fiducials into a dictionary'
    try:
        fiducials_json= load_json(fiducials_path)
        fiducials= {f['id']:np.array(f['position']) for f in fiducials_json['markups'][0]['controlPoints']}
    except Exception as e:
        print(f' - Error reading fiducials ({fiducials_path}): {e}')
        fiducials= {}
    
    if check_fiducials is not None:
        if not fiducials.keys() == check_fiducials.keys():
            print(f'The fiducial IDs are different: ({fiducials.keys()}) vs. ({check_fiducials.keys()})')
            fiducials= {}
    
    return fiducials

def get_soft_ur_points(softmask_sitk, valid_mask_sitk=None, den_threshold=0.25):
    mask= sitk.GetArrayFromImage(softmask_sitk) 
    assert len(mask.shape) == 3, 'Mask should have 3 dimensions: (z, y, x)'
    pos_idx= np.indices(mask.shape).astype(float)[-2:]
    intr= lambda a: int(np.round(a, 0))
                
    mask[mask < 0.]= 0.
    denominator= mask.sum(axis=(-1, -2))
    x_means= (mask*pos_idx[1]).sum(axis=(-1, -2))/denominator
    y_means= (mask*pos_idx[0]).sum(axis=(-1, -2))/denominator
    
    valid= np.isfinite(x_means) & ~np.isnan(x_means)
    x_means= x_means[valid]
    y_means= y_means[valid]
    
    points= [(x, y, z) for z, (x,y) in enumerate(zip(x_means, y_means))]
    if valid_mask_sitk is not None:
        points= [p for p in points if valid_mask_sitk[intr(p[0]), intr(p[1]), intr(p[2])]]
    points_sitk= [softmask_sitk.TransformContinuousIndexToPhysicalPoint(p) for p in points]
    
    print(denominator[valid], denominator[valid] > den_threshold)
    return [p for p, v in zip(points_sitk, denominator[valid] > den_threshold) if v]
    
def get_ur_points(ur_msk_sitk, mask):
    'Gets a list of slice points from a urethra segmentation mask'
    try:
        if isinstance(ur_msk_sitk, str):
            ur_msk_sitk= sitk.ReadImage(ur_msk_sitk)
        ur_msk= sitk.GetArrayFromImage(ur_msk_sitk)
        intr= lambda a: int(np.round(a, 0))
        ur_mean_pos, ur_mean_pos_raw= [], []
        for sl in np.unique(np.argwhere(ur_msk)[:,0]):
            point= [float(sl), *np.mean(np.argwhere(ur_msk[sl]), axis=0)]
            if mask[intr(point[0]), intr(point[1]), intr(point[2])]: 
                ur_mean_pos.append(ur_msk_sitk.TransformContinuousIndexToPhysicalPoint(point)[::-1])
                ur_mean_pos_raw.append(point)
    except Exception as e:
        ur_mean_pos, ur_mean_pos_raw= [], [], None
        print(' - Error reading urethra segmentation masks:', e)
    return ur_mean_pos, ur_mean_pos_raw, ur_msk_sitk

def torch2sitk(img, c=[0], mask=False, ref=None, f=lambda a: a):
    'Transforms torch image to sitk image (requires torch)'
    img_sitk= sitk.GetImageFromArray(f(img.cpu().numpy()[0, c[0] if len(c)==1 else c]), isVector=len(c)!=1)
    img_sitk= img_sitk > 0.5 if mask else img_sitk
    if ref is not None:
        img_sitk.SetSpacing(ref.GetSpacing())
        img_sitk.SetOrigin(ref.GetOrigin())
        img_sitk.SetDirection(ref.GetDirection())
    return img_sitk

def sitk_transform_points(tfm_sitk, P, fix=False):
    return np.stack([tfm_sitk.TransformPoint(np.double(p)) for p in P], axis=0)

def distance0(ps1, ps2):
    'Distance between two points'
    return np.sqrt(np.sum(np.array(ps1) - np.array(ps2))**2)

def distance(ps1, ps2):
    'Distance between two paired sets of points or a point and a set of points'
    if not len(ps1) or not len(ps2) or len(ps1) != len(ps2): return [np.nan]
    else: return np.sqrt(np.sum((np.array(ps1) - np.array(ps2))**2, axis=1))

def point_average_error(points1, points2):
    'Mean average distance between two sets of paired points'
    if not len(points1) or not len(points2) or len(points1) != len(points2):
        return np.nan
    else:
        return np.mean(distance(points1, points2))
    
def point_max_min_distance(points1, points2):
    'Returns the maximum minimum distance between two sets of points'
    if not len(points1) or not len(points2):
        return np.nan
    else:
        return max([np.max([np.min(distance(p1, points2)) for p1 in points1]),
                    np.max([np.min(distance(points1, p2)) for p2 in points2])])
    
def point_hd95(points1, points2, percentile=95):
    'Returns the HD95 distance between two sets of points'
    if not len(points1) or not len(points2):
        return np.nan
    else:
        return max([np.percentile([np.min(distance(p1, points2)) for p1 in points1], percentile),
                    np.percentile([np.min(distance(points1, p2)) for p2 in points2], percentile)])
    
def point_abd(points1, points2, percentile=95):
    'Returns the average distance between two sets of points'
    if not len(points1) or not len(points2):
        return np.nan
    else:
        return max(
            np.mean([np.min(distance(p1, points2)) for p1 in points1]), 
            np.mean([np.min(distance(points1, p2)) for p2 in points2]) )

def transform_sitk(img, tfm, interpolator=sitk.sitkBSpline, extrapolate=False):
    'Conveniance funtion for applying sitk transformations'
    resampler= sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetReferenceImage(img)
    resampler.SetTransform(tfm)
    if extrapolate: resampler.UseNearestNeighborExtrapolatorOn()
    return resampler.Execute(img)

def MI_sitk(img1, img2, mask=None):
    'Compute Mattes Mutual Information'
    rm= sitk.ImageRegistrationMethod()
    rm.SetMetricAsMattesMutualInformation()
    if mask is not None: rm.SetMetricFixedMask(mask)
    #rm.SetMetricSamplingPercentage(1.)
    return rm.MetricEvaluate(img1, img2)

def CC_sitk(img1, img2, mask=None):
    'Compute cross-correlation coeficient'
    rm= sitk.ImageRegistrationMethod()
    rm.SetMetricAsCorrelation()
    if mask is not None: rm.SetMetricFixedMask(mask)
    #rm.SetMetricSamplingPercentage(1.)
    return rm.MetricEvaluate(img1, img2)


def DSC_sitk(*imgs):
    'Compute DSC coeficient'
    om= sitk.LabelOverlapMeasuresImageFilter()
    om.Execute(*[sitk.Cast(img, sitk.sitkUInt16) for img in imgs])
    return om.GetDiceCoefficient()

def surface_distance_metrics(mask1, mask2, fully_connected=False, P=95, fill_holes=False):
    '''
        Returns the Average Boundary Distance (ABD) and 
        95th percentile Haussdorf Distance (HD95) between provided masks
        Implemented following: https://promise12.grand-challenge.org/Details/
    '''
    #Fill holes in masks if requested
    #Hole artifacts appearing in one of the masks can lead to unfairly high distances
    if fill_holes:
        closing_filter= sitk.BinaryMorphologicalClosingImageFilter()
        closing_filter.SetKernelRadius(int(fill_holes))
        mask1= closing_filter.Execute(mask1)
        mask2= closing_filter.Execute(mask2)
    
    #Extract surfaces
    surf_filter= sitk.BinaryContourImageFilter()
    surf_filter.SetFullyConnected(fully_connected)
    surf1= surf_filter.Execute(mask1)
    surf2= surf_filter.Execute(mask2)
    
    #Compute distance maps
    dist_filter= sitk.DanielssonDistanceMapImageFilter()
    dist_filter.SetInputIsBinary(True)
    dist_filter.SetSquaredDistance(False)
    dist_filter.SetUseImageSpacing(True)
    dist1= dist_filter.Execute(surf1)
    dist2= dist_filter.Execute(surf2)
    
    #From here onwards, convert to numpy for simplicity
    #Mask distance maps with the opposite surfaces
    distances1= sitk.GetArrayFromImage(dist1)[sitk.GetArrayFromImage(surf2).astype(bool)].astype(np.float64)
    distances2= sitk.GetArrayFromImage(dist2)[sitk.GetArrayFromImage(surf1).astype(bool)].astype(np.float64)
    
    #Compute required metrics
    hd95= np.max([np.percentile(distances1, P), np.percentile(distances2, P)])
    abd= np.max([np.mean(distances1), np.mean(distances2)])
    
    return hd95, abd
        
def print_metric(a, *bs, metric= lambda a, b: np.nan, name=None, 
                 labels=None, combine=np.mean):
    'Plots the results of a metric before and after registration'
    results= [ metric(a, b) for b in bs ] 
    res0= combine(results[0])
    print(f" - {metric.__name__ if name is None else name}: ", end='')
    if labels is None or len(labels) != len(bs):
        labels= range(len(bs))
    for i, (resN, label) in enumerate(zip(results, labels)):
        res= combine(resN)
        print(f"{label}: {res:.4f}", end='')
        if i != 0:
            print(f"({(res-res0)/res0*100:.2f}%)", end='')
            
        if i == (len(results) -1):
            print('')
        else:
            print(', ', end='')
    return results

def dump(obj):
    '''
        Shows all attributes of an object
        Useful for exploring a new library
    '''
    for attr in dir(obj): print("obj.%s = %r" % (attr, getattr(obj, attr)))

def make_dirs(*dirs): 
    '''
        Makes all provided dirs if they did not exist
    '''
    for dir in dirs: 
        if not os.path.exists(dir): os.makedirs(dir)
    
def get_gradient_features(image, combine=True):
    '''
        Return the average gradient in x and y directions for a given `image`
    '''
    image_grad= sitk.GradientImageFilter().Execute(image)
    if combine:
        image_grad= 0.5* (sitk.VectorIndexSelectionCast(image_grad, 0) + \
                          sitk.VectorIndexSelectionCast(image_grad, 1) )
    return image_grad

def load_json(path, **kwargs):
    'Load json file'
    with open(path, "r") as f:
        data = json.load(f, **kwargs)
    return data

def save_json(data, path, indent: int = 4, **kwargs):
    'Save json file'
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, **kwargs)

def slice_wise_binary_smoothing(mask, radius=2, iterations=1):
    '''
        Applies binaty smoothing to a SITK binary image in slice-by-slice manner
        Radius is provided in mm
        
        Arguments
        ---------
        mask: sitk Image
            Mask to which smoothing will be applied
        radius: float, default 2
            Smoothing radius in mm
        iterations: int, default 1
            Number of smoothing iterations
    '''
    #Compute actual radius
    actual_radius= round(radius/mask.GetSpacing()[0])
    
    #Load required libs
    from skimage.morphology import disk
    from scipy.ndimage.morphology import binary_opening, binary_closing
    selem= disk(actual_radius)[np.newaxis]
        
    #Apply
    mask_arr= sitk.GetArrayFromImage(mask)
    #plot3(mask_arr)
    mask_arr= binary_opening(mask_arr, selem, iterations=iterations)
    #plot3(mask_arr)
    mask_arr= binary_closing(mask_arr, selem, iterations=iterations)
    #plot3(mask_arr)
    mask_new= sitk.GetImageFromArray(mask_arr.astype(np.uint8))
    mask_new.CopyInformation(mask)
    
    return mask_new

def rescale_intensity(image, thres=[0.1, 99.9]):
    '''
        Clips the intensity of an image and rescales it between 0 and 1 for better display
        
        Arguments
        ----------
        image: array
            Image to normalize
        thresh: tuple or list of two floats between 0 and 100
            Clip the image between the thresh[0]th and thresh[1]th percentiles
            
        Returns
        -------
        normalized_image: array
    '''
    val_l, val_h = np.percentile(image, thres)
    image[image < val_l] = val_l
    image[image > val_h] = val_h
    return (image.astype(np.float32) - val_l) / (val_h - val_l + 1e-6)

def make_grid(img):
    '''
        Creates an alternating grid of non-adjacent -1 and 1 tiles with the same
        properties as the input image.
    '''
    grid= np.zeros_like(sitk.GetArrayViewFromImage(img), dtype=np.float64)
    grid[:]= -1
    grid[::2]*= -1; grid[:,::2]*= -1; grid[:,:,::2]*= -1
    grid_sitk= sitk.GetImageFromArray(grid)
    grid_sitk.CopyInformation(img)
    return grid_sitk

def resample_image(img, size, spacing, mode='center', mask=None, return_as_dict=False,
                   img_interpolator=sitk.sitkBSpline, mask_interpolator=sitk.sitkLabelGaussian):
    '''
        Crop **and resample** an image to a given `size` (and optionally a mask too) around
        some central point that is computed according to `mode` (the center of the image by default)
        
        WARNING: All physical information of the image will be lost, but it can be recovered
        by setting return_as_dict=True and using that information with, `undo_resample_image`
        
        Parameters
        ----------
        img: SimpleITK Image
            The image to crop
        size: tuple of 3 ints
            The output size of the crop
        spacing: tuple of 3 ints
            The output spacing of the crop
        mode: str or (x, y, z) tuple, default 'center'
            One of {'roi', 'center'} or a custom phyisical position (x, y, z)
        mask: SimpleITK Image or None, default None
            The mask to crop. Used for computing the centroid of the ROI if `mode='roi'`
        return_as_dict: bool, default False
            Return the image AND a dictionary containing all the transformation information,
            which can be used with undo_resample_image to revert the transformation and gain
            back the lost physical information too!
        img_interpolator: SimpleITK Interpolator, default sitk.sitkBSpline
            Interpolator to be used with `img`
        label_interpolator: SimpleITK Interpolator, default sitk.sitkLabelGaussian
            Interpolator to be used with `mask`
    '''
    #If mask is provided, check that it is in the same space as the image
    if mask is not None:
        if not np.allclose(img.GetSpacing(), mask.GetSpacing()) or \
           not np.allclose(img.GetSize(), mask.GetSize()):
            print(f'Warning: Image and mask are in different frames of reference. ',
                  f'Size: image {img.GetSize()} vs mask {mask.GetSize()}. ',
                  f'Spacing: image {img.GetSpacing()} vs mask {mask.GetSpacing()}. ',
                  'Check the output to make sure it is correct.')
    
    #Save image properties
    original_spacing, original_size= img.GetSpacing(), img.GetSize()
    original_dir, original_origin= img.GetDirection(), img.GetOrigin()
    
    #Reset image properties
    img_copy= sitk.Image(img)
    img_copy.SetOrigin((0,)*3)
    img_copy.SetDirection(np.eye(3).flatten())
    if mask is not None:
        mask_copy= sitk.Image(mask)
        mask_copy.SetOrigin((0,)*3)
        mask_copy.SetDirection(np.eye(3).flatten())
        
    #Set up a shift to center the downscaled image
    if mode == 'center' or mask == None:
        offset= [ int((sz*sp-SZ*SP)/2) for SP, SZ, sp, sz in zip(
               original_spacing, original_size, spacing, size)]
    elif mode == 'roi':
        ma_centroid= (sitk.VectorIndexSelectionCast(mask_copy, 0) if mask_copy.GetNumberOfComponentsPerPixel() > 1 
                                                                  else mask_copy) > 0.5
        f= sitk.LabelShapeStatisticsImageFilter()
        f.Execute(ma_centroid)
        centroid= f.GetCentroid(1)
        offset= - np.array(centroid) + np.array(size)*np.array(spacing)/2
    else:
        offset= np.array(mode)
        
    #Build final transform
    translation= sitk.TranslationTransform(3, offset)
    
    #Reample
    img_out= sitk.Resample(img_copy, size, outputSpacing=spacing, interpolator=img_interpolator,
                           transform=translation.GetInverse())
    if mask is not None:
        mask_out= sitk.Resample(mask_copy, size, outputSpacing=spacing, interpolator=mask_interpolator,
                                transform=translation.GetInverse())
    
    #Return
    if not return_as_dict:
        if mask is None:  return img_out
        else:             return img_out, mask_out
    else:
        info_dict= {'original_spacing': original_spacing, 'transform': translation,
                'original_size': original_size, 'original_direction': original_dir,
                'original_origin': original_origin, 'spacing':spacing}
        if mask is None:  return img_out, info_dict
        else:             return img_out, mask_out, info_dict
    
def undo_resample_image(img, spacing, original_size, original_spacing, original_origin,
                        original_direction, transform, interpolator=sitk.sitkBSpline):
    '''
        Applies the opposite operation to resample_image using the parameteres given by 
        the dictionary returned by resample_image(..., return_as_dict=True)
    '''    
    #Reample
    img.SetSpacing(spacing)
    img_out= sitk.Resample(img, original_size, outputSpacing=original_spacing, 
                           interpolator=interpolator, transform=transform)
    img_out.SetOrigin(original_origin)
    img_out.SetDirection(original_direction)
    
    return img_out

def crop_image(img, mask=None, size=(150,150,20), mode='center'):
    '''
        Crop an image to a given `size` (and optionally a mask too) around some central point
        that is computed according to `mode` (the center of the image by default)
        
        Parameters
        ---------
        img: SimpleITK Image
            The image to crop
        mask: SimpleITK Image or None
            The mask to crop. Used for computing the centroid of the ROI if `mode='roi'`
        size: tuple of 3 ints
            The output size of the crop
        mode: str or (x, y, z) tuple, default 'center'
            One of {'roi', 'center'} or a custom phyisical position (x, y, z)
    '''
    # assert mask is None or np.allclose(img.GetSpacing(), mask.GetSpacing()) and \
    #                        np.allclose(img.GetSize(), mask.GetSize()), \
    #     'Image and mask cannot be in different frames of reference'
    
    #Get some useful properties
    spacing_orig, size_orig, origin= img.GetSpacing(), img.GetSize(), img.GetOrigin()
        
    #Set up a shift to center the downscaled image
    #We will work here in image coordinates
    if mode == 'center' or mask == None:
        centroid= img.TransformContinuousIndexToPhysicalPoint(np.array(size_orig)/2)
    elif mode == 'roi':
        ma_centroid= (sitk.VectorIndexSelectionCast(mask, 0) if mask.GetNumberOfComponentsPerPixel() > 1 else mask) > 0.5
        f= sitk.LabelShapeStatisticsImageFilter()
        f.Execute(ma_centroid)
        centroid= f.GetCentroid(1)
    else:
        centroid= np.array(mode)
                
    #We will now work in voxel coordinates
    #Get center voxel position and prepare slicing indices
    #Also, add constant pad to the image beforehand before slicing to make sure we don't over-slice
    tr= img.TransformPhysicalPointToContinuousIndex(centroid)
    pad= [int(n) for n in 1 + np.array(size)/2]
    slices= [slice( np.round(tr[d]-size[d]/2+pad[d],0).astype(int), 
                    np.round(tr[d]+size[d]/2+pad[d],0).astype(int) ) for d in range(3)]
    
    #Get center crop
    img_tf= sitk.ConstantPad(img, pad, pad)[slices[0], slices[1], slices[2]]
    mask_tf= sitk.ConstantPad(mask, pad, pad)[slices[0], slices[1], slices[2]] if mask is not None else None
    
    return img_tf, mask_tf, np.array(centroid)

def read_dicom(path: str, extension: str='dcm', axis: str='z', 
               content_tag: Optional[Union[tuple, Callable[[str], str]]]=(0x8, 0x33)):
    '''
        Read (possibly) multiple images from a single DICOM directory.
        Indpendent images will be indentified by using the `content_tag` DICOM tag (see description below).
        Slice ordering will be decied according to the Image Position Patient `axis`-value DICOM tag.
        
        Paramters
        ---------
        path: str
            Path to de DICOM directory
        extension: str, default 'dcm'
            Extension of the DICOM slices
        axis: str or None, deafult 'z'
            Axis along which to read the image
            Set to None to use Slice Location instead (this may not always exist)
        content_tag: tuple, callable, or None, default (0x8, 0x33)
            DICOM tag to use for identifying different images within a DICOM directory.
            Some examples:
             - (0x8, 0x33) Content Time (default, useful for several MRI sequences)
             - (0x18, 0x9087) Diffusion b-value (useful for MRI b-values, but usually unused)
             - (0x20, 0x12) Acquisition Number (Useful for some MRI DCE)
             - (0x20, 0x100) Temporal Position Identifier (Useful for some MRI DCE)
            It can also be a callable that takes the slice name and returns the content_id
            An example:
             - lambda path: os.path.basename(path).split('-')[0]
            Set to None to read all slices as a single image. 
             
        Returns
        -------
        images: dict
            Dictionary with all images (SimpleITK Image instances).
            Keys are all the different `content_tag` values, and the values are the actual images
    '''
    import pydicom
    reader = sitk.ImageSeriesReader()
    file_names, positions, slice_spacings, images= defaultdict(list), defaultdict(list), {}, {}
    axis_id= {'x':0, 'y':1, 'z':2}[axis] if axis is not None else -1
    
    def get_content_id(dcm, file):
        if content_tag is None:
            return 'default'
        elif isinstance(content_tag, tuple):
            if content_tag in dcm:
                return str(dcm[content_tag].value)
            else:
                return 'default'
        else:
            return content_tag(file)

    #The dicom slices must be explored to set the correct order
    for i, file in enumerate([f for f in glob.glob(os.path.join(path,'*.%s'%extension))]):
        try:
            dcm= pydicom.dcmread(file)
        except Exception as e:
            print('\t\t- Error reading %s: %s'%(os.path.split(file)[-1], str(e)))
            continue
        
        #For debugging
#         print('%03d'%i, 'Slice location:', dcm[(0x20, 0x1041)].value , 
#                         '| Instance number:', dcm[(0x20, 0x13)].value, 
#                         '| Image position z-value:', float(dcm[(0x20, 0x32)].value[{'x':0, 'y':1, 'z':2}[axis]]) , 
#                         '| Patient position:', dcm[(0x18, 0x5100)].value)
        
        #Read required information from each slice and store it
        if axis_id != -1:
            position= float(dcm[(0x20, 0x32)].value[axis_id]) #Image Position Patient z-value
        else:
            position= float(dcm[((0x20, 0x1041))].value)
        content_id= get_content_id(dcm, file) #Content identifier
        sbs= dcm[(0x18, 0x88)].value if (0x18, 0x88) in dcm else None #Spacing between slices

        #Save info
        file_names[content_id].append((position, file)) 
        slice_spacings[content_id]= sbs
        
    #Read each of the detected images (according to content_id)
    for cid, slices in file_names.items():
        #Sort all image lists by sl and read them
        file_names_sorted= [name for pos, name in sorted(slices, key=lambda tup: tup[0], reverse=False)]
        reader.SetFileNames(file_names_sorted)
        images[cid]= reader.Execute()

        #Print some warnings if inter-slice spacing is messed up or if there are too few slices
        if sbs is not None and abs(images[cid].GetSpacing()[-1] - slice_spacings[cid]) > 0.01:
            print('Warning: Detected spacing (%.2f) is different from actual spacing (%.2f) for id: %s'%(
                    images[cid].GetSpacing()[-1], slice_spacings[cid], cid))
        if images[cid].GetSize()[-1] < 10:
            print('Warning: Few slices (%d)'%images[cid].GetSize()[-1])

    keys= list(images.keys())
    return images #if len(keys) > 1 else images[keys[0]]

def detect_border(mask, threshold=3):
    '''
        Detects if a mask is at (using threshold) or near the border of the image
        returns: a binary vector indicating if the mask touches the border in:
        [ax0down, ax1down, ax2down, ax0up, ax1up, ax2up]
        
        Parameters
        ----------
        mask: SimpleITK image
        threshold: int, default 3
            Detect border intersection if mask is within at least 3 voxels of the border
            
        Returns
        -------
        out: np.array
            A binary vector indicating if the mask touches the border in:
            [ax0down, ax1down, ax2down, ax0up, ax1up, ax2up]
    '''
    indices= np.argwhere(mask > 0.5)
    
    border_max= indices > np.stack(
            [np.array(mask.shape)-2-threshold]*indices.shape[0], axis=0)
    border_min= indices < (np.ones(indices.shape) + threshold)
    
    final_border_min= np.any(border_min, axis=0)
    final_border_max= np.any(border_max, axis=0)
    
    return np.concatenate([final_border_min, final_border_max])

def get_minimum_slices_transform(img, min_slices):
    '''
        Adds extra slices to either side of the z-axis, so that the
        resulting image is centered, keeps physical information AND
        has at least `min_slices` slices
        
        Parameters
        ----------
        img: SimpleITK Image
        min_slices: int
        
        Returns
        -------
        out_size: tuple of ints
        tfm: SimpleITK Transform
    '''
    z_shift= -(min_slices - img.GetSize()[2]) * img.GetSpacing()[2]/2
    out_size= (*img.GetSize()[:2], min_slices)
    tfm= sitk.Similarity3DTransform()
    tfm.SetTranslation([0, 0, z_shift])
    return out_size, tfm

def register(fixed_image, moving_image, mode='spline', fixed_image_mask=None, lr=200,
             show_progress=False, verbose=True, optimizer='bfgs', metric='mattes',
             iters=200, control_point_separation=80., scale_factors=[1,2]):
    '''
        Register `moving_image` to `fixed_image` using variety of methods, metrics and
        configurations. Default values should be fine for many problems. Please make
        sure to first convert both images to the same size and spacing, using PatientResampler
        for instance.
        
        Parameters
        ----------
        fixed_image: SimpleITK Image
            Reference image
        moving_image: SimpleITK Image
            Image that will be transformed to match `fixed_image`
        mode: str, default 'spline'
            One of: 'spline', 'rigid'
        fixed_image_mask: SimpleITK Image or None, default None
            Mask (of same size as the rest of the images) with voxels to consider for registration
            set to 1, and the rest set to 0, or None to not apply any mask
        lr: float, default 200
            Learning rate of the optimization algorithm
        show_progress: bool, default False
            Show a convergence plot during training
        verbose: bool, default True
            Show some information after training
        optimizer: str, dfault 'bfgs'
            One of: 'bfgs', 'sgd', 'linesgd'
        metric: str, default 'mattes'
            One of: 'mattes', 'mi', 'corr', 'mse'
        iters: int, default 100
            Maximum number of iterations per scale
        control_point_separation: float, default 40.
            Distance between spline control points.
        scale_factors: list of ints, default [1,2]
            Number of refinement levels. At each level, the number of control points is multiplied
            by each scale factor. Also, the registration is peformed at a factor that is determined
            by reading scale_factors in reverse.
        
        Returns
        -------
        initial_transform: SimpleITK Transform
        metric_value: float
    '''
    l_mult= lambda l,k: [i*k for i in l]
    
    fixed_image= sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image= sitk.Cast(moving_image, sitk.sitkFloat32)
    
    R = sitk.ImageRegistrationMethod()
    registration_tracker= RegistrationTracker()
    
    #Set registration mode
    if mode == 'spline':
        # Determine the number of BSpline control points using the physical spacing we 
        # want for the finest resolution control grid. 
        grid_physical_spacing = [control_point_separation]*3 # A control point every 50mm
        image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
        mesh_size = [int(image_size/grid_spacing + 0.5) \
                     for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]

        # The starting mesh size will be 1/4 of the original, it will be refined by 
        # the multi-resolution framework.
        mesh_size = [max(1, int(sz/4 + 0.5)) for sz in mesh_size]
        initial_transform = sitk.BSplineTransformInitializer(image1 = fixed_image, 
                                                             transformDomainMeshSize = mesh_size, order=3)    
        # Instead of the standard SetInitialTransform we use the BSpline specific method which also
        # accepts the scaleFactors parameter to refine the BSpline mesh. In this case we start with 
        # the given mesh_size at the highest pyramid level then we double it in the next lower level and
        # in the full resolution image we use a mesh that is four times the original size.
        R.SetInitialTransformAsBSpline(initial_transform,
                                                         inPlace=True,
                                                         scaleFactors=scale_factors)
            
        # Multi-resolution framework.            
        shrink_factors= l_mult(scale_factors[::-1],2)
        R.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
        R.SetSmoothingSigmasPerLevel(smoothingSigmas=shrink_factors)
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    elif mode == 'rigid':
        #A reasonable guesstimate for the initial translational alignment can be obtained by using 
        #the CenteredTransformInitializer (functional interface to the CenteredTransformInitializerFilter).

        #The resulting transformation is centered with respect to the fixed image and the translation 
        #aligns the centers of the two images. There are two options for defining the centers of the images, 
        #either the physical centers of the two data sets (GEOMETRY), or the centers defined by the intensity moments (MOMENTS).

        #Two things to note about this filter, it requires the fixed and moving image have the same type 
        #even though it is not algorithmically required, and its return type is the generic SimpleITK.Transform.
        initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(fixed_image,moving_image.GetPixelID()), 
                                                              moving_image, 
                                                              sitk.Euler3DTransform(), 
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
        #Set the initial moving and optimized transforms.
        R.SetInitialTransform(initial_transform)
    else:
        raise ValueError('`mode` must be either `spline` or `rigid`')
    
    #Set metric
    if metric.lower() == 'mse':
        R.SetMetricAsMeanSquares()
    elif metric.lower() == 'corr':
        R.SetMetricAsCorrelation()
    elif metric.lower() == 'mattes':
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    elif metric.lower() == 'mi':
        R.SetMetricAsJointHistogramMutualInformation()
    else:
        raise ValueError('Unknown metric: %s'%metric)
        
    # Settings for metric sampling, usage of a mask is optional. When given a mask the sample points will be 
    # generated inside that region. Also, this implicitly speeds things up as the mask is smaller than the
    # whole image.
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.1)
    if fixed_image_mask:
        R.SetMetricFixedMask(fixed_image_mask)

    R.SetInterpolator(sitk.sitkLinear)
    
    #Set optimizer
    # Use the LBFGS2 instead of LBFGS. The latter cannot adapt to the changing control grid resolution.
    if optimizer.lower() == 'bfgs':
        R.SetOptimizerAsLBFGS2(solutionAccuracy=1e-6, numberOfIterations=iters, deltaConvergenceTolerance=0.001)
    elif optimizer.lower() == 'sgd':
        R.SetOptimizerAsGradientDescent(learningRate=lr, numberOfIterations=iters,
                                                          estimateLearningRate=R.Never)
    elif optimizer.lower() == 'linesgd':
        R.SetOptimizerAsGradientDescentLineSearch(learningRate=lr, numberOfIterations=iters)
    else:
        raise ValueError('Unknown optimizer: %s'%optimizer)
        
    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.
    #Add event handlers
    if show_progress:
        R.AddCommand(sitk.sitkStartEvent, registration_tracker.start_plot)
        R.AddCommand(sitk.sitkEndEvent, registration_tracker.end_plot)
        R.AddCommand(sitk.sitkMultiResolutionIterationEvent, 
                                       registration_tracker.update_multires_iterations) 
        R.AddCommand(sitk.sitkIterationEvent, 
                                       lambda: registration_tracker.plot_values(R))
    
    R.Execute(fixed_image, moving_image)
    
    #Resample
#     moving_resampled = sitk.Resample(moving_image, fixed_image, initial_transform, 
#                                      sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    
    if verbose:
        print('Optimizer\'s stopping condition, {0}'.format(R.GetOptimizerStopConditionDescription()))
        print('Final metric value: {0}'.format(R.GetMetricValue()))
        if mode == 'spline': print('Used a mesh size of:', mesh_size)
    
    return initial_transform, R.GetMetricValue()

try:
    from IPython.display import clear_output
except:
    pass
class RegistrationTracker():
    def __init__(self):
        self.start_plot()
    
    def start_plot(self):
        self.metric_values = []
        self.multires_iterations = []

    def end_plot(self):
        # Close figure, we don't want to get a duplicate of the plot latter on.
        plt.close()

    def plot_values(self, R):
        '''
             Callback invoked when the IterationEvent happens, update our data and display new figure.
             
             Parameters
             ----------
             R: Object
                 Any objetc with a method GetMetricValue() that reurtuns a float
        '''
        self.metric_values.append(R.GetMetricValue())                                       
        # Clear the output area (wait=True, to reduce flickering), and plot current data
        clear_output(wait=True)
        # Plot the similarity metric values
        plt.plot(self.metric_values, 'r')
        plt.plot(self.multires_iterations, [self.metric_values[i] for i in self.multires_iterations], 'b*')
        plt.xlabel('Iteration Number',fontsize=12)
        plt.ylabel('Metric Value',fontsize=12)
        plt.show()

    def update_multires_iterations(self):
        self.multires_iterations.append(len(self.metric_values))