import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
import numbers

""" Spatial transformer layers and other useful layers """

def get_conv(ndims=3):
    if ndims == 1:   return nn.Conv1d
    elif ndims == 2: return nn.Conv2d
    elif ndims == 3: return nn.Conv3d
    else: raise RuntimeError('Only 1, 2, and 3 dimensions are supported. Received %d'%ndims)

def get_fconv(dim=3):
    if dim == 1:   return nnf.conv1d
    elif dim == 2: return nnf.conv2d
    elif dim == 3: return nnf.conv3d
    else: raise RuntimeError('Only 1, 2, and 3 dimensions are supported. Received %d'%dim)

class NonLearnableConv(nn.Module):
    def __init__(self, channels, kernel_size=3, dim=3, sigma=1, mode='dilation', padding='same'):
        '''
            Apply a non-learnable filter to a feature map
            
            Parameters
            ----------
            channels: int
                Number of input channels
            kernel_size: int, default 3
                Kernel size. If mode == 'sobel', only 3 is supported
            dim: int, default 3
                Dimensionality of the input data, one of {1,2,3}
            sigma: float, default 1
                Std of the gaussian if mode == 'gaussian'
            mode: str, default 'dilation'
                One of 'dilation', 'mean', 'gaussian', 'sobel'
            padding: str, default 'same'
                Padding mode. See Pytorch documentation on convolutions
        '''
        super().__init__()
        
        #Extend dimensions of kernel_size and sigma
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        
        #Create kernel
        if mode in ['dilation', 'mean']:
            kernel= torch.ones(kernel_size)
        elif mode in ['gaussian']:
            # The gaussian kernel is the product of the gaussian function of each dimension.
            kernel = 1
            meshgrids = torch.meshgrid([ torch.arange(size, dtype=torch.float32)
                                         for size in kernel_size ])
            for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
                mean = (size - 1) / 2
                kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                          torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        elif mode in ['sobel']:
            sobel_x = torch.tensor( [[
                                     [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                     [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                                     [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
                                    ]], dtype=torch.float)
            kernel= torch.stack([sobel_x, sobel_x.permute(0, 1, 3, 2), sobel_x.permute(0, 3, 1, 2)], axis=0)
            #kernel= torch.repeat_interleave(kernel, channels, 1) #Not needed if self.groups= channels
            assert dim == 3 and kernel_size[0]==3, f'Not implemented {dim=}, {kernel_size}'
        else:
            raise ValueError('Unrecognized mode: %s'%mode)
            
        # Make sure sum of values in gaussian kernel equals 1.
        if mode in ['mean', 'gaussian']:
            kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        if mode in ['dilation', 'mean', 'gaussian']:
            kernel = kernel.view(1, 1, *kernel.size())
            kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.mode= mode
        self.padding= padding
        self.conv= get_fconv(dim)

    def forward(self, input):
        out= self.conv(input, weight=self.weight, groups=self.groups, padding=self.padding)
        return out > 0 if self.mode in ['dilation'] else out

class Identity(nn.Module):
    def __init__(self, ndims=3):
        """
		Creates an identity transform
        """
        super().__init__()
        self.ndims= ndims

    def forward(self, flow):
        # create identity grid
        size = flow.shape[2:]
        vectors = [
            torch.arange(0, s, dtype=flow.dtype, device=flow.device) for s in size
        ]
        grids = torch.meshgrid(vectors)
        identity = torch.stack(grids)  # z, y, x
        identity = identity.expand(
            flow.shape[0], *[-1] * (self.ndims + 1)
        )  # add batch
        return identity


class FlowComposition(nn.Module):
    """
    A flow composer, composing two flows /transformations / displacement fields.
    """

    def __init__(self):
        """
        instantiates the FlowComposition
        """
        super().__init__()
        self.transformer = SpatialTransformer()

    def forward(self, *args):
        """
        compose any number of flows
        
        Parameters:
            *args: flows, in order from left to right
        """
        if len(args) == 0:
            raise Exception("Can not compose 0 flows")
        elif len(args) == 1:
            return args[0]
        else:
            composition = self.compose(args[0], args[1])
            return self.forward(composition, *args[2:])

    def compose(self, flow0, flow1):
        """
        compose the flows
        
        Parameters:
            flow0: the first flow
            flow1: the next flow
        """
        return flow0 + self.transformer(flow1, flow0)


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, mode="bilinear"):
        """
        Instantiates the spatial transformer. 
        A spatial transformer transforms a src image with a flow of displacement vectors.
        
        Parameters:
            mode: interpolation mode
        """
        super().__init__()
        self.identity = Identity()
        self.grid_sampler = GridSampler(mode=mode)

    def forward(self, src, flow, mode=None, padding_mode="border"):
        """
        Transforms the src with the flow 
        Parameters:
            src: Tensor (B x C x D x H x W)
            flow: Tensor  (B x C x D x H x W) of displacement vextors. Channel 0 indicates the flow in the depth dimension.
            mode: interpolation mode. If not specified, take mode from init function
            padding_mode: 'zeros', 'boarder', 'reflection'
        """

        # map from displacement vectors to absolute coordinates
        coordinates = self.identity(flow) + flow
        return self.grid_sampler(src, coordinates, mode=mode, padding_mode=padding_mode)


class AffineSpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer for affine input
    """

    def __init__(self, mode="bilinear", ndims=3):
        """
        Instantiates the spatial transformer. 
        A spatial transformer transforms a src image with a flow of displacement vectors.
        
        Parameters:
            mode: interpolation mode
        """
        super().__init__()
        self.identity = Identity()
        self.grid_sampler = GridSampler(mode=mode)
        self.ndims = ndims

    def forward(self, src, affine, mode=None, padding_mode="border"):
        """
        Transforms the src with the flow 
        Parameters:
            src: Tensor (B x C x D x H x W)
            affine: Tensor  (B x 4 x 4) the affine transformation matrix
            mode: interpolation mode. If not specified, take mode from init function
            padding_mode: 'zeros', 'boarder', 'reflection'
        """
        coordinates = self.identity(src)

        # add homogenous coordinate
        coordinates = torch.cat(
            (
                coordinates,
                torch.ones(
                    coordinates.shape[0],
                    1,
                    *coordinates.shape[2:],
                    device=coordinates.device,
                    dtype=coordinates.dtype
                ),
            ),
            dim=1,
        )

        # center the coordinate grid, so that rotation happens around the center of the domain
        size = coordinates.shape[2:]
        for i in range(self.ndims):
            coordinates[:, i] -= size[i] / 2.0

        # permute for batched matric multiplication
        coordinates = (
            coordinates.permute(0, 2, 3, 4, 1)
            if self.ndims == 3
            else coordinates.permute(0, 2, 3, 1)
        )
        # we need to do this for each member of the batch separately
        for i in range(len(coordinates)):
            coordinates[i] = torch.matmul(coordinates[i], affine[i])
        coordinates = (
            coordinates.permute(0, -1, 1, 2, 3)
            if self.ndims == 3
            else coordinates.permute(0, -1, 1, 2)
        )
        # de-homogenize
        coordinates = coordinates[:, : self.ndims]

        # un-center the coordinate grid
        for i in range(self.ndims):
            coordinates[:, i] += size[i] / 2

        return self.grid_sampler(src, coordinates, mode=mode, padding_mode=padding_mode)


class GridSampler(nn.Module):
    """
    A simple Grid sample operation
    """

    def __init__(self, mode="bilinear", ndims=3):
        """
        Instantiates the grid sampler.
        The grid sampler samples a grid of values at coordinates.
        
        Parameters:
            mode: interpolation mode
        """
        super().__init__()
        self.mode = mode
        self.ndims = ndims

    def forward(self, values, coordinates, mode=None, padding_mode="border"):
        """
        Transforms the src with the flow 
        Parameters:
            src: Tensor (B x C x D x H x W)
            flow: Tensor  (B x C x D x H x W) of displacement vectors. Channel 0 indicates the flow in the depth dimension.
            mode: interpolation mode. If not specified, take mode from init function
            padding_mode: 'zeros', 'boarder', 'reflection'
        """
        mode = mode if mode else self.mode

        # clone the coordinate field as we will modift it.
        coordinates = coordinates.clone()
        # normalize coordinates to be within [-1..1]
        size = values.shape[2:]
        for i in range(len(size)):
            coordinates[:, i, ...] = 2 * (coordinates[:, i, ...] / (size[i] - 1) - 0.5)

        # put coordinate channels in last position and
        # reverse channels (in-build pytorch function indexes axis D x H x W and pixel coordinates z,y,x)
        if self.ndims == 2:
            coordinates = coordinates.permute(0, 2, 3, 1)
            coordinates = coordinates[..., [1, 0]]
        elif self.ndims == 3:
            coordinates = coordinates.permute(0, 2, 3, 4, 1)
            coordinates = coordinates[..., [2, 1, 0]]

        # sample
        return nnf.grid_sample(
            values,
            coordinates,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=True,  # align = True is nessesary to behave similar to indexing the transformation.
        )

class FlowIntegration(nn.Module):
    """
    Integrates a displacement vector field via scaling and squaring.
    """

    def __init__(self, nsteps, downsize=1):
        """ 
        Parameters:
            nsteps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
        """
        super().__init__()

        assert nsteps >= 0, "nsteps should be >= 0, found: %d" % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer()

        # configure optional resize layers
        resize = downsize > 1
        self.resize = ResizeTransform(downsize) if resize else None
        self.fullsize = ResizeTransform(1.0 / downsize) if resize else None

    def forward(self, flow):
        # resize
        if self.resize:
            flow = self.resize(flow)

        # scaling ...
        flow = flow * self.scale

        # and squaring ...
        for _ in range(self.nsteps):
            flow = flow + self.transformer(flow, flow)

        # resize back to full size
        if self.fullsize:
            flow = self.fullsize(flow)
        return flow


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize):
        super().__init__()
        self.factor = 1.0 / vel_resize

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, scale_factor=self.factor)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, scale_factor=self.factor)

        # don't do anything if resize is 1
        return x
        
class FlowPredictor(nn.Module):
    """
    A layer intended for flow prediction. Initialied with small weights for faster training.
    """

    def __init__(self, in_channels, ndims=3):
        super().__init__()
        """
        instantiates the flow prediction layer.
        
        Parameters:
            in_channels: input channels
        """
        # configure cnn
        self.cnn = nn.Sequential(
            get_conv(ndims=ndims)(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            get_conv(ndims=ndims)(in_channels, in_channels, kernel_size=3, padding=1),
        )

        # init final cnn layer with small weights and bias
        self.cnn[-1].weight = nn.Parameter(torch.normal(0, 1e-5, self.cnn[-1].weight.shape))
        self.cnn[-1].bias = nn.Parameter(torch.zeros(self.cnn[-1].bias.shape))

    def forward(self, x):
        """
        predicts the transformation. 
        
        Parameters:
            x: the input
            
        Return:
            pos_flow, neg_flow: the positive and negative flow
        """
        # predict the flow
        return self.cnn(x)