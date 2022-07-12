import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

class NCC(nn.Module):
    """
    Local (over window) normalized cross correlation loss. Normalized to window [0,1], with 0 being perfect match.

    We follow the NCC definition from the paper "VoxelMorph: A Learning Framework for Deformable Medical Image Registration",
    which implements it via the coefficient of determination (R2 score). 
    This is strictly the squared normalized cross-correlation, or squared cosine similarity.

    NCC over two image pacthes I, J of size N is calculated as
    NCC(I, J) = 1/N * [sum_n=1^N (I_n - mean(I)) * (J_n - mean(J))]^2 / [var(I) * var(J)]

    The output is rescaled to the interval [0..1], best match at 0.

    """

    def __init__(self, window=5, ndims=3):
        super().__init__()
        self.win = window
        self.ndims= ndims

    def forward(self, y_true, y_pred):
        def compute_local_sums(I, J):
            # calculate squared images
            I2 = I * I
            J2 = J * J
            IJ = I * J

            # take sums
            I_sum = conv_fn(I, filt, stride=stride, padding=padding)
            J_sum = conv_fn(J, filt, stride=stride, padding=padding)
            I2_sum = conv_fn(I2, filt, stride=stride, padding=padding)
            J2_sum = conv_fn(J2, filt, stride=stride, padding=padding)
            IJ_sum = conv_fn(IJ, filt, stride=stride, padding=padding)

            # take means
            win_size = np.prod(filt.shape)
            u_I = I_sum / win_size
            u_J = J_sum / win_size

            # calculate cross corr components
            cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
            I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
            J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

            return I_var, J_var, cross

        # get dimension of volume
        channels = y_true.shape[1]

        # set filter
        filt = torch.ones(channels, channels, *([self.win] * self.ndims)).type_as(y_true)

        # get convolution function
        conv_fn = getattr(F, "conv%dd" % self.ndims)
        stride = 1
        padding = self.win // 2

        # calculate cc
        var0, var1, cross = compute_local_sums(y_true, y_pred)
        cc = cross * cross / (var0 * var1 + 1e-5)

        # mean and invert for minimization
        return 1-torch.mean(cc)


class MaskedNCC(nn.Module):
    """
    Masked Normalized Cross-coralation. 
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.ncc = NCC(**kwargs)

    def forward(self, y_true, y_pred, mask):
        # we first null out any areas outside of the masks, so that the barders are the same
        y_true = y_true * mask
        y_pred = y_pred * mask
        # perform ncc
        return self.ncc(y_true, y_pred)


class MSE(nn.Module):
    """
    Mean squared error loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class MaskedMSE(nn.Module):
    """
    Masked Mean squared error. 
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, mask):
        sq_error = (y_true - y_pred) ** 2
        masked_sq_error = sq_error * mask
        return torch.mean(masked_sq_error)


class GradNorm(nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty="l2", reduction="mean", ndims=3):
        super().__init__()
        self.penalty = penalty
        self.ndims = ndims
        self.reduction = reduction

    def forward(self, flow):
        # pad flow
        flow = F.pad(flow, [0, 1] * self.ndims, mode="replicate")
        # get finite differences
        if self.ndims == 2:
            dy = torch.abs(flow[:, :, 1:, :-1] - flow[:, :, :-1, :-1])
            dx = torch.abs(flow[:, :, :-1, 1:] - flow[:, :, :-1, :-1])
        elif self.ndims == 3:
            dz = torch.abs(flow[:, :, 1:, :-1, :-1] - flow[:, :, :-1, :-1, :-1])
            dy = torch.abs(flow[:, :, :-1, 1:, :-1] - flow[:, :, :-1, :-1, :-1])
            dx = torch.abs(flow[:, :, :-1, :-1, 1:] - flow[:, :, :-1, :-1, :-1])

        # square
        if self.penalty == "l2":
            dx = dx ** 2
            dy = dy ** 2
            if self.ndims == 3:
                dz = dz ** 2

        d = dx + dy + (dz if self.ndims == 3 else 0)
        d /= self.ndims
        if self.reduction == "none":
            # mean over channels. Keep spatial dimensions
            return torch.mean(d, dim=1, keepdim=True)
        elif self.reduction == "mean":
            return torch.mean(d)
        elif self.reduction == "sum":
            return torch.sum(d)


class JacobianDeterminant(nn.Module):
    def __init__(self, reduction="mean", preserve_size=False, ndims=3):
        super().__init__()
        self.idty = torchreg.nn.Identity()
        self.reduction = reduction
        self.ndims = ndims
        self.preserve_size = preserve_size

    def forward(self, flow):
        """
        calculates the area of each pixel after the flow is applied
        """

        def determinant_2d(x, y):
            return x[:, [0]] * y[:, [1]] - x[:, [1]] * y[:, [0]]
        def determinant_3d(x, y, z):
            return (x[:, [0]] * y[:, [1]] * z[:, [2]] +
                    x[:, [2]] * y[:, [0]] * z[:, [1]] +
                    x[:, [1]] * y[:, [2]] * z[:, [0]] -
                    x[:, [2]] * y[:, [1]] * z[:, [0]] -
                    x[:, [1]] * y[:, [0]] * z[:, [2]] -
                    x[:, [0]] * y[:, [2]] * z[:, [1]])
        
        if self.preserve_size:
            flow = F.pad(flow, [0, 1] * self.ndims, mode="replicate")

        # map to target domain
        transform = flow + self.idty(flow)
        
        # get finite differences
        if self.ndims == 2:
            dx = torch.abs(transform[:, :, 1:, :-1] - transform[:, :, :-1, :-1])
            dy = torch.abs(transform[:, :, :-1, 1:] - transform[:, :, :-1, :-1])
            jacdet = determinant_2d(dx, dy)
        elif self.ndims == 3:
            dx = torch.abs(transform[:, :, 1:, :-1, :-1] - transform[:, :, :-1, :-1, :-1])
            dy = torch.abs(transform[:, :, :-1, 1:, :-1] - transform[:, :, :-1, :-1, :-1])
            dz = torch.abs(transform[:, :, :-1, :-1, 1:] - transform[:, :, :-1, :-1, :-1])
            jacdet = determinant_3d(dx, dy, dz)
            
        if self.reduction == "none":
            return jacdet
        elif self.reduction == "mean":
            return torch.mean(jacdet)
        elif self.reduction == "sum":
            return torch.sum(jacdet)


class DiceOverlap(nn.Module):
    def __init__(self, classes, mean_over_classes=True, ignore_zero_class=False):
        """
        calculates the mean dice overlap of the given classes
        This loss metric is not suitable for training. Use the SoftdiceOverlap for training instead.
        Parameters:
            classes: list of classes to consider, e.g. [0, 1]
            mean_over_classes: should the mean dive overlap over the classes be returned, or one value per class? Default True.
        """
        super().__init__()
        self.classes = classes
        self.mean_over_classes = mean_over_classes
        self.ignore_zero_class = ignore_zero_class

    def cast_to_int(self, t):
        if t.dtype == torch.float:
            return t.round().int()
        else:
            return t

    def forward(self, y_true, y_pred):
        y_true = self.cast_to_int(y_true)
        y_pred = self.cast_to_int(y_pred)
        dice_overlaps = []
        for label in self.classes:
            mask0 = y_true == label
            mask1 = y_pred == label
            intersection = (mask0 * mask1).sum()
            union = mask0.sum() + mask1.sum()
            dice_overlap = 2.0 * intersection / (union + 1e-6)
            dice_overlaps.append(dice_overlap)

        dice_overlaps = torch.stack(dice_overlaps)
        dice_overlaps = dice_overlaps[1:] if self.ignore_zero_class else dice_overlaps
        return dice_overlaps.mean() if self.mean_over_classes else dice_overlaps


class SoftDiceOverlap(nn.Module):
    def __init__(self, ignore_zero_class=False, ndims=3):
        """
        calculates the mean soft dice overlap of one-hot encoded feature maps.
        This loss metric is suitable for training
        """
        super().__init__()
        self.ignore_zero_class= ignore_zero_class
        self.ndims= ndims

    def forward(self, y_true, y_pred):
        # calculate union
        union = y_true * y_pred

        # sum over B, D, H, W
        sum_dims = [0, 2, 3] if self.ndims == 2 else [0, 2, 3, 4]
        s_union = torch.sum(union, dim=sum_dims)
        s_y_true = torch.sum(y_true, dim=sum_dims)
        s_y_pred = torch.sum(y_pred, dim=sum_dims)

        # calculate dice per class, mean over classes
        dice = 2 * s_union / (s_y_true + s_y_pred + 1e-6)
        return torch.mean(dice[1:] if self.ignore_zero_class else dice)
    
class MaskedSoftDiceOverlap(nn.Module):
    def __init__(self, ignore_zero_class=False):
        """
        calculates the mean soft dice overlap of one-hot encoded feature maps.
        This loss metric is suitable for training
        """
        self.ignore_zero_class= ignore_zero_class
        super().__init__()

    def forward(self, y_true, y_pred, mask):
        # calculate union
        union = y_true * y_pred

        # sum over B, D, H, W
        sum_dims = [0, 2, 3] if self.ndims == 2 else [0, 2, 3, 4]
        s_union = torch.sum(union * mask, dim=sum_dims)
        s_y_true = torch.sum(y_true * mask, dim=sum_dims)
        s_y_pred = torch.sum(y_pred * mask, dim=sum_dims)

        # calculate dice per class, mean over classes
        dice = 2 * s_union / (s_y_true + s_y_pred + 1e-6)
        return torch.mean(dice[1:] if self.ignore_zero_class else dice)

#Image metrics
class SegmentationDiceOverlap(nn.Module):
    """
    calculates the mean dice overlap of the given classes

    WARNING:
    return of forward function is a tensor containing the dice overlap per image pair.
    If an image does not have a segmentation mask, -1 is returned at it's position instead.

    Parameters:
        classes: list of classes to consider, e.g. [0, 1]
    """

    def __init__(self, classes):
        super().__init__()
        self.dice_metric = DiceOverlap(classes=classes)

    def forward(self, annoated_image0, annoated_image1):
        dice_overlaps = []

        for i in range(len(annoated_image0)):
            # calculate segmentation dice overlap, if a segmentation mask is available
            if (
                annoated_image0.segmentation[i] is not None
                and annoated_image1.segmentation[i] is not None
            ):
                dice_overlap = self.dice_metric(
                    annoated_image0.segmentation[i], annoated_image1.segmentation[i]
                )
                dice_overlaps.append(dice_overlap)
            else:
                dice_overlaps.append(
                    torch.tensor(-1).type_as(annoated_image0.intensity)
                )
        return torch.stack(dice_overlaps)


class TargetRegistrationError(nn.Module):
    def __init__(self, norm="l1"):
        super().__init__()
        self.norm = norm

    def forward(self, annoated_image0, annoated_image1):
        target_registration_errors = []

        for i in range(len(annoated_image0)):
            if (
                annoated_image0.landmarks[i] is not None
                and annoated_image1.landmarks[i] is not None
            ):
                # calculate euclidean landmark error
                euclidean_dist = torch.mean(
                    torch.sum(
                        (annoated_image0.landmarks[i] - annoated_image1.landmarks[i])
                        ** 2,
                        dim=0,
                    )
                    ** 0.5
                )

                if self.norm == "l2":
                    euclidean_dist = euclidean_dist ** 2

                target_registration_errors.append(euclidean_dist)
            else:
                target_registration_errors.append(
                    torch.tensor(-1).type_as(annoated_image0.intensity)
                )

        return torch.stack(target_registration_errors)
