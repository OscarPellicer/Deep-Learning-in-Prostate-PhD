'''
Source: https://github.com/connorlee77/pytorch-mutual-information
Adapted to be dimensionality-agnostic (1D / 2D / 3D images transparently supported)
Also added some small QoL improvements
'''

import os
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn

class MutualInformation(nn.Module):

    def __init__(self, sigma=0.012, num_bins=256, normalize=True, value_range=[0., 1.], **kwargs):
        super().__init__(**kwargs)

        self.sigma = 2*sigma**2
        self.num_bins = num_bins
        self.normalize = normalize
        self.epsilon = 1e-10
        self.value_range= value_range
        
        self.bins = nn.Parameter(torch.linspace(*value_range, num_bins).float(), requires_grad=False)

    def marginalPdf(self, values):

        residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2))

        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization
        
        return pdf, kernel_values

    def jointPdf(self, kernel_values1, kernel_values2):

        joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2) 
        normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + self.epsilon
        pdf = joint_kernel_values / normalization

        return pdf

    def getMutualInformation(self, input1, input2, plot=False):
        '''
            input1: B, C, H, (W), (D)
            input2: B, C, H, (W), (D)

            return: scalar
        '''
        assert((input1.shape == input2.shape))

        B, C, DIMS= input1.shape[0], input1.shape[1], torch.tensor(input1.shape[2:])
        x1 = input1.view(B, torch.prod(DIMS), C)
        x2 = input2.view(B, torch.prod(DIMS), C)
        
        pdf_x1, kernel_values1 = self.marginalPdf(x1)
        pdf_x2, kernel_values2 = self.marginalPdf(x2)
        pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

        H_x1 = -torch.sum(pdf_x1*torch.log2(pdf_x1 + self.epsilon), dim=1)
        H_x2 = -torch.sum(pdf_x2*torch.log2(pdf_x2 + self.epsilon), dim=1)
        H_x1x2 = -torch.sum(pdf_x1x2*torch.log2(pdf_x1x2 + self.epsilon), dim=(1,2))

        mutual_information = H_x1 + H_x2 - H_x1x2
        
        if self.normalize:
            mutual_information = 2*mutual_information/(H_x1+H_x2)
            
        if plot: 
            self._plot(pdf_x1x2, -mutual_information)

        return - mutual_information
    
    def _plot(self, h, mi):
        hgram_plot= h.detach().cpu().numpy()
        nzs= hgram_plot > 0
        hgram_plot[nzs]= np.log(hgram_plot[nzs])
        plt.figure(figsize=(6,6))
        plt.imshow(hgram_plot.T, origin='lower')
        plt.title(f'Histogram between transformed MR and US (MI = {mi.cpu().detach().numpy()[0]:.4f})')
        plt.show()

    def forward(self, input1, input2, plot=False):
        '''
            input1: B, C, H, (W), (D)
            input2: B, C, H, (W), (D)

            return: scalar
        '''
        return self.getMutualInformation(input1, input2, plot=plot)


if __name__ == '__main__':
    
    import skimage.io
    from PIL import Image
    from sklearn.metrics import normalized_mutual_info_score
    from torchvision import transforms
    
    device = 'cuda:0'

    ### Create test cases ###
    img1 = Image.open('grad.jpg').convert('L')
    img2 = img1.rotate(10)

    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    mi_true_1 = normalized_mutual_info_score(arr1.ravel(), arr2.ravel())
    mi_true_2 = normalized_mutual_info_score(arr2.ravel(), arr2.ravel())

    img1 = transforms.ToTensor() (img1).unsqueeze(dim=0).to(device)
    img2 = transforms.ToTensor() (img2).unsqueeze(dim=0).to(device)

    # Pair of different images, pair of same images
    input1 = torch.cat([img1, img2])
    input2 = torch.cat([img2, img2])

    MI = MutualInformation(num_bins=256, sigma=0.4, normalize=True).to(device)
    mi_test = MI(input1, input2)

    mi_test_1 = mi_test[0].cpu().numpy()
    mi_test_2 = mi_test[1].cpu().numpy()

    print('Image Pair 1 | sklearn MI: {}, this MI: {}'.format(mi_true_1, mi_test_1))
    print('Image Pair 2 | sklearn MI: {}, this MI: {}'.format(mi_true_2, mi_test_2))

    assert(np.abs(mi_test_1 - mi_true_1) < 0.05)
    assert(np.abs(mi_test_2 - mi_true_2) < 0.05)