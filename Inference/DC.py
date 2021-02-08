import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as FFT 
import scipy.io as scio
import numpy as np


class lPhi(nn.Module):
    def __init__(self, conv_x):
        super(lPhi, self).__init__()
        self.lg = LG(conv_x)

    def forward(self,x_r, x_i):
        x = torch.cat([x_r, x_i], dim = 1)
        x = x.permute(0, 2, 3, 1).contiguous() ## Nb * Nx * Ny * 2 
        x = torch.view_as_complex(x) # for FFT. 

        phi = x.angle()
        phi = torch.unsqueeze(phi, 1)
        expPhi_r = torch.cos(phi)
        expPhi_i = torch.sin(phi)

        a_r = self.lg(expPhi_r)  ## first term. (delta(1j * phi)
        a_i = self.lg(expPhi_i)  

        ## b_r = a_r * expPhi_r + a_i * expPhi_i    ## first term  multiply the second term (exp(-1j * phi) = cos(phi) - j * sin(phi)))
        b_i = a_i * expPhi_r - a_r * expPhi_i

        return b_i

class LG(nn.Module):
    def __init__(self, conv_x):
        super(LG, self).__init__()
        self.weight_x = nn.Parameter(conv_x,requires_grad=False)

    def forward(self, tensor_image):
        out = F.conv2d(tensor_image, self.weight_x, bias=None,stride=1,padding=3)  ## 7 * kernel, padding 3 zeros. 

        h, w = out.shape[2], out.shape[3]
        out[:, :, [0, h-1], :] = 0
        out[:, :, :, [0, w-1]] = 0
        return out

## complex operations
def FFT2D(x):
    return FFT.fft2(FFT.fftshift(x, dim=(-2, -1)))

def IFFT2D(x):
    return FFT.ifftshift(FFT.ifft2(x), dim=(-2, -1))


def data_consistency(self, k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    WM   - learnable weighting matrix; initial value: 0s. 
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0 
    return out    

class DataConsistencyInKspace(nn.Module):
    """ Create data consistency operator
    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.
    """

    def __init__(self, noise_lvl=None, norm=None):
        super(DataConsistencyInKspace, self).__init__()
        self.normalized = norm == 'ortho'   
        self.noise_lvl = noise_lvl

    def forward(self, x_r, x_i, k0_r, k0_i, mask):
        ## x_r: Nb * 1 * Nx * Ny, the same with x_r
        ###WM = torch.cat([WM_r, WM_i], dim = 1)
        ## WM = WM.permute(0, 2, 3, 1)  ## Nb * Nx * Ny * 2
        ## WM = torch.view_as_complex(WM)

        k0 = torch.cat([k0_r, k0_i], dim = 1)
        k0 = k0.permute(0, 2, 3, 1) ## Nb * Nx * Ny * 2
        ## k0 = torch.view_as_complex(k0) # Nb * Nx * Ny, complex

        mask = mask.permute(0, 2, 3, 1) # Nb * Nx * Ny * 1, real
        mask = mask.expand(-1, -1, -1, 2)

        x = torch.cat([x_r, x_i], dim = 1)
        x = x.permute(0, 2, 3, 1).contiguous() ## Nb * Nx * Ny * 2 
        x = torch.view_as_complex(x) # for FFT. 

        k = IFFT2D(x)
        k = torch.view_as_real(k).contiguous()  #Nb * Nx * Ny * 2 

        out = data_consistency(self, k, k0, mask, noise_lvl=None) ## out shape: Nb * Nx * Ny * 2 real; 

        out = torch.view_as_complex(out.contiguous()) # Nb * Nx * Ny, complex

        x_res = FFT2D(out) ## complex nb * Nx * Ny 

        x_res = torch.view_as_real(x_res).contiguous()  ## real nb * Nx * Ny * 2

        x_res = x_res.permute(0, 3, 1, 2)
        x_r = x_res[:, 0, :, :]
        x_i = x_res[:, 1, :, :]
        return torch.unsqueeze(x_r, 1), torch.unsqueeze(x_i, 1)
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        WM   - learnable weighting matrix; initial value: 0s.  Nb * 1 * Nx * Ny (complex)
        """
"""
        if x.dim() == 4: # input is 2D
            x    = x.permute(0, 2, 3, 1)
            k0   = k0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)
        elif x.dim() == 5: # input is 3D
            x    = x.permute(0, 4, 2, 3, 1)
            k0   = k0.permute(0, 4, 2, 3, 1)
            mask = mask.permute(0, 4, 2, 3, 1)
"""
"""
        k = torch.fft(x, 2, normalized=self.normalized)
        out = data_consistency(k, k0, mask, self.noise_lvl)
        x_res = torch.ifft(out, 2, normalized=self.normalized)

        if x.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            x_res = x_res.permute(0, 4, 2, 3, 1)
""" 

if __name__ == '__main__':
    LGOP =  scio.loadmat("LG_operator.mat", verify_compressed_data_integrity=False)
    conv_op = LGOP['LG_operator']
    conv_op = np.array(conv_op)
    lphi_OP = lPhi(conv_op)

    matTest = scio.loadmat("test_py_lphi.mat", verify_compressed_data_integrity=False)
    image = matTest['test']
    image = np.array(image)
    image = torch.from_numpy(image).contiguous() ## size 128 * 256 * 256 * 2

    image_r = image[:,:,:,0]
    image_i = image[:,:,:,1]

    image_r = image_r.float()
    image_i = image_i.float()

    image_i = image_i.permute(2, 0, 1)
    image_r = image_r.permute(2, 0, 1)

    image_r = torch.unsqueeze(image_r, 1)
    image_i = torch.unsqueeze(image_i, 1)

    recons = lphi_OP(image_r, image_i)
    
    recons = recons.to('cpu')
    recons = recons.numpy()

    print('Saving results')
    path = "recon" + "test_py.mat"
    scio.savemat(path, {'recons':recons})

