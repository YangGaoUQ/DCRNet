import torch
import torch.nn as nn 
import torch.nn.functional as F
from DC import * 

class GenWM(nn.Module):
    def __init__(self, num_ch = 32):
        super(GenWM, self).__init__()
        self.cconv1 = CConv2d_BN_RELU(2, num_ch, use_CCBAM = False)
        self.cconv2 = CConv2d(num_ch, 1)

    def forward(self, x_r, x_i, k0_r, k0_i):
        ## x_r: Nb * 1 * Nx * Ny, the same with x_r
        x = torch.cat([x_r, x_i], dim = 1)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.view_as_complex(x)

        k = IFFT2D(x)
        k = torch.view_as_real(k).contiguous()  ## complex nb * Nx * Ny * 2
        k = k.permute(0, 3, 1, 2)
        k_r = k[:,0,:, :]
        k_i = k[:,1,:, :]

        x_r = torch.cat([x_r, k0_r], dim = 1)
        x_i = torch.cat([x_i, k0_i], dim = 1)

        x_r, x_i = self.cconv1(x_r, x_i)
        WM_r, WM_i = self.cconv2(x_r, x_i)

        return WM_r, WM_i


class ReconNet(nn.Module):
    def __init__(self, EncodingDepth = 5):
        super(ReconNet, self).__init__()
        initial_num_layers = 64
        No_channels = 1
        self.EncodingDepth = EncodingDepth
        self.init1 = CConv2d_BN_RELU(No_channels, initial_num_layers)

        self.midLayers = []
        temp = list(range(1, EncodingDepth + 1))
        for encodingLayer in temp:
            self.midLayers.append(Basic_block(initial_num_layers, initial_num_layers))
        self.midLayers = nn.ModuleList(self.midLayers)

        self.FinalConv =  CConv2d(initial_num_layers, No_channels)

        self.dc = DataConsistencyInKspace()

    def forward(self, x_r, x_i, k0_r, k0_i, mask):
        INPUT_r = x_r
        INPUT_i = x_i
        x_r, x_i = self.init1(x_r, x_i)

        temp = list(range(1, self.EncodingDepth + 1))
        for encodingLayer in temp:
            temp_conv = self.midLayers[encodingLayer - 1]
            x_r, x_i = temp_conv(x_r, x_i)
        
        x_r, x_i = self.FinalConv(x_r, x_i)
        x_r = x_r + INPUT_r  ## 
        x_i = x_i + INPUT_i ## 
        
        x_r, x_i = self.dc(x_r, x_i, k0_r, k0_i, mask)

        return x_r, x_i



class Basic_block(nn.Module):
    def __init__(self, num_in, num_out):
        super(Basic_block, self).__init__()
        self.cconv1 = CConv2d_BN_RELU(num_in, num_out, use_CCBAM = False)
        self.cconv2 = CConv2d_BN_RELU(num_out, num_out, use_CCBAM = False)

    def forward(self, x_r, x_i):
        INPUT_r = x_r
        INPUT_i = x_i
        x_r, x_i = self.cconv1(x_r, x_i)
        x_r = x_r + INPUT_r
        x_i = x_i + INPUT_i
        x_r, x_i = self.cconv2(x_r, x_i)
        return x_r, x_i


class CCBAM(nn.Module):
    def __init__(self, chs):
        super(CCBAM, self).__init__()
        self.ca = ChannelAttention(chs)  # chs: number of channels
        self.sa = SpatialAttention()

    def forward(self, x_r, x_i):
        INPUT_r = x_r
        INPUT_i = x_i
        c_r, c_i = self.ca(x_r, x_i)
        x_r = c_r * x_r - c_i * x_i
        x_i = c_i * x_r + c_r * x_i

        x_r = x_r + INPUT_r
        x_i = x_i + INPUT_i

        INPUT_r = x_r
        INPUT_i = x_i

        s_r, s_i = self.sa(x_r, x_i)
        x_r = s_r * x_r - s_i * x_i
        x_i = s_i * x_r + s_r * x_i

        x_r = x_r + INPUT_r
        x_i = x_i + INPUT_i
        
        return x_r, x_i


class ChannelAttention(nn.Module):
    def __init__(self, in_chs, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP1 = CConv2d(in_chs, in_chs // ratio, ks = 1, pad = 0, bs = False)
        self.relu = nn.ReLU()
        self.sharedMLP2 = CConv2d(in_chs // ratio, in_chs, ks = 1, pad = 0, bs = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_r, x_i):
        avg_r = self.avg_pool(x_r)
        avg_i = self.avg_pool(x_i)
        avg_r, avg_i = self.sharedMLP1(avg_r, avg_i)
        avg_r = self.relu(avg_r)
        avg_i = self.relu(avg_i)
        avg_r, avg_i = self.sharedMLP2(avg_r, avg_i)

        max_r = self.max_pool(x_r)
        max_i = self.max_pool(x_i)
        max_r, max_i = self.sharedMLP1(max_r, max_i)
        max_r = self.relu(max_r)
        max_i = self.relu(max_i)
        max_r, max_i = self.sharedMLP2(max_r, max_i)

        x_r = avg_r + max_r
        x_i = avg_i + max_i

        return self.sigmoid(x_r), self.sigmoid(x_i)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = CConv2d(2, 1, ks = kernel_size, pad = padding, bs = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_r, x_i):
        avg_r = torch.mean(x_r, dim=1, keepdim=True)
        avg_i = torch.mean(x_i, dim=1, keepdim=True)
        max_r, _ = torch.max(x_r, dim=1, keepdim=True)
        max_i, _ = torch.max(x_i, dim=1, keepdim=True)

        x_r = torch.cat([avg_r, max_r], dim = 1)
        x_i = torch.cat([avg_i, max_i], dim = 1)

        x_r, x_i = self.conv(x_r, x_i)

        return self.sigmoid(x_r), self.sigmoid(x_i)


## complex convolution; 
class CConv2d_BN_RELU(nn.Module):
    def __init__(self, num_in, num_out, ks = 3, pad = 1, use_CCBAM = False):
        super(CConv2d_BN_RELU, self).__init__()
        self.conv_r = nn.Conv2d(num_in, num_out, ks, padding= pad)
        self.conv_i = nn.Conv2d(num_in, num_out, ks, padding= pad)
        self.bn_r = nn.BatchNorm2d(num_out)
        self.bn_i = nn.BatchNorm2d(num_out)

        self.use_CCBAM = use_CCBAM
        if use_CCBAM:
            self.cbam = CCBAM(num_out)

        self.relu_r = nn.ReLU(inplace = True)
        self.relu_i = nn.ReLU(inplace = True)

    def forward(self, x_r, x_i):
        x_rr = self.conv_r(x_r)
        x_ri = self.conv_i(x_r)
        x_ir = self.conv_r(x_i)
        x_ii = self.conv_i(x_i)
        x_r = x_rr - x_ii 
        x_i = x_ri + x_ir
        x_r = self.bn_r(x_r)
        x_i = self.bn_i(x_i)

        if self.use_CCBAM:
            x_r, x_i = self.cbam(x_r, x_i)

        x_r = self.relu_r(x_r)
        x_i = self.relu_i(x_i)
        return x_r, x_i


## complex convolution; 
class CConv2d(nn.Module):
    def __init__(self, num_in, num_out, ks = 1, pad = 0, bs = True,  use_CCBAM = False):
        super(CConv2d, self).__init__()
        self.conv_r = nn.Conv2d(num_in, num_out, ks, bias = bs, padding= pad)
        self.conv_i = nn.Conv2d(num_in, num_out, ks, bias = bs, padding= pad)
        
    def forward(self, x_r, x_i):
        x_rr = self.conv_r(x_r)
        x_ri = self.conv_i(x_r)
        x_ir = self.conv_r(x_i)
        x_ii = self.conv_i(x_i)
        x_r = x_rr - x_ii 
        x_i = x_ri + x_ir

        return x_r, x_i



def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=1e-2)
        #nn.init.zeros_(m.bias)   
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, mean=0, std=1e-2)
        #nn.init.zeros_(m.bias)   
    if isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)   

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

#################### For Code Test ##################################
## before running the training codes, verify the network architecture. 
if __name__ == '__main__':
    unet = ReconNet(5)
    unet.apply(weights_init)
    print(unet.state_dict)
    print(get_parameter_number(unet))
    x_r = torch.randn(2,1,48,48, dtype=torch.float)
    x_i = torch.randn(2,1,48,48, dtype=torch.float)
    print('input' + str(x_r.size()))
    print(x_r.dtype)
    y_r, y_i = unet(x_r, x_i, x_r, x_i, x_r)
    print(torch.max(y_r - y_i))
    print('output'+str(y_r.size()))
