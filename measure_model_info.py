from models.swin_transformer import SwinGenerator
from monai.networks.nets import UNet, UNETR
from fvcore.nn import FlopCountAnalysis
import copy
import torch

img_size = 256
swin_gen = SwinGenerator(
        img_size=img_size,
        window_size=int(img_size/32),
        in_chans = 1,
        out_ch=1,
        )

#base_channel = 16
#channels=(base_channel, base_channel*2, base_channel*4, base_channel*8, base_channel*16),
#base_channel = 96
base_channel = 16
channels=(base_channel, base_channel*2, base_channel*4, base_channel*8, base_channel*24, base_channel*36, base_channel*48, base_channel*64)
strides = tuple([2]*(len(channels)-1))
unet = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=channels,
        strides=strides,
        num_res_units=2,
        )

def params(model):
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_parameters

def flops(model, resolution=224):
    new_model = copy.deepcopy(model)
    device = next(new_model.parameters()).device
    tensor = (torch.rand(1,1,resolution,resolution, device=device), )
    flops = FlopCountAnalysis(new_model, tensor)
    del new_model
    return flops.total() / 1e9

def main():
    model_list = [unet]
    string_list = []
    for model in model_list:
        n_params = params(model)/1.0e6
        n_flops = flops(model, resolution=256)
        string = f"Params(M): {n_params} FLOPs(G): {n_flops}"
        print(string)
        string_list.append(string)
    print('Summary......')
    for string in string_list:
        print(string)
    print('done!')

if __name__ == '__main__':
    main()