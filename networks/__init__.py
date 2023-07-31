import torch.nn as nn
from networks.networks import UNet, scSERDUNet, scSERDUNet3, Weight_Adaptive


def set_gpu(network, gpu_ids):
    network.to(gpu_ids[0])  # Default to the 1st GPU
    network = nn.DataParallel(network, device_ids=gpu_ids)  # Parallel computing on multiple GPU

    return network


def get_generator(name, opts, ic = 1, unet_depth = 4):
    # (2) DuRDN / default_depth = 4
    if name == 'DuRDN4':
        network = scSERDUNet(n_channels=ic, n_filters=opts.net_filter, n_denselayer=opts.n_denselayer, growth_rate=opts.growth_rate, norm=opts.norm, dropout=opts.dropout)

    # (2) DuRDN
    elif name == 'DuRDN3':
        network = scSERDUNet3(n_channels=ic, n_filters=opts.net_filter, n_denselayer=opts.n_denselayer, growth_rate=opts.growth_rate, norm=opts.norm, dropout=opts.dropout)

   # (1) UNet
    elif name == 'UNet':
        network = UNet(in_channels=ic, residual=False, depth=unet_depth, wf=opts.UNet_filters, norm=opts.norm, dropout=opts.dropout)

    elif name == "Weight":
        network = Weight_Adaptive(n_channels=2, n_filters=32, num_layers=4, growthrate=32)



    num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('Number of parameters of Generator: {}'.format(num_param))

    return set_gpu(network, opts.gpu_ids)