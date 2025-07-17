import models.archs.CWNet as CWNet

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    nf = opt_net['nf']
    n_l_blocks = opt_net['n_l_blocks']
    n_h_blocks = opt_net['n_h_blocks']

    if which_model == 'CWNet':
        netG = CWNet.CWNet(nc=nf, n_l_blocks=n_l_blocks, n_h_blocks=n_h_blocks)
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

