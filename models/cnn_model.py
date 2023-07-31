import os

from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from networks import get_generator
from networks.networks import gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, psnr, mse, nmse, nmae
from skimage.metrics import structural_similarity as ssim
from utils.data_patch_util import *
import scipy.io as scio
from scipy.sparse import coo_matrix


class CNNModel(nn.Module):
    def __init__(self, opts):
        super(CNNModel, self).__init__()
        self.loss_names = []
        self.networks = []
        self.optimizers = []
        self.lr_1 = opts.lr_1  # Projection-based loss
        self.lr_2 = opts.lr_2  # Image-based loss
        self.lr_3 = opts.lr_3


        # Loss Name
        self.loss_names += ['loss_G0']  # Image-based loss
        self.loss_names += ['loss_G1_1']
        self.loss_names += ['loss_G2_1']
        self.loss_names += ['loss_G3_1']
        self.loss_names += ['loss_G3_2']
        self.loss_names += ['loss_G3_3']
        self.loss_names += ['loss_G3_4']
        self.loss_names += ['loss_G4_1']
        self.loss_names += ['loss_G4_2']
        self.loss_names += ['loss_G4_3']

        # Network
        self.net_G0 = get_generator('DuRDN4', opts, ic=1)  # Image-based prediction
        self.net_G1_1 = get_generator('DuRDN3', opts, ic=1)
        self.net_G2_1 = get_generator('DuRDN3', opts, ic=2)
        self.net_G3_1 = get_generator('DuRDN3', opts, ic=3) # 1st iteration
        self.net_G3_2 = get_generator('DuRDN3', opts, ic=4) # 1st iteration
        self.net_G3_3 = get_generator('DuRDN3', opts, ic=5) # 1st iteration
        self.net_G3_4 = get_generator('DuRDN3', opts, ic=6) # 1st iteration
        self.net_G4_1 = get_generator('DuRDN3', opts, ic=2) # 1st iteration
        self.net_G4_2 = get_generator('DuRDN3', opts, ic=2) # 1st iteration
        self.net_G4_3 = get_generator('DuRDN3', opts, ic=2) # 1st iteration
        self.net_W_1 = get_generator('Weight', opts)
        self.net_W_2 = get_generator('Weight', opts)
        self.net_W_3 = get_generator('Weight', opts)
        self.net_W_4 = get_generator('Weight', opts)

        self.networks.append(self.net_G0)
        self.networks.append(self.net_G1_1)
        self.networks.append(self.net_G2_1)
        self.networks.append(self.net_G3_1)
        self.networks.append(self.net_G3_2)
        self.networks.append(self.net_G3_3)
        self.networks.append(self.net_G3_4)
        self.networks.append(self.net_G4_1)
        self.networks.append(self.net_G4_2)
        self.networks.append(self.net_G4_3)
        self.networks.append(self.net_W_1)
        self.networks.append(self.net_W_2)
        self.networks.append(self.net_W_3)
        self.networks.append(self.net_W_4)


        # Optimizer
        self.optimizer_G0 = torch.optim.Adam(self.net_G0.parameters(), lr=self.lr_2, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_G1_1 = torch.optim.Adam(self.net_G1_1.parameters(), lr=self.lr_1, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_G2_1 = torch.optim.Adam(self.net_G2_1.parameters(), lr=self.lr_1, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_G3_1 = torch.optim.Adam(self.net_G3_1.parameters(), lr=self.lr_1, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_G3_2 = torch.optim.Adam(self.net_G3_2.parameters(), lr=self.lr_1, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_G3_3 = torch.optim.Adam(self.net_G3_3.parameters(), lr=self.lr_1, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_G3_4 = torch.optim.Adam(self.net_G3_4.parameters(), lr=self.lr_1, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_G4_1 = torch.optim.Adam(self.net_G4_1.parameters(), lr=self.lr_1, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_G4_2 = torch.optim.Adam(self.net_G4_2.parameters(), lr=self.lr_1, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_G4_3 = torch.optim.Adam(self.net_G4_3.parameters(), lr=self.lr_1, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_W_1 = torch.optim.Adam(self.net_W_1.parameters(), lr=self.lr_3, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_W_2 = torch.optim.Adam(self.net_W_2.parameters(), lr=self.lr_3, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_W_3 = torch.optim.Adam(self.net_W_3.parameters(), lr=self.lr_3, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_W_4 = torch.optim.Adam(self.net_W_4.parameters(), lr=self.lr_3, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizers.append(self.optimizer_G0)
        self.optimizers.append(self.optimizer_G1_1)
        self.optimizers.append(self.optimizer_G2_1)
        self.optimizers.append(self.optimizer_G3_1)
        self.optimizers.append(self.optimizer_G3_2)
        self.optimizers.append(self.optimizer_G3_3)
        self.optimizers.append(self.optimizer_G3_4)
        self.optimizers.append(self.optimizer_G4_1)
        self.optimizers.append(self.optimizer_G4_2)
        self.optimizers.append(self.optimizer_G4_3)
        self.optimizers.append(self.optimizer_W_1)
        self.optimizers.append(self.optimizer_W_2)
        self.optimizers.append(self.optimizer_W_3)
        self.optimizers.append(self.optimizer_W_4)

        # Loss Function
        self.criterion = nn.L1Loss()  # L1 loss function.py

        # Options
        self.opts = opts


    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))  # Choose GPU for CUDA computing; For input setting

    def system_matrix(self):
        # Read in the sparse matrix in coo_matrix format
        SM = coo_matrix(scio.loadmat('./sm/sm601_20.mat')['sm'])

        # Extract the values, indices, and shape
        values = torch.FloatTensor(SM.data)
        indices = torch.LongTensor(np.vstack((SM.row, SM.col)))
        shape = torch.Size(SM.shape)

        # Build the system matrix in the torch sparse format
        self.SM = torch.sparse.FloatTensor(indices, values, shape).to(self.device).float().unsqueeze(0).unsqueeze(0)  #  [1, 1, 32*32*20, 72*72*40], nnz = 28768054
        # self.SM = torch.sparse.FloatTensor(indices, values, shape).to(self.device).float().unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)  #  [B, 1, 32*32*20, 72*72*40]

        # convert the sparse SM to dense SM
        self.SM_dense = self.SM.to_dense()  # [1, 1, 32*32*20, 72*72*40]

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    # LR decay can be realized here
    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]


    def set_input(self, data):
        # self.Amap = data['Amap'].to(self.device).float()
        self.Recon_FD_FA  = data['Recon_FD_FA_EM'].to(self.device).float()
        # self.Recon_FD_LA  = data['Recon_FD_LA_EM'].to(self.device).float()
        # self.Recon_LD_FA  = data['Recon_LD_FA_EM'].to(self.device).float()
        self.Recon_LD_LA  = data['Recon_LD_LA_EM'].to(self.device).float()
        self.Mask = data['Mask'].to(self.device).float()

        self.Proj_FD_FA  = data['Proj_FD_FA_EM'].to(self.device).float()
        self.Proj_FD_LA  = data['Proj_FD_LA_EM'].to(self.device).float()
        # self.Proj_LD_FA  = data['Proj_LD_FA_EM'].to(self.device).float()
        self.Proj_LD_LA  = data['Proj_LD_LA_EM'].to(self.device).float()
        self.Mask_Proj  = data['Mask_Proj'].to(self.device).float()

        self.opts_drop = data['opts_drop'][0].numpy()  # Training: True; Testing: False
        self.sm_size = self.Recon_FD_FA.size(0)  # Batch size


    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            errors_ret[name] = float(getattr(self, name))  # get self.loss_G_L1
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self):
        # Initialization
        self.Recon_LD_LA.requires_grad_(True)
        self.Proj_LD_LA.requires_grad_(True)

        # Image-based prediction
        self.Recon_FD_FA_pred = self.net_G0(self.Recon_LD_LA, self.opts_drop)
        self.Recon_FD_FA_pred_FP = torch.matmul(self.SM_dense, self.Recon_FD_FA_pred.flip(4).rot90(-2, [2, 3]).permute(0, 1, 4, 3, 2).reshape(self.sm_size, 1, 40 * 72 * 72, 1)).reshape(self.sm_size, 1, 20, 32, 32).permute(0, 1, 4, 3, 2)  # [B,1, 32, 32, 20]
        self.Recon_FD_FA_pred_FP_normLDLA = self.Recon_FD_FA_pred_FP / self.Recon_FD_FA_pred_FP.detach().mean() * self.Proj_LD_LA.detach().mean()

        # Projection domain prediction
        # Iteration 0
        self.Proj_FD_LA_pred0 = self.net_G1_1(self.Proj_LD_LA, self.opts_drop)

        self.Proj_FD_FA_pred0 = self.net_G2_1(torch.cat((self.Proj_LD_LA, self.Recon_FD_FA_pred_FP_normLDLA), 1), self.opts_drop)
        self.Proj_FD_FA_pred0_DC = self.net_W_1(self.Proj_FD_LA_pred0, self.Proj_FD_FA_pred0, self.Mask_Proj)
        self.Proj_FD_FA_pred0_DC_normLDLA = self.Proj_FD_FA_pred0_DC / self.Proj_FD_FA_pred0_DC.detach().mean() * self.Proj_LD_LA.detach().mean()


        # Iteration 1
        self.Proj_FD_LA_pred1 = self.net_G4_1(torch.cat((self.Proj_FD_LA_pred0, self.Proj_FD_FA_pred0_DC), 1), self.opts_drop)

        self.Proj_FD_FA_pred1 = self.net_G3_1(torch.cat((self.Proj_LD_LA, self.Recon_FD_FA_pred_FP_normLDLA, self.Proj_FD_FA_pred0_DC_normLDLA), 1), self.opts_drop)
        self.Proj_FD_FA_pred1_DC = self.net_W_2(self.Proj_FD_LA_pred1, self.Proj_FD_FA_pred1, self.Mask_Proj)
        self.Proj_FD_FA_pred1_DC_normLDLA = self.Proj_FD_FA_pred1_DC / self.Proj_FD_FA_pred1_DC.detach().mean() * self.Proj_LD_LA.detach().mean()


        # Iteration 2
        self.Proj_FD_LA_pred2 = self.net_G4_2(torch.cat((self.Proj_FD_LA_pred1, self.Proj_FD_FA_pred1_DC), 1), self.opts_drop)

        self.Proj_FD_FA_pred2 = self.net_G3_2(torch.cat((self.Proj_LD_LA, self.Recon_FD_FA_pred_FP_normLDLA, self.Proj_FD_FA_pred0_DC_normLDLA, self.Proj_FD_FA_pred1_DC_normLDLA), 1), self.opts_drop)
        self.Proj_FD_FA_pred2_DC = self.net_W_3(self.Proj_FD_LA_pred2, self.Proj_FD_FA_pred2, self.Mask_Proj)
        self.Proj_FD_FA_pred2_DC_normLDLA = self.Proj_FD_FA_pred2_DC / self.Proj_FD_FA_pred2_DC.detach().mean() * self.Proj_LD_LA.detach().mean()


        # Iteration 3
        self.Proj_FD_LA_pred3 = self.net_G4_3(torch.cat((self.Proj_FD_LA_pred2, self.Proj_FD_FA_pred2_DC), 1), self.opts_drop)

        self.Proj_FD_FA_pred3 = self.net_G3_3(torch.cat((self.Proj_LD_LA, self.Recon_FD_FA_pred_FP_normLDLA, self.Proj_FD_FA_pred0_DC_normLDLA, self.Proj_FD_FA_pred1_DC_normLDLA, self.Proj_FD_FA_pred2_DC_normLDLA), 1), self.opts_drop)
        self.Proj_FD_FA_pred3_DC = self.net_W_4(self.Proj_FD_LA_pred3, self.Proj_FD_FA_pred3, self.Mask_Proj)
        self.Proj_FD_FA_pred3_DC_normLDLA = self.Proj_FD_FA_pred3_DC / self.Proj_FD_FA_pred3_DC.detach().mean() * self.Proj_LD_LA.detach().mean()



        # Iteration 4
        self.Proj_FD_FA_pred  = self.net_G3_4(torch.cat((self.Proj_LD_LA, self.Recon_FD_FA_pred_FP_normLDLA, self.Proj_FD_FA_pred0_DC_normLDLA, self.Proj_FD_FA_pred1_DC_normLDLA, self.Proj_FD_FA_pred2_DC_normLDLA, self.Proj_FD_FA_pred3_DC_normLDLA), 1), self.opts_drop)


    def update(self):
        # Zero Gradient
        self.optimizer_G0.zero_grad()
        self.optimizer_G1_1.zero_grad()
        self.optimizer_G2_1.zero_grad()
        self.optimizer_G3_1.zero_grad()
        self.optimizer_G3_2.zero_grad()
        self.optimizer_G3_3.zero_grad()
        self.optimizer_G3_4.zero_grad()
        self.optimizer_G4_1.zero_grad()
        self.optimizer_G4_2.zero_grad()
        self.optimizer_G4_3.zero_grad()
        self.optimizer_W_1.zero_grad()
        self.optimizer_W_2.zero_grad()
        self.optimizer_W_3.zero_grad()
        self.optimizer_W_4.zero_grad()

        # Calculate Loss
        loss_G0   = self.criterion(self.Recon_FD_FA_pred, self.Recon_FD_FA)
        loss_G1_1 = self.criterion(self.Proj_FD_LA_pred0, self.Proj_FD_LA)
        loss_G2_1 = self.criterion(self.Proj_FD_FA_pred0, self.Proj_FD_FA)
        loss_G3_1 = self.criterion(self.Proj_FD_FA_pred1, self.Proj_FD_FA)
        loss_G3_2 = self.criterion(self.Proj_FD_FA_pred2, self.Proj_FD_FA)
        loss_G3_3 = self.criterion(self.Proj_FD_FA_pred3, self.Proj_FD_FA)
        loss_G3_4 = self.criterion(self.Proj_FD_FA_pred,  self.Proj_FD_FA)
        loss_G4_1 = self.criterion(self.Proj_FD_LA_pred1, self.Proj_FD_LA)
        loss_G4_2 = self.criterion(self.Proj_FD_LA_pred2, self.Proj_FD_LA)
        loss_G4_3 = self.criterion(self.Proj_FD_LA_pred3, self.Proj_FD_LA)
        self.loss_G0   = loss_G0.item()
        self.loss_G1_1 = loss_G1_1.item()
        self.loss_G2_1 = loss_G2_1.item()
        self.loss_G3_1 = loss_G3_1.item()
        self.loss_G3_2 = loss_G3_2.item()
        self.loss_G3_3 = loss_G3_3.item()
        self.loss_G3_4 = loss_G3_4.item()
        self.loss_G4_1 = loss_G4_1.item()
        self.loss_G4_2 = loss_G4_2.item()
        self.loss_G4_3 = loss_G4_3.item()


        # Backward and update
        total_loss = loss_G0 + loss_G1_1 + loss_G2_1 + loss_G3_1 + loss_G3_2 + loss_G3_3 + loss_G3_4 + loss_G4_1 + loss_G4_2 + loss_G4_3
        total_loss.backward()

        self.optimizer_G0.step()
        self.optimizer_G1_1.step()
        self.optimizer_G2_1.step()
        self.optimizer_G3_1.step()
        self.optimizer_G3_2.step()
        self.optimizer_G3_3.step()
        self.optimizer_G3_4.step()
        self.optimizer_G4_1.step()
        self.optimizer_G4_2.step()
        self.optimizer_G4_3.step()
        self.optimizer_W_1.step()
        self.optimizer_W_2.step()
        self.optimizer_W_3.step()
        self.optimizer_W_4.step()

    @property
    def loss_summary(self):
        message = ''
        message += 'loss_G3_4: {:.4e}'.format(self.loss_G3_4)
        return message


    # learning rate decay
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()  # learning rate update
        # Extract the information
        self.lr_1 = self.optimizers[1].param_groups[0]['lr']  # Extract the current learning rate

    def save(self, filename, epoch, total_iter):  # Save the net/optimizer state data
        state = {}  # dict
        state['net_G0'] = self.net_G0.module.state_dict()
        state['net_G1_1'] = self.net_G1_1.module.state_dict()
        state['net_G2_1'] = self.net_G2_1.module.state_dict()
        state['net_G3_1'] = self.net_G3_1.module.state_dict()
        state['net_G3_2'] = self.net_G3_2.module.state_dict()
        state['net_G3_3'] = self.net_G3_3.module.state_dict()
        state['net_G3_4'] = self.net_G3_4.module.state_dict()
        state['net_G4_1'] = self.net_G4_1.module.state_dict()
        state['net_G4_2'] = self.net_G4_2.module.state_dict()
        state['net_G4_3'] = self.net_G4_3.module.state_dict()
        state['net_W_1'] = self.net_W_1.module.state_dict()
        state['net_W_2'] = self.net_W_2.module.state_dict()
        state['net_W_3'] = self.net_W_3.module.state_dict()
        state['net_W_4'] = self.net_W_4.module.state_dict()

        state['opt_G0'] = self.optimizer_G0.state_dict()
        state['opt_G1_1'] = self.optimizer_G1_1.state_dict()
        state['opt_G2_1'] = self.optimizer_G2_1.state_dict()
        state['opt_G3_1'] = self.optimizer_G3_1.state_dict()
        state['opt_G3_2'] = self.optimizer_G3_2.state_dict()
        state['opt_G3_3'] = self.optimizer_G3_3.state_dict()
        state['opt_G3_4'] = self.optimizer_G3_4.state_dict()
        state['opt_G4_1'] = self.optimizer_G4_1.state_dict()
        state['opt_G4_2'] = self.optimizer_G4_2.state_dict()
        state['opt_G4_3'] = self.optimizer_G4_3.state_dict()
        state['opt_W_1'] = self.optimizer_W_1.state_dict()
        state['opt_W_2'] = self.optimizer_W_2.state_dict()
        state['opt_W_3'] = self.optimizer_W_3.state_dict()
        state['opt_W_4'] = self.optimizer_W_4.state_dict()

        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)
        print('Saved {}'.format(filename))


    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.net_G0.module.load_state_dict(checkpoint['net_G0'])
        self.net_G1_1.module.load_state_dict(checkpoint['net_G1_1'])
        self.net_G2_1.module.load_state_dict(checkpoint['net_G2_1'])
        self.net_G3_1.module.load_state_dict(checkpoint['net_G3_1'])
        self.net_G3_2.module.load_state_dict(checkpoint['net_G3_2'])
        self.net_G3_3.module.load_state_dict(checkpoint['net_G3_3'])
        self.net_G3_4.module.load_state_dict(checkpoint['net_G3_4'])
        self.net_G4_1.module.load_state_dict(checkpoint['net_G4_1'])
        self.net_G4_2.module.load_state_dict(checkpoint['net_G4_2'])
        self.net_G4_3.module.load_state_dict(checkpoint['net_G4_3'])
        self.net_W_1.module.load_state_dict(checkpoint['net_W_1'])
        self.net_W_2.module.load_state_dict(checkpoint['net_W_2'])
        self.net_W_3.module.load_state_dict(checkpoint['net_W_3'])
        self.net_W_4.module.load_state_dict(checkpoint['net_W_4'])

        if train:
            self.optimizer_G0.load_state_dict(checkpoint['opt_G0'])
            self.optimizer_G1_1.load_state_dict(checkpoint['opt_G1_1'])
            self.optimizer_G2_1.load_state_dict(checkpoint['opt_G2_1'])
            self.optimizer_G3_1.load_state_dict(checkpoint['opt_G3_1'])
            self.optimizer_G3_2.load_state_dict(checkpoint['opt_G3_2'])
            self.optimizer_G3_3.load_state_dict(checkpoint['opt_G3_3'])
            self.optimizer_G3_4.load_state_dict(checkpoint['opt_G3_4'])
            self.optimizer_G4_1.load_state_dict(checkpoint['opt_G4_1'])
            self.optimizer_G4_2.load_state_dict(checkpoint['opt_G4_2'])
            self.optimizer_G4_3.load_state_dict(checkpoint['opt_G4_3'])
            self.optimizer_W_1.load_state_dict(checkpoint['opt_W_1'])
            self.optimizer_W_2.load_state_dict(checkpoint['opt_W_2'])
            self.optimizer_W_3.load_state_dict(checkpoint['opt_W_3'])
            self.optimizer_W_4.load_state_dict(checkpoint['opt_W_4'])

        print('Loaded {}'.format(checkpoint_file))
        return checkpoint['epoch'], checkpoint['total_iter']


    # -------------- Evaluation, Calculate PSNR ---------------
    def evaluate(self, loader):
        val_bar = tqdm(loader)
        val_bar.set_description(desc='Evaluating images ...')

        # For calculating metrics
        avg_nmse = AverageMeter()
        avg_nmae = AverageMeter()
        avg_ssim = AverageMeter()
        avg_psnr = AverageMeter()

        for data in val_bar:
            self.set_input(data)  # [batch_szie=1, 1, 48, 72, 72]
            self.forward()

            # Non-megativity & Mean-normalization
            self.Proj_FD_FA_pred[self.Proj_FD_FA_pred < 0] = 0  # non-negativity

            # Calculate the metrics; z_range can be used to calculate the mean
            # Proj
            nmse_ = nmse(self.Proj_FD_FA_pred, self.Proj_FD_FA)
            nmae_ = nmae(self.Proj_FD_FA_pred, self.Proj_FD_FA)
            ssim_ = ssim(self.Proj_FD_FA_pred[0, 0, ...].cpu().numpy(), self.Proj_FD_FA[0, 0, ...].cpu().numpy())
            psnr_ = psnr(self.Proj_FD_FA_pred, self.Proj_FD_FA)
            avg_nmse.update(nmse_)
            avg_nmae.update(nmae_)
            avg_ssim.update(ssim_)
            avg_psnr.update(psnr_)

            # Descrip show NMSE, NMAE, SSIM here
            message = 'NMSE: {:4f}, NMAE: {:4f}, SSIM: {:4f}, PSNR: {:4f}'.format(avg_nmse.avg, avg_nmae.avg, avg_ssim.avg, avg_psnr.avg)
            val_bar.set_description(desc=message)

        # Calculate the average metrics
        self.nmse = avg_nmse.avg
        self.nmae = avg_nmae.avg
        self.ssim = avg_ssim.avg
        self.psnr = avg_psnr.avg


    # --------------- Save the images ------------------------------
    def save_images(self, loader, folder):
        val_bar = tqdm(loader)
        val_bar.set_description(desc='Saving images ...')

        # Load data for each batch
        index = 0
        for data in val_bar:
            index += 1
            self.set_input(data)  # [batch_szie=1, 1, 64, 64, 64]
            self.forward()

            # Non-megativity & Mean-normalization
            self.Proj_FD_FA_pred[self.Proj_FD_FA_pred < 0] = 0  # non-negativity

            # --------------- Mkdir folder -------------------
            # Original 64x64x64 images
            if not os.path.exists(os.path.join(folder, 'Proj_FD_FA_pred')):
                os.mkdir(os.path.join(folder, 'Proj_FD_FA_pred'))
            if not os.path.exists(os.path.join(folder, 'Proj_FD_FA')):
                os.mkdir(os.path.join(folder, 'Proj_FD_FA'))
            if not os.path.exists(os.path.join(folder, 'Proj_LD_LA')):
                os.mkdir(os.path.join(folder, 'Proj_LD_LA'))

            # save image
            save_nii(self.Proj_FD_FA_pred.squeeze().cpu().numpy(),   os.path.join(folder, 'Proj_FD_FA_pred',  'Proj_FD_FA_pred_'   + str(index) + '.nii'))
            save_nii(self.Proj_FD_FA.squeeze().cpu().numpy(),   os.path.join(folder, 'Proj_FD_FA',  'Proj_FD_FA_'   + str(index) + '.nii'))
            save_nii(self.Proj_LD_LA.squeeze().cpu().numpy(),   os.path.join(folder, 'Proj_LD_LA',  'Proj_LD_LA_'   + str(index) + '.nii'))









