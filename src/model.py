import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Things that doesn't yet seem to be supported in pytorch:
# padding = same
# spatial softmax
class AutoEncoder_Dynamics(nn.Module):
    def __init__(self, img_res=32, z_dim=2, u_dim=2):
        super(AutoEncoder_Dynamics, self).__init__()
        
        self.img_res = img_res
        self.x_dim = img_res*img_res
        self.z_dim = z_dim
        self.u_dim = u_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding=2), # kernel_size different than original
            nn.ReLU(),
            nn.Conv2d(8, 8, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 8, 5, padding=2),
            nn.ReLU(),
            SpatialSoftmax(img_res, img_res, 8),
            nn.Linear(8*2, 256), # Spatial softmax will result in 2 values per channel 1 along height and 1 along widt.
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.z_dim),
            )
        
#         self.encoder = self.encoder.float()
        self.dynamics = nn.Sequential(
            nn.Linear(self.z_dim + self.u_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.z_dim),
            nn.ReLU(),
        )
#         self.dynamics = self.dynamics.float()
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
#         self.decoder = self.decoder.float()
        self.environment = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2), # kernel_size different than original
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * self.x_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
#         self.environment = self.environment.float()
        self.last_layer = nn.Linear(512 + 512, self.x_dim)
#         self.last_layer = self.last_layer.float()

        self.dummy_param = nn.Parameter(torch.tensor(1.))
    
    def encode(self, x_t, x_tplus):
        x_full = torch.cat((x_t, x_tplus), dim=0)
        input_enc = torch.reshape(x_full, [-1, 1, self.img_res, self.img_res])
        z_full = self.encoder(input_enc)
        return x_full, z_full

    def predict_dynamics(self, z_t, u_t):
        input_dyn = torch.cat((z_t, u_t), dim=1) #TODO: Do I have to use torch.identity after concatenation? why/not?
        z_hat_tplus = self.dynamics(input_dyn)
        return z_hat_tplus
        
    def compute_grammian(self, z_t, u_t, z_hat_tplus, retain_graph_after_last_grad=True):
        z_hat_tplus_zero = z_hat_tplus[:, 0]
        z_hat_tplus_one = z_hat_tplus[:, 1]

        grad_zh0_zt = torch.autograd.grad(z_hat_tplus_zero, z_t,
                                    grad_outputs=torch.ones(z_hat_tplus_zero.size()).to(self.dummy_param.device),
                                    retain_graph=True)[0]
        
        grad_zh1_zt = torch.autograd.grad(z_hat_tplus_one, z_t,
                                    grad_outputs=torch.ones(z_hat_tplus_one.size()).to(self.dummy_param.device),
                                    retain_graph=True)[0]
        
        grad_zh0_ut = torch.autograd.grad(z_hat_tplus_zero, u_t,
                                    grad_outputs=torch.ones(z_hat_tplus_zero.size()).to(self.dummy_param.device),
                                    retain_graph=True)[0]

        grad_zh1_ut = torch.autograd.grad(z_hat_tplus_one, u_t,
                                    grad_outputs=torch.ones(z_hat_tplus_one.size()).to(self.dummy_param.device),
                                    retain_graph=retain_graph_after_last_grad)[0]

        A = torch.stack([grad_zh0_zt, grad_zh1_zt], dim=1) # N x D_z_hat x D_z  (D_z_hat = D_z = 2)
        B = torch.stack([grad_zh0_ut, grad_zh1_ut], dim=1) # N x D_z_hat x D_c  (D_c = 2)
        c = self.__expand_dims(z_hat_tplus) - torch.bmm(A, self.__expand_dims(z_t)) - torch.bmm(B, self.__expand_dims(u_t))
        AT = torch.transpose(A, 1, 2) # Preserve the batch dimension 0 and transpose dimentions 1 and 2
        BT = torch.transpose(B, 1, 2)
        
        G = torch.bmm(A, torch.bmm(B, torch.bmm(BT, AT))) # N x D_z x D_z (remember D_z_hat = D_z)
        offset_for_invertible = (0.0001 * torch.eye(G.size()[1])).expand_as(G).to(self.dummy_param.device)
        with torch.no_grad(): # Is this needed? Probably not...
            G_inv = torch.inverse(G + offset_for_invertible) # N x D_z x D_z
            
        return G_inv, A, c
        
    def forward(self, x_t, x_tplus, x_empty, u_t):
        '''
        x_t, x_tplus, x_empty must be of shape [N, C*H*W] where, N = batch_size, 
        C = Channels, H = Height, W = Width of image.
        u is of shape [N, D_c] where D_c = Control Dimension.
        '''
        batch_size = u_t.size()[0]

        x_full, z_full = self.encode(x_t, x_tplus)
        
        z_t = z_full[:batch_size, :]
        z_tplus = z_full[batch_size:, :]

        z_hat_tplus = self.predict_dynamics(z_t, u_t)
        
        input_dec = torch.cat((z_t, z_hat_tplus), dim=0) #TODO: Again, should I use torch.identity here?
        output_dec = self.decoder(input_dec)
        
        x_empty_full = torch.cat((x_empty, x_empty), dim=0)
        input_env = torch.reshape(x_empty_full, [-1, 1, self.img_res, self.img_res]) #TODO: identity?
        output_env = self.environment(input_env)
        
        input_last = torch.cat((output_dec, output_env), dim=1)
        x_hat_full = self.last_layer(input_last)
        
        return x_full, z_t, z_tplus, x_hat_full, z_hat_tplus
        
    def compute_loss(self, u_t, x_full, z_t, z_tplus, x_hat_full, z_hat_tplus, L2_weight):
        '''
        From a typical pytorch code principles, perhaps this should be a different class
        than the net class itself but that's ok for now.
        '''
        G_inv, _, _ = self.compute_grammian(z_t, u_t, z_hat_tplus)

        z_diff = self.__expand_dims(z_hat_tplus) - self.__expand_dims(z_tplus) # N x D_z x 1
        z_diff_T = torch.transpose(z_diff, 1, 2) # N x 1 x D_z
        
        predict_loss_G = torch.sum(torch.abs(torch.bmm(z_diff_T, torch.bmm(G_inv, z_diff)))) # N x 1 before sum, scalar after sum
        predict_loss_L2 = F.mse_loss(z_hat_tplus, z_tplus,  reduction='mean')
        predict_loss = predict_loss_G * (1 - L2_weight) + predict_loss_L2 * L2_weight
        
        recon_loss = F.mse_loss(x_hat_full, x_full, reduction='mean')
        total_loss = predict_loss + recon_loss
        
        return total_loss, predict_loss_G, predict_loss_L2, recon_loss
    
    def __expand_dims(self, input):
        return input.unsqueeze(input.dim())
        
class CollisionChecker(nn.Module):
    def __init__(self, img_res=32, z_dim=2, u_dim=2):
        super(CollisionChecker, self).__init__()
        self.img_res = img_res
        self.x_dim = img_res*img_res
        self.z_dim = z_dim
        self.u_dim = u_dim
        conv_filters = 10
        padding=3
        kernel_size=7
        fc_dim = 128
        self.imageConvLayer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv_filters, kernel_size=kernel_size, padding=padding), # kernel_size different than original
            nn.ReLU(),
            nn.Conv2d(conv_filters, conv_filters, kernel_size, padding),
            nn.ReLU(),
            nn.Conv2d(conv_filters, conv_filters, kernel_size, padding),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_filters*self.x_dim, 4*fc_dim),
            nn.ReLU()
        )
        
        self.latentDenseLayer = nn.Sequential(
            nn.Linear(2*self.z_dim, 4*fc_dim),
            nn.ReLU(),
        )
        
        self.finalDenseLayer = nn.Sequential(
            nn.Linear(8*fc_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, 1)
        )
    
    def image_representation(self, x):
        inputs_img = torch.reshape(x, [-1, 1, self.img_res, self.img_res])
        img_dense_out = self.imageConvLayer(inputs_img)
        
    def forward(self, z1, z2, img_dense_out):        
        inputs_lat = torch.cat((z1, z2), dim=1)
        lat_dense_out = self.latentDenseLayer(inputs_lat)
        
        inputs_final = torch.cat((lat_dense_out, img_dense_out), dim=1)
        collision_prediction = self.finalDenseLayer(inputs_final)
        
        return collision_prediction
    
    def compute_loss(self, labels, logits):
        return F.binary_cross_entropy_with_logits(logits, labels, reduction='sum')
    

class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3)

        feature = feature.view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        return expected_xy.view(-1, self.channel*2)