from model import AutoEncoder_Dynamics
from dataset_image import LSBMPDataset
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import time
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import set_random_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_random_seed(42)

enc_dyn_net = AutoEncoder_Dynamics()
enc_dyn_net.to(device)

x_dim = 32*32
u_dim = 2
img_res = 32

batch_size = 256
lr = 1e-4
epochs = 100
data_train_path = '../data/train'
data_test_path = '../data/test'
save_every = 5
mse_weight = 0
l2_weight = 0.1

dynamics_train = LSBMPDataset(data_train_path)
dynamics_test = LSBMPDataset(data_test_path)

dyn_train_loader = DataLoader(dynamics_train, batch_size=batch_size, shuffle=True)
dyn_test_loader = DataLoader(dynamics_test, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.Adam(enc_dyn_net.parameters(), lr=lr)

expt_prefix = 'AutoEncoder-Training-'
expt_name = expt_prefix + time.strftime('%m-%d-%H-%M-%S')
writer = SummaryWriter('runs/' + expt_name)

def train_one_epoch(model, epoch):
    model.train()
    running_loss = np.array([0.0, 0.0, 0.0, 0.0])
    for i, (x_t, x_tplus, x_empty, u_t) in enumerate(dyn_train_loader, 0):
        x_t = x_t.to(device)
        x_tplus = x_tplus.to(device)
        x_empty = x_empty.to(device)
        u_t = u_t.to(device)
        u_t.requires_grad_()
            
        x_full, z_t, z_tplus, x_hat_full, z_hat_tplus = enc_dyn_net(x_t, x_tplus, x_empty, u_t)
        # l2_weight = 1.0 if epoch < 10 else 0.0 # Can use a more sophisticated L2_weight formulation
        total_loss, predict_loss_G, predict_loss_L2, recon_loss = enc_dyn_net.compute_loss(u_t, x_full, z_t, z_tplus, x_hat_full, z_hat_tplus, l2_weight, mse_weight)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        running_loss += np.array([total_loss.item(), predict_loss_G.item(), predict_loss_L2.item(), recon_loss.item()])
        
    avg_loss = running_loss / (i + 1)
    writer.add_scalar('train_loss', avg_loss[0], epoch)
    writer.add_scalar('train_loss_G', avg_loss[1], epoch)
    writer.add_scalar('train_loss_L2', avg_loss[2], epoch)
    writer.add_scalar('train_recon_loss', avg_loss[3], epoch)
            
            
def test_one_epoch(model, epoch):
    model.eval()
    running_loss = np.array([0.0, 0.0, 0.0, 0.0])
    
    with torch.no_grad():
        for i, (x_t, x_tplus, x_empty, u_t) in enumerate(dyn_test_loader, 0):
            x_t = x_t.to(device)
            x_tplus = x_tplus.to(device)
            x_empty = x_empty.to(device)
            u_t = u_t.to(device)
            u_t.requires_grad_()     
                        
            x_full, z_t, z_tplus, x_hat_full, z_hat_tplus = enc_dyn_net(x_t, x_tplus, x_empty, u_t)
            # l2_weight = 1.0 if epoch < 10 else 0.0 # Can use a more sophisticated L2_weight formulation
            total_loss, predict_loss_G, predict_loss_L2, recon_loss = enc_dyn_net.compute_loss(u_t, x_full, z_t, z_tplus, x_hat_full, z_hat_tplus, l2_weight, mse_weight)
                    
            running_loss += np.array([total_loss.item(), predict_loss_G.item(), predict_loss_L2.item(), recon_loss.item()])

    
        
    avg_loss = running_loss / (i + 1)
    writer.add_scalar('test_loss', avg_loss[0], epoch)
    writer.add_scalar('test_loss_G', avg_loss[1], epoch)
    writer.add_scalar('test_loss_L2', avg_loss[2], epoch)
    writer.add_scalar('test_recon_loss', avg_loss[3], epoch)
    writer.add_text('test_latent1', z_t[0].__repr__(), epoch)
    writer.add_text('test_latent2', z_tplus[0].__repr__(), epoch)
    writer.add_image('test_x_t', x_t[:1], epoch)
    writer.add_image('test_x_tplus', x_tplus[:1], epoch)
    writer.add_image('test_x_hat', x_hat_full[:1], epoch)
    i = x_hat_full.shape[0] // 2
    writer.add_image('test_x_hat_tplus', x_hat_full[i:i + 1], epoch)
            
for epoch in tqdm(range(epochs)):
    train_one_epoch(enc_dyn_net, epoch)
    test_one_epoch(enc_dyn_net, epoch)
    # torch.save(enc_dyn_net.state_dict(), "latest.ckpt")
    # if epoch % save_every == 0:
    #     torch.save(enc_dyn_net.state_dict(), f"epoch_{epoch}.ckpt")
os.makedirs("../checkpoints", exist_ok=True)
torch.save(enc_dyn_net.state_dict(), "../checkpoints/dyn_model.pth")

writer.close()
