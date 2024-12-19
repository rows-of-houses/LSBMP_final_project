from model import AutoEncoder_Dynamics
from dataset import NumpyCsvDataSet
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc_dyn_net = AutoEncoder_Dynamics()
enc_dyn_net.train()
enc_dyn_net.to(device)

x_dim = 32*32
u_dim = 2
img_res = 32

batch_size = 256
lr = 1e-4
epochs = 100
data_path = '/home/dmitry/Documents/planning_course/LSBMP/data/geometric_data.csv'

dynamics_training = NumpyCsvDataSet(data_path, indeces=np.arange(4500))
dynamics_test = NumpyCsvDataSet(data_path, indeces=np.arange(4500, 5000))

dyn_training_loader = DataLoader(dynamics_training, batch_size=batch_size, shuffle=True)
dyn_test_loader = DataLoader(dynamics_training, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.Adam(enc_dyn_net.parameters(), lr=lr)

expt_prefix = 'AutoEncoderDynamics-Training-'
expt_name = expt_prefix + time.strftime('%m-%d-%H-%M-%S')
writer = SummaryWriter('runs/' + expt_name)

def train_one_epoch(model, epoch):
    model.train()
    running_loss = np.array([0.0, 0.0, 0.0, 0.0])
    for i, data in enumerate(dyn_training_loader, 0):
        data = data.float()
        x_t = data[:, :x_dim].to(device)
        x_tplus = data[:, x_dim:2*x_dim].to(device)
        x_empty = data[:, 2*x_dim:3*x_dim].to(device)
        u_t = data[:, 3*x_dim:].to(device)
        u_t.requires_grad_()
            
        x_full, z_t, z_tplus, x_hat_full, z_hat_tplus = enc_dyn_net(x_t, x_tplus, x_empty, u_t)
        l2_weight = 1.0 if epoch < 10 else 0.0 # Can use a more sophisticated L2_weight formulation
        total_loss, predict_loss_G, predict_loss_L2, recon_loss = enc_dyn_net.compute_loss(u_t, x_full, z_t, z_tplus, x_hat_full, z_hat_tplus, l2_weight)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        running_loss += np.array([total_loss.item(), predict_loss_G.item(), predict_loss_L2.item(), recon_loss.item()])
        if i % 100 == 0:
            avg_loss = running_loss / 100
            print ('[%d, %5d] loss: %.3f' % (epoch+1, i+1, avg_loss[0]))
            index = epoch * len(dyn_training_loader) + i
            writer.add_scalar('training_loss', avg_loss[0], index)
            writer.add_scalar('training_loss_G', avg_loss[1], index)
            writer.add_scalar('training_loss_L2', avg_loss[2], index)
            writer.add_scalar('training_recon_loss', avg_loss[3], index)
            running_loss[:] = 0.0
            
            
def test_one_epoch(model, epoch):
    model.train()
    running_loss = np.array([0.0, 0.0, 0.0, 0.0])
    for i, data in enumerate(dyn_test_loader, 0):
        data = data.float()
        x_t = data[:, :x_dim].to(device)
        x_tplus = data[:, x_dim:2*x_dim].to(device)
        x_empty = data[:, 2*x_dim:3*x_dim].to(device)
        u_t = data[:, 3*x_dim:].to(device)
        u_t.requires_grad_()
            
        x_full, z_t, z_tplus, x_hat_full, z_hat_tplus = enc_dyn_net(x_t, x_tplus, x_empty, u_t)
        l2_weight = 1.0 if epoch < 10 else 0.0 # Can use a more sophisticated L2_weight formulation
        total_loss, predict_loss_G, predict_loss_L2, recon_loss = enc_dyn_net.compute_loss(u_t, x_full, z_t, z_tplus, x_hat_full, z_hat_tplus, l2_weight)
                
        running_loss += np.array([total_loss.item(), predict_loss_G.item(), predict_loss_L2.item(), recon_loss.item()])
        if i % 100 == 0:
            avg_loss = running_loss / 100
            print ('[%d, %5d] test loss: %.3f' % (epoch+1, i+1, avg_loss[0]))
            index = epoch * len(dyn_training_loader) + i
            writer.add_scalar('testing_loss', avg_loss[0], index)
            writer.add_scalar('testing_loss_G', avg_loss[1], index)
            writer.add_scalar('testing_loss_L2', avg_loss[2], index)
            writer.add_scalar('testing_recon_loss', avg_loss[3], index)
            running_loss[:] = 0.0
            

for epoch in tqdm(range(epochs)):
    train_one_epoch(enc_dyn_net, epoch)
    test_one_epoch(enc_dyn_net, epoch)
# torch.save(enc_dyn_net.state_dict(), PATH)
