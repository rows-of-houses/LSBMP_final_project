from model import CollisionChecker
from dataset_collision import CollisionDataset
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

model = CollisionChecker()
model.train()
model.to(device)

x_dim = 32*32
u_dim = 2
img_res = 32

batch_size = 256
lr = 1e-4
epochs = 100
data_train_path = '../data/train_collision'
data_test_path = '../data/test_collision'

trainset = CollisionDataset(data_train_path)
testset = CollisionDataset(data_test_path)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

expt_prefix = 'AutoEncoderCollision-Training-'
expt_name = expt_prefix + time.strftime('%m-%d-%H-%M-%S')
writer = SummaryWriter('runs/' + expt_name)

def train_one_epoch(model, epoch):
    model.train()
    running_loss = 0
    for i, (z1, z2, x_empty, labels) in enumerate(train_loader, 0):
        z1 = z1.to(device)
        z2 = z2.to(device)
        x_empty = x_empty.to(device)
            
        img_dense_out = model.image_representation(x_empty)
        logits = model(z1, z2, img_dense_out)
        total_loss = model.compute_loss(labels, logits)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
        if i % 100 == 0:
            avg_loss = running_loss / 100
            print ('[%d, %5d] loss: %.3f' % (epoch+1, i+1, avg_loss))
            index = epoch * len(dyn_train_loader) + i
            writer.add_scalar('train_loss', avg_loss, index)
            running_loss = 0.0
            
            
def test_one_epoch(model, epoch):
    model.eval()
    running_loss = 0
    accuracy = 0
    with torch.no_grad():
        for i, (z1, z2, x_empty, labels) in enumerate(test_loader, 0):
            z1 = z1.to(device)
            z2 = z2.to(device)
            x_empty = x_empty.to(device)
                
            img_dense_out = model.image_representation(x_empty)
            logits = model(z1, z2, img_dense_out)
            total_loss = model.compute_loss(labels, logits)
            running_loss += total_loss.item()

            accuracy += ((logits > 0) == labels).sum()
    avg_loss = running_loss * batch_size / len(testset)
    avg_accuracy = accuracy / len(testset)
    print ('[%d, %5d] loss: %.3f accuracy: %.3f' % (epoch+1, i+1, avg_loss, avg_accuracy))
    index = epoch * len(dyn_train_loader)
    writer.add_scalar('test_loss', avg_loss, index)
    writer.add_scalar('test_accuracy', avg_accuracy, index)
            

for epoch in tqdm(range(epochs)):
    train_one_epoch(model, epoch)
    test_one_epoch(model, epoch)
os.makedirs("../checkpoints", exist_ok=True)
torch.save(enc_dyn_net.state_dict(), "../checkpoints/collision_model.pth")
