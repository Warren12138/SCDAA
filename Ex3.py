import torch
import torch.nn as nn
from torch.autograd import Variable as V
from lib.Exercise1_1 import LQRSolver
from torch.utils.data import TensorDataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import time 

Proj_dtype = torch.double
Proj_device = 'cpu' 

class DGMNN2(nn.Module):
    def __init__(self):
        super(DGMNN2, self).__init__()
        self.layer1 = nn.Linear(3, 100)
        self.layer2 = nn.Linear(100, 200)
        self.layer3 = nn.Linear(200, 400)
        self.layer4 = nn.Linear(400, 400)
        self.layer5 = nn.Linear(400, 400)
        self.layer6 = nn.Linear(400, 200)
        self.layer7 = nn.Linear(200, 100)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.output = nn.Linear(100, 1)
        
    def forward(self, x):

        out1 = self.tanh(self.layer1(x))

        out2 = self.tanh(self.layer2(out1))

        out3 = self.relu(self.layer3(out2))

        out4 = self.relu(self.layer4(out3))

        out5 = self.relu(self.layer5(out4)+out3)

        out6 = self.relu(self.layer6(out5)+out2)

        out7 = self.tanh(self.layer7(out6)+out1)

        return self.output(out7)
def get_hessian(grad,x):
    Hessian = torch.tensor([], device = Proj_device)
    
    for i in range(len(x)):
        hessian = torch.tensor([], device = Proj_device)
        for j in range(len(grad[i])):
            u_xxi = torch.autograd.grad(grad[i][j], x, grad_outputs=torch.ones_like(grad[i][j]), retain_graph=True,create_graph=True, allow_unused=True)[0]           
            hessian = torch.cat((hessian, u_xxi[i].unsqueeze(0)))
        Hessian = torch.cat((Hessian, hessian.unsqueeze(0)),dim = 0)
        #print(Hessian)
    return Hessian

def get_hessian_(model,t,x):
    Hessian = torch.tensor([], device = Proj_device)
    for i in range(len(t)):
        x_i = V(x[i],requires_grad=True)
        input = torch.cat(((t[i]).unsqueeze(0), x_i),dim=0)
        u_in = model(input)
        grad = torch.autograd.grad(u_in, x_i, grad_outputs=torch.ones_like(u_in), create_graph=True, retain_graph=True)[0]
        hessian = torch.tensor([], device = Proj_device)
        for j in range(len(grad)):
            u_xxi = torch.autograd.grad(grad[j], x_i, grad_outputs=torch.ones_like(grad[j]), retain_graph=True,create_graph=True, allow_unused=True)[0]           
            hessian = torch.cat((hessian, u_xxi.unsqueeze(0)))
        Hessian = torch.cat((Hessian, hessian.unsqueeze(0)),dim = 0)
    return Hessian

def pde_residual(model, t, x):
    
    input = torch.cat((t.unsqueeze(1), x),dim=1)
    
    u = model(input)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    u_xx = get_hessian(u_x,x)

#    u_xx = get_hessian_(model,t,x)
    
    residual = u_t + 0.5 * torch.einsum('bii->b', sigma @ sigma.transpose(1,2) @ u_xx) + (u_x.unsqueeze(1) @ (H @ x.unsqueeze(1).transpose(1,2)) + u_x.unsqueeze(1) @ M @ alpha + x.unsqueeze(1) @ C @ x.unsqueeze(1).transpose(1,2) + alpha.transpose(1,2) @ D @ alpha).squeeze()
    
    return residual

def boundary_condition(model,t,x):

    
    T_input = T * torch.ones_like(t)

    input = torch.cat((T_input.unsqueeze(1), x),dim=1)
    u = model(input)

    return u - (x.unsqueeze(1) @ R @ x.unsqueeze(1).transpose(1,2)).squeeze()

def total_residual(model, t, x):
    
    residual_loss = pde_residual(model, t, x).pow(2).mean()
    boundary_loss = boundary_condition(model,t,x).pow(2).mean()
    
    return residual_loss + boundary_loss

def new_data(num_samples):
    #num_samples = 10000
    t_samples = T * torch.rand(num_samples, dtype=Proj_dtype, device = Proj_device, requires_grad=True)
    x_ends = torch.tensor([-3,3], dtype = Proj_dtype)
    x_samples = x_ends[0] + (x_ends[1]- x_ends[0]) * torch.rand(num_samples , 2, dtype=Proj_dtype, device = Proj_device, requires_grad=True)
    return t_samples,x_samples

H = torch.tensor([[1.2, 0.8], [-0.6, 0.9]], dtype=Proj_dtype, device = Proj_device)
M = torch.tensor([[0.5,0.7], [0.3,1.0]], dtype=Proj_dtype, device = Proj_device)
sigma = torch.tensor([[[0.08],[0.11]]], dtype=Proj_dtype, device = Proj_device)
alpha = torch.tensor([[[1],[1]]], dtype=Proj_dtype, device = Proj_device)
C = torch.tensor([[1.6, 0.0], [0.0, 1.1]], dtype=Proj_dtype, device = Proj_device)
D = torch.tensor([[0.5, 0.0], [0.0, 0.7]], dtype=Proj_dtype, device = Proj_device)
R = torch.tensor([[0.9, 0.0], [0.0, 1.0]], dtype=Proj_dtype, device = Proj_device)
T = torch.tensor(1.0, dtype=Proj_dtype, device = Proj_device)

model_DGM = DGMNN2().double()

#stat_dict = torch.load('model2_DGM_state_dict.pt', map_location=torch.device('cpu'))
#model_DGM.load_state_dict(stat_dict)

#model_DGM = DGMNN().float().to(Proj_device)
# Prepare for training
optimizer_DGM = torch.optim.Adam(model_DGM.parameters(), lr=0.01)
scheduler_DGM = lr_scheduler.ExponentialLR(optimizer_DGM, gamma=0.9)

epoch_losses = []

Batch_size = 8
epochs = 100

for batch in range(Batch_size):
  
    print(f'Batch {batch+1}/{Batch_size}'+'\n')
    
    t_data,x_data = new_data(10000)
    dataset = TensorDataset(t_data,x_data)
    dataloader = DataLoader(dataset, batch_size=10000, shuffle=True)

    for epoch in range(epochs):

        model_DGM.train()
        total_loss = 0
        
        for batch_idx, (t_data_,x_data_) in enumerate(dataloader):
            optimizer_DGM.zero_grad()
            t_v = V(t_data_,requires_grad=True)
            x_v = V(x_data_,requires_grad=True)
            loss = total_residual(model_DGM, t_v, x_v) 
            loss.backward(retain_graph=False)
            #loss.backward(retain_graph=True)
            optimizer_DGM.step()
            total_loss += loss.item()
        epoch_losses.append(total_loss / len(dataloader))
        
        scheduler_DGM.step()
        if epoch == 0:
            print(f'Epoch {epoch+1}/{epochs} \t Loss: {total_loss / len(dataloader)}')
        if(epoch+1)% 10 == 0:
            torch.save(model_DGM.state_dict(), 'model3_DGM_state_dict.pt')
        if (epoch+1) % 1 == 0:
            print(f'Epoch {epoch+1}/{epochs} \t Loss: {total_loss / len(dataloader)}')

    print('\n')