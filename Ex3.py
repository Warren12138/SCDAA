import torch
import torch.nn as nn
import torch.nn.init as init

from lib.Exercise1_1 import LQRSolver
from torch.utils.data import TensorDataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import os
from datetime import datetime

Proj_dtype = torch.double
Proj_device = 'cpu'

class DGMhiddenlayerYYBver(nn.Module):

    # From the original paper of Justin's, presented by Yuebo Yang Apr.9th 2024
    
    def __init__(self, input_f, output_f, activation = 'tanh'):
        
        super(DGMhiddenlayerYYBver, self).__init__()

        self.input_f = input_f
        self.output_f = output_f

        # Params

        # Zl's

        self.Uzl = nn.Parameter(torch.Tensor(output_f, input_f))
        self.Wzl = nn.Parameter(torch.Tensor(output_f, output_f))
        self.bzl = nn.Parameter(torch.Tensor(output_f))

        # Gl's

        self.Ugl = nn.Parameter(torch.Tensor(output_f, input_f))
        self.Wgl = nn.Parameter(torch.Tensor(output_f, output_f))
        self.bgl = nn.Parameter(torch.Tensor(output_f))

        # Rl's

        self.Url = nn.Parameter(torch.Tensor(output_f, input_f))
        self.Wrl = nn.Parameter(torch.Tensor(output_f, output_f))
        self.brl = nn.Parameter(torch.Tensor(output_f))

        # Hl's

        self.Uhl = nn.Parameter(torch.Tensor(output_f, input_f))
        self.Whl = nn.Parameter(torch.Tensor(output_f, output_f))
        self.bhl = nn.Parameter(torch.Tensor(output_f))


        if activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = None 

        self.init_method = 'normal' # or 'uniform'

        self._initialize_params()

    def _initialize_params(self):

        if self.init_method == 'uniform':
            for param in self.parameters():
                if param.dim() > 1:
                    init.xavier_uniform_(param)  
                else:
                    init.constant_(param, 0) 
                    
        if self.init_method == 'normal':
            for param in self.parameters():
                if param.dim() > 1:
                    init.xavier_normal_(param)  
                else:
                    init.constant_(param, 0) 

    def forward(self, x, S1, Sl):

        Zl = self.activation(torch.mm(x, self.Uzl.t())+ torch.mm(Sl, self.Wzl.t()) + self.bzl)

        Gl = self.activation(torch.mm(x, self.Ugl.t())+ torch.mm(S1, self.Wgl.t()) + self.bgl)

        Rl = self.activation(torch.mm(x, self.Url.t())+ torch.mm(Sl, self.Wrl.t()) + self.brl)

        Hl = self.activation(torch.mm(x, self.Uhl.t())+ torch.mm(torch.mul(Sl,Rl), self.Whl.t()) + self.bhl)

        Sl_1 = torch.mul((1-Gl),Hl) + torch.mul(Zl,Sl)

        return Sl_1


class DGMNN_YYBver(nn.Module):

    # From the original paper of Justin's, presented by Yuebo Yang Apr.9th 2024

    def __init__(self, init_method = 'uniform'):
        super(DGMNN_YYBver, self).__init__()

        self.nodenum = 50

        self.layer1 = DGMhiddenlayerYYBver(3, self.nodenum)
        self.layer2 = DGMhiddenlayerYYBver(3, self.nodenum)
        self.layer3 = DGMhiddenlayerYYBver(3, self.nodenum)

        self.tanh = nn.Tanh()

        # Params

        # S1's

        self.W1 = nn.Parameter(torch.Tensor(self.nodenum, 3))
        self.b1 = nn.Parameter(torch.Tensor(self.nodenum))

        # Output's

        self.W = nn.Parameter(torch.Tensor(1, self.nodenum))
        self.b = nn.Parameter(torch.Tensor(1))

        self.activation = torch.tanh

        self.init_method = 'normal' # or 'uniform'

        self._initialize_params()

    def _initialize_params(self):

        if self.init_method == 'uniform':
            for param in self.parameters():
                if param.dim() > 1:
                    init.xavier_uniform_(param)  
                else:
                    init.constant_(param, 0) 
                    
        if self.init_method == 'normal':
            for param in self.parameters():
                if param.dim() > 1:
                    init.xavier_normal_(param)  
                else:
                    init.constant_(param, 0) 

        
    def forward(self, x):

        S_1 = self.activation(torch.mm(x, self.W1.t()) + self.b1)
        # l=1
        S_2 = self.layer1(x,S_1,S_1)
        # l=2
        S_3 = self.layer2(x,S_1,S_2)
        # l=3
        S_4 = self.layer3(x,S_1,S_3)

        output = torch.mm(S_4, self.W.t()) + self.b

        return output

def get_hessian(grad,x):
    Hessian = torch.tensor([], device = Proj_device)
    
    for i in range(len(x)):
        hessian = torch.tensor([], device = Proj_device)
        for j in range(len(grad[i])):
            u_xxi = torch.autograd.grad(grad[i][j], x, grad_outputs=torch.ones_like(grad[i][j]), retain_graph=True,create_graph=True, allow_unused=True)[0]           
            hessian = torch.cat((hessian, u_xxi[i].unsqueeze(0)))
        Hessian = torch.cat((Hessian, hessian.unsqueeze(0)),dim = 0)
        # print(Hessian)
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

    t_samples = T * torch.rand(num_samples, dtype=Proj_dtype, device = Proj_device, requires_grad=False)
    x_ends = torch.tensor([-3,3], dtype = Proj_dtype)
    x_samples = x_ends[0] + (x_ends[1]- x_ends[0]) * torch.rand(num_samples , 2, dtype=Proj_dtype, device = Proj_device, requires_grad=False)
    return t_samples,x_samples

def main():
    model_DGM = DGMNN_YYBver().double()
    optimizer_DGM = torch.optim.Adam(model_DGM.parameters(), lr=0.0001)
    scheduler_DGM = lr_scheduler.ExponentialLR(optimizer_DGM, gamma=0.9)


    continue_training = input("Do you want to continue training or start a new one? (c/n): ").lower() == 'c'

    model_save_path = 'model3_DGM_state_dict.pt'
    optimizer_save_path = 'optimizer_DGM_state.pt'

    if continue_training and os.path.exists(model_save_path) and os.path.exists(optimizer_save_path):
        
        model_DGM.load_state_dict(torch.load(model_save_path))
        optimizer_DGM.load_state_dict(torch.load(optimizer_save_path))
        print("Continuing training from saved state.")
    else:
        print("Starting training from scratch.")

    epoch_losses = []

    iterations = 50
    epochs = 100

    patience = 10

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    filename = f'Ex3_training_loss_{timestamp}.dat'

    with open('filename', 'w') as f:

        for iteration in range(iterations):
            print(f'Iteration {iteration+1}/{iterations}'+'\n')
            
            t_data,x_data = new_data(1000)
            dataset = TensorDataset(t_data,x_data)
            dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

            best_loss = float('inf')
            patience_counter = 0  

            for epoch in range(epochs):

                model_DGM.train()
                total_loss = 0
                
                for batch_idx, (_t_data,_x_data) in enumerate(dataloader):
                    optimizer_DGM.zero_grad()
                    t_data = _t_data.clone().requires_grad_(True)
                    x_data = _x_data.clone().requires_grad_(True)
                    loss = total_residual(model_DGM, t_data, x_data) 
                    loss.backward()
                    optimizer_DGM.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(dataloader)
                epoch_losses.append(avg_loss)
                
                scheduler_DGM.step()

                f.write(f'Iteration {iteration+1}, Epoch {epoch+1}, Loss: {avg_loss}\n')

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0  
                    torch.save(model_DGM.state_dict(), model_save_path)
                    torch.save(optimizer_DGM.state_dict(), optimizer_save_path)
                else:
                    patience_counter += 1

                if epoch == 0 or (epoch+1) % 5 == 0:
                    print(f'Epoch {epoch+1}/{epochs} \t Loss: {avg_loss}')
                
                
                if patience_counter >= patience:
                    print(f'Early stopping triggered at epoch {epoch+1}')
                    break  
            print('\n')
            
    model_DGM.eval()

if __name__ == '__main__':

    H = torch.tensor([[1.2, 0.8], [-0.6, 0.9]], dtype=Proj_dtype, device = Proj_device)
    M = torch.tensor([[0.5,0.7], [0.3,1.0]], dtype=Proj_dtype, device = Proj_device)
    sigma = torch.tensor([[[0.08],[0.11]]], dtype=Proj_dtype, device = Proj_device)
    alpha = torch.tensor([[[1],[1]]], dtype=Proj_dtype, device = Proj_device)
    C = torch.tensor([[1.6, 0.0], [0.0, 1.1]], dtype=Proj_dtype, device = Proj_device)
    D = torch.tensor([[0.5, 0.0], [0.0, 0.7]], dtype=Proj_dtype, device = Proj_device)
    R = torch.tensor([[0.9, 0.0], [0.0, 1.0]], dtype=Proj_dtype, device = Proj_device)
    T = torch.tensor(1.0, dtype=Proj_dtype, device = Proj_device)
    
    main()
