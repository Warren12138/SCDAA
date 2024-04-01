
import torch
from Exercise1_1_MPS import LQRSolver_MPS
from Exercise1_1 import LQRSolver

H = torch.tensor([[1.2, 0.8], [-0.6, 0.9]], dtype=torch.float32, device = 'mps')
M = torch.tensor([[0.5,0.7], [0.3,1.0]], dtype=torch.float32, device = 'mps')
sigma = torch.tensor([[[0.8],[1.1]]], dtype=torch.float32, device = 'mps') 
C = torch.tensor([[1.6, 0.0], [0.0, 1.1]], dtype=torch.float32, device = 'mps')  # Positive semi-definite
D = torch.tensor([[0.5, 0.0], [0.0, 0.7]], dtype=torch.float32, device = 'mps')  # Positive definite
R = torch.tensor([[0.9, 0.0], [0.0, 1.0]], dtype=torch.float32, device = 'mps')  # Positive semi-definite
T = torch.tensor(1.0, dtype=torch.float32, device = 'mps')
method = 'rk4'
solver = LQRSolver_MPS(H, M, sigma, C, D, R, T, method)

N = int(512)
batch_size_MC = int(100)

t0 = torch.tensor(0.1,dtype = torch.float32, device = 'mps')

time_grid_for_MC = torch.linspace(t0,T,N,dtype = torch.float32, device = 'mps')

X0 = 0.5*torch.ones([1,1,2], dtype=torch.float32, device = 'mps')
dt_for_MC = time_grid_for_MC[1:]-time_grid_for_MC[:-1]
S = solver.solve_riccati_ode(time_grid_for_MC.unsqueeze(0)).squeeze()
multX = - M@torch.linalg.inv(D)@M.T
multa = - torch.linalg.inv(D)@M.T@S


#Implicit code by Yuebo Yang Mar.25.2024


I = torch.eye(len(X0.squeeze()),dtype = torch.float32, device = 'mps')

A = (I - dt_for_MC.unsqueeze(-1).unsqueeze(-1)*(H + multX @ S[1:]))

diagA = torch.diagonal(A,dim1=-2, dim2=-1).flatten()

A_subs = A.flip(dims=[-1])
A_subs_f = torch.diagonal(A_subs, dim1=-2, dim2=-1).flatten()
u_A_subs_f = A_subs_f[::2] 
l_A_subs_f = A_subs_f[1::2] 

A_subs_0s = torch.zeros_like(u_A_subs_f,dtype = torch.float32, device = 'mps')[:-1]
A_u = torch.zeros(len(X0.squeeze()) -1 + u_A_subs_f.numel()*2, dtype = torch.float32, device = 'mps')
A_l = torch.zeros(len(X0.squeeze()) -1 + u_A_subs_f.numel()*2, dtype = torch.float32, device = 'mps')
A_u[len(X0.squeeze())::2] = u_A_subs_f
A_u[len(X0.squeeze())+1::2] = A_subs_0s
A_l[len(X0.squeeze())::2] = l_A_subs_f
A_l[len(X0.squeeze())+1::2] = A_subs_0s

negsub = -torch.ones(diagA.numel(), dtype = torch.float32, device = 'mps')
matrix_negsub = torch.diag(negsub, diagonal=-len(X0.squeeze()))
matrix_A_u = torch.diag(A_u, diagonal=1)
matrix_A_l = torch.diag(A_l, diagonal=-1)
matrix_diag = torch.diag(torch.cat((torch.ones(len(X0.squeeze()), dtype = torch.float32, device = 'mps'), diagA)))
AA = matrix_diag+matrix_A_l+matrix_A_u+matrix_negsub

b = torch.cat((X0.squeeze(),sigma.squeeze().repeat(N-1)*torch.sqrt(dt_for_MC).repeat_interleave(len(X0.squeeze()))*torch.randn(len(sigma.squeeze().repeat(N-1)),dtype=torch.float32, device = 'mps')))

#X_0_N = (torch.inverse(AA)@b).reshape(N,1,2)

X_0_N = torch.linalg.tensorsolve(AA,b).reshape(N,1,2)

alpha = multa@X_0_N.transpose(1,2)

int_ = X_0_N@C@X_0_N.transpose(1,2) + alpha.transpose(1,2)@D@alpha

J = X_0_N[-1]@R@X_0_N[-1].T + torch.tensor(0.5, dtype=torch.float32, device = 'mps')*dt_for_MC@((int_.squeeze(1)[1:]+int_.squeeze(1)[:-1]))

print(J)