# Multiprocessing Version 
# Apple Silicon Metal Performance Shaders Accelerated

import numpy as np
import torch
from Exercise1_1 import LQRSolver
from torch.multiprocessing import Pool, set_start_method

def MonteCarloSampler(iteration, params):
    
    C = params['C'].float().to("mps")
    D = params['D'].float().to("mps")
    N = torch.tensor(params['N']).to("mps")
    R = params['R'].float().to("mps")
    S = params['S'].float().to("mps")
    X0 = params['X0'].float().to("mps")
    H = params['H'].float().to("mps")
    dt = params['dt'].float().to("mps")
    multX = params['multX'].float().to("mps")
    multa = params['multa'].float().to("mps")
    sig = params['sig'].float().to("mps")
    
    J_sample = []
    
    for ii in range(10):
        
        X_0_N = X0
        
        for i in range(N-1):
            
            X_next = ((torch.eye(2).float().to("mps")+ dt[i]*(H + multX@S[i]))@ X_0_N[i:].transpose(1,2)+sig*torch.sqrt(dt[i])*torch.randn(1).float().to("mps")).transpose(1,2)
            X_0_N = torch.cat((X_0_N, X_next), dim=0)

        alp = multa@X_0_N.transpose(1,2)
        int_ = alp.transpose(1,2)@D@alp + X_0_N@C@X_0_N.transpose(1,2)
        J = X_0_N[-1]@R@X_0_N[-1].T + torch.tensor(0.5)*dt@((int_.squeeze(1)[1:]+int_.squeeze(1)[:-1]))

        J_sample.append(J)
        
    J_sample = torch.stack(J_sample)


    return torch.mean(J_sample).unsqueeze(0).cpu()

if __name__ == '__main__':
    
    try:
        set_start_method('spawn')  
    except RuntimeError:
        pass
    
    H = torch.tensor([[1.0, 2.0], [-2.0, -3.0]], dtype=torch.double)
    M = torch.tensor([[1.0,0.0], [0.0,1.0]], dtype=torch.double)
    sigma = torch.tensor([[[0.5249],[0.4072]]], dtype=torch.double) 
    C = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.double)  # Positive semi-definite
    D = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.double)  # Positive definite
    R = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.double)  # Positive semi-definite
    T = torch.tensor(1.0, dtype=torch.double)
    method = 'rk4'
    
    solver = LQRSolver(H, M, sigma, C, D, R, T, method)
    
    N = 5000
    t0 = torch.tensor(0.1,dtype = torch.double)
    
    time_grid_for_MC = torch.linspace(t0,T,N,dtype = torch.double)
    
    X0 = 0.5*torch.ones([1,1,2], dtype=torch.double)
    dt_for_MC = time_grid_for_MC[1:]-time_grid_for_MC[:-1]
    S = solver.solve_riccati_ode(time_grid_for_MC.unsqueeze(0)).squeeze()
    multpX = - M@torch.linalg.inv(D)@M.T
    multpalp = -torch.linalg.inv(D)@M.T@S
    
    params = {
    'C': C,
    'D': D,
    'N': N,
    'R': R,
    'S': S,
    'X0': X0,
    'H': H,
    'dt': dt_for_MC,
    'multX': multpX,
    'multa':multpalp,
    'S': S,
    'sig': sigma,
    }
    
    iterations = list(range(500))
    
    with Pool(processes = 12) as pool:  
        J_list = pool.starmap(MonteCarloSampler, [(iteration, params) for iteration in iterations])
        
    J_list_tensor = torch.cat(J_list, dim=0)  
    print(torch.sort(J_list_tensor)[0][:20])