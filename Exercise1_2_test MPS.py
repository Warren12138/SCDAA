#Multiprocessing Version

import numpy as np
import torch
from Exercise1_1_MPS import LQRSolver_MPS
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, set_start_method, Value, Lock
import time
import sys

def MonteCarloSampler(iteration, params):
    
    C = params['C'].to('mps')
    D = params['D'].to('mps')
    N = params['N']
    R = params['R'].to('mps')
    S = params['S'].to('mps')
    X0 = params['X0'].to('mps')
    H = params['H'].to('mps')
    dt = params['dt'].to('mps')
    multX = params['multX'].to('mps')
    multa = params['multa'].to('mps')
    sig = params['sig'].to('mps')

    X_0_N = X0
    
    for i in range(N-1):
        
        X_next = ((torch.eye(2, dtype=torch.float32, device = 'mps')+ dt[i]*(H + multX@S[i]))@ X_0_N[i:].transpose(1,2)+sig*torch.sqrt(dt[i])*torch.randn(1,dtype=torch.float32, device = 'mps')).transpose(1,2)
        X_0_N = torch.cat((X_0_N, X_next), dim=0)

    alp = multa@X_0_N.transpose(1,2)
    int_ = X_0_N@C@X_0_N.transpose(1,2) + alp.transpose(1,2)@D@alp
    J = X_0_N[-1]@R@X_0_N[-1].T + torch.tensor(0.5, dtype=torch.float32, device = 'mps')*dt@((int_.squeeze(1)[1:]+int_.squeeze(1)[:-1]))
    
    return J

if __name__ == '__main__':
    
    # try:
    #     set_start_method('spawn')  
    # except RuntimeError:
    #     pass

    mp.set_start_method('forkserver')
    
    start_time = time.perf_counter()

    H = torch.tensor([[1.2, 0.8], [-0.6, 0.9]], dtype=torch.float32, device = 'mps')
    M = torch.tensor([[0.5,0.7], [0.3,1.0]], dtype=torch.float32, device = 'mps')
    sigma = torch.tensor([[[0.8],[1.1]]], dtype=torch.float32, device = 'mps') 
    C = torch.tensor([[1.6, 0.0], [0.0, 1.1]], dtype=torch.float32, device = 'mps')  # Positive semi-definite
    D = torch.tensor([[0.5, 0.0], [0.0, 0.7]], dtype=torch.float32, device = 'mps')  # Positive definite
    R = torch.tensor([[0.9, 0.0], [0.0, 1.0]], dtype=torch.float32, device = 'mps')  # Positive semi-definite
    T = torch.tensor(1.0, dtype=torch.float32, device = 'mps')
    method = 'rk4'
    
    solver = LQRSolver_MPS(H, M, sigma, C, D, R, T, method)
    
    N = int(5e3)
    #batch_size_MC = int(5e3)
    batch_size_MC = int(100)
    t0 = torch.tensor(0.1,dtype = torch.float32, device = 'mps')
    
    time_grid_for_MC = torch.linspace(t0,T,N,dtype = torch.float32, device = 'mps')
    
    X0 = 0.5*torch.ones([1,1,2], dtype=torch.float32, device = 'mps')
    dt_for_MC = time_grid_for_MC[1:]-time_grid_for_MC[:-1]
    S = solver.solve_riccati_ode(time_grid_for_MC.unsqueeze(0)).squeeze()
    multp_X = - M@torch.linalg.inv(D)@M.T
    multp_alp = - torch.linalg.inv(D)@M.T@S
    
    params = {
    'C': C.to('cpu'),
    'D': D.to('cpu'),
    'N': N,
    'R': R.to('cpu'),
    'S': S.to('cpu'),
    'X0': X0.to('cpu'),
    'H': H.to('cpu'),
    'dt': dt_for_MC.to('cpu'),
    'multX': multp_X.to('cpu'),
    'multa':multp_alp.to('cpu'),
    'S': S.to('cpu'),
    'sig': sigma.to('cpu'),
    }
    
    J_list = []

    times_MC = 1
    
    pool = Pool(processes = 8)

    for i in range(times_MC):

        #iterations = list(range(batch_size_MC))    

        #J_sample = pool.starmap(MonteCarloSampler, [(iteration, params) for iteration in iterations])

        J_sample = []
        iteration = 0
        for ii in range(batch_size_MC):
            J_sample.append(MonteCarloSampler(iteration, params))

        J_sample = torch.stack(J_sample)

        J_list.append(torch.mean(J_sample).unsqueeze(0))

    J_list_tensor = torch.cat(J_list, dim=0)  

    print(f"\nThe optimal value from the numerical solution is: \n{solver.value_function(t0.unsqueeze(0),X0).item()} \n ")

    print(f"The optimal values from {times_MC} time(s) of Monte Carlo Simulation (batch size {batch_size_MC} for each run) is: \n \n{torch.sort(J_list_tensor)[0]} \n")

    end_time = time.perf_counter()

    print(f"Running time: {end_time - start_time} s.")