#Multiprocessing Version

import numpy as np
import torch
from Exercise1_1 import LQRSolver
from torch.multiprocessing import Pool, set_start_method

def MonteCarloSampler(iteration, params):
    
    C = params['C']
    D = params['D']
    N = params['N']
    R = params['R']
    S = params['S']
    X0 = params['X0']
    H = params['H']
    dt = params['dt']
    multX = params['multX']
    multa = params['multa']
    sig = params['sig']
         
    X_0_N = X0
    
    for i in range(N-1):
        
        X_next = ((torch.eye(2)+ dt[i]*(H + multX@S[i]))@ X_0_N[i:].transpose(1,2)+sig*torch.sqrt(dt[i])*torch.randn(1)).transpose(1,2)
        X_0_N = torch.cat((X_0_N, X_next), dim=0)

    alp = multa@X_0_N.transpose(1,2)
    int_ = X_0_N@C@X_0_N.transpose(1,2) + alp.transpose(1,2)@D@alp
    J = X_0_N[-1]@R@X_0_N[-1].T + torch.tensor(0.5)*dt@((int_.squeeze(1)[1:]+int_.squeeze(1)[:-1]))

    return J

if __name__ == '__main__':
    
    try:
        set_start_method('spawn')  
    except RuntimeError:
        pass
    
    H = torch.tensor([[1.2, 0.8], [-0.6, 0.9]], dtype=torch.double)
    M = torch.tensor([[0.5,0.7], [0.3,1.0]], dtype=torch.double)
    sigma = torch.tensor([[[0.8],[1.1]]], dtype=torch.double) 
    C = torch.tensor([[1.6, 0.0], [0.0, 1.1]], dtype=torch.double)  # Positive semi-definite
    D = torch.tensor([[0.5, 0.0], [0.0, 0.7]], dtype=torch.double)  # Positive definite
    R = torch.tensor([[0.9, 0.0], [0.0, 1.0]], dtype=torch.double)  # Positive semi-definite
    T = torch.tensor(1.0, dtype=torch.double)
    method = 'rk4'
    
    solver = LQRSolver(H, M, sigma, C, D, R, T, method)
    
    N = 5000

    t0 = torch.tensor(0.1,dtype = torch.double)
    
    time_grid_for_MC = torch.linspace(t0,T,N,dtype = torch.double)
    
    X0 = 0.5*torch.ones([1,1,2], dtype=torch.double)
    dt_for_MC = time_grid_for_MC[1:]-time_grid_for_MC[:-1]
    S = solver.solve_riccati_ode(time_grid_for_MC.unsqueeze(0)).squeeze()
    multp_X = - M@torch.linalg.inv(D)@M.T
    multp_alp = - torch.linalg.inv(D)@M.T@S
    
    params = {
    'C': C,
    'D': D,
    'N': N,
    'R': R,
    'S': S,
    'X0': X0,
    'H': H,
    'dt': dt_for_MC,
    'multX': multp_X,
    'multa':multp_alp,
    'S': S,
    'sig': sigma,
    }
    
    J_list = []

    times_MC = 20
    batch_size_MC = 5000

    for i in range(times_MC):

        iterations = list(range(batch_size_MC))  
        
        with Pool(processes = 10) as pool:  

            J_sample = pool.starmap(MonteCarloSampler, [(iteration, params) for iteration in iterations])

        J_sample = torch.stack(J_sample)

        J_list.append(torch.mean(J_sample).unsqueeze(0))

    J_list_tensor = torch.cat(J_list, dim=0)  

    print(f"\nThe optimal value from the numerical solution is: \n{solver.value_function(t0.unsqueeze(0),X0).item()} \n ")

    print(f"The optimal values from {times_MC} time(s) of Monte Carlo Simulation (batch size {batch_size_MC} for each run) is：\n \n{torch.sort(J_list_tensor)[0]} \n")

