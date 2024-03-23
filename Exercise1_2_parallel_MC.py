#Multiprocessing Version
import sys
import numpy as np
import torch
from Exercise1_1 import LQRSolver
from torch.multiprocessing import Pool, set_start_method
import time
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

def print_progress(outer, total_outer, inner, total_inner, t_i, x_i):
    sys.stdout.write('\x1b[2J\x1b[H') 
    if outer >= 0: 
        print(f"Monte Carlo Simulation (S_size {batch_size_MC} T_step_num {N}) \n Total progress (t = {t_i} s):")
        percent_outer = outer / total_outer  
        bar_length_outer = int(30 * percent_outer)
        bar_outer = '[' + '=' * bar_length_outer + ' ' * (30 - bar_length_outer) + ']'
        print(f"{bar_outer} {percent_outer * 100:.2f}% ({outer}/{total_outer})")
    
    if inner >= 0:  
        formatted_x = ', '.join(f'{item:.4f}' for item in x_i.flatten())
        print(f"Inner progress (x = [{formatted_x}]):")
        percent_inner = (inner + 1) / total_inner
        bar_length_inner = int(30 * percent_inner)
        bar_inner = '[' + '=' * bar_length_inner + ' ' * (30 - bar_length_inner) + ']'
        print(f"{bar_inner} {percent_inner * 100:.2f}% ({inner+1}/{total_inner})")


if __name__ == '__main__':
    
    try:
        set_start_method('spawn')  
    except RuntimeError:
        pass
    
    N = int(sys.argv[2])
    batch_size_MC = int(sys.argv[3])
    
    load_interval_setting = torch.load(sys.argv[1]+'/interval_setting.pt')

    t_ends = load_interval_setting['t_ends']
    t_num = load_interval_setting['t_num']
    x_ends = load_interval_setting['x_ends']
    x_num = load_interval_setting['x_num']

    t_batch_i = torch.linspace(t_ends[0],t_ends[1],t_num,dtype=torch.double)

    x1 = torch.linspace(x_ends[0][0],x_ends[0][1],x_num[0],dtype=torch.double)
    x2 = torch.linspace(x_ends[1][0],x_ends[1][1],x_num[1],dtype=torch.double)

    x_batch_i = torch.cartesian_prod(x1, x2).unsqueeze(1)

    file_path_Ex1_2 = f'Exercise1_2'
    init_for_solver = torch.load(file_path_Ex1_2+'/'+'initialization_for_solver.pt')
    
    H = init_for_solver['H']
    M = init_for_solver['M']
    sigma = init_for_solver['sigma'] 
    C = init_for_solver['C'] 
    D = init_for_solver['D'] 
    R = init_for_solver['R']
    T = init_for_solver['T']
    method = init_for_solver['method']
    
    solver = LQRSolver(H, M, sigma, C, D, R, T, method)

    J_tensor_file = torch.tensor([])
    torch.save(J_tensor_file, sys.argv[1]+'/value_MC.pt')

    for outer in range(len(t_batch_i)):

        t0 = t_batch_i[outer]
        time_grid_for_MC = torch.linspace(t0,T,N,dtype = torch.double)
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
        'X0': 0,
        'H': H,
        'dt': dt_for_MC,
        'multX': multp_X,
        'multa':multp_alp,
        'S': S,
        'sig': sigma,
        }

        pool = Pool(processes=12)

        for inner in range(len(x_batch_i)):

            print_progress(outer, len(t_batch_i), inner, len(x_batch_i), t_batch_i[outer], x_batch_i[inner])  

            X0 = x_batch_i[inner].unsqueeze(0)
            params['X0'] = X0
    
            J_list = []

            times_MC = 10

            for i in range(times_MC):

                iterations = list(range(batch_size_MC))  

                J_sample = pool.starmap(MonteCarloSampler, [(iteration, params) for iteration in iterations])

                J_sample = torch.stack(J_sample)

                J_list.append(torch.mean(J_sample).unsqueeze(0))

            J_list_tensor = torch.cat(J_list, dim=0)
            #J_list_unadded = torch.load(sys.argv[1]+'/value_MC.pt')
            J_tensor_file = torch.cat((J_tensor_file,J_list_tensor.unsqueeze(0)),dim=0)

        print_progress(outer + 1, len(t_batch_i), -1, len(x_batch_i), t_batch_i[outer], x_batch_i[inner])  
    
    torch.save(J_tensor_file, sys.argv[1]+'/value_MC.pt')
    pool.close()
    pool.join()

    print(f"\nMonte Carlo Simulation (S_size {batch_size_MC} T_step_num {N}) Finished.")
