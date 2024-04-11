
import torch
from Exercise1_1_MPS import LQRSolver_MPS
from Exercise1_1_CPU import LQRSolver
from torch.multiprocessing import Pool, set_start_method, Value, Lock
import time

def implicit(it,params):

    #Implicit code by Yuebo Yang Mar.25.2024

    device = params['device']
    AA = params['AA'].to(device)

    C = params['C'].to(device)
    D = params['D'].to(device)
    N = params['N']
    R = params['R'].to(device)

    Step_limit = params['Step_limit']
    X0 = params['X0'].to(device)
    dt = params['dt'].to(device)
    multa = params['multa'].to(device)
    sig = params['sig'].to(device)
    
    b = torch.cat((X0.squeeze(),sig.squeeze().repeat(N-1)*torch.sqrt(dt).repeat_interleave(len(X0.squeeze()))*torch.randn(len(sig.squeeze().repeat(N-1)),dtype=torch.float32, device = device)))

    if N//Step_limit >= 1:

        X_0_N = torch.clone(X0.squeeze().repeat(N))

        for i in range(N//Step_limit):

            b_in_batch = torch.clone(b[i*Step_limit*2:(i+1)*Step_limit*2])

            if i == 0:

                _X_0 = torch.zeros_like(b_in_batch[:2],dtype = torch.float32, device = device)
            else:
                _X_0 = torch.clone(X_0_N[i*(Step_limit)*2-2:i*(Step_limit)*2])

            b_in_batch[:2] = b_in_batch[:2]+_X_0

            AA_in_batch = torch.clone(AA[i*Step_limit*2:(i+1)*Step_limit*2,i*Step_limit*2:(i+1)*Step_limit*2])

            X_0_N[i*Step_limit*2:(i+1)*Step_limit*2] = (torch.inverse(AA_in_batch)@b_in_batch)

        if (N%Step_limit != 0):

            _X_0 = torch.clone(X_0_N[-(N%Step_limit+1)*2:-(N%Step_limit)*2])

            #final_section

            b_fin = torch.clone(b[-(N%Step_limit)*2:])

            b_fin[:2] = torch.clone(b_fin[:2]+_X_0)

            AA_fin = AA[-(N%Step_limit)*2:,-(N%Step_limit)*2:]

            X_0_N[-(N%Step_limit)*2:] = (torch.inverse(AA_fin)@b_fin)

        X_0_N = X_0_N.reshape(N,1,2)

    else:

        X_0_N = (torch.inverse(AA)@b).reshape(N,1,2)

    alpha = multa@X_0_N.transpose(1,2)

    int_ = X_0_N@C@X_0_N.transpose(1,2) + alpha.transpose(1,2)@D@alpha

    J = X_0_N[-1]@R@X_0_N[-1].T + torch.tensor(0.5, dtype=torch.float32, device = device)*dt@((int_.squeeze(1)[1:]+int_.squeeze(1)[:-1]))
    
    return J.to('cpu')

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


    device = 'cpu'
    #device = 'mps'


    H = torch.tensor([[1.2, 0.8], [-0.6, 0.9]], dtype=torch.float32, device = device)
    M = torch.tensor([[0.5,0.7], [0.3,1.0]], dtype=torch.float32, device = device)
    sigma = torch.tensor([[[0.8],[1.1]]], dtype=torch.float32, device = device) 
    C = torch.tensor([[1.6, 0.0], [0.0, 1.1]], dtype=torch.float32, device = device)  # Positive semi-definite
    D = torch.tensor([[0.5, 0.0], [0.0, 0.7]], dtype=torch.float32, device = device)  # Positive definite
    R = torch.tensor([[0.9, 0.0], [0.0, 1.0]], dtype=torch.float32, device = device)  # Positive semi-definite
    T = torch.tensor(1.0, dtype=torch.float32, device = device)
    method = 'euler'
    solver = LQRSolver(H, M, sigma, C, D, R, T, method)
    #solver = LQRSolver_MPS(H, M, sigma, C, D, R, T, method)
    #sol2 = LQRSolver(H, M, sigma, C, D, R, T, 'rk4')
    #sol2 = LQRSolver_MPS(H, M, sigma, C, D, R, T, 'rk4')

    N = int(5000)

    Step_limit = 512

    batch_size_MC = int(50)

    t0 = torch.tensor(0.1,dtype = torch.float32, device = device)

    time_grid_for_MC = torch.linspace(t0,T,N,dtype = torch.float32, device = device)

    X0 = 0.5*torch.ones([1,1,2], dtype=torch.float32, device = device)
    dt_for_MC = time_grid_for_MC[1:]-time_grid_for_MC[:-1]
    #S1 = sol2.solve_riccati_ode(time_grid_for_MC.unsqueeze(0)).squeeze()
    S = solver.solve_riccati_ode(time_grid_for_MC.unsqueeze(0)).squeeze()

    #print(torch.sum(torch.abs(S1-S)))

    multX = - M@torch.linalg.inv(D)@M.T
    multa = - torch.linalg.inv(D)@M.T@S

    J_list = []

    times_MC = 1

    J_sample = []

    #Implicit code by Yuebo Yang Mar.25.2024

    I = torch.eye(len(X0.squeeze()),dtype = torch.float32, device = device)

    S = S.reshape(N,2,2)

    A = (I - dt_for_MC.unsqueeze(-1).unsqueeze(-1)*(H + multX @ S[1:]))

    diagA = torch.diagonal(A,dim1=-2, dim2=-1).flatten()

    A_subs = A.flip(dims=[-1])
    A_subs_f = torch.diagonal(A_subs, dim1=-2, dim2=-1).flatten()
    u_A_subs_f = A_subs_f[::2] 
    l_A_subs_f = A_subs_f[1::2] 

    A_subs_0s = torch.zeros_like(u_A_subs_f,dtype = torch.float32, device = device)[:-1]
    A_u = torch.zeros(len(X0.squeeze()) -1 + u_A_subs_f.numel()*2, dtype = torch.float32, device = device)
    A_l = torch.zeros(len(X0.squeeze()) -1 + u_A_subs_f.numel()*2, dtype = torch.float32, device = device)
    A_u[len(X0.squeeze())::2] = u_A_subs_f
    A_u[len(X0.squeeze())+1::2] = A_subs_0s
    A_l[len(X0.squeeze())::2] = l_A_subs_f
    A_l[len(X0.squeeze())+1::2] = A_subs_0s

    negsub = -torch.ones(diagA.numel(), dtype = torch.float32, device = device)
    matrix_negsub = torch.diag(negsub, diagonal=-len(X0.squeeze()))
    matrix_A_u = torch.diag(A_u, diagonal=1)
    matrix_A_l = torch.diag(A_l, diagonal=-1)
    matrix_diag = torch.diag(torch.cat((torch.ones(len(X0.squeeze()), dtype = torch.float32, device = device), diagA)))

    AA = matrix_diag+matrix_A_l+matrix_A_u+matrix_negsub
    #AA = AA.to('cpu')
    #nonzero_indices = torch.nonzero(AA, as_tuple=True)
    #values = AA[nonzero_indices]

    #AA_sparse = torch.sparse_coo_tensor(torch.vstack(nonzero_indices), values, AA.size())

    params = {
    'device': device,
    'AA': AA,
    'C': C.to('cpu'),
    'D': D.to('cpu'),
    'N': N,
    'R': R.to('cpu'),
    'S': S.to('cpu'),
    'Step_limit': Step_limit,
    'X0': X0.to('cpu'),
    'H': H.to('cpu'),
    'dt': dt_for_MC.to('cpu'),
    'multX': multX.to('cpu'),
    'multa': multa.to('cpu'),
    'sig': sigma.to('cpu'),

    }

    start_time = time.perf_counter()

    pool = Pool(processes = 8)

    iterations = list(range(batch_size_MC)) 
    iteration = 0
    #J_sample = pool.starmap(implicit_YYB, [(iteration, params) for iteration in iterations])
    J_sample = pool.starmap(MonteCarloSampler, [(iteration, params) for iteration in iterations])

    # for ii in range(batch_size_MC):
        
    #     J = implicit_YYB(iteration,params)
    #     J_sample.append(J)

    J_sample = torch.stack(J_sample)

    J_list.append(torch.mean(J_sample).unsqueeze(0))

    J_list_tensor = torch.cat(J_list, dim=0)  

    print(f"The optimal values from {times_MC} time(s) of Monte Carlo Simulation (batch size {batch_size_MC} for each run) is: \n \n{torch.sort(J_list_tensor)[0]} \n")

    end_time = time.perf_counter()

    print(f"Running time: {end_time - start_time} s.")
