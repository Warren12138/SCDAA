import os
import numpy as np
import torch
from Exercise1_1 import LQRSolver


H = torch.tensor([[1.2, 0.8], [-0.6, 0.9]], dtype=torch.double)
M = torch.tensor([[0.5,0.7], [0.3,1.0]], dtype=torch.double)
sigma = torch.tensor([[[0.8],[1.1]]], dtype=torch.double) 
C = torch.tensor([[1.6, 0.0], [0.0, 1.1]], dtype=torch.double)  # Positive semi-definite
D = torch.tensor([[0.5, 0.0], [0.0, 0.7]], dtype=torch.double)  # Positive definite
R = torch.tensor([[0.9, 0.0], [0.0, 1.0]], dtype=torch.double)  # Positive semi-definite
T = torch.tensor(1.0, dtype=torch.double)
method = 'rk4'

file_path_Ex1_2 = f'Exercise1_2'
os.makedirs(file_path_Ex1_2, exist_ok=True)

initialization_for_solver = {
    'H': H,
    'M': M,
    'sigma': sigma,
    'C': C,
    'D': D,
    'R': R,
    'T': T,
    'method': method
}

torch.save(initialization_for_solver, file_path_Ex1_2+'/'+'initialization_for_solver.pt')

solver = LQRSolver(H, M, sigma, C, D, R, T, method)

# Setting for Error Comparison

# Initialization

t_ends = [0.1,0.9]
t_num = 9
x_ends = [[0.1,0.9],[0.1,0.9]]
x_num = [5,5] # alleviate the calculation workload.

# load_interval_setting = torch.load('Exercise1_2/value_numerical/?x?/'+'interval_setting.pt')
# t_ends = load_interval_setting['t_ends']
# t_num = load_interval_setting['t_num']
# x_ends = load_interval_setting['x_ends']
# x_num = load_interval_setting['x_num']

file_path_numerical = file_path_Ex1_2 + '/' + f'value_numerical/{x_num[0]}x{x_num[1]}'
os.makedirs(file_path_numerical, exist_ok=True)

# Establish meshgrid structure from setting.

t_batch_i = torch.linspace(t_ends[0],t_ends[1],t_num,dtype=torch.double)
t_batch = t_batch_i.repeat_interleave(x_num[0]*x_num[1])

x1 = torch.linspace(x_ends[0][0],x_ends[0][1],x_num[0],dtype=torch.double)
x2 = torch.linspace(x_ends[1][0],x_ends[1][1],x_num[1],dtype=torch.double)

x_batch_i = torch.cartesian_prod(x1, x2).unsqueeze(1)

X1 = x_batch_i[:, 0, 0].view(x_num[0], x_num[1])
X2 = x_batch_i[:, 0, 1].view(x_num[0], x_num[1])

x_batch = torch.cartesian_prod(x1, x2).unsqueeze(1).repeat(t_num, 1, 1)

# Calculation value funtion from numerical solution.

value_numerical = solver.value_function(t_batch,x_batch)

#Preparing data for graph.

value_min = torch.min(value_numerical).numpy()
value_max = torch.max(value_numerical).numpy()
value_numerical_reshape = value_numerical.reshape([t_num,x_num[0]*x_num[1]]).view(t_num, x_num[0], x_num[1])

interval_setting = {
    't_ends': t_ends,
    't_num': t_num,
    'x_ends': x_ends,
    'x_num': x_num
}

# Save the interval_setting and the corresponding obtained value data.

torch.save(interval_setting, file_path_numerical+'/'+'interval_setting.pt')
torch.save(value_numerical, file_path_numerical+'/'+'value_numerical.pt')

# Setting for Monte Carlo
# According to the requirements, build two folders for storing data.

load_interval_setting = torch.load('Exercise1_2/value_numerical/5x5/'+'interval_setting.pt')
t_ends = load_interval_setting['t_ends']
t_num = load_interval_setting['t_num']
x_ends = load_interval_setting['x_ends']
x_num = load_interval_setting['x_num']

# Fixed sampling size.
FSS = int(1e5)
file_path_MC_FSS = file_path_Ex1_2 + '/' + f'value_MC/{x_num[0]}x{x_num[1]}/FSS_1e5'
#Varied time step number.
#FSS_VTSN = [int(x) for x in[1e0,1e1,5e1,1e2,5e2,1e3,5e3]]
FSS_VTSN = [int(x) for x in[1e0,1e1,5e1]]
# Fixed time step number.
FTSN = int(5e3)
file_path_MC_FTSN = file_path_Ex1_2 + '/' + f'value_MC/{x_num[0]}x{x_num[1]}/FTSN_5e3'
#Varied sampling size.
#FTSN_VSS = [int(x) for x in[1e1,5e1,1e2,5e2,1e3,5e3,1e4,5e4,1e5]]
FTSN_VSS = [int(x) for x in[1e1,5e1,1e2]]

t_batch_i = torch.linspace(t_ends[0],t_ends[1],t_num,dtype=torch.double)

x1 = torch.linspace(x_ends[0][0],x_ends[0][1],x_num[0],dtype=torch.double)
x2 = torch.linspace(x_ends[1][0],x_ends[1][1],x_num[1],dtype=torch.double)

x_batch_i = torch.cartesian_prod(x1, x2).unsqueeze(1)

interval_setting = {
    't_ends': t_ends,
    't_num': t_num,
    'x_ends': x_ends,
    'x_num': x_num
}

# for i in FSS_VTSN:
#     if i == 1:
#         trvlthg = ''
#     else:
#         trvlthg = 's'
#     path_FSS_VTSN_i = file_path_MC_FSS+'/'+ str(i) + '_step'+ trvlthg
#     os.makedirs(path_FSS_VTSN_i, exist_ok=True)
#     torch.save(interval_setting, path_FSS_VTSN_i+'/'+'interval_setting.pt')
#     script_path = 'Exercise1_2(functional).py'
#     #script_path = 'test.py'
#     os.system(f'python "{script_path}" "{path_FSS_VTSN_i}" "{str(i)}" "{str(FSS)}"')
    
 
# for i in FTSN_VSS:
#     path_FTSN_VSS_i = file_path_MC_FTSN+'/'+ str(i) + '_samples'
#     os.makedirs(path_FTSN_VSS_i, exist_ok=True)
#     torch.save(interval_setting, path_FTSN_VSS_i+'/'+'interval_setting.pt')
#     script_path = 'Exercise1_2(functional).py'
#     os.system(f'python "{script_path}" "{path_FTSN_VSS_i}" "{str(FTSN)}" "{str(i)}"')

shell_script_path = "run_MCs.sh"

with open(shell_script_path, 'w') as shell_script:
    shell_script.write("#!/bin/bash\n\n")
    
    for i in FSS_VTSN:
        if i == 1:
            trvlthg = ''
        else:
            trvlthg = 's'
        path_FSS_VTSN_i = f"{file_path_MC_FSS}/{i}_step{trvlthg}"
        os.makedirs(path_FSS_VTSN_i, exist_ok=True)
        torch.save(interval_setting, path_FSS_VTSN_i+'/interval_setting.pt')
        
        script_path = 'Exercise1_2\(functional\).py'
   
        shell_script.write(f'python {script_path} {path_FSS_VTSN_i} {i} {FSS} &\n')
    
 
    for i in FTSN_VSS:
        path_FTSN_VSS_i = f"{file_path_MC_FTSN}/{i}_samples"
        os.makedirs(path_FTSN_VSS_i, exist_ok=True)
        torch.save(interval_setting, path_FTSN_VSS_i+'/'+'interval_setting.pt')
        
        script_path = 'Exercise1_2\(functional\).py'
        shell_script.write(f'python {script_path} {path_FTSN_VSS_i} {FTSN} {i} &\n')


os.system(f"chmod +x {shell_script_path}")

