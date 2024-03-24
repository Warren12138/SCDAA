# SCDAA Coursework 2023-24 
## Group Members

S1973765 Yu Wang

S2559490 Zheyan Lu

S2601126 Yuebo Yang

## How to run the code

The suggested way is firstly to clone this project to a local folder and then build an environment by installing all the dependencies noted in the requirements.txt. Then run the codes along the instructions of each.

## Linear Quadratic Regulator
### Exercise 1.1

For Exercise 1.1 there are a __*.ipynb file__ and a __*.py file__. 

#### The Exercise1_1.ipynb file contains:

- `LQRsolver`: a class required to be written, which can be __initialised__ with default time horizon T = 1 (which can also take value from users' input) and with the matrices specifying the LQR problem which are:
    - `H` (`torch.Size([n,n]) torch.tensor`): the linear relations of dynamics between the total `n` state processes.
    - `M` (`torch.Size([n,m]) torch.tensor`): the influences from `m` control variables to `n` state processes. 
    - `sigma`(`torch.Size([n,d]) torch.tensor`): the diffusion matrix from `d` Wiener processes to `n` state processes.
    - `C` (`torch.Size([n,n]) torch.tensor`): the contribution matrix from state processes to runnning reward.
    - `D` (`torch.Size([m,m]) torch.tensor`): the contribution matrix from the final value of state processes to runnning reward.
    - `R` (`torch.Size([n,n]) torch.tensor`): the contribution matrix from the final values of state processes to terminal reward.
    - `method` (`string`): the indicator for using euler scheme or 4th order Runge-Kutta4th order Runge-Kutta scheme.
    
    __Declaration for dimensions of matrices__:
    Here `n` should be compatible with the dimension of state variable space and m is the dimension of control variable space.
    Note that `n = m = 2` and `d = 1` in this exercise but can be extended to a higher dimension.

    There are __3 main methods__ built in `LQRsolver` including

    - `solve_riccati_ode`: __a numerical solver for Ricatti ODE__ which requires
    
        __input__ of
    
        - `time_grids`  `torch.Size([batch_size,l]) torch.Tensor`
        
        and __returns__
        
        - the values of solution function $S(t)$  `torch.Size([batch_size,l,n,n]) torch.tensor`
        
        (Two numerical methods are tried to be provided as options: Euler scheme and 4th order Runge-Kutta scheme)
    
    - `value_function`: __a computation of value function__ which requires

        __inputs__ (in sequence) of 
        
        - `t_batch`  `torch.Size([batch_size]) torch.tensor` whose entries took value initially from [0,1] but would be scaled by the given T in the further calculation
        - `x_batch`  `torch.Size([batch_size,1,n]) torch.tensor`
        
        - `sol_method` `string` the indicator for using interpolation or using direct calculation for each t in t_batch (default setting is using interpolation)
        
        and __returns__
        
        - values of value_function `torch.Size([batch_size]) torch.tensor`
        
    - `markov_control`: __a computation of Markov control function__ which requires
    
        __inputs__ (in sequence) of 
        
        - `t_batch`  `torch.Size([batch_size]) torch.tensor` whose entries took value initially from [0,1] but would be scaled by the given T in the further calculation
        - `x_batch`  `torch.Size([batch_size,1,n]) torch.tensor`
        
        - `sol_method` `string` the indicator for using interpolation or using direct calculation for each t in t_batch (default setting is using interpolation)
        
        and __returns__
        
        - values of Markov_control_function  `torch.Size([batch_size,n]) torch.tensor`
        
    
- __A runnable sample__ including: 

    - a whole set of matrices for __initialisation__ and an instance created based on the __initialisation__ from the cunstom class `LQRsolver`
    - an example of __calculation of value function__ with given t_batch and x_batch with __interpolation__ and another example with __direct calculation__
    - an example of __calculation of Markov control function__ with given t_batch and x_batch with __interpolation__ and another example with __direct calculation__

#### The Exercise1_1.py file is just for storing the `LQRsolver` class in a callable way for the rest part fo the coursework. Repetatively, it contains:
- `LQRsolver`: the class we wrote.

    And ther are  __3 main methods__ built in, including:

    - `solve_riccati_ode`
    - `value_function`
    - `markov_control`
    
    
### Exercise 1.2
Something
## Supervised learning, checking the NNs are good enough
### Exercise 2.1
Something
### Exercise 2.2
Something
## Deep Galerkin approximation for a linear PDE
### Exercise 3.1
Something
## Deep Galerkin approximation for a linear PDE
### Exercise 3.1
Something

