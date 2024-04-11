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
#### Exercise 1.2 Simulation Analysis 

For Exercise 1.2 there is a __*.ipynb file__. 

#### Overview

This document details the setup, execution, and analysis of Monte Carlo simulations for Exercise 1.2. The simulations evaluate numerical solutions' convergence under varying parameters, focusing on sample size and time steps, to understand their effects on simulation reliability and accuracy.

#### File Structure

- `Exercise1_2/`: Main directory.
  - `value_numerical/`: Stores solver-obtained numerical values.
  - `value_MC/`: Contains Monte Carlo simulation results.
    - `FSS_1e5/` and `FTSN_5e3/`: Subdirectories for fixed sampling size and fixed time step number experiments, respectively.
  - `lib/`: Contains essential Python scripts.
    - `Exercise1_2_parallel_MC.py`: Runs parallel Monte Carlo simulations.

#### Simulation Settings

Two settings explore the impact of varying parameters:

- **Fixed Sampling Size (FSS):** Analyzes time discretization granularity's effect by varying time steps with a fixed sample number.
- **Fixed Time Step Number (FTSN):** Examines the influence of increasing sample size on solution accuracy with a constant number of time steps.

#### Running Simulations

Execute the `run_MCs.sh` script from the root directory to start all specified simulations. Results will be stored in the `value_MC/` directory, under corresponding sub-folders for each experiment.

#### Analysis and Visualization

Post-simulation, data is analyzed to observe convergence behaviors:

- **Sample Size Impact:** Log-log plots depict how solution accuracy improves with larger sample sizes.
- **Time Steps Impact:** Similar plots show accuracy changes with more granular time steps.

Plots are saved in EPS and PNG formats for quality and accessibility.


#### Conclusion

This README guides through Monte Carlo simulations for Exercise 1.2, highlighting the importance of sample size and time steps on numerical solution convergence and accuracy.


## Supervised learning, checking the NNs are good enough
### Exercise 2.1 and Exercise 2.2
In this section of our project, we explore the approximation of Linear Quadratic Regulator (LQR) solutions using neural networks. We focus on two main aspects: approximating the value function and the optimal control policy through deep learning models. Below, we outline the methodology, code structure, and results obtained from our experiments.

#### Overview

The goal of Exercise 2 is to demonstrate the capability of neural networks in approximating complex functions within the context of control theory. Specifically, we address the LQR problem, a fundamental problem in control theory for designing optimal controllers.

#### Methodology

1. **Data Generation**: We generate synthetic data that represents different states and time samples for the LQR problem. This data serves as the input for training our neural networks.
   
2. **Neural Network Models**:
    - `ValueFunctionNN`: A model that approximates the value function of the LQR problem.
    - `MarkovControlNN`: A model designed to approximate the optimal control policy.

3. **Training**: Both models are trained using a dataset comprising state, time, and either value or control samples. We use the Mean Squared Error (MSE) as the loss function and Adam as the optimizer.

4. **Visualization**: After training, we visually compare the neural network approximations with the theoretical solutions to evaluate the models' performance.

#### Repository Structure

- `new_data()`: Function to generate training data.
- `ValueFunctionNN`: The neural network class for the value function approximation.
- `MarkovControlNN`: The neural network class for the control policy approximation.
- Training loops for both neural networks, including loss calculation and optimization steps.
- Visualization scripts for comparing the neural network approximations with the analytical solutions.

#### Results

The training process demonstrates the neural networks' ability to learn the underlying patterns in the LQR solutions. The final visualizations provide a clear comparison between the approximations made by the neural networks and the actual solutions, showcasing the effectiveness of deep learning in solving such problems.


## Deep Galerkin approximation for a linear PDE
### Exercise 3.1
For Exercise 3.1 there are 3 __*.ipynb file__ and a __*.py file__. 

#### Overview

This section of the project applies deep learning techniques to solve Linear Quadratic Regulator (LQR) problems. Using Deep Generative Models (DGM), we aim to approximate the value function and control actions for a given LQR setup. The project showcases the power of neural networks in handling complex dynamic systems and control tasks.

The main components of this project include model definition, training, and evaluation against Monte Carlo simulations for verification. Here is a brief guide on how to use the scripts included in this project:

#### Model Training

1. **Define the Neural Network Models**: The models for approximating the value function and control actions are defined in `DGMNN.py` and `MarkovControlNN.py`, respectively.

2. **Generate Training Data**: Use the `new_data` function to generate synthetic data samples for training the models.

3. **Train the Models**: Run the training script with appropriate hyperparameters and data. Training progress will be saved, and the model's state dict can be exported for later use.

#### Evaluation

- **Load Trained Models**: Load the models' state dicts and prepare them for evaluation.

- **Compare with Monte Carlo Simulations**: Evaluate the models' outputs against the results from Monte Carlo simulations for verification.

#### Visualization

- Plot the training loss over epochs to assess the learning progress.
- Compare the predicted value functions and control actions with those from Monte Carlo simulations visually.

## Policy iteration with DGM
### Exercise 4.1
For Exercise 4.1 there are one __*.ipynb file__ . 

#### Overview

This section of the project combines the Policy Iteration Algorithm with Deep Galerkin Methods introduced in section 3. The task involves approximating the value function $v$ and the Markov controls $a$ using neural networks, represented as $`v(·, ·; θ_{val})`$ and $`a(·, ·; θ_{act})`$.

The process is iterative, starting with a given Markov control function approximated by initially `torch.tensor([[1., 1.]])` then $`a(·, ·; θ_{act})`$, leading to an update $θ_val$. Then with $θ_{val}$ fixed, we update $θ_{act}$ to minimize the Hamiltonian.

#### Policy iteration

1. **Introduce Neural Network Methods from 3.1** The models for approximating the value function and control actions are defined in `DGMNN_YYBver`.

2. **Updating value function and control** We implement the policy iteration algorithm using the Deep Galerkin Method (DGM) for approximating the value function and Markov controls. The primary objective is to iteratively improve upon a policy until it converges to the optimal solution. We run the iteration while the difference is large, i.e. larger than 0.01, and the iteration time does not exceed 5.

#### Evaluation

- **Compare with Numerical Solution**

- Compare the according updated value functions and control actions with those from Numerical solution output visually.
- Print the update control actions and the according output from the Numerical solution.






