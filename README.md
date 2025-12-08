# Code for paper *Bayesian Deep Generative Models for Multiplex Networks with Multiscale Overlapping Clusters*

## Code
- `class_M.py`: verify if a matrix is from class $M$
- `utils.py`: utility functions
- `simulate.py`: simulate data
- `spec_init.py`: spectral initialization
- `sample.py`: Gibbs samplers
- `run_sim.py`: posterior computation for simulation
- `plot_sim.py`: plot simulation outputs
- `run_sens.py`: posterior computation for sensitivity analysis
- `plot_sens.py`: plot sensitivity analysis outputs
- `run_large_p_small_n.py`: posterior computation for large-$p_K$ small-$N$ regime
- `plot_large_p_small_n.py`: plot large-$p_K$ small-$N$ regime outputs
- `run_hsbm.py`: posterior computation for simulated data from hierarchical stochastic block model
- `run_hsbm_hcd.py`: fitting [hierarchical community detection by recursive partitioning (HCD)](https://www.tandfonline.com/doi/full/10.1080/01621459.2020.1833888) for simulated data from hierarchical stochastic block model
- `plot_hsbm.py`: plot outputs of this comparison with HCD over simulated data from hierarchical stochastic block model
- `run_app.py`: posterior computation for application to brain connectivity networks
- `plot_app.py`: plot outputs of application to brain connectivity networks
- `run_mask.py`: posterior computation for masked brain connectivity networks (i.e. application task on prediction of missing edges)
- `run_mask_dcmm.py`: fitting [degree corrected mixed membership model (DCMM)](https://www.sciencedirect.com/science/article/abs/pii/S0304407622002081) for masked brain connectivity networks
- `plot_mask.py`: plot outputs of this comparison with DCMM over prediction of missing edges in masked brain connectivity networks

## Data
- (data files placed here)

## Output
- `simulation/`: all simulation outputs, including:
  - main simulation study
  - sensitivity analysis
  - large-$p_K$ small-$N$ regime
  - comparison with HCD
- `application/`: all application outputs, including:
  - main application to brain connectivity networks
  - prediction of missing edges
