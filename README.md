# Stochastic Processes: Comparative Simulation

Simulation and comparison of four anomalous diffusion models, implemented from scratch using Cholesky decomposition of analytical covariance matrices.

## Models

- **Fractional Brownian Motion (fBm)**: correlated increments with Hurst exponent h=0.7, exhibiting long-range dependence
- **Exponentially Tempered fBm**: fBm with exponential cutoff at large lag times (tau=90), using the confluent hypergeometric function 1F1
- **Power-law Tempered fBm**: fBm with power-law tempering via the Gauss hypergeometric function 2F1
- **Ornstein-Uhlenbeck Process**: mean-reverting process with exponential covariance decay

## What it does

Each model is sampled by constructing its theoretical covariance matrix and applying Cholesky decomposition to generate correlated trajectories. Empirical mean squared displacement (MSD) is then computed across M=50 independent realizations and compared against the analytical MSD, validating that the simulation matches theory.

## Output

A single figure with 4 subplots, each showing empirical vs. theoretical MSD for one process.

## Dependencies

```bash
pip install numpy scipy matplotlib
```
