# Repository Structure

This document provides an overview of the `gw-mist` project repository's organization and purpose.

## Directory Overview

- **`mist-base/`**: Foundational code for the project, including models, simulators, and utilities.
- **`notebooks/`**: Jupyter notebooks for weekly research, experimentation, and visualization.
- **`papers/`**: Relevant academic papers and literature.
- **`src/`**: Main source code for utilities and helper functions.

```
gw-mist/
├── .gitignore               # Specifies intentionally untracked files to ignore
├── PROJECT_PLAN.md          # Project planning and timeline document
├── README.md                # Main project README file
├── REPOSITORY_STRUCTURE.md  # This file, documenting the repository structure
├── mist-base/               # Foundational project code
│   ├── GW/                  # Gravitational Wave specific data and simulators
│   ├── models/              # Neural network architectures
│   ├── notebooks/           # Base notebooks for examples
│   ├── simulators/          # Data simulation and distortion tools
│   └── utils/               # Utility functions for data handling and training
├── notebooks/               # Weekly research and development notebooks
│   ├── week1/               # Initial explorations with GW data and libraries
│   ├── week2/               # Introduction to MIST and PyTorch basics
│   ├── week3/               # Advanced simulations and training scripts
│   ├── week4/               # Complex distortion analysis and network development
│   └── week5/               # Plotting, background trainers, and final scripts
├── papers/                  # Academic papers and literature
└── src/                     # Main source code
    └── utils/               # Utility scripts and helper functions
```

## `mist-base/`

This directory contains the core components of the MIST framework.

- **`mist-base/GW/`**: Contains tools for simulating Gravitational Wave events, specifically tailored to GW150814.
  - `gw150814_simulator.py`: A simulator for the GW150814 event, including data loading, filtering, and waveform generation.
- **`mist-base/models/`**: Defines the neural network architectures used in the project.
  - `gof_net.py`: Goodness-of-fit networks, including 1D and 2D architectures.
- **`mist-base/simulators/`**: Includes simulators for generating synthetic data with various types of distortions.
  - `additive.py`: A simulator for generating data with additive noise and distortions.
- **`mist-base/utils/`**: A collection of utility functions supporting data loading, processing, and model training.
  - `data.py`: Contains PyTorch `Dataset` and `DataModule` classes for handling data.

## `src/`

This directory holds the main source code for the project's utilities.

- **`src/utils/`**: Contains helper scripts for data generation and other tasks.
  - `generators.py`: A script with various data generation utilities.

## Notebooks

### `notebooks/week1`
- `cosmic_mixing_desk.ipynb`: An interactive notebook for exploring and manipulating Gravitational Wave data, including filtering and sound playback.
- `gwpy_sandbox.ipynb`: A sandbox environment for experimenting with the `gwpy` library for gravitational-wave data analysis.
- `Likelihoods.ipynb`: A notebook focused on understanding and implementing likelihood calculations.
- `simulator_sandbox.ipynb`: A sandbox for testing and experimenting with the data simulators.

### `notebooks/week2`
- `mist_basics.ipynb`: An introduction to the fundamental concepts and usage of the MIST framework.
- `torch_basics.ipynb`: A primer on the basics of PyTorch for building and training neural networks.

### `notebooks/week3`
- `comb_plotter.ipynb`: A notebook for visualizing "comb" data or signals.
- `comb_sandbox.ipynb`: A sandbox for experimenting with comb-filter-related simulations.
- `gw_mist_sandbox.ipynb`: A dedicated sandbox for hands-on experimentation with the `gw-mist` project.
- `gw_trainer.py`: A Python script for training gravitational-wave detection models.

### `notebooks/week4`
- `distortions_sandbox_complex_BCE.ipynb`: A sandbox for analyzing complex distortions using Binary Cross-Entropy loss.
- `distortions_sandbox_complex.ipynb`: A general sandbox for exploring and analyzing complex signal distortions.

### `notebooks/week5`
- `complex_plotter.ipynb`: A notebook for plotting and visualizing complex data and results.
- `distortions_BCE_background_trainer.ipynb`: A notebook for training models on distorted data with background noise, using BCE loss.
- `distortions_sandbox_complex_background.ipynb`: A sandbox for experimenting with complex distortions in the presence of background noise.
- `makeall.sh`: A shell script for running a sequence of processing or training tasks.
- `network_plotmaker.py`: A Python script for generating plots related to neural network performance and architecture.
