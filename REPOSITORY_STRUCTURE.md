# Repository Structure

This document describes the organization of the gw-mist project repository.

## Directory Overview

```
gw-mist/
├── PROJECT_PLAN.md              # Detailed 8-week project plan
├── README.md                    # Main project documentation
├── REPOSITORY_STRUCTURE.md      # This file
├── src/                         # Main source code
│   ├── models/                  # Neural network architectures
│   ├── simulators/              # Data simulation and distortion injection
│   ├── utils/                   # Utility functions and helpers
│   └── data/                    # Data loading and preprocessing
├── notebooks/                   # Jupyter notebooks organized by week
│   ├── week1-fundamentals/      # GW basics and data analysis
│   ├── week2-technical/         # PyTorch and codebase review
│   ├── week3-sbi/              # SBI framework understanding
│   ├── week4-multidetector/    # Multi-detector implementation
│   ├── week5-frequency/        # Frequency domain extension
│   ├── week6-validation/       # Testing and validation
│   ├── week7-distortions/      # Realistic distortion models
│   └── week8-final/            # Final results and documentation
├── tests/                       # Unit tests and integration tests
├── docs/                        # Additional documentation
├── examples/                    # Example scripts and tutorials
├── scripts/                     # Utility scripts for data processing
├── data/                        # Data directory (not tracked in git)
│   ├── raw/                     # Raw GW data files
│   ├── processed/               # Preprocessed data
│   └── results/                 # Model outputs and results
└── config/                      # Configuration files
```

## Source Code Organization (`src/`)

### `models/`
Contains neural network architectures and training modules:
- `gof_networks.py` - Goodness-of-fit network architectures (single/multi-detector)
- `frequency_networks.py` - Complex-valued networks for frequency domain
- `lightning_modules.py` - PyTorch Lightning training modules
- `losses.py` - Custom loss functions

### `simulators/`
Data simulation and distortion injection:
- `gw_simulator.py` - Gravitational wave event simulation
- `distortions.py` - Distortion injection models (additive, correlated)
- `glitch_models.py` - LIGO glitch simulation
- `waveform_errors.py` - Systematic waveform mismodelling

### `utils/`
Utility functions and helpers:
- `data_utils.py` - Data loading and preprocessing utilities
- `gw_utils.py` - Gravitational wave specific utilities
- `training_utils.py` - Training and evaluation helpers
- `visualization.py` - Plotting and visualization functions

### `data/`
Data handling and dataset classes:
- `datasets.py` - PyTorch Dataset classes
- `loaders.py` - Data loading functions
- `preprocessing.py` - Data preprocessing pipelines

## Notebooks Organization

Each week has its own subdirectory containing relevant Jupyter notebooks:

### Week 1-3: Learning Phase
- `week1-fundamentals/`: GW physics, Bayesian inference, data analysis basics
- `week2-technical/`: PyTorch tutorials, codebase exploration
- `week3-sbi/`: SBI framework deep dive, existing method analysis

### Week 4-6: Implementation Phase
- `week4-multidetector/`: Multi-detector architecture development
- `week5-frequency/`: Frequency domain implementation
- `week6-validation/`: Performance testing and validation

### Week 7-8: Distortion Development
- `week7-distortions/`: Realistic distortion model development
- `week8-final/`: Final testing, documentation, and results

## Data Directory (`data/`)

**Note**: This directory should not be tracked in version control due to large file sizes.

### `raw/`
- GW strain data from LIGO/Virgo
- Detector noise curves and PSDs
- Glitch catalogs and metadata
- Posterior samples from GW events

### `processed/`
- Preprocessed strain data
- Generated training datasets
- Cached simulation results

### `results/`
- Trained model checkpoints
- Performance evaluation results
- Generated plots and figures
- Final analysis outputs

## Configuration (`config/`)

Configuration files for different aspects of the project:
- `training_config.yaml` - Training hyperparameters
- `data_config.yaml` - Data processing parameters
- `model_config.yaml` - Model architecture specifications
- `detector_config.yaml` - Detector-specific settings

## Testing (`tests/`)

Organized unit tests and integration tests:
- `test_models.py` - Neural network architecture tests
- `test_simulators.py` - Simulation and distortion injection tests
- `test_utils.py` - Utility function tests
- `test_integration.py` - End-to-end pipeline tests

## Documentation (`docs/`)

Additional documentation:
- API reference documentation
- Mathematical derivations
- Method descriptions
- Performance benchmarks

## Examples (`examples/`)

Self-contained example scripts:
- `basic_training.py` - Simple training example
- `multi_detector_analysis.py` - Multi-detector analysis example
- `frequency_domain_demo.py` - Frequency domain method demo
- `distortion_injection_demo.py` - Distortion injection examples

## Scripts (`scripts/`)

Utility scripts for common tasks:
- `download_data.py` - Download GW data from GWOSC
- `preprocess_data.py` - Batch preprocessing scripts
- `train_model.py` - Model training scripts
- `evaluate_model.py` - Model evaluation scripts

## Development Workflow

1. **Week-by-week development**: Use corresponding notebook directories for exploration
2. **Code development**: Implement reusable code in `src/` modules
3. **Testing**: Write tests in `tests/` for all new functionality
4. **Documentation**: Update documentation as features are added
5. **Examples**: Create examples for major features in `examples/`

## Version Control Guidelines

### Tracked Files
- All source code (`src/`)
- Notebooks with cleared outputs
- Documentation and configuration files
- Tests and examples

### Not Tracked
- Data files (`data/` directory)
- Model checkpoints (large files)
- Temporary files and caches
- Personal configuration files

### Git Best Practices
- Commit frequently with descriptive messages
- Use branches for feature development
- Clear notebook outputs before committing
- Keep commits focused and atomic