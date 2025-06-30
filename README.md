# Machine Learning Tools for Gravitational Wave Mismodelling Detection

A machine learning framework for detecting mismodelling in gravitational wave data analysis using simulation-based inference (SBI) goodness-of-fit methods.

## Project Overview

This project extends the simulation-based inference framework presented in [Anau Montel et al. 2024](https://arxiv.org/pdf/2412.15100) to gravitational wave astrophysics. The goal is to develop robust machine learning tools that can identify various types of distortions and systematic errors in gravitational wave detection and parameter estimation.

### Key Features

- **Multi-detector Analysis**: Coherent analysis across LIGO/Virgo detector network
- **Frequency Domain Processing**: Handle complex-valued frequency domain data streams
- **Realistic Distortion Models**: Inject various types of mismodelling relevant to GW science
- **Robust Detection**: High-accuracy identification of systematic errors and glitches

## Quick Start

### Prerequisites

```bash
# Core dependencies
pip install torch pytorch-lightning
pip install numpy scipy matplotlib
pip install gwpy
pip install jax jaxlib
pip install ripple jimgw
```

## Project Structure

```
gw-mist/
├── mist-base/              # Collated version of the original mist/ library including GW example
├── src/                    # Main source code
├── notebooks/              # Weekly development notebooks
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
├── examples/               # Example scripts and tutorials
├── config/                 # Configuration files
├── papers/                 # Relevant research papers. Stored as arxiv-2xxx.xxxx.pdf for easy reference.
└── data/                   # Data directory (not tracked)
```

See [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md) for detailed organization.

## Development Timeline

This is an 8-week summer research project organized in three phases:

### Phase 1: Learning (Weeks 1-3)
- **Week 1**: GW fundamentals and Bayesian inference
- **Week 2**: PyTorch/Lightning and codebase review
- **Week 3**: SBI framework deep dive

### Phase 2: Implementation (Weeks 4-6)
- **Week 4**: Multi-detector extensions
- **Week 5**: Frequency domain processing
- **Week 6**: Testing and validation

### Phase 3: Distortion Development (Weeks 7-8)
- **Week 7**: Realistic distortion models (glitches, waveform errors)
- **Week 8**: Final testing and documentation

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for detailed weekly objectives and deliverables.

## Key Components

### Neural Network Architectures (`src/models/`)

- **Network1D**: 1D ResNet-based goodness-of-fit networks
- **Multi-detector Networks**: Coherent analysis across detector network
- **Complex-valued Networks**: Frequency domain processing capabilities
- **Background Subtraction**: UNet-based background modeling

### Simulation Framework (`src/simulators/`)

- **GW Event Simulation**: Realistic gravitational wave event generation
- **Distortion Injection**: Various mismodelling scenarios
  - Additive distortions (bin-by-bin)
  - Correlated distortions (spatially coherent)
  - Glitch injection (from LIGO catalogs)
  - Waveform systematic errors
- **Multi-detector Coherence**: Proper sky localization and detector response

### Data Handling (`src/data/`)

- **Efficient Loading**: On-the-fly data generation during training
- **Preprocessing Pipelines**: Whitening, filtering, downsampling
- **Multi-format Support**: Time domain, frequency domain, spectrograms

### Development Workflow
1. Use weekly notebook directories for exploration
2. Implement reusable code in `src/` modules
3. Write tests for new functionality
4. Update documentation as features are added

## Documentation

### Mathematical Background
- Detailed method descriptions in `docs/methods/`
- Mathematical derivations and theoretical background

### Tutorials
- Step-by-step tutorials in `examples/`
- Weekly notebook progressions in `notebooks/`

### Git Workflow
1. Create feature branches for development
2. Use descriptive commit messages
3. Clear notebook outputs before committing
4. Submit pull requests for review

## References

1. [Anau Montel et al. (2024)](https://arxiv.org/pdf/2412.15100) - "Tests for model misspecification in simulation-based inference"
2. [LIGO/Virgo Data Analysis Guide](https://cplberry.com/2020/02/09/gw-data-guides/)
3. [GWOSC - Gravitational Wave Open Science Center](https://www.gw-openscience.org/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
