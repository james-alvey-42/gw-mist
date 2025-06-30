# Machine Learning Tools for Gravitational Wave Mismodelling Detection
## 8-Week Summer Research Project Plan

### Project Overview
This project extends simulation-based inference (SBI) goodness-of-fit methods to detect mismodelling in gravitational wave data analysis. Building on the framework presented in [Anau Montel et al. 2024](https://arxiv.org/pdf/2412.15100), the aim is to develop machine learning tools that can identify various types of distortions and systematic errors in GW detection.

### Project Goals
1. **Extend current methods** from single-detector time-domain to multi-detector and frequency-domain analysis
2. **Develop realistic distortion models** relevant to gravitational wave science
3. **Create robust detection framework** for identifying mismodelling in GW data

---

## Phase 1: Learning Phase (Weeks 1-3)

### Week 1: Gravitational Wave Fundamentals
**Objectives:**
- Understand Bayesian inference in GW context
- Learn a bit about GW detector physics and data characteristics
- Familiarize with LIGO/Virgo data formats and analysis

**Goals:**
- [ ] Review core elements of Bayesian inference and parameter estimation, and connect them to GWs
- [ ] Run throught GW150914 simulator
- [ ] Document understanding of detector noise and signal characteristics
- [ ] Set up development environment with required packages

**Resources:**
- [LIGO/Virgo data analysis guide](https://cplberry.com/2020/02/09/gw-data-guides/)
- `gwpy` and `jimgw` documentation
- GW150914 event data and analysis

### Week 2: Technical Foundations
**Objectives:**
- Cover PyTorch and PyTorch Lightning frameworks
- Understand the existing codebase structure
- Implement basic neural network training pipelines

**Goals:**
- [ ] PyTorch Lightning training examples
- [ ] Review PyTorch elements of `mist/` repository
- [ ] Document basic examples from existing codebase and connect to original paper

**Resources:**
- PyTorch Lightning documentation
- Existing code in `mist/models/` and `mist/utils/`
- Simple GW modeling code: `mist/GW/gw150814_simulator.py`

### Week 3: SBI Mismodelling Framework
**Objectives:**
- Deep dive into SBI goodness-of-fit methodology
- Understand distortion injection and detection principles
- Analyze current additive and correlated distortion models

**Goals:**
- [ ] Summary of SBI framework methodology
- [ ] Analysis of existing distortion models
- [ ] Identify limitations of current single-detector approach

**Resources:**
- Paper: Anau Montel et al. (2024) "Tests for model misspecification in simulation-based inference"
- Code: `mist/simulators/additive.py`
- Jupyter notebooks in `mist_OG/notebooks/`

## Phase 2: Implementation Phase (TBC: Weeks 4-6)

### Week 4: Multi-Detector Extension
**Objectives:**
- Generalize network architectures for multiple detectors
- Implement coherent analysis across H1, L1
- Handle detector-specific noise characteristics

**Goals:**
- [ ] Multi-detector data loading and preprocessing
- [ ] Extended Network1D architecture for multi-detector input
- [ ] Coherent distortion injection across detector network

**Technical Goals:**
- Modify `Network1D` to handle multi-detector input tensors
- Extend `GW150814_Additive` for coherent multi-detector simulation
- Implement proper sky localization for multi-detector analysis

### Week 5: Frequency Domain Extension
**Objectives:**
- Adapt methods to complex-valued frequency domain data
- Implement frequency-domain distortion models
- Document methods for injecting sensible frequency domain distortions

**Goals:**
- [ ] Complex-valued neural network architectures
- [ ] Frequency-domain distortion injection methods
- [ ] Comparison framework for time vs frequency domain

**Technical Goals:**
- Create frequency-domain versions of distortion simulators
- Adapt network processing steps for complex valued data
- Inject distortions that are consistent with proper frequency-domain filtering and masking

### Week 6: Testing and Validation
**Objectives:**
- Validate multi-detector and frequency-domain implementations
- Compare performance across different data representations
- Optimize network hyperparameters

**Goals:**
- [ ] Performance benchmarks for all implementations
- [ ] Hyperparameter optimization results
- [ ] Validation on real LIGO/Virgo data

## Phase 3: Distortion Development (Weeks 7-8)

### Week 7: Realistic Distortion Models
**Objectives:**
- Implement LIGO glitch injection based on real glitch catalogs
- Model systematic waveform errors (e.g., precession effects)
- Develop model-independent frequency-domain distortions

**Goals:**
- [ ] Glitch injection framework using LIGO glitch database
- [ ] Waveform mismodelling scenarios (precession, eccentricity)
- [ ] Generic frequency-domain distortion models

**Technical Implementation:**
- Interface with LIGO glitch catalogs
- Implement waveform parameter perturbations
- Create broadband and narrowband frequency distortions

### Week 8: Final Testing and Documentation
**Objectives:**
- Comprehensive testing on range of developed distortion types
- Performance analysis and method comparison
- Complete project documentation and tutorials

**Goals:**
- [ ] Final performance evaluation report
- [ ] Complete code documentation and API reference
- [ ] Tutorial notebooks for all major features
- [ ] Project presentation and future work recommendations

**Final Validation:**
- Test on held-out real data
- Demonstrate robustness across different GW events
- Quantify false positive/negative rates

## Future Extensions
- Application to next-generation detectors (Einstein Telescope, Cosmic Explorer)
- Integration with parameter estimation pipelines
- Extension to other astrophysical signals (neutron star mergers)
- Application to existing GWOSC event catalog, including events with known glitches