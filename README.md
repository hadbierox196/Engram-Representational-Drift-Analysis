# Engram Representational Drift Analysis

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-013243.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.3%2B-11557c.svg)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Computational neuroscience analysis of memory engram stability and representational drift in longitudinal calcium imaging data**

## Table of Contents
- [Overview](#overview)
- [What This Project Does](#what-this-project-does)
- [Key Features](#key-features)
- [Scientific Background](#scientific-background)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis Pipeline](#analysis-pipeline)
- [Output & Visualizations](#output--visualizations)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Mathematical Models](#mathematical-models)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project implements a comprehensive computational pipeline for analyzing **representational drift** in neural memory engrams using simulated longitudinal calcium imaging data. It models how neural representations of memories change over time and quantifies the stability of these representations across different contexts.

**Representational drift** refers to the phenomenon where neural activity patterns representing the same memory gradually change over time, even though the memory itself remains intact. This project provides tools to measure, model, and visualize this drift.

---

## What This Project Does

### Core Functionality:
1. **Simulates longitudinal calcium imaging data** from neural populations across multiple sessions
2. **Tracks neural activity patterns** (engrams) representing different memory contexts
3. **Quantifies representational drift** using population vector correlation analysis
4. **Fits exponential decay models** to measure memory stability and half-life
5. **Tests independence** of drift between different memory contexts
6. **Analyzes dimensionality** of neural representations over time
7. **Generates comprehensive visualizations** of all analyses

### Real-World Application:
This type of analysis is crucial for understanding:
- How memories remain stable despite neural turnover
- Why some memories fade while others persist
- How different experiences are encoded in the brain
- The neural basis of memory consolidation and forgetting

---

## Key Features
-  **Realistic Neural Data Simulation**: Generates biologically-plausible calcium imaging data
-  **Population Vector Analysis**: Computes correlations between neural activity patterns
-  *Exponential Drift Modeling**: Fits decay models to quantify memory half-life
-  **Independence Testing**: Statistical tests for correlation between contexts
-  **Dimensionality Analysis**: PCA-based assessment of representational complexity
-  **Comprehensive Visualizations**: 5 multi-panel publication-quality figures
-  **Statistical Rigor**: Permutation tests, R² calculations, and confidence intervals
-  **Professional Plotting**: Clean, annotated figures suitable for presentations

---

##  Scientific Background

### What are Memory Engrams?
**Engrams** are the physical neural substrates of memories - specific patterns of neural activity that encode experiences. When you remember something, you're reactivating the same (or similar) pattern of neurons that fired during the original experience.

### What is Representational Drift?
Even though we can remember events from years ago, the specific neurons encoding those memories change over time. This is **representational drift** - the neural code slowly evolves while the memory persists.

### Why Does This Matter?
Understanding drift helps explain:
- **Memory stability vs. flexibility**: How memories can be stable yet adaptable
- **Forgetting mechanisms**: Why drift sometimes leads to memory loss
- **Context discrimination**: How the brain maintains distinct memories
- **Neural plasticity**: The ongoing reorganization of brain circuits

---

## 🛠️ Technologies Used

### Core Libraries:
- **Python 3.7+**: Primary programming language
- **NumPy**: Numerical computing and array operations
- **Matplotlib**: Data visualization and figure generation
- **Seaborn**: Statistical data visualization
- **SciPy**: Scientific computing, statistics, and optimization
- **Scikit-learn**: Machine learning (PCA, regression)

### Key Algorithms:
- Population vector correlation analysis
- Exponential decay curve fitting
- Principal Component Analysis (PCA)
- Permutation testing
- Linear regression

---

##  Prerequisites

### Required Knowledge:
- Basic Python programming
- Understanding of arrays/matrices
- Basic statistics (correlation, regression)
- (Optional) Neuroscience background helpful but not required

### System Requirements:
- **Python**: Version 3.7 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: ~100MB for code and outputs
- **OS**: Windows, macOS, or Linux

### Python Packages:
```bash
numpy >= 1.19.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
scipy >= 1.5.0
scikit-learn >= 0.23.0
```

---

## 🚀 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/engram-drift-analysis.git
cd engram-drift-analysis
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# OR using conda
conda create -n engram python=3.8
conda activate engram
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```text
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.5.0
scikit-learn>=0.23.0
```

### Step 4: Verify Installation
```bash
python -c "import numpy, matplotlib, scipy, sklearn; print('All packages installed successfully!')"
```

---

## 💻 Usage

### Quick Start:
```bash
python engram_drift_analysis.py
```

### What Happens:
1. Simulates 30 sessions of neural recordings (100 neurons, 20 trials each)
2. Analyzes drift for two different memory contexts (A and B)
3. Generates 5 publication-quality figures
4. Prints comprehensive statistical summary to console

### Expected Runtime:
- **Simulation**: ~5 seconds
- **Analysis**: ~10 seconds
- **Visualization**: ~5 seconds
- **Total**: ~20 seconds

### Customization:
Modify key parameters at the top of the script:

```python
# Adjust these parameters
n_sessions = 30              # Number of imaging sessions
n_cells = 100                # Number of neurons recorded
n_trials = 20                # Trials per session
drift_rate_A = 0.02          # Drift speed for context A
drift_rate_B = 0.025         # Drift speed for context B
correlation = 0.3            # Correlation between contexts
```

---

##  Analysis Pipeline

### **Part 1: Data Generation**
- Simulates neural activity for two memory contexts
- Models realistic calcium imaging noise
- Implements gradual representational drift

### **Part 2: Population Vector Analysis**
- Computes mean activity patterns per session
- Calculates correlation matrices across all session pairs

### **Part 3: Correlation Matrix Visualization**
- Heatmaps showing similarity between sessions
- Comparison between contexts A and B

### **Part 4: Temporal Lag Analysis**
- Extracts correlation as a function of time lag
- Reveals decay of similarity over time

### **Part 5: Exponential Drift Modeling**
- Fits exponential decay: `r(t) = r₀ × exp(-t/τ)`
- Calculates memory half-life: `t₁/₂ = τ × ln(2)`
- Provides R² goodness-of-fit metrics

### **Part 6: Independence Testing**
- Tests if contexts drift independently or together
- Permutation-based statistical testing
- Computes cross-context drift correlations

### **Part 7: Dimensionality Analysis**
- PCA-based effective dimensionality
- Participation ratio calculation
- Stability index over time

### **Part 8: Summary Visualization**
- Comprehensive 9-panel summary figure
- All key metrics in one view

---

##  Output & Visualizations

### Generated Files:

#### 1. **drift_correlation_matrices.png**
- Correlation heatmaps for both contexts
- Difference map showing relative stability
- **Interpretation**: Darker off-diagonal = faster drift

#### 2. **drift_model_fits.png**
- Correlation vs. lag with exponential fits
- Half-life comparison bar chart
- **Interpretation**: Longer half-life = more stable memory

#### 3. **drift_independence_analysis.png**
- Cross-context correlation over time
- Permutation test results
- Drift magnitude and speed
- **Interpretation**: Tests if memories drift together

#### 4. **drift_additional_analyses.png**
- Dimensionality evolution
- Participation ratio trends
- Local stability metrics
- Cumulative drift
- **Interpretation**: Multi-dimensional view of stability

#### 5. **drift_summary_figure.png**
- 9-panel comprehensive summary
- All key findings in one figure
- **Interpretation**: Publication-ready overview

### Console Output:
```
======================================================================
ENGRAM REPRESENTATIONAL DRIFT ANALYSIS
======================================================================

Generating simulated longitudinal engram data...
  Sessions: 30
  Cells: 100
  Trials per session: 20
  Drift rate A: 0.02
  Drift rate B: 0.025

✓ Data generated
  Context A shape: (30, 20, 100)
  Context B shape: (30, 20, 100)

[... detailed statistics for each analysis step ...]

ANALYSIS SUMMARY
======================================================================

Drift Model Results:

  Context A:
    Initial correlation: 0.987
    Decay constant (τ): 24.53 sessions
    Half-life: 17.01 sessions
    R²: 0.994

  Context B:
    Initial correlation: 0.984
    Decay constant (τ): 19.87 sessions
    Half-life: 13.77 sessions
    R²: 0.991

Drift Independence:
  Mean drift correlation: 0.287
  P-value: 0.0012
  Interpretation: Contexts drift IN CORRELATION

======================================================================
ALL ANALYSES COMPLETE
======================================================================
```

---

##  Key Findings

### Typical Results:

1. **Memory Stability**:
   - Context A half-life: ~17 sessions
   - Context B half-life: ~14 sessions
   - ➡️ Different memories have different stability

2. **Drift Characteristics**:
   - Exponential decay fits R² > 0.99
   - ➡️ Drift follows predictable dynamics

3. **Context Independence**:
   - Drift correlation: ~0.3
   - P-value: ~0.001
   - ➡️ Contexts drift partially together (shared mechanisms)

4. **Dimensionality**:
   - Effective dimensions: ~5-8 (from 100 neurons)
   - ➡️ Memories use low-dimensional representations

---

## Project Structure

```
engram-drift-analysis/
│
├── engram_drift_analysis.py      # Main analysis script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── LICENSE                        # MIT License
│
├── outputs/                       # Generated figures (created on run)
│   ├── drift_correlation_matrices.png
│   ├── drift_model_fits.png
│   ├── drift_independence_analysis.png
│   ├── drift_additional_analyses.png
│   └── drift_summary_figure.png
│
└── docs/                          # Additional documentation
    ├── METHODS.md                 # Detailed methodology
    └── THEORY.md                  # Theoretical background
```

---

##  Mathematical Models

### 1. Exponential Decay Model:
```
r(Δt) = r₀ × exp(-Δt / τ)
```
Where:
- `r(Δt)` = correlation at time lag Δt
- `r₀` = initial correlation
- `τ` = decay time constant
- `t₁/₂ = τ × ln(2)` = half-life

### 2. Population Vector Correlation:
```
ρ(i,j) = corr(v⃗ᵢ, v⃗ⱼ)
```
Where:
- `v⃗ᵢ` = mean population activity vector at session i
- `ρ(i,j)` = Pearson correlation between sessions i and j

### 3. Participation Ratio:
```
PR = (Σλᵢ)² / Σ(λᵢ²)
```
Where:
- `λᵢ` = eigenvalues from PCA
- PR = effective dimensionality

### 4. Drift Magnitude:
```
D(t) = ||v⃗(t+1) - v⃗(t)||₂
```
Where:
- `||·||₂` = Euclidean norm
- `D(t)` = drift magnitude at time t

---

## 🚧 Future Enhancements

### Planned Features:
- [ ] **Real data import**: Support for actual calcium imaging formats (TIFF, HDF5)
- [ ] **Cell tracking**: Account for cell identification errors across sessions
- [ ] **Behavioral integration**: Link drift to performance metrics
- [ ] **Multi-context comparison**: Extend beyond 2 contexts
- [ ] **Interactive visualization**: Web-based dashboard
- [ ] **Statistical bootstrapping**: More robust confidence intervals
- [ ] **GPU acceleration**: Faster computation for large datasets
- [ ] **Automated reporting**: Generate PDF reports

### Research Extensions:
- Investigate drift during memory consolidation
- Compare drift across brain regions
- Model the effect of sleep/wake cycles
- Analyze drift in disease models

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Reporting Bugs:
1. Check existing issues first
2. Create a new issue with:
   - Clear description
   - Steps to reproduce
   - Expected vs. actual behavior
   - System information

### Suggesting Enhancements:
1. Open an issue describing the feature
2. Explain the use case
3. Provide examples if possible

### Pull Requests:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style:
- Follow PEP 8 guidelines
- Add docstrings to functions
- Include comments for complex logic
- Update README for new features

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 [Hassan Farooq]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```


---

##  Acknowledgments

- Inspired by research from the Tonegawa Lab (MIT) on memory engrams
- Based on drift analysis methods from Driscoll et al. (2017) Nature
- Visualization techniques adapted from computational neuroscience literature
- Special thanks to the open-source scientific Python community

---

## References

### Key Papers:
1. Driscoll et al. (2017). "Representational drift in primary visual cortex." *Nature*
2. Josselyn & Tonegawa (2020). "Memory engrams: Recalling the past and imagining the future." *Science*
3. Rule et al. (2019). "Causes and consequences of representational drift." *Current Opinion in Neurobiology*

### Documentation:
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [SciPy Reference](https://docs.scipy.org/doc/scipy/reference/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/engram-drift-analysis&type=Date)](https://star-history.com/#yourusername/engram-drift-analysis&Date)

---
##  Project Stats

![GitHub repo size](https://img.shields.io/github/repo-size/yourusername/engram-drift-analysis)
![GitHub code size](https://img.shields.io/github/languages/code-size/yourusername/engram-drift-analysis)
![Lines of code](https://img.shields.io/tokei/lines/github/yourusername/engram-drift-analysis)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/engram-drift-analysis)

---

<div align="center">



[⬆ Back to Top](#engram-representational-drift-analysis)

</div>
