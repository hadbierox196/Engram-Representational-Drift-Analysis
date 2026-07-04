# Engram Representational Drift Analysis

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-013243.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.3%2B-11557c.svg)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Simulates and quantifies **representational drift** — the surprising fact that the neurons encoding a memory keep changing over time, even while the memory itself stays intact.

---

## What it does

- Simulates longitudinal calcium imaging data for neural populations across many sessions
- Tracks how the neural pattern (engram) for a given memory context evolves over time
- Quantifies drift using population-vector correlation between sessions
- Fits an exponential decay model to estimate each memory's "half-life"
- Tests whether two different memories drift independently or together
- Measures the effective dimensionality of the neural code over time
- Produces 5 publication-quality figures summarizing everything

---

## Background

An **engram** is the specific pattern of neuron activity that encodes a memory — remembering something means partially reactivating that pattern. Oddly, even long-stable memories are encoded by neurons whose individual activity patterns keep drifting: the *population code* remains recognizable, but the *cells* carrying it change.

Understanding drift matters because it explains:
- How memories stay stable while individual neurons keep changing
- Why some memories fade while others persist
- How the brain keeps different memories distinct from each other
- The link between ongoing synaptic plasticity and long-term memory

---

## Installation

```bash
git clone https://github.com/yourusername/engram-drift-analysis.git
cd engram-drift-analysis
python -m venv env && source env/bin/activate   # Windows: env\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt**
```text
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.5.0
scikit-learn>=0.23.0
```

---

## Usage

```bash
python engram_drift_analysis.py
```

Simulates 30 sessions × 100 neurons × 20 trials for two memory contexts, runs the full analysis, and prints a statistical summary — done in about 20 seconds.

### Customization

```python
n_sessions = 30              # number of imaging sessions
n_cells = 100                # neurons recorded
n_trials = 20                # trials per session
drift_rate_A = 0.02          # drift speed, context A
drift_rate_B = 0.025         # drift speed, context B
correlation = 0.3            # correlation between contexts
```

---

## Pipeline

1. **Simulate** two memory contexts with realistic noise and gradual drift
2. **Compute** population-vector correlations between every pair of sessions
3. **Visualize** correlation matrices to see how similarity fades with time
4. **Fit** an exponential decay model to extract decay rate and half-life
5. **Test** whether the two contexts' drift is independent or correlated (permutation test)
6. **Analyze** dimensionality (PCA) and stability of the code over time

---

## Outputs

| File | Shows |
|---|---|
| `drift_correlation_matrices.png` | Session-by-session similarity heatmaps for both contexts |
| `drift_model_fits.png` | Correlation vs. time lag with exponential fits + half-life comparison |
| `drift_independence_analysis.png` | Whether the two memories drift together or independently |
| `drift_additional_analyses.png` | Dimensionality, participation ratio, local stability, cumulative drift |
| `drift_summary_figure.png` | 9-panel overview of all key findings |

### Example results

| Finding | Result |
|---|---|
| Context A half-life | ~17 sessions |
| Context B half-life | ~14 sessions |
| Exponential fit quality | R² > 0.99 |
| Cross-context drift correlation | ~0.3 (p ≈ 0.001) — partially shared drift |
| Effective dimensionality | ~5–8 dimensions (from 100 neurons) |

**Takeaway:** representational drift follows a clean, predictable exponential decay; different memories can have different stability; and drift in separate memories is often partially correlated rather than fully independent — consistent with shared underlying plasticity mechanisms.

---

## Math, briefly

**Exponential decay model:** `r(Δt) = r₀ · e^(−Δt/τ)`, half-life `t½ = τ · ln(2)`

**Population vector correlation:** `ρ(i,j) = corr(v̄ᵢ, v̄ⱼ)` between session mean activity vectors

**Participation ratio (effective dimensionality):** `PR = (Σλᵢ)² / Σλᵢ²`

**Drift magnitude:** `D(t) = ‖v̄(t+1) − v̄(t)‖₂`

---

## Roadmap

- Import real calcium imaging data (TIFF/HDF5)
- Handle cell-tracking errors across sessions
- Link drift to behavioral performance
- Extend beyond two memory contexts
- Bootstrapped confidence intervals

---

## Contributing

Issues and PRs welcome — please follow PEP 8, add docstrings, and briefly explain the scientific motivation behind new features.

## License

MIT — see [LICENSE](LICENSE).

## References

- Driscoll et al. (2017) — *Representational drift in primary visual cortex*, Nature
- Josselyn & Tonegawa (2020) — *Memory engrams: Recalling the past and imagining the future*, Science
- Rule et al. (2019) — *Causes and consequences of representational drift*, Current Opinion in Neurobiology

---

<div align="center">

**Understanding how memories stay stable even as the brain keeps changing**

</div>
