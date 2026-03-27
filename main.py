"""
Assignment 4: Engram Representational Drift Analysis
Longitudinal calcium imaging analysis of memory engrams
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

print("="*70)
print("ENGRAM REPRESENTATIONAL DRIFT ANALYSIS")
print("="*70)

#%% PART 1: GENERATE SIMULATED LONGITUDINAL DATA

def generate_longitudinal_engram_data(n_sessions=30, n_cells=100, 
                                     n_trials_per_session=20,
                                     drift_rate_A=0.02, drift_rate_B=0.025,
                                     correlation_between_contexts=0.3):
    """
    Generate simulated longitudinal calcium imaging data
    
    Parameters:
    -----------
    n_sessions : int
        Number of imaging sessions
    n_cells : int
        Number of neurons
    n_trials_per_session : int
        Trials per session per context
    drift_rate_A : float
        Drift rate for context A (per session)
    drift_rate_B : float
        Drift rate for context B
    correlation_between_contexts : float
        Correlation between A and B drift
        
    Returns:
    --------
    data : dict
        Contains neural activity for both contexts across sessions
    """
    
    print("\nGenerating simulated longitudinal engram data...")
    print(f"  Sessions: {n_sessions}")
    print(f"  Cells: {n_cells}")
    print(f"  Trials per session: {n_trials_per_session}")
    print(f"  Drift rate A: {drift_rate_A}")
    print(f"  Drift rate B: {drift_rate_B}")
    
    # Initialize engram patterns
    # Each context has a distinct population code
    engram_A_initial = np.random.rand(n_cells) > 0.7  # 30% of cells active
    engram_B_initial = np.random.rand(n_cells) > 0.7
    
    # Ensure some overlap but not complete
    overlap = np.random.rand(n_cells) > 0.85
    engram_B_initial[overlap] = engram_A_initial[overlap]
    
    # Storage for all sessions
    activity_A = []  # (n_sessions, n_trials, n_cells)
    activity_B = []
    
    engram_A = engram_A_initial.copy().astype(float)
    engram_B = engram_B_initial.copy().astype(float)
    
    for session in range(n_sessions):
        # Drift the engrams
        if session > 0:
            # Independent drift component
            drift_noise_A = np.random.randn(n_cells) * drift_rate_A
            drift_noise_B = np.random.randn(n_cells) * drift_rate_B
            
            # Correlated drift component
            shared_drift = np.random.randn(n_cells) * drift_rate_A
            
            engram_A += (1 - correlation_between_contexts) * drift_noise_A + \
                       correlation_between_contexts * shared_drift
            engram_B += (1 - correlation_between_contexts) * drift_noise_B + \
                       correlation_between_contexts * shared_drift
            
            # Keep bounded
            engram_A = np.clip(engram_A, 0, 1)
            engram_B = np.clip(engram_B, 0, 1)
        
        # Generate trials for this session
        trials_A = []
        trials_B = []
        
        for trial in range(n_trials_per_session):
            # Context A trial
            activity = engram_A + np.random.randn(n_cells) * 0.1
            activity = np.clip(activity, 0, None)
            trials_A.append(activity)
            
            # Context B trial
            activity = engram_B + np.random.randn(n_cells) * 0.1
            activity = np.clip(activity, 0, None)
            trials_B.append(activity)
        
        activity_A.append(np.array(trials_A))
        activity_B.append(np.array(trials_B))
    
    data = {
        'context_A': np.array(activity_A),  # (n_sessions, n_trials, n_cells)
        'context_B': np.array(activity_B),
        'n_sessions': n_sessions,
        'n_cells': n_cells,
        'n_trials': n_trials_per_session,
        'drift_rate_A': drift_rate_A,
        'drift_rate_B': drift_rate_B,
        'correlation': correlation_between_contexts
    }
    
    print(f"\n✓ Data generated")
    print(f"  Context A shape: {data['context_A'].shape}")
    print(f"  Context B shape: {data['context_B'].shape}")
    
    return data


# Generate data
n_sessions = 30
n_cells = 100
n_trials = 20

data = generate_longitudinal_engram_data(
    n_sessions=n_sessions,
    n_cells=n_cells,
    n_trials_per_session=n_trials,
    drift_rate_A=0.02,
    drift_rate_B=0.025,
    correlation_between_contexts=0.3
)


#%% PART 2: COMPUTE POPULATION VECTOR CORRELATIONS

print("\n" + "="*70)
print("PART 2: Population Vector Correlation Analysis")
print("="*70)

def compute_population_vectors(activity):
    """
    Compute mean population vector for each session
    
    Parameters:
    -----------
    activity : array (n_sessions, n_trials, n_cells)
        Neural activity
        
    Returns:
    --------
    pop_vectors : array (n_sessions, n_cells)
        Mean population vector per session
    """
    return np.mean(activity, axis=1)


def compute_correlation_matrix(pop_vectors):
    """
    Compute correlation between all pairs of sessions
    
    Parameters:
    -----------
    pop_vectors : array (n_sessions, n_cells)
        Population vectors
        
    Returns:
    --------
    corr_matrix : array (n_sessions, n_sessions)
        Correlation matrix
    """
    n_sessions = pop_vectors.shape[0]
    corr_matrix = np.corrcoef(pop_vectors)
    return corr_matrix


print("\nComputing population vectors...")

pop_vectors_A = compute_population_vectors(data['context_A'])
pop_vectors_B = compute_population_vectors(data['context_B'])

print(f"✓ Population vectors computed")
print(f"  Context A: {pop_vectors_A.shape}")
print(f"  Context B: {pop_vectors_B.shape}")

print("\nComputing correlation matrices...")

corr_matrix_A = compute_correlation_matrix(pop_vectors_A)
corr_matrix_B = compute_correlation_matrix(pop_vectors_B)

print(f"✓ Correlation matrices computed")


#%% PART 3: VISUALIZE CORRELATION MATRICES

print("\nVisualizing correlation matrices...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Context A
ax = axes[0]
im = ax.imshow(corr_matrix_A, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xlabel('Session', fontsize=11)
ax.set_ylabel('Session', fontsize=11)
ax.set_title('Context A: Population Vector Correlation', 
            fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Correlation')

# Context B
ax = axes[1]
im = ax.imshow(corr_matrix_B, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xlabel('Session', fontsize=11)
ax.set_ylabel('Session', fontsize=11)
ax.set_title('Context B: Population Vector Correlation', 
            fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Correlation')

# Difference
ax = axes[2]
diff = corr_matrix_A - corr_matrix_B
im = ax.imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
ax.set_xlabel('Session', fontsize=11)
ax.set_ylabel('Session', fontsize=11)
ax.set_title('Difference (A - B)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Correlation Difference')

plt.tight_layout()
plt.savefig('drift_correlation_matrices.png', dpi=150, bbox_inches='tight')
plt.show()


#%% PART 4: CORRELATION VS LAG ANALYSIS

print("\n" + "="*70)
print("PART 4: Correlation vs Temporal Lag")
print("="*70)

def compute_correlation_vs_lag(corr_matrix):
    """
    Extract correlation as a function of temporal lag
    
    Parameters:
    -----------
    corr_matrix : array (n_sessions, n_sessions)
        Correlation matrix
        
    Returns:
    --------
    lags : array
        Temporal lags
    correlations_mean : array
        Mean correlation at each lag
    correlations_sem : array
        SEM of correlations at each lag
    """
    n_sessions = corr_matrix.shape[0]
    max_lag = n_sessions - 1
    
    lags = []
    correlations_mean = []
    correlations_sem = []
    
    for lag in range(max_lag + 1):
        # Extract all pairs with this lag
        corr_values = []
        for i in range(n_sessions - lag):
            j = i + lag
            corr_values.append(corr_matrix[i, j])
        
        lags.append(lag)
        correlations_mean.append(np.mean(corr_values))
        correlations_sem.append(np.std(corr_values) / np.sqrt(len(corr_values)))
    
    return np.array(lags), np.array(correlations_mean), np.array(correlations_sem)


print("\nComputing correlation vs lag...")

lags_A, corr_mean_A, corr_sem_A = compute_correlation_vs_lag(corr_matrix_A)
lags_B, corr_mean_B, corr_sem_B = compute_correlation_vs_lag(corr_matrix_B)

print(f"✓ Correlation vs lag computed")


#%% PART 5: FIT DRIFT MODEL

print("\n" + "="*70)
print("PART 5: Exponential Drift Model Fitting")
print("="*70)

def exponential_decay(t, r0, tau):
    """
    Exponential decay model: r(t) = r0 * exp(-t/tau)
    
    Parameters:
    -----------
    t : array
        Time lags
    r0 : float
        Initial correlation
    tau : float
        Decay time constant (engram half-life = tau * ln(2))
        
    Returns:
    --------
    r : array
        Predicted correlation
    """
    return r0 * np.exp(-t / tau)


def fit_drift_model(lags, correlations):
    """
    Fit exponential decay model to correlation data
    
    Returns:
    --------
    params : tuple
        (r0, tau) - initial correlation and decay constant
    half_life : float
        Engram half-life in sessions
    """
    # Initial guess
    p0 = [1.0, 10.0]
    
    # Fit
    params, covariance = curve_fit(exponential_decay, lags, correlations, 
                                   p0=p0, maxfev=10000)
    
    r0, tau = params
    half_life = tau * np.log(2)
    
    # Compute R²
    predicted = exponential_decay(lags, *params)
    residuals = correlations - predicted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((correlations - np.mean(correlations))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return params, half_life, r_squared


print("\nFitting exponential decay models...")

# Fit Context A
params_A, half_life_A, r2_A = fit_drift_model(lags_A, corr_mean_A)
print(f"\nContext A:")
print(f"  Initial correlation (r0): {params_A[0]:.3f}")
print(f"  Decay constant (τ): {params_A[1]:.2f} sessions")
print(f"  Half-life: {half_life_A:.2f} sessions")
print(f"  R²: {r2_A:.3f}")

# Fit Context B
params_B, half_life_B, r2_B = fit_drift_model(lags_B, corr_mean_B)
print(f"\nContext B:")
print(f"  Initial correlation (r0): {params_B[0]:.3f}")
print(f"  Decay constant (τ): {params_B[1]:.2f} sessions")
print(f"  Half-life: {half_life_B:.2f} sessions")
print(f"  R²: {r2_B:.3f}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Correlation vs lag with fits
ax = ax1
ax.errorbar(lags_A, corr_mean_A, yerr=corr_sem_A,
           fmt='o', capsize=5, markersize=8, linewidth=2,
           color='red', label='Context A (data)', alpha=0.7)

ax.errorbar(lags_B, corr_mean_B, yerr=corr_sem_B,
           fmt='s', capsize=5, markersize=8, linewidth=2,
           color='blue', label='Context B (data)', alpha=0.7)

# Plot fits
t_fit = np.linspace(0, max(lags_A), 100)
ax.plot(t_fit, exponential_decay(t_fit, *params_A),
       'r-', linewidth=2, label=f'Context A fit (τ={params_A[1]:.1f})')
ax.plot(t_fit, exponential_decay(t_fit, *params_B),
       'b-', linewidth=2, label=f'Context B fit (τ={params_B[1]:.1f})')

ax.set_xlabel('Temporal Lag (sessions)', fontsize=12)
ax.set_ylabel('Population Vector Correlation', fontsize=12)
ax.set_title('Representational Drift Over Time', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)

# Half-life comparison
ax = ax2
contexts = ['Context A', 'Context B']
half_lives = [half_life_A, half_life_B]
colors = ['red', 'blue']

bars = ax.bar(contexts, half_lives, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

for i, (bar, hl) in enumerate(zip(bars, half_lives)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
           f'{hl:.1f}', ha='center', fontsize=12, fontweight='bold')

ax.set_ylabel('Engram Half-Life (sessions)', fontsize=12)
ax.set_title('Memory Stability Comparison', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('drift_model_fits.png', dpi=150, bbox_inches='tight')
plt.show()


#%% PART 6: TEST INDEPENDENCE OF DRIFT

print("\n" + "="*70)
print("PART 6: Testing Independence of Drift Between Contexts")
print("="*70)

def compute_cross_context_correlation(pop_vectors_A, pop_vectors_B):
    """
    Compute correlation between contexts at each session
    """
    n_sessions = pop_vectors_A.shape[0]
    cross_corr = []
    
    for session in range(n_sessions):
        corr = np.corrcoef(pop_vectors_A[session], pop_vectors_B[session])[0, 1]
        cross_corr.append(corr)
    
    return np.array(cross_corr)


def compute_drift_vectors(pop_vectors):
    """
    Compute drift vectors (change between consecutive sessions)
    """
    drift_vectors = []
    for i in range(len(pop_vectors) - 1):
        drift = pop_vectors[i+1] - pop_vectors[i]
        drift_vectors.append(drift)
    
    return np.array(drift_vectors)


def test_drift_correlation(drift_A, drift_B):
    """
    Test if drift is correlated between contexts
    """
    n_steps = min(len(drift_A), len(drift_B))
    
    correlations = []
    for i in range(n_steps):
        corr = np.corrcoef(drift_A[i], drift_B[i])[0, 1]
        correlations.append(corr)
    
    mean_corr = np.mean(correlations)
    
    # Permutation test
    n_permutations = 1000
    null_correlations = []
    
    for _ in range(n_permutations):
        # Shuffle drift vectors
        shuffled_drift_B = drift_B[np.random.permutation(n_steps)]
        
        perm_corr = []
        for i in range(n_steps):
            corr = np.corrcoef(drift_A[i], shuffled_drift_B[i])[0, 1]
            perm_corr.append(corr)
        
        null_correlations.append(np.mean(perm_corr))
    
    null_correlations = np.array(null_correlations)
    p_value = np.mean(np.abs(null_correlations) >= np.abs(mean_corr))
    
    return mean_corr, p_value, null_correlations


print("\nComputing cross-context correlation...")
cross_corr = compute_cross_context_correlation(pop_vectors_A, pop_vectors_B)

print("\nComputing drift vectors...")
drift_A = compute_drift_vectors(pop_vectors_A)
drift_B = compute_drift_vectors(pop_vectors_B)

print("\nTesting drift correlation...")
mean_drift_corr, p_value, null_dist = test_drift_correlation(drift_A, drift_B)

print(f"\n✓ Drift independence test complete")
print(f"  Mean drift correlation: {mean_drift_corr:.3f}")
print(f"  P-value: {p_value:.4f}")
print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Cross-context correlation over time
ax = axes[0, 0]
ax.plot(range(n_sessions), cross_corr, 'o-', linewidth=2, markersize=8, color='purple')
ax.set_xlabel('Session', fontsize=11)
ax.set_ylabel('Cross-Context Correlation', fontsize=11)
ax.set_title('Context A vs B Similarity Over Time', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Drift correlation distribution
ax = axes[0, 1]
ax.hist(null_dist, bins=30, alpha=0.7, color='gray', edgecolor='black',
       label='Null distribution')
ax.axvline(mean_drift_corr, color='red', linewidth=3, linestyle='--',
          label=f'Observed ({mean_drift_corr:.3f})')
ax.set_xlabel('Mean Drift Correlation', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'Permutation Test (p={p_value:.4f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Drift magnitude over time
ax = axes[1, 0]
drift_mag_A = np.linalg.norm(drift_A, axis=1)
drift_mag_B = np.linalg.norm(drift_B, axis=1)

ax.plot(range(len(drift_mag_A)), drift_mag_A, 'o-', linewidth=2, 
       markersize=6, color='red', label='Context A', alpha=0.7)
ax.plot(range(len(drift_mag_B)), drift_mag_B, 's-', linewidth=2,
       markersize=6, color='blue', label='Context B', alpha=0.7)
ax.set_xlabel('Session Transition', fontsize=11)
ax.set_ylabel('Drift Magnitude', fontsize=11)
ax.set_title('Drift Speed Over Time', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Scatter: drift A vs drift B (example session)
ax = axes[1, 1]
example_session = n_sessions // 2

ax.scatter(drift_A[example_session], drift_B[example_session],
          alpha=0.6, s=50, c='purple', edgecolors='black')

# Linear fit
slope, intercept = np.polyfit(drift_A[example_session], 
                              drift_B[example_session], 1)
x_fit = np.linspace(drift_A[example_session].min(), 
                    drift_A[example_session].max(), 100)
ax.plot(x_fit, slope * x_fit + intercept, 'r--', linewidth=2,
       label=f'r={np.corrcoef(drift_A[example_session], drift_B[example_session])[0,1]:.2f}')

ax.set_xlabel('Context A Drift (per neuron)', fontsize=11)
ax.set_ylabel('Context B Drift (per neuron)', fontsize=11)
ax.set_title(f'Drift Correlation (Session {example_session})', 
            fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('drift_independence_analysis.png', dpi=150, bbox_inches='tight')
plt.show()


#%% PART 7: ADDITIONAL ANALYSES

print("\n" + "="*70)
print("PART 7: Additional Analyses")
print("="*70)

# Dimensionality analysis
print("\nAnalyzing representational dimensionality over time...")

def compute_dimensionality(activity_data, n_components=10):
    """Compute effective dimensionality using PCA"""
    n_sessions = activity_data.shape[0]
    
    explained_variance = []
    participation_ratio = []
    
    for session in range(n_sessions):
        # Mean activity across trials
        data = activity_data[session]  # (n_trials, n_cells)
        
        if data.shape[0] > 1:
            pca = PCA(n_components=min(n_components, data.shape[0]-1))
            pca.fit(data)
            
            explained_variance.append(pca.explained_variance_ratio_)
            
            # Participation ratio
            eigenvalues = pca.explained_variance_
            pr = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)
            participation_ratio.append(pr)
    
    return explained_variance, participation_ratio


ev_A, pr_A = compute_dimensionality(data['context_A'])
ev_B, pr_B = compute_dimensionality(data['context_B'])

print(f"✓ Dimensionality analysis complete")

# Engram stability index
print("\nComputing engram stability index...")

def compute_stability_index(corr_matrix, window=5):
    """
    Stability index: mean correlation within a temporal window
    """
    n_sessions = corr_matrix.shape[0]
    stability = []
    
    for i in range(n_sessions - window):
        window_corr = []
        for j in range(i, i + window):
            for k in range(j+1, i + window):
                window_corr.append(corr_matrix[j, k])
        stability.append(np.mean(window_corr))
    
    return np.array(stability)


stability_A = compute_stability_index(corr_matrix_A)
stability_B = compute_stability_index(corr_matrix_B)

print(f"✓ Stability index computed")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Explained variance over time
ax = axes[0, 0]
ev_mean_A = [np.sum(ev[:3]) for ev in ev_A]
ev_mean_B = [np.sum(ev[:3]) for ev in ev_B]

ax.plot(range(len(ev_mean_A)), ev_mean_A, 'o-', linewidth=2,
       markersize=6, color='red', label='Context A', alpha=0.7)
ax.plot(range(len(ev_mean_B)), ev_mean_B, 's-', linewidth=2,
       markersize=6, color='blue', label='Context B', alpha=0.7)
ax.set_xlabel('Session', fontsize=11)
ax.set_ylabel('Cumulative Var. Explained (PC1-3)', fontsize=11)
ax.set_title('Dimensionality Over Time', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Participation ratio
ax = axes[0, 1]
ax.plot(range(len(pr_A)), pr_A, 'o-', linewidth=2,
       markersize=6, color='red', label='Context A', alpha=0.7)
ax.plot(range(len(pr_B)), pr_B, 's-', linewidth=2,
       markersize=6, color='blue', label='Context B', alpha=0.7)
ax.set_xlabel('Session', fontsize=11)
ax.set_ylabel('Participation Ratio', fontsize=11)
ax.set_title('Effective Dimensionality', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Stability index
ax = axes[1, 0]
ax.plot(range(len(stability_A)), stability_A, 'o-', linewidth=2,
       markersize=6, color='red', label='Context A', alpha=0.7)
ax.plot(range(len(stability_B)), stability_B, 's-', linewidth=2,
       markersize=6, color='blue', label='Context B', alpha=0.7)
ax.set_xlabel('Session', fontsize=11)
ax.set_ylabel('Stability Index (5-session window)', fontsize=11)
ax.set_title('Local Stability Over Time', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Cumulative drift
ax = axes[1, 1]
cumulative_drift_A = np.cumsum(np.linalg.norm(drift_A, axis=1))
cumulative_drift_B = np.cumsum(np.linalg.norm(drift_B, axis=1))

ax.plot(range(len(cumulative_drift_A)), cumulative_drift_A, 'o-', linewidth=2,
       markersize=6, color='red', label='Context A', alpha=0.7)
ax.plot(range(len(cumulative_drift_B)), cumulative_drift_B, 's-', linewidth=2,
       markersize=6, color='blue', label='Context B', alpha=0.7)
ax.set_xlabel('Session', fontsize=11)
ax.set_ylabel('Cumulative Drift', fontsize=11)
ax.set_title('Total Representational Change', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('drift_additional_analyses.png', dpi=150, bbox_inches='tight')
plt.show()


#%% PART 8: SUMMARY FIGURE

print("\n" + "="*70)
print("PART 8: Summary Figure")
print("="*70)

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

# 1. Correlation matrices
ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(corr_matrix_A, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
ax1.set_title('Context A Correlation Matrix', fontsize=11, fontweight='bold')
ax1.set_xlabel('Session', fontsize=9)
ax1.set_ylabel('Session', fontsize=9)
plt.colorbar(im, ax=ax1, fraction=0.046)

ax2 = fig.add_subplot(gs[0, 1])
im = ax2.imshow(corr_matrix_B, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
ax2.set_title('Context B Correlation Matrix', fontsize=11, fontweight='bold')
ax2.set_xlabel('Session', fontsize=9)
ax2.set_ylabel('Session', fontsize=9)
plt.colorbar(im, ax=ax2, fraction=0.046)

# 2. Drift curves with fits
ax3 = fig.add_subplot(gs[0, 2])
ax3.errorbar(lags_A, corr_mean_A, yerr=corr_sem_A,
            fmt='o', markersize=6, color='red', alpha=0.6)
ax3.errorbar(lags_B, corr_mean_B, yerr=corr_sem_B,
            fmt='s', markersize=6, color='blue', alpha=0.6)
t_fit = np.linspace(0, max(lags_A), 100)
ax3.plot(t_fit, exponential_decay(t_fit, *params_A), 'r-', linewidth=2)
ax3.plot(t_fit, exponential_decay(t_fit, *params_B), 'b-', linewidth=2)
ax3.set_xlabel('Lag (sessions)', fontsize=9)
ax3.set_ylabel('Correlation', fontsize=9)
ax3.set_title('Representational Drift', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 3. Half-life comparison
ax4 = fig.add_subplot(gs[1, 0])
ax4.bar(['Context A', 'Context B'], [half_life_A, half_life_B],
       color=['red', 'blue'], alpha=0.7, edgecolor='black')
ax4.set_ylabel('Half-Life (sessions)', fontsize=9)
ax4.set_title('Memory Stability', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# 4. Cross-context correlation
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(range(n_sessions), cross_corr, 'o-', color='purple', linewidth=2, markersize=6)
ax5.set_xlabel('Session', fontsize=9)
ax5.set_ylabel('Cross-Context Correlation', fontsize=9)
ax5.set_title('Context Similarity Over Time', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 5. Drift independence test
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(null_dist, bins=30, alpha=0.7, color='gray', edgecolor='black')
ax6.axvline(mean_drift_corr, color='red', linewidth=2, linestyle='--')
ax6.set_xlabel('Drift Correlation', fontsize=9)
ax6.set_ylabel('Count', fontsize=9)
ax6.set_title(f'Independence Test (p={p_value:.3f})', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

# 6. Drift magnitude
ax7 = fig.add_subplot(gs[2, 0])
ax7.plot(drift_mag_A, 'o-', color='red', linewidth=2, markersize=5, alpha=0.7, label='A')
ax7.plot(drift_mag_B, 's-', color='blue', linewidth=2, markersize=5, alpha=0.7, label='B')
ax7.set_xlabel('Transition', fontsize=9)
ax7.set_ylabel('Drift Magnitude', fontsize=9)
ax7.set_title('Drift Speed', fontsize=11, fontweight='bold')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# 7. Participation ratio
ax8 = fig.add_subplot(gs[2, 1])
ax8.plot(pr_A, 'o-', color='red', linewidth=2, markersize=5, alpha=0.7, label='A')
ax8.plot(pr_B, 's-', color='blue', linewidth=2, markersize=5, alpha=0.7, label='B')
ax8.set_xlabel('Session', fontsize=9)
ax8.set_ylabel('Participation Ratio', fontsize=9)
ax8.set_title('Effective Dimensionality', fontsize=11, fontweight='bold')
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3)

# 8. Stability index
ax9 = fig.add_subplot(gs[2, 2])
ax9.plot(stability_A, 'o-', color='red', linewidth=2, markersize=5, alpha=0.7, label='A')
ax9.plot(stability_B, 's-', color='blue', linewidth=2, markersize=5, alpha=0.7, label='B')
ax9.set_xlabel('Session', fontsize=9)
ax9.set_ylabel('Stability Index', fontsize=9)
ax9.set_title('Local Stability', fontsize=11, fontweight='bold')
ax9.legend(fontsize=8)
ax9.grid(True, alpha=0.3)

plt.suptitle('Engram Representational Drift: Comprehensive Analysis',
            fontsize=14, fontweight='bold')

plt.savefig('drift_summary_figure.png', dpi=150, bbox_inches='tight')
plt.show()


#%% SUMMARY

print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

print(f"\nDataset:")
print(f"  Sessions: {n_sessions}")
print(f"  Cells: {n_cells}")
print(f"  Trials per session: {n_trials}")

print(f"\nDrift Model Results:")
print(f"\n  Context A:")
print(f"    Initial correlation: {params_A[0]:.3f}")
print(f"    Decay constant (τ): {params_A[1]:.2f} sessions")
print(f"    Half-life: {half_life_A:.2f} sessions")
print(f"    R²: {r2_A:.3f}")

print(f"\n  Context B:")
print(f"    Initial correlation: {params_B[0]:.3f}")
print(f"    Decay constant (τ): {params_B[1]:.2f} sessions")
print(f"    Half-life: {half_life_B:.2f} sessions")
print(f"    R²: {r2_B:.3f}")

print(f"\nDrift Independence:")
print(f"  Mean drift correlation: {mean_drift_corr:.3f}")
print(f"  P-value: {p_value:.4f}")
print(f"  Interpretation: Contexts drift {'INDEPENDENTLY' if p_value > 0.05 else 'IN CORRELATION'}")

print(f"\nStability Metrics:")
print(f"  Mean stability (A): {np.mean(stability_A):.3f}")
print(f"  Mean stability (B): {np.mean(stability_B):.3f}")
print(f"  Mean participation ratio (A): {np.mean(pr_A):.2f}")
print(f"  Mean participation ratio (B): {np.mean(pr_B):.2f}")

print("\n" + "="*70)
print("ALL ANALYSES COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  - drift_correlation_matrices.png")
print("  - drift_model_fits.png")
print("  - drift_independence_analysis.png")
print("  - drift_additional_analyses.png")
print("  - drift_summary_figure.png")
