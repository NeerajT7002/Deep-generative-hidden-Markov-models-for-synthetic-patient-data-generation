import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for server/CLI environments
import matplotlib.pyplot as plt

# Set clean aesthetic styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'grid.color': '#eeeeee',
    'grid.linestyle': '--',
    'grid.alpha': 0.7
})

def plot_real_vs_synthetic_trajectories(real_data: np.ndarray, 
                                        synth_data: np.ndarray, 
                                        feature_names: list, 
                                        save_path: str = None) -> plt.Figure:
    """
    Plots mean 24-hour trajectories with ±1 standard deviation shaded envelopes
    comparing real patient ICU vitals with DG-HMM synthetic vitals.
    
    Args:
        real_data: Shape (N_real, seq_len, D)
        synth_data: Shape (N_synth, seq_len, D)
        feature_names: List of D feature names
        save_path: Optional path to save the generated figure
    """
    num_features = len(feature_names)
    seq_len = real_data.shape[1]
    time_steps = np.arange(1, seq_len + 1)
    
    # 2 rows x 3 columns for 6 features
    cols = 3
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8), sharex=True)
    axes = axes.flatten()
    
    for i, name in enumerate(feature_names):
        ax = axes[i]
        
        # Real statistics
        real_mean = np.nanmean(real_data[:, :, i], axis=0)
        real_std = np.nanstd(real_data[:, :, i], axis=0)
        
        # Synthetic statistics
        synth_mean = np.nanmean(synth_data[:, :, i], axis=0)
        synth_std = np.nanstd(synth_data[:, :, i], axis=0)
        
        # Plot Real data
        ax.plot(time_steps, real_mean, color='#1f77b4', linewidth=2.0, label='Real MIMIC-III (Mean)')
        ax.fill_between(time_steps, real_mean - real_std, real_mean + real_std, 
                        color='#1f77b4', alpha=0.18, label='Real ±1 Std')
        
        # Plot Synthetic data
        ax.plot(time_steps, synth_mean, color='#d62728', linestyle='--', linewidth=2.0, label='DG-HMM Synthetic (Mean)')
        ax.fill_between(time_steps, synth_mean - synth_std, synth_mean + synth_std, 
                        color='#d62728', alpha=0.18, label='Synthetic ±1 Std')
        
        ax.set_title(name, fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel('Stay Duration (Hours)', fontsize=10)
        ax.set_ylabel('Measurement Value', fontsize=10)
        ax.grid(True)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
            
    # Hide unused subplots if any
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle('Temporal Trajectory Comparison: Real vs. DG-HMM Synthetic Vitals (24-Hour ICU Stay)', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory comparison plot to: {save_path}")
        
    return fig


def plot_transition_matrix(A_matrix: np.ndarray, 
                           state_labels: list = None, 
                           save_path: str = None) -> plt.Figure:
    """
    Plots an annotated heatmap of the learned HMM transition matrix A.
    
    Args:
        A_matrix: Transition matrix of shape (K, K)
        state_labels: Optional labels for each hidden state
        save_path: Optional path to save the generated figure
    """
    K = A_matrix.shape[0]
    if state_labels is None:
        state_labels = [f"State {k}" for k in range(K)]
        
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(A_matrix, cmap='YlGnBu', vmin=0.0, vmax=1.0)
    
    # Colorbar
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Transition Probability P(S_t | S_{t-1})', rotation=-90, va="bottom", fontsize=10)
    
    # Tick marks
    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(K))
    ax.set_xticklabels(state_labels, fontsize=10, rotation=25, ha="right")
    ax.set_yticklabels(state_labels, fontsize=10)
    
    ax.set_xlabel('Next State (S_t)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Current State (S_{t-1})', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Learned HMM Latent State Transition Matrix', fontsize=12, fontweight='bold', pad=12)
    
    # Annotate probabilities inside cells
    for i in range(K):
        for j in range(K):
            val = A_matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=11, fontweight='bold')
            
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved transition matrix heatmap to: {save_path}")
        
    return fig


def plot_state_progression(synthetic_states: np.ndarray, 
                           state_labels: list = None, 
                           save_path: str = None, 
                           num_patients: int = 6) -> plt.Figure:
    """
    Plots the sequence of hidden clinical states over 24 hours for sample synthetic patients.
    
    Args:
        synthetic_states: Shape (N, seq_len) with integer state indices
        state_labels: Optional labels for each hidden state
        save_path: Optional path to save figure
        num_patients: Number of patient timelines to display
    """
    num_samples, seq_len = synthetic_states.shape
    num_display = min(num_patients, num_samples)
    
    K = int(np.max(synthetic_states)) + 1
    if state_labels is None:
        state_labels = [f"State {k}" for k in range(K)]
        
    # State color palette
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728', '#9467bd', '#8c564b'][:K]
    cmap = matplotlib.colors.ListedColormap(colors)
    bounds = np.arange(K + 1) - 0.5
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(synthetic_states[:num_display], cmap=cmap, norm=norm, aspect='auto')
    
    ax.set_yticks(np.arange(num_display))
    ax.set_yticklabels([f"Synth Patient #{i+1}" for i in range(num_display)], fontsize=10)
    ax.set_xticks(np.arange(0, seq_len, 2))
    ax.set_xticklabels(np.arange(1, seq_len + 1, 2), fontsize=10)
    
    ax.set_xlabel('ICU Stay Duration (Hours)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Decoded Latent Clinical State Progression Over Time', fontsize=12, fontweight='bold', pad=10)
    
    # Legend
    patches = [matplotlib.patches.Patch(color=colors[k], label=state_labels[k]) for k in range(K)]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="Clinical States")
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved state progression plot to: {save_path}")
        
    return fig


def plot_loss_curves(loss_history: dict, save_path: str = None) -> plt.Figure:
    """
    Plots training loss convergence curves for total joint loss and individual components.
    
    Args:
        loss_history: Dict containing lists: 'total', 'recon', 'hmm', 'kl'
        save_path: Optional path to save figure
    """
    epochs = np.arange(1, len(loss_history['total']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    
    # Total Joint Loss
    ax1.plot(epochs, loss_history['total'], color='#2b5c8f', linewidth=2.2, marker='o', markersize=4)
    ax1.set_title('Joint Training Objective: L_total', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Total Loss', fontsize=10)
    ax1.grid(True)
    
    # Component Losses
    ax2.plot(epochs, loss_history['recon'], label='Reconstruction (MSE)', color='#d62728', linewidth=1.8)
    ax2.plot(epochs, loss_history['hmm'], label='HMM Negative Log-Likelihood', color='#2ca02c', linewidth=1.8)
    ax2.plot(epochs, loss_history['kl'], label='KL Divergence (Regularizer)', color='#ff7f0e', linewidth=1.8)
    
    ax2.set_title('Decomposed Loss Components', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Component Loss', fontsize=10)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True)
    
    plt.suptitle('DG-HMM Optimization Convergence', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved loss curve plot to: {save_path}")
        
    return fig


def plot_feature_distributions(real_data: np.ndarray, 
                               synth_data: np.ndarray, 
                               feature_names: list, 
                               save_path: str = None) -> plt.Figure:
    """
    Plots marginal histograms comparing distributions of all vital features.
    
    Args:
        real_data: Shape (N_real, seq_len, D)
        synth_data: Shape (N_synth, seq_len, D)
        feature_names: List of D feature names
        save_path: Optional path to save figure
    """
    num_features = len(feature_names)
    real_flat = real_data.reshape(-1, num_features)
    synth_flat = synth_data.reshape(-1, num_features)
    
    cols = 3
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, name in enumerate(feature_names):
        ax = axes[i]
        
        # Histograms
        ax.hist(real_flat[:, i], bins=25, density=True, alpha=0.55, color='#1f77b4', label='Real MIMIC-III')
        ax.hist(synth_flat[:, i], bins=25, density=True, alpha=0.55, color='#d62728', label='DG-HMM Synthetic')
        
        ax.set_title(name, fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel('Measurement Value', fontsize=10)
        ax.set_ylabel('Probability Density', fontsize=10)
        ax.grid(True)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
            
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle('Marginal Distribution Fidelity: Real vs. Synthetic Patient Vitals', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature distribution plot to: {save_path}")
        
    return fig


def plot_counterfactual_scenarios(scenarios_dict: dict, feature_names: list, 
                                  save_path: str = None) -> plt.Figure:
    """
    Plots controllable counterfactual patient trajectories under different clinical narratives.
    
    Args:
        scenarios_dict: Dict with keys = scenario title, 
                        values = dict('vitals': np.ndarray of shape (seq_len, D), 
                                      'states': np.ndarray of shape (seq_len,))
        feature_names: List of D feature names
        save_path: Optional path to save figure
    """
    num_scenarios = len(scenarios_dict)
    fig, axes = plt.subplots(num_scenarios, 2, figsize=(16, 4 * num_scenarios), sharex=True)
    
    # State color map: 0=Stable (Green), 1=Moderate (Blue), 2=Guarded (Orange), 3=Critical (Red)
    state_colors = {0: '#2ca02c', 1: '#1f77b4', 2: '#ff7f0e', 3: '#d62728'}
    state_labels = {0: 'Stable', 1: 'Moderate', 2: 'Guarded', 3: 'Critical'}
    
    # Track indices for Heart Rate, Systolic BP, Diastolic BP
    hr_idx = feature_names.index('Heart Rate') if 'Heart Rate' in feature_names else 0
    sbp_idx = feature_names.index('Systolic BP') if 'Systolic BP' in feature_names else 1
    dbp_idx = feature_names.index('Diastolic BP') if 'Diastolic BP' in feature_names else 2
    
    for row_idx, (title, data) in enumerate(scenarios_dict.items()):
        vitals = data['vitals']  # (T, D)
        states = data['states']  # (T,)
        T = len(states)
        hours = np.arange(1, T + 1)
        
        # Left subplot: Cardiovascular Dynamics (Heart Rate & Blood Pressure)
        ax_left = axes[row_idx, 0] if num_scenarios > 1 else axes[0]
        ax_right = axes[row_idx, 1] if num_scenarios > 1 else axes[1]
        
        # Plot vitals
        ax_left.plot(hours, vitals[:, hr_idx], color='#d62728', lw=2.5, marker='o', markersize=4, label='Heart Rate (BPM)')
        ax_left.plot(hours, vitals[:, sbp_idx], color='#1f77b4', lw=2.5, marker='s', markersize=4, label='Systolic BP (mmHg)')
        ax_left.plot(hours, vitals[:, dbp_idx], color='#17becf', lw=2.0, linestyle='--', label='Diastolic BP (mmHg)')
        
        # Color background by clinical hidden state
        for t in range(T):
            s = int(states[t])
            ax_left.axvspan(t + 0.5, t + 1.5, color=state_colors[s], alpha=0.15)
            ax_right.axvspan(t + 0.5, t + 1.5, color=state_colors[s], alpha=0.15)
            
        ax_left.set_title(f"{title} - Cardiovascular Response", fontsize=11, fontweight='bold')
        ax_left.set_ylabel("Pressure / Rate", fontsize=10)
        ax_left.grid(True)
        ax_left.legend(loc='upper right', fontsize=8)
        
        # Right subplot: Discrete Hidden State Progression Timeline
        ax_right.step(hours, states, where='mid', color='#333333', lw=2.5, marker='D', markersize=5)
        ax_right.set_yticks([0, 1, 2, 3])
        ax_right.set_yticklabels(['State 0\n(Stable)', 'State 1\n(Moderate)', 'State 2\n(Guarded)', 'State 3\n(Critical)'])
        ax_right.set_title(f"{title} - Latent Markov State Path", fontsize=11, fontweight='bold')
        ax_right.set_ylabel("Clinical State", fontsize=10)
        ax_right.grid(True)
        
        if row_idx == num_scenarios - 1:
            ax_left.set_xlabel("ICU Stay Timeline (Hours 1 to 24)", fontsize=11)
            ax_right.set_xlabel("ICU Stay Timeline (Hours 1 to 24)", fontsize=11)
            
    plt.suptitle("Controllable Counterfactual Patient Simulations (Clinical Path Steering)", 
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved counterfactual simulation plot to: {save_path}")
        
    return fig

