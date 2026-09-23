import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root is in sys.path for direct command line execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.dghmm import DGHMM
from src.utils.data_loader import get_dataloader, preprocess_mimic_demo, VitalsScaler

def compute_model_parameters_count(model: torch.nn.Module) -> int:
    """Returns the total count of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_bic_aic(data: np.ndarray, 
                     candidate_states: list = [2, 3, 4, 5, 6, 8], 
                     epochs: int = 20, 
                     batch_size: int = 16,
                     latent_dim: int = 8,
                     device: str = 'cpu',
                     save_plot_path: str = None) -> dict:
    """
    Adaptive HMM State Selection using Bayesian Information Criterion (BIC)
    and Akaike Information Criterion (AIC) as formulated in Momenzadeh & Oshaghi (2026, Page 4):
    
        BIC = -2 * log P(X | theta) + k * log(n)
        AIC = -2 * log P(X | theta) + 2 * k
        
    where k is the number of trainable parameters and n is the sample size (number of patient sequences).
    
    Args:
        data (np.ndarray): Scaled patient sequences of shape (N, T, D)
        candidate_states (list): List of state counts K to test
        epochs (int): Training epochs per candidate
        batch_size (int): Mini-batch size
        latent_dim (int): Latent continuous manifold dimension
        device (str): Computation device
        save_plot_path (str): Optional path to save evaluation plot
        
    Returns:
        results (dict): Dictionary with BIC/AIC scores, parameter counts, and optimal K
    """
    N, T, D = data.shape
    loader = get_dataloader(data, batch_size=batch_size, shuffle=True)
    
    bic_scores = []
    aic_scores = []
    log_likelihoods = []
    param_counts = []
    
    print("\n" + "=" * 70)
    print(" ADAPTIVE STATE SELECTION (BIC / AIC) - Momenzadeh & Oshaghi (2026)")
    print("=" * 70)
    print(f" {'K (States)':<10} | {'Log-Likelihood':<16} | {'Params (k)':<12} | {'BIC Score':<14} | {'AIC Score':<14}")
    print("-" * 70)
    
    for K in candidate_states:
        # Initialize model for state count K
        model = DGHMM(
            input_dim=D,
            latent_dim=latent_dim,
            num_states=K,
            hidden_dims_enc=[64, 32],
            hidden_dims_dec=[32, 64],
            num_mixtures=3,
            lambda_hmm=0.1,
            beta_kl=0.01
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        model.train()
        
        # Fast convergence training
        for ep in range(epochs):
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss, _ = model.compute_loss(batch, state_mode='viterbi')
                loss.backward()
                optimizer.step()
                
        # Evaluate total sequence log-likelihood
        model.eval()
        total_ll = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                z, _, _ = model.encoder(batch)
                batch_ll = model.hmm.compute_sequence_log_likelihood(z)
                total_ll += batch_ll
                
        k_params = compute_model_parameters_count(model)
        
        # BIC = -2 * LL + k * log(N)
        # AIC = -2 * LL + 2 * k
        bic = -2.0 * total_ll + float(k_params) * np.log(N * T)
        aic = -2.0 * total_ll + 2.0 * float(k_params)
        
        bic_scores.append(bic)
        aic_scores.append(aic)
        log_likelihoods.append(total_ll)
        param_counts.append(k_params)
        
        print(f" {K:<10} | {total_ll:<16.2f} | {k_params:<12} | {bic:<14.2f} | {aic:<14.2f}")
        
    print("-" * 70)
    
    # Identify optimal states (minimum BIC / AIC)
    best_bic_idx = int(np.argmin(bic_scores))
    optimal_k_bic = candidate_states[best_bic_idx]
    best_aic_idx = int(np.argmin(aic_scores))
    optimal_k_aic = candidate_states[best_aic_idx]
    
    print(f"Optimal State Selection under BIC Criterion: K = {optimal_k_bic}")
    print(f"Optimal State Selection under AIC Criterion: K = {optimal_k_aic}")
    print("=" * 70)
    
    # Generate presentation plot
    if save_plot_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_plot_path)), exist_ok=True)
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        # Normalize scores for visual comparability
        bic_norm = (np.array(bic_scores) - np.min(bic_scores)) / (np.ptp(bic_scores) + 1e-8)
        aic_norm = (np.array(aic_scores) - np.min(aic_scores)) / (np.ptp(aic_scores) + 1e-8)
        
        ax1.plot(candidate_states, bic_norm, marker='o', linewidth=2.2, color='#1f77b4', label='Normalized BIC (Primary)')
        ax1.plot(candidate_states, aic_norm, marker='s', linewidth=2.0, linestyle='--', color='#ff7f0e', label='Normalized AIC')
        ax1.axvline(x=optimal_k_bic, color='#2ca02c', linestyle=':', linewidth=2.0, label=f'Optimal K={optimal_k_bic} (Elbow/Min)')
        
        ax1.set_title('Adaptive State Selection via Bayesian Information Criterion (BIC)', fontsize=12, fontweight='bold', pad=12)
        ax1.set_xlabel('Number of Latent States (K)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Normalized Criterion Penalty', fontsize=11, fontweight='bold')
        ax1.set_xticks(candidate_states)
        ax1.legend(loc='upper right', frameon=True)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=300)
        plt.close()
        print(f"Saved state selection curve to: {save_plot_path}")
        
    return {
        'candidate_states': candidate_states,
        'bic_scores': [round(b, 2) for b in bic_scores],
        'aic_scores': [round(a, 2) for a in aic_scores],
        'log_likelihoods': [round(ll, 2) for ll in log_likelihoods],
        'param_counts': param_counts,
        'optimal_k_bic': optimal_k_bic,
        'optimal_k_aic': optimal_k_aic
    }

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mimic_path = os.path.join(base_dir, "mimic-iii-clinical-database-demo-1.4")
    raw_data, feature_names = preprocess_mimic_demo(mimic_path, seq_len=24)
    scaler = VitalsScaler()
    scaled_data = scaler.fit_transform(raw_data)
    
    plot_path = os.path.join(base_dir, "outputs", "plots", "bic_aic_state_selection.png")
    evaluate_bic_aic(scaled_data, candidate_states=[2, 3, 4, 5, 6, 8], epochs=15, save_plot_path=plot_path)
