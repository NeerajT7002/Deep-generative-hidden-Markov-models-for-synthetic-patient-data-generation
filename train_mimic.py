import os
import json
import torch
import numpy as np
import pandas as pd
from src.utils.data_loader import preprocess_mimic_demo, get_dataloader, VitalsScaler
from src.models.dghmm import DGHMM
from src.evaluation.fidelity import compute_ks_score, compute_correlation_preservation
from src.evaluation.temporal import (
    compute_autocorrelation, 
    compute_dtw_distance,
    compute_skewness_error,
    compute_kurtosis_error
)
from src.evaluation.privacy import evaluate_mia_resistance, compute_min_distance, compute_k_anonymity
from src.evaluation.clinical_validity import (
    compute_clinical_rule_compliance,
    compute_physiological_bounds_adherence
)

from src.evaluation.visualization import (
    plot_real_vs_synthetic_trajectories,
    plot_transition_matrix,
    plot_state_progression,
    plot_loss_curves,
    plot_feature_distributions
)

def main():
    print("=" * 78)
    print(" DG-HMM: Deep Generative Hidden Markov Models for Patient Generation")
    print(" Rigorous Paper Parity Pipeline (Momenzadeh & Oshaghi, BMC Med Inform 2026)")
    print("=" * 78)

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")
    if device.type == "cuda":
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # 2. Paths and Directories
    repo_root = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(repo_root, "mimic-iii-clinical-database-demo-1.4")
    outputs_dir = os.path.join(repo_root, "outputs")
    plots_dir = os.path.join(outputs_dir, "plots")
    checkpoints_dir = os.path.join(outputs_dir, "checkpoints")
    data_dir = os.path.join(outputs_dir, "data")

    for d in [plots_dir, checkpoints_dir, data_dir]:
        os.makedirs(d, exist_ok=True)

    seq_len = 24       # 24-hour ICU stay window
    latent_dim = 8     # Dimensionality of latent continuous manifold z_t
    num_states = 4     # 4 Clinical States: Stable, Moderate, Guarded, Critical (validated via BIC)
    epochs = 100       # Training epochs for stable latent manifold convergence
    batch_size = 16    # Mini-batch size
    learning_rate = 0.002

    state_labels = [
        "State 0 (Stable)",
        "State 1 (Moderate)",
        "State 2 (Guarded)",
        "State 3 (Critical)"
    ]

    # 3. Data Preprocessing & Normalization
    print("\n[Step 1/5] Loading & Preprocessing MIMIC-III Clinical Vitals...")
    raw_data, feature_names = preprocess_mimic_demo(base_path, seq_len=seq_len, use_cache=True)
    input_dim = raw_data.shape[-1]
    print(f"Loaded {raw_data.shape[0]} ICU stays with {input_dim} vitals over {seq_len} hours.")
    print(f"Vitals Monitored: {feature_names}")

    # Scale vitals to [0, 1] range to balance gradient dynamics
    scaler = VitalsScaler(feature_range=(0.0, 1.0))
    scaled_data = scaler.fit_transform(raw_data)

    # Train / Test split
    np.random.seed(42)
    torch.manual_seed(42)
    indices = np.arange(len(scaled_data))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(scaled_data))

    train_scaled = scaled_data[indices[:split_idx]]
    test_scaled = scaled_data[indices[split_idx:]]
    train_raw = raw_data[indices[:split_idx]]
    test_raw = raw_data[indices[split_idx:]]
    print(f"Dataset Split: {len(train_scaled)} Training stays, {len(test_scaled)} Validation stays")

    train_loader = get_dataloader(train_scaled, batch_size=min(batch_size, len(train_scaled)), shuffle=True)

    # 4. Initialize DG-HMM Model Architecture with Empirical Logit Calibration
    print("\n[Step 2/5] Initializing Pure DG-HMM Architecture (with Empirical Logit Calibration)...")
    train_means = np.mean(train_scaled, axis=(0, 1))
    train_means = np.clip(train_means, 1e-4, 1.0 - 1e-4)
    init_bias = torch.tensor(np.log(train_means / (1.0 - train_means)), dtype=torch.float32)

    model = DGHMM(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_states=num_states,
        hidden_dims_enc=[128, 64],
        hidden_dims_dec=[64, 128],
        num_mixtures=3,
        use_flows=False,    # GMM latent emissions
        lambda_hmm=0.1,     # Balanced gradient scale for HMM NLL
        beta_kl=0.005,      # VAE regularizer as specified in paper (beta_3 in [0.001, 0.1])
        bias_init=init_bias # Initialize decoder output bias for accurate physiological baselines
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)

    # 5. Joint Training Loop
    print("\n[Step 3/5] Joint Model Training (Reconstruction + Latent HMM + KL)...")
    loss_history = {'total': [], 'recon': [], 'hmm': [], 'kl': []}

    model.train()
    for epoch in range(1, epochs + 1):
        batch_total, batch_recon, batch_hmm, batch_kl = [], [], [], []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            loss, details = model.compute_loss(batch, state_mode='viterbi')
            loss.backward()
            optimizer.step()

            batch_total.append(details['loss_total'])
            batch_recon.append(details['loss_recon'])
            batch_hmm.append(details['loss_hmm'])
            batch_kl.append(details['loss_kl'])

        scheduler.step()

        epoch_total = float(np.mean(batch_total))
        epoch_recon = float(np.mean(batch_recon))
        epoch_hmm = float(np.mean(batch_hmm))
        epoch_kl = float(np.mean(batch_kl))

        loss_history['total'].append(epoch_total)
        loss_history['recon'].append(epoch_recon)
        loss_history['hmm'].append(epoch_hmm)
        loss_history['kl'].append(epoch_kl)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | "
                  f"Total Loss: {epoch_total:.4f} | "
                  f"Recon MSE: {epoch_recon:.4f} | "
                  f"HMM NLL: {epoch_hmm:.4f} | "
                  f"KL Reg: {epoch_kl:.4f}")

    # Fit empirical latent distribution per state (Page 4: B_j(z_t) = N(z_t; mu_j, Sigma_j))
    model.fit_state_latent_priors(train_loader, device=device)

    # Save trained checkpoint
    checkpoint_path = os.path.join(checkpoints_dir, "dghmm_mimic.pt")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'latent_state_means': model.latent_state_means,
        'latent_state_cholesky': model.latent_state_cholesky,
        'feature_names': feature_names,
        'latent_dim': latent_dim,
        'num_states': num_states
    }, checkpoint_path)
    print(f"Model checkpoint saved to: {checkpoint_path}")

    # 6. Generate Synthetic Clinical Patient Stays (Pure DG-HMM Generative Sampling)
    print("\n[Step 4/5] Generating Synthetic Electronic Health Records (Pure Neural Decoder)...")
    num_synthetic = 60
    synth_scaled_torch, synth_states_torch = model.generate(
        num_samples=num_synthetic, 
        seq_len=seq_len, 
        device=device
    )

    synth_scaled = synth_scaled_torch.cpu().numpy()
    synth_states = synth_states_torch.cpu().numpy()

    # Denormalize back to physiological clinical units
    synth_denorm = scaler.inverse_transform(synth_scaled)
    print(f"Generated {num_synthetic} synthetic stays of shape: {synth_denorm.shape}")

    # Export generated cohort to CSV
    export_rows = []
    for p_id in range(num_synthetic):
        for h in range(seq_len):
            row = {
                'synthetic_patient_id': p_id + 1,
                'hour': h + 1,
                'hidden_state_id': int(synth_states[p_id, h]),
                'clinical_state': state_labels[int(synth_states[p_id, h])]
            }
            for f_idx, feat in enumerate(feature_names):
                row[feat] = round(float(synth_denorm[p_id, h, f_idx]), 2)
            export_rows.append(row)

    df_synth = pd.DataFrame(export_rows)
    synth_csv_path = os.path.join(data_dir, "synthetic_patient_cohort.csv")
    df_synth.to_csv(synth_csv_path, index=False)
    np.save(os.path.join(data_dir, "synthetic_states.npy"), synth_states)
    print(f"Exported synthetic patient cohort to: {synth_csv_path}")

    # 7. Comprehensive Quantitative Evaluation Across All 4 Paper Tables
    print("\n[Step 5/5] Quantitative Evaluation (Genuine Unclipped Metrics)...")
    
    # Table 2: Statistical Fidelity
    ks_score = compute_ks_score(raw_data, synth_denorm)
    corr_pres = compute_correlation_preservation(raw_data, synth_denorm)
    
    # Table 3: Temporal Consistency & Higher-Order Dynamics
    real_acf_1 = compute_autocorrelation(raw_data, lag=1)
    synth_acf_1 = compute_autocorrelation(synth_denorm, lag=1)
    lag_10 = min(10, seq_len - 1)
    real_acf_10 = compute_autocorrelation(raw_data, lag=lag_10)
    synth_acf_10 = compute_autocorrelation(synth_denorm, lag=lag_10)
    
    dtw_dist = compute_dtw_distance(raw_data, synth_denorm, max_samples=30)
    skew_err = compute_skewness_error(raw_data, synth_denorm)
    kurt_err = compute_kurtosis_error(raw_data, synth_denorm)
    
    # Table 4: Clinical Validity
    rule_compliance_pct = compute_clinical_rule_compliance(synth_denorm, feature_names)
    phys_bounds_pct = compute_physiological_bounds_adherence(synth_denorm, feature_names)
    
    # Table 5: Privacy Protection
    mia_res = evaluate_mia_resistance(train_raw, test_raw, synth_denorm)
    min_dist = compute_min_distance(train_raw, synth_denorm)
    k_anon = compute_k_anonymity(synth_denorm)

    # 8. Publication-Quality Visualizations
    print("\nGenerating Presentation Visualizations...")
    
    # Trajectories
    plot_real_vs_synthetic_trajectories(
        raw_data, synth_denorm, feature_names,
        save_path=os.path.join(plots_dir, "real_vs_synthetic_trajectories.png")
    )

    # Transition matrix
    A_matrix = model.hmm.get_transition_matrix().detach().cpu().numpy()
    plot_transition_matrix(
        A_matrix, state_labels=state_labels,
        save_path=os.path.join(plots_dir, "hmm_transition_matrix.png")
    )

    # State timelines
    plot_state_progression(
        synth_states, state_labels=state_labels,
        save_path=os.path.join(plots_dir, "state_progression_timelines.png")
    )

    # Loss convergence
    plot_loss_curves(
        loss_history,
        save_path=os.path.join(plots_dir, "training_loss_curves.png")
    )

    # Marginal distributions
    plot_feature_distributions(
        raw_data, synth_denorm, feature_names,
        save_path=os.path.join(plots_dir, "marginal_distributions.png")
    )

    # Save metrics JSON summary
    metrics_summary = {
        "dataset": "MIMIC-III Clinical Database Demo (24-hour ICU Stays)",
        "cohort_size": {
            "real_stays": int(raw_data.shape[0]),
            "synthetic_stays": int(num_synthetic),
            "sequence_length_hours": int(seq_len),
            "features_count": int(input_dim)
        },
        "model": {
            "architecture": "Pure DG-HMM (Deep Generative Hidden Markov Model)",
            "latent_dim": latent_dim,
            "hidden_states": num_states,
            "training_epochs": epochs,
            "final_joint_loss": round(loss_history['total'][-1], 4),
            "final_recon_mse": round(loss_history['recon'][-1], 4)
        },
        "metrics": {
            "table_2_statistical_fidelity": {
                "kolmogorov_smirnov_score": round(ks_score, 4),
                "paper_target_ks": 0.014,
                "correlation_preservation": round(corr_pres, 4),
                "paper_target_corr": 0.942
            },
            "table_3_temporal_consistency": {
                "synthetic_autocorrelation_acf1": round(synth_acf_1, 4),
                "synthetic_autocorrelation_acf10": round(synth_acf_10, 4),
                "paper_target_acf10": 0.876,
                "dynamic_time_warping_distance": round(dtw_dist, 4),
                "paper_target_dtw": 0.890,
                "skewness_error": round(skew_err, 4),
                "paper_target_skewness": 0.067,
                "kurtosis_error": round(kurt_err, 4),
                "paper_target_kurtosis": 0.094
            },
            "table_4_clinical_validity": {
                "clinical_rule_compliance_pct": rule_compliance_pct,
                "paper_target_rule_compliance_pct": 96.3,
                "physiological_bounds_adherence_pct": phys_bounds_pct,
                "paper_target_phys_bounds_pct": 94.8
            },
            "table_5_privacy_protection": {
                "mia_resistance_pct": round(mia_res * 100.0, 2),
                "paper_target_mia_pct": 89.4,
                "min_distance_rms": round(min_dist, 4),
                "paper_target_min_dist": 0.420,
                "k_anonymity_score": k_anon,
                "paper_target_k_anon": 7.8
            }
        }
    }

    metrics_json_path = os.path.join(outputs_dir, "metrics_summary.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Saved evaluation metrics summary to: {metrics_json_path}")

    # Print Formatted Evaluation Report Table with Paper Comparison
    print("\n" + "=" * 84)
    print(" QUANTITATIVE EVALUATION vs. PAPER BENCHMARKS (Momenzadeh & Oshaghi 2026)")
    print("=" * 84)
    print(f" {'Category':<18} | {'Evaluation Metric':<28} | {'Our DG-HMM':<14} | {'Paper Benchmark':<16}")
    print("-" * 84)
    print(f" {'Table 2 (Fidelity)':<18} | {'Kolmogorov-Smirnov (KS)':<28} | {ks_score:<14.4f} | {'0.014 ± 0.002':<16}")
    print(f" {'Table 2 (Fidelity)':<18} | {'Correlation Preservation':<28} | {corr_pres:<14.4f} | {'0.942 ± 0.012':<16}")
    print(f" {'Table 3 (Temporal)':<18} | {'Autocorrelation (ACF@1)':<28} | {synth_acf_1:<14.4f} | {'N/A':<16}")
    print(f" {'Table 3 (Temporal)':<18} | {'Autocorrelation (ACF@10)':<28} | {synth_acf_10:<14.4f} | {'0.876 ± 0.023':<16}")
    print(f" {'Table 3 (Temporal)':<18} | {'DTW Alignment Distance':<28} | {dtw_dist:<14.4f} | {'0.890 ± 0.110':<16}")
    print(f" {'Table 3 (Temporal)':<18} | {'Skewness Error':<28} | {skew_err:<14.4f} | {'0.067 ± 0.009':<16}")
    print(f" {'Table 3 (Temporal)':<18} | {'Kurtosis Error':<28} | {kurt_err:<14.4f} | {'0.094 ± 0.012':<16}")
    print(f" {'Table 4 (Clinical)':<18} | {'Clinical Rule Compliance':<28} | {f'{rule_compliance_pct:.1f}%':<14} | {'96.3 ± 1.8%':<16}")
    print(f" {'Table 4 (Clinical)':<18} | {'Physiological Bounds':<28} | {f'{phys_bounds_pct:.1f}%':<14} | {'94.8 ± 2.1%':<16}")
    print(f" {'Table 5 (Privacy)':<18} | {'MIA Attack Resistance':<28} | {f'{mia_res * 100.0:.1f}%':<14} | {'89.4 ± 2.1%':<16}")
    print(f" {'Table 5 (Privacy)':<18} | {'Min Record Distance (RMS)':<28} | {min_dist:<14.4f} | {'0.420 ± 0.030':<16}")
    print(f" {'Table 5 (Privacy)':<18} | {'k-Anonymity Score':<28} | {k_anon:<14} | {'7.8 ± 0.5':<16}")
    print("=" * 84)

    print("\nPresentation Visualizations Generated:")
    print(f" 1. Vitals Trajectories Plot : {os.path.join(plots_dir, 'real_vs_synthetic_trajectories.png')}")
    print(f" 2. HMM Transition Heatmap   : {os.path.join(plots_dir, 'hmm_transition_matrix.png')}")
    print(f" 3. State Progression Plot   : {os.path.join(plots_dir, 'state_progression_timelines.png')}")
    print(f" 4. Loss Convergence Curve   : {os.path.join(plots_dir, 'training_loss_curves.png')}")
    print(f" 5. Distribution Curves      : {os.path.join(plots_dir, 'marginal_distributions.png')}")
    print(f" 6. Synthetic Cohort CSV     : {synth_csv_path}")
    print(f" 7. Model Checkpoint         : {checkpoint_path}")
    print("=" * 84)

if __name__ == "__main__":
    main()
