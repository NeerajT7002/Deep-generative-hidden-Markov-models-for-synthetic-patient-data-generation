import os
import json
import torch
import numpy as np
import pandas as pd
from src.utils.data_loader import preprocess_mimic_demo, get_dataloader, VitalsScaler
from src.models.dghmm import DGHMM
from src.evaluation.fidelity import compute_ks_score, compute_correlation_preservation
from src.evaluation.temporal import compute_autocorrelation, compute_dtw_distance
from src.evaluation.privacy import evaluate_mia_resistance, compute_min_distance, compute_k_anonymity

from src.evaluation.visualization import (
    plot_real_vs_synthetic_trajectories,
    plot_transition_matrix,
    plot_state_progression,
    plot_loss_curves,
    plot_feature_distributions
)

def main():
    print("=" * 70)
    print(" DG-HMM: Deep Generative Hidden Markov Models for Patient Generation")
    print(" First Review Demonstration Pipeline (Group 6)")
    print("=" * 70)

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

    # Model Hyperparameters (aligned with Momenzadeh & Oshaghi 2026 specifications)
    seq_len = 24       # 24-hour ICU stay window
    latent_dim = 8     # Dimensionality of latent continuous manifold z_t
    num_states = 4     # 4 Clinical States: Stable, Moderate, Guarded, Critical
    epochs = 50        # Training epochs for deep convergence
    batch_size = 16    # Mini-batch size
    learning_rate = 0.001

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
    indices = np.arange(len(scaled_data))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(scaled_data))

    train_scaled = scaled_data[indices[:split_idx]]
    test_scaled = scaled_data[indices[split_idx:]]
    train_raw = raw_data[indices[:split_idx]]
    test_raw = raw_data[indices[split_idx:]]
    print(f"Dataset Split: {len(train_scaled)} Training stays, {len(test_scaled)} Validation stays")

    train_loader = get_dataloader(train_scaled, batch_size=min(batch_size, len(train_scaled)), shuffle=True)

    # 4. Initialize DG-HMM Model Architecture
    print("\n[Step 2/5] Initializing DG-HMM Architecture...")
    model = DGHMM(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_states=num_states,
        hidden_dims_enc=[128, 64],
        hidden_dims_dec=[64, 128],
        num_mixtures=3,
        lambda_hmm=0.08,    # Balanced gradient scale
        beta_kl=0.005       # VAE regularizer as specified in paper (beta_3 in [0.001, 0.1])
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)

    # 5. Training Loop
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


    # Save trained checkpoint
    checkpoint_path = os.path.join(checkpoints_dir, "dghmm_mimic.pt")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'feature_names': feature_names,
        'latent_dim': latent_dim,
        'num_states': num_states
    }, checkpoint_path)
    print(f"Model checkpoint saved to: {checkpoint_path}")

    # 6. Generate Synthetic Clinical Patient Stays
    print("\n[Step 4/5] Generating Synthetic Electronic Health Records...")
    num_synthetic = 50
    synth_scaled_torch, synth_states_torch = model.generate(
        num_samples=num_synthetic, 
        seq_len=seq_len, 
        device=device,
        reference_data=torch.tensor(train_scaled, dtype=torch.float32, device=device)
    )


    synth_scaled = synth_scaled_torch.cpu().numpy()
    synth_states = synth_states_torch.cpu().numpy()

    # Denormalize back to physiological clinical units (BPM, mmHg, deg F, %)
    synth_denorm = scaler.inverse_transform(synth_scaled)
    print(f"Successfully generated {num_synthetic} synthetic patient stays of shape: {synth_denorm.shape}")

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

    # 7. Comprehensive Quantitative Evaluation
    print("\n[Step 5/5] Quantitative Evaluation for First Review...")
    
    # Statistical Fidelity
    ks_score = compute_ks_score(raw_data, synth_denorm)
    corr_pres = compute_correlation_preservation(raw_data, synth_denorm)
    
    # Temporal Consistency
    real_acf = compute_autocorrelation(raw_data, lag=1)
    synth_acf = compute_autocorrelation(synth_denorm, lag=1)
    dtw_dist = compute_dtw_distance(raw_data, synth_denorm, max_samples=30)
    
    # Privacy Protection
    mia_res = evaluate_mia_resistance(train_raw, test_raw, synth_denorm)
    min_dist = compute_min_distance(train_raw, synth_denorm)
    k_anon = compute_k_anonymity(synth_denorm)

    # 8. Publication-Quality Presentation Plots
    print("\nGenerating Presentation Visualizations...")
    
    # Figure 1: 24-Hour Vitals Trajectories (Real vs. Synthetic)
    plot_real_vs_synthetic_trajectories(
        raw_data, synth_denorm, feature_names,
        save_path=os.path.join(plots_dir, "real_vs_synthetic_trajectories.png")
    )

    # Figure 2: Learned HMM Transition Probability Heatmap
    A_matrix = model.hmm.get_transition_matrix().detach().cpu().numpy()
    plot_transition_matrix(
        A_matrix, state_labels=state_labels,
        save_path=os.path.join(plots_dir, "hmm_transition_matrix.png")
    )

    # Figure 3: Decoded Hidden State Timelines
    plot_state_progression(
        synth_states, state_labels=state_labels,
        save_path=os.path.join(plots_dir, "state_progression_timelines.png")
    )

    # Figure 4: Joint Optimization Loss Curves
    plot_loss_curves(
        loss_history,
        save_path=os.path.join(plots_dir, "training_loss_curves.png")
    )

    # Figure 5: Marginal Distribution Fidelity
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
            "architecture": "DG-HMM (Deep Generative Hidden Markov Model)",
            "latent_dim": latent_dim,
            "hidden_states": num_states,
            "training_epochs": epochs,
            "final_joint_loss": round(loss_history['total'][-1], 4),
            "final_recon_mse": round(loss_history['recon'][-1], 4)
        },
        "metrics": {
            "statistical_fidelity": {
                "kolmogorov_smirnov_score": round(ks_score, 4),
                "paper_benchmark_ks": 0.014,
                "correlation_preservation": round(corr_pres, 4),
                "paper_benchmark_corr": 0.942
            },
            "temporal_consistency": {
                "real_autocorrelation_lag1": round(real_acf, 4),
                "synthetic_autocorrelation_lag1": round(synth_acf, 4),
                "dynamic_time_warping_distance": round(dtw_dist, 4),
                "paper_benchmark_dtw": 0.89
            },
            "privacy_protection": {
                "mia_resistance_pct": round(mia_res * 100.0, 2),
                "paper_benchmark_mia_pct": 89.4,
                "min_distance": round(min_dist, 4),
                "paper_benchmark_min_dist": 0.42,
                "k_anonymity_score": int(k_anon),
                "paper_benchmark_k_anon": 7.8
            }
        }
    }

    metrics_json_path = os.path.join(outputs_dir, "metrics_summary.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Saved evaluation metrics summary to: {metrics_json_path}")

    # Print Formatted Evaluation Report Table with Paper Comparison
    print("\n" + "=" * 78)
    print(" QUANTITATIVE RESULTS vs. PAPER BENCHMARKS (Momenzadeh & Oshaghi 2026)")
    print("=" * 78)
    print(f" {'Metric Category':<22} | {'Evaluation Metric':<28} | {'Our DG-HMM':<10} | {'Paper (Table)':<12}")
    print("-" * 78)
    print(f" {'Fidelity':<22} | {'Kolmogorov-Smirnov (KS)':<28} | {ks_score:<10.4f} | {'0.014 (Tab 2)':<12}")
    print(f" {'Fidelity':<22} | {'Correlation Preservation':<28} | {corr_pres:<10.4f} | {'0.942 (Tab 2)':<12}")
    print(f" {'Temporal Dynamics':<22} | {'Synthetic Autocorr':<28} | {synth_acf:<10.4f} | {'0.876 (Tab 3)':<12}")
    print(f" {'Temporal Dynamics':<22} | {'DTW Distance':<28} | {dtw_dist:<10.4f} | {'0.890 (Tab 3)':<12}")
    print(f" {'Privacy':<22} | {'MIA Attack Resistance':<28} | {f'{mia_res * 100.0:.1f}%':<10} | {'89.4% (Tab 5)':<12}")
    print(f" {'Privacy':<22} | {'Minimum Record Distance':<28} | {min_dist:<10.4f} | {'0.420 (Tab 5)':<12}")
    print(f" {'Privacy':<22} | {'k-Anonymity Score':<28} | {k_anon:<10} | {'7.8 (Tab 5)':<12}")
    print("=" * 78)


    print("\nGenerated Presentation Deliverables:")
    print(f" 1. Vitals Trajectories Plot : {os.path.join(plots_dir, 'real_vs_synthetic_trajectories.png')}")
    print(f" 2. HMM Transition Heatmap   : {os.path.join(plots_dir, 'hmm_transition_matrix.png')}")
    print(f" 3. State Progression Plot   : {os.path.join(plots_dir, 'state_progression_timelines.png')}")
    print(f" 4. Loss Convergence Curve   : {os.path.join(plots_dir, 'training_loss_curves.png')}")
    print(f" 5. Distribution Curves      : {os.path.join(plots_dir, 'marginal_distributions.png')}")
    print(f" 6. Synthetic Cohort CSV     : {synth_csv_path}")
    print(f" 7. Model Weights Checkpoint : {checkpoint_path}")
    print("=" * 78)

if __name__ == "__main__":
    main()

