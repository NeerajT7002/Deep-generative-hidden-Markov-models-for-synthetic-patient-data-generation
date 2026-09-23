import os
import json
import torch
import numpy as np
import pandas as pd
from src.utils.data_loader import preprocess_mimic_demo, VitalsScaler
from src.models.dghmm import DGHMM
from src.evaluation.clinical_utility import evaluate_tstr
from src.evaluation.visualization import plot_counterfactual_scenarios

def main():
    print("=" * 80)
    print(" DG-HMM NOVEL EXTENSIONS: BEYOND THE PAPER (FINAL REVIEW INNOVATIONS)")
    print(" Group 6 | Probabilistic Reasoning (Semester 5)")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mimic_path = os.path.join(base_dir, "mimic-iii-clinical-database-demo-1.4")
    checkpoint_path = os.path.join(base_dir, "outputs", "checkpoints", "dghmm_mimic.pt")
    plots_dir = os.path.join(base_dir, "outputs", "plots")
    data_dir = os.path.join(base_dir, "outputs", "data")
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # 1. Load Data & Scaler
    print("\n[Step 1/4] Loading Preprocessed Clinical Data...")
    raw_data, feature_names = preprocess_mimic_demo(mimic_path, seq_len=24, use_cache=True)
    scaler = VitalsScaler(feature_range=(0.0, 1.0))
    scaled_data = scaler.fit_transform(raw_data)

    # Train / Test split (80/20)
    np.random.seed(42)
    indices = np.arange(len(raw_data))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(raw_data))

    train_raw = raw_data[indices[:split_idx]]
    test_raw = raw_data[indices[split_idx:]]
    train_scaled = scaled_data[indices[:split_idx]]
    test_scaled = scaled_data[indices[split_idx:]]

    # 2. Load Trained DG-HMM Model
    print("\n[Step 2/4] Loading Trained DG-HMM Checkpoint...")
    input_dim = raw_data.shape[-1]
    latent_dim = 8
    num_states = 4

    model = DGHMM(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_states=num_states,
        hidden_dims_enc=[128, 64],
        hidden_dims_dec=[64, 128],
        num_mixtures=3
    ).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        if 'latent_state_means' in checkpoint:
            model.latent_state_means = checkpoint['latent_state_means']
            model.latent_state_cholesky = checkpoint['latent_state_cholesky']
        print(f"Loaded trained DG-HMM weights from: {checkpoint_path}")
    else:
        print("Warning: No checkpoint found, using initialized weights.")

    model.eval()

    # =========================================================================
    # INNOVATION 1: CONTROLLABLE "WHAT-IF" COUNTERFACTUAL PATIENT SIMULATION
    # =========================================================================
    print("\n[Step 3/4] INNOVATION 1: Generating Controllable Counterfactual Patient Scenarios...")
    seq_len = 24
    
    # Define 3 specific clinical counterfactual pathways
    # State 0: Stable, State 1: Moderate, State 2: Guarded, State 3: Critical
    scenarios_definitions = {
        "Scenario A: Acute Septic Shock Decompensation": (
            [0] * 8 + [3] * 8 + [2] * 8,
            "Patient starts Stable (Hrs 1-8), suffers rapid Septic Shock Decompensation to Critical (Hrs 9-16), and stabilizes under Vasopressor ICU therapy (Hrs 17-24)."
        ),
        "Scenario B: Post-Surgical Trauma & Gradual Recovery": (
            [3] * 6 + [2] * 8 + [0] * 10,
            "Patient arrives in Critical condition post-op (Hrs 1-6), stabilizes to Guarded (Hrs 7-14), and recovers to Stable discharge readiness (Hrs 15-24)."
        ),
        "Scenario C: Chronic Refractory Hemodynamic Instability": (
            [1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1],
            "Chronic ICU patient displaying persistent vitals oscillation between Moderate and Guarded without acute collapse."
        )
    }

    scenarios_plot_data = {}
    counterfactual_rows = []

    for s_idx, (title, (state_seq, description)) in enumerate(scenarios_definitions.items()):
        target_states_tensor = torch.tensor([state_seq], dtype=torch.long, device=device)
        synth_vitals_scaled = model.generate_counterfactual(target_states_tensor, device=device)
        synth_vitals_denorm = scaler.inverse_transform(synth_vitals_scaled.cpu().numpy())[0] # (24, 6)

        scenarios_plot_data[title] = {
            'vitals': synth_vitals_denorm,
            'states': np.array(state_seq)
        }

        for h in range(seq_len):
            row = {
                'scenario_name': title,
                'clinical_description': description,
                'hour': h + 1,
                'hidden_state_id': state_seq[h],
                'clinical_state': ['Stable', 'Moderate', 'Guarded', 'Critical'][state_seq[h]]
            }
            for f_idx, feat in enumerate(feature_names):
                row[feat] = round(float(synth_vitals_denorm[h, f_idx]), 2)
            counterfactual_rows.append(row)

    df_counterfactual = pd.DataFrame(counterfactual_rows)
    counterfactual_csv_path = os.path.join(data_dir, "counterfactual_patient_scenarios.csv")
    df_counterfactual.to_csv(counterfactual_csv_path, index=False)
    print(f"Exported counterfactual patient scenarios to: {counterfactual_csv_path}")

    counterfactual_plot_path = os.path.join(plots_dir, "counterfactual_trajectories.png")
    plot_counterfactual_scenarios(scenarios_plot_data, feature_names, save_path=counterfactual_plot_path)

    # =========================================================================
    # INNOVATION 2: DOWNSTREAM CLINICAL UTILITY — TSTR BENCHMARK
    # =========================================================================
    print("\n[Step 4/4] INNOVATION 2: Downstream Clinical Utility Benchmark (TSTR vs. TRTR)...")
    # Load calibrated synthetic cohort from CSV or generate fresh batch
    synth_csv_path = os.path.join(data_dir, "synthetic_patient_cohort.csv")
    if os.path.exists(synth_csv_path):
        df_synth = pd.read_csv(synth_csv_path)
        n_pts = df_synth['synthetic_patient_id'].nunique()
        synthetic_denorm = df_synth[feature_names].values.reshape(n_pts, seq_len, len(feature_names))
        print(f"Loaded calibrated synthetic cohort ({n_pts} stays) from: {synth_csv_path}")
    else:
        synth_scaled_tensor, _ = model.generate(
            num_samples=60, 
            seq_len=24, 
            device=device
        )
        synthetic_denorm = scaler.inverse_transform(synth_scaled_tensor.cpu().numpy())


    tstr_plot_path = os.path.join(plots_dir, "tstr_roc_curves.png")
    tstr_results = evaluate_tstr(
        real_train=train_raw,
        real_test=test_raw,
        synthetic_cohort=synthetic_denorm,
        feature_names=feature_names,
        save_plot_path=tstr_plot_path
    )

    tstr_json_path = os.path.join(base_dir, "outputs", "tstr_clinical_utility.json")
    with open(tstr_json_path, 'w') as f:
        json.dump(tstr_results, f, indent=4)
    print(f"Saved TSTR benchmark results to: {tstr_json_path}")
    print(f"Saved TSTR ROC Curves to: {tstr_plot_path}")

    # =========================================================================
    # PRESENTATION REPORT SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print(" FINAL REVIEW NOVEL EXTENSIONS: RESULTS SUMMARY")
    print("=" * 80)
    print(f" Innovation 1: Controllable Counterfactual Simulation (Clinical Path Steering)")
    print(f"   - Scenarios Tested : Acute Septic Shock, Post-Op Trauma Recovery, Chronic Instability")
    print(f"   - Deliverables     : {counterfactual_csv_path}")
    print(f"                        {counterfactual_plot_path}")
    print("-" * 80)
    print(f" Innovation 2: Downstream Clinical Utility Benchmark (TSTR vs. TRTR)")
    print(f"   - Task Evaluated   : 24-Hour Acute Hemodynamic Instability Prediction in ICU")
    print(f"   - TRTR AUROC (Trained on Real, Tested on Real)        : {tstr_results['trtr']['auroc']:.4f}")
    print(f"   - TSTR AUROC (Trained on Synthetic, Tested on Real)   : {tstr_results['tstr']['auroc']:.4f}")
    print(f"   - Relative Clinical Decision Utility Preserved        : {tstr_results['relative_utility_pct']:.2f}%")
    print(f"   - Deliverables     : {tstr_plot_path}")
    print(f"                        {tstr_json_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
