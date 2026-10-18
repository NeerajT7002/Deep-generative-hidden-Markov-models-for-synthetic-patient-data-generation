#!/usr/bin/env python3
"""
DG-HMM Clinical Data Generation Web Application
Group 6 | Probabilistic Reasoning (Semester 5)
Reference: Momenzadeh & Oshaghi (2026), BMC Medical Informatics and Decision Making, 26:86

Flask server providing model inference, patient vital sign visualization,
and benchmark analytics against published paper results.
"""

import os
import json
import torch
import numpy as np
from flask import Flask, render_template, jsonify, request

# Project imports
from src.models.dghmm import DGHMM
from src.utils.data_loader import preprocess_mimic_demo, VitalsScaler
from src.evaluation.clinical_validity import compute_physiological_bounds_adherence, compute_clinical_rule_compliance

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(REPO_ROOT, "outputs", "checkpoints", "dghmm_mimic.pt")
METRICS_PATH = os.path.join(REPO_ROOT, "outputs", "metrics_summary.json")

# Global model state
DEVICE = 'cpu'
MODEL = None
SCALER = None
RAW_DATA = None
FEATURE_NAMES = None
STATE_NAMES = ['Stable', 'Moderate', 'Guarded', 'Critical']

def load_system():
    global MODEL, SCALER, RAW_DATA, FEATURE_NAMES
    raw_data, feature_names = preprocess_mimic_demo(
        os.path.join(REPO_ROOT, "mimic-iii-clinical-database-demo-1.4"),
        seq_len=24,
        use_cache=True
    )
    RAW_DATA = raw_data
    FEATURE_NAMES = feature_names

    scaler = VitalsScaler(feature_range=(0.0, 1.0))
    scaler.fit(raw_data)
    SCALER = scaler

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
    ).to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        if 'latent_state_means' in checkpoint:
            model.latent_state_means = checkpoint['latent_state_means']
            model.latent_state_cholesky = checkpoint['latent_state_cholesky']
        print(f"Flask App: Successfully loaded DG-HMM weights from {CHECKPOINT_PATH}")
    else:
        print("Flask App: Warning - No checkpoint found, using uncalibrated weights.")
        
    model.eval()
    MODEL = model

# Initialize on startup
load_system()

@app.route('/')
def index():
    return render_template('index.html', feature_names=FEATURE_NAMES)

@app.route('/dataset')
def dataset_page():
    """Renders the dedicated, standalone full-screen dataset window."""
    return render_template('dataset.html')

@app.route('/api/cohort_dataset', methods=['GET'])
def get_cohort_dataset():
    """Serves the generated synthetic ICU patient cohort records with optional filtering."""
    csv_path = os.path.join(REPO_ROOT, "outputs", "data", "synthetic_patient_cohort.csv")
    if not os.path.exists(csv_path):
        return jsonify({"error": "Dataset not found", "records": []}), 404
    
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    patient_id = request.args.get('patient_id')
    state_id = request.args.get('state_id')
    
    if patient_id and patient_id != 'all':
        try:
            df = df[df['synthetic_patient_id'] == int(patient_id)]
        except ValueError:
            pass
            
    if state_id and state_id != 'all':
        try:
            df = df[df['hidden_state_id'] == int(state_id)]
        except ValueError:
            pass

    # Compute Pulse Pressure and Shock Index on the fly for complete clinical view
    df['Pulse Pressure'] = (df['Systolic BP'] - df['Diastolic BP']).round(1)
    df['Shock Index'] = (df['Heart Rate'] / df['Systolic BP'].clip(lower=1.0)).round(2)
    
    records = df.to_dict(orient='records')
    
    return jsonify({
        "total_records": len(records),
        "total_patients": int(df['synthetic_patient_id'].nunique()),
        "records": records[:1000],  # safety cap for quick UI rendering
        "features": FEATURE_NAMES
    })

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    metrics_data = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics_data = json.load(f).get("metrics", {})

    t2 = metrics_data.get("table_2_statistical_fidelity", {})
    t3 = metrics_data.get("table_3_temporal_consistency", {})
    t4 = metrics_data.get("table_4_clinical_validity", {})
    t5 = metrics_data.get("table_5_privacy_protection", {})

    # Published reference paper benchmarks (Momenzadeh & Oshaghi 2026, Tables 2-5)
    benchmarks = {
        "ks_score": {"ours": t2.get("kolmogorov_smirnov_score", 0.4535), "paper": "0.014 ± 0.002", "status": "Evaluated"},
        "correlation": {"ours": t2.get("correlation_preservation", 0.5691), "paper": "0.942 ± 0.012", "status": "Evaluated"},
        "dtw": {"ours": t3.get("dynamic_time_warping_distance", 0.3335), "paper": "0.890 ± 0.110", "status": "Evaluated"},
        "rule_compliance": {"ours": t4.get("clinical_rule_compliance_pct", 100.0), "paper": "96.3 ± 1.8%", "status": "Evaluated"},
        "bounds_adherence": {"ours": t4.get("physiological_bounds_adherence_pct", 100.0), "paper": "94.8 ± 2.1%", "status": "Evaluated"},
        "mia_resistance": {"ours": t5.get("mia_resistance_pct", 51.85), "paper": "89.4 ± 2.1%", "status": "Evaluated"},
        "min_distance": {"ours": t5.get("min_distance_rms", 0.1953), "paper": "0.420 ± 0.030", "status": "Evaluated"},
        "k_anonymity": {"ours": t5.get("k_anonymity_score", 1.0), "paper": "7.8 ± 0.5", "status": "Evaluated"}
    }
    return jsonify(benchmarks)

@app.route('/api/generate_patient', methods=['POST'])
def generate_patient():
    """Generates a novel 24-hr patient stay via pure neural decoder."""
    with torch.no_grad():
        scaled_synth, states = MODEL.generate(num_samples=1, seq_len=24, device=DEVICE)
        synth_denorm = SCALER.inverse_transform(scaled_synth.cpu().numpy())[0] # (24, 6)
        states_seq = states.cpu().numpy()[0].tolist()

    records = []
    for h in range(24):
        rec = {
            "hour": h + 1,
            "state_id": states_seq[h],
            "state_name": STATE_NAMES[states_seq[h]],
            "Heart Rate": round(float(synth_denorm[h, 0]), 1),
            "Systolic BP": round(float(synth_denorm[h, 1]), 1),
            "Diastolic BP": round(float(synth_denorm[h, 2]), 1),
            "Respiratory Rate": round(float(synth_denorm[h, 3]), 1),
            "SpO2": round(float(synth_denorm[h, 4]), 1),
            "Temperature": round(float(synth_denorm[h, 5]), 2),
            "Pulse Pressure": round(float(synth_denorm[h, 1] - synth_denorm[h, 2]), 1),
            "Shock Index": round(float(synth_denorm[h, 0] / max(synth_denorm[h, 1], 1.0)), 2)
        }
        records.append(rec)

    # Verification checks
    patient_3d = synth_denorm.reshape(1, 24, -1)
    bounds_adh = compute_physiological_bounds_adherence(patient_3d, FEATURE_NAMES)
    rule_comp = compute_clinical_rule_compliance(patient_3d, FEATURE_NAMES)

    return jsonify({
        "records": records,
        "features": FEATURE_NAMES,
        "bounds_adherence": bounds_adh,
        "rule_compliance": rule_comp,
        "transition_summary": f"Markov Path: {' -> '.join([STATE_NAMES[s][:4] for s in states_seq[::4]])}"
    })

@app.route('/api/real_sample', methods=['GET'])
def get_real_sample():
    """Returns a random real ICU patient stay from MIMIC-III for side-by-side overlay."""
    idx = int(request.args.get('idx', np.random.randint(0, len(RAW_DATA))))
    stay = RAW_DATA[idx]
    records = []
    for h in range(24):
        records.append({
            "hour": h + 1,
            "Heart Rate": round(float(stay[h, 0]), 1),
            "Systolic BP": round(float(stay[h, 1]), 1),
            "Diastolic BP": round(float(stay[h, 2]), 1),
            "Respiratory Rate": round(float(stay[h, 3]), 1),
            "SpO2": round(float(stay[h, 4]), 1),
            "Temperature": round(float(stay[h, 5]), 2)
        })
    return jsonify({
        "patient_index": idx,
        "records": records,
        "features": FEATURE_NAMES
    })

if __name__ == '__main__':
    print("\nStarting DG-HMM Clinical AI Web Dashboard on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False)
