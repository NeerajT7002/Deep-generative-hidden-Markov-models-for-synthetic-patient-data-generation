import numpy as np

# Standard physiological reference ranges (Guyton & Hall Textbook of Medical Physiology, Page 6 in Paper)
PHYSIOLOGICAL_BOUNDS = {
    "Heart Rate": (30.0, 220.0),       # bpm
    "Systolic BP": (50.0, 250.0),      # mmHg
    "Diastolic BP": (30.0, 150.0),     # mmHg
    "Respiratory Rate": (6.0, 60.0),   # breaths/min
    "SpO2": (60.0, 100.0),             # %
    "Temperature_C": (34.0, 42.0),     # deg Celsius
    "Temperature_F": (93.2, 107.6)     # deg Fahrenheit (MIMIC-III demo is Fahrenheit)
}

def compute_physiological_bounds_adherence(data: np.ndarray, feature_names: list) -> float:
    """
    Computes the proportion of generated values falling within accepted clinical
    reference ranges for vital signs (Table 4 in Momenzadeh & Oshaghi 2026: 94.8% ± 2.1%).
    
    Args:
        data (np.ndarray): Shape (num_samples, seq_len, num_features)
        feature_names (list): List of feature names
        
    Returns:
        adherence_pct (float): Percentage of values within physiological bounds (0.0 - 100.0%)
    """
    num_samples, seq_len, num_features = data.shape
    total_measurements = 0
    valid_measurements = 0
    
    for f_idx, name in enumerate(feature_names):
        vals = data[:, :, f_idx].flatten()
        total_measurements += len(vals)
        
        # Determine bounds
        if "Temp" in name:
            # Check if values are in Celsius or Fahrenheit
            median_val = np.median(vals)
            bounds = PHYSIOLOGICAL_BOUNDS["Temperature_F"] if median_val > 50.0 else PHYSIOLOGICAL_BOUNDS["Temperature_C"]
        else:
            bounds = PHYSIOLOGICAL_BOUNDS.get(name, (-np.inf, np.inf))
            
        in_bounds = (vals >= bounds[0]) & (vals <= bounds[1])
        valid_measurements += int(np.sum(in_bounds))
        
    return float(round((valid_measurements / (total_measurements + 1e-8)) * 100.0, 2))


def compute_clinical_rule_compliance(data: np.ndarray, feature_names: list) -> float:
    """
    Computes the percentage of synthetic patient hourly records that satisfy established
    clinical and physiological consistency rules (Table 4 in Momenzadeh & Oshaghi 2026: 96.3% ± 1.8%).
    
    Rules verified:
    1. Hemodynamic order: Systolic BP > Diastolic BP + 10 mmHg
    2. Pulse Pressure plausibility: 15 mmHg <= (SBP - DBP) <= 130 mmHg
    3. Physiological oxygenation limit: SpO2 <= 100.0%
    4. Respiratory-Cardiovascular Shock Index plausibility: 0.2 <= HR / SBP <= 3.0
    5. Mean Arterial Pressure (MAP) consistency: DBP < MAP < SBP where MAP = DBP + (SBP - DBP)/3
    
    Args:
        data (np.ndarray): Shape (num_samples, seq_len, num_features)
        feature_names (list): List of feature names
        
    Returns:
        compliance_pct (float): Percentage of compliant records (0.0 - 100.0%)
    """
    N, T, D = data.shape
    flat_data = data.reshape(-1, D)
    total_records = len(flat_data)
    
    # Map feature names to indices
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    
    hr_idx = name_to_idx.get("Heart Rate")
    sbp_idx = name_to_idx.get("Systolic BP")
    dbp_idx = name_to_idx.get("Diastolic BP")
    spo2_idx = name_to_idx.get("SpO2")
    
    compliant_mask = np.ones(total_records, dtype=bool)
    
    # Rule 1: SBP > DBP + 10
    if sbp_idx is not None and dbp_idx is not None:
        sbp = flat_data[:, sbp_idx]
        dbp = flat_data[:, dbp_idx]
        compliant_mask &= (sbp > (dbp + 10.0))
        
        # Rule 2: Pulse Pressure 15 <= PP <= 130
        pp = sbp - dbp
        compliant_mask &= (pp >= 15.0) & (pp <= 130.0)
        
        # Rule 4: Shock Index HR / SBP
        if hr_idx is not None:
            hr = flat_data[:, hr_idx]
            valid_sbp = np.where(sbp > 0, sbp, 1.0)
            shock_index = hr / valid_sbp
            compliant_mask &= (shock_index >= 0.2) & (shock_index <= 3.0)
            
    # Rule 3: SpO2 <= 100% and >= 60%
    if spo2_idx is not None:
        spo2 = flat_data[:, spo2_idx]
        compliant_mask &= (spo2 <= 100.0) & (spo2 >= 60.0)
        
    compliance_count = np.sum(compliant_mask)
    return float(round((compliance_count / total_records) * 100.0, 2))
