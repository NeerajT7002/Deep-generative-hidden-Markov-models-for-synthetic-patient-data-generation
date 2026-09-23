import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def evaluate_mia_resistance(real_train: np.ndarray, real_test: np.ndarray, 
                            synthetic_data: np.ndarray) -> float:
    """
    Simulates a Membership Inference Attack (MIA) against the generative model (van Breugel et al. 2023).
    The attacker classifier learns from nearest-neighbor proximity features to distinguish training records from holdout test records.
    MIA Resistance is mathematically defined as: 1.0 - 2.0 * |Attacker_Accuracy - 0.5|.
    An attacker at random chance (50% accuracy) gives 100% resistance (perfect privacy).
    
    Args:
        real_train: (N_train, seq_len, D)
        real_test: (N_test, seq_len, D)
        synthetic_data: (N_synth, seq_len, D)
        
    Returns:
        mia_resistance: Score in [0.0, 1.0] (higher is better privacy preservation)
    """
    n_train = len(real_train)
    n_test = len(real_test)
    n_synth = len(synthetic_data)
    
    flat_train = real_train.reshape(n_train, -1)
    flat_test = real_test.reshape(n_test, -1)
    flat_synth = synthetic_data.reshape(n_synth, -1)
    
    # Standardize features for invariant distance metric
    mean_val = np.mean(flat_train, axis=0, keepdims=True)
    std_val = np.std(flat_train, axis=0, keepdims=True) + 1e-6
    
    flat_train_norm = (flat_train - mean_val) / std_val
    flat_test_norm = (flat_test - mean_val) / std_val
    flat_synth_norm = (flat_synth - mean_val) / std_val
    
    # Compute minimum and mean distance to synthetic cohort for each train and test record
    dist_train_min = np.min(np.linalg.norm(flat_train_norm[:, None, :] - flat_synth_norm[None, :, :], axis=-1), axis=1)
    dist_test_min = np.min(np.linalg.norm(flat_test_norm[:, None, :] - flat_synth_norm[None, :, :], axis=-1), axis=1)
    
    dist_train_mean = np.mean(np.linalg.norm(flat_train_norm[:, None, :] - flat_synth_norm[None, :, :], axis=-1), axis=1)
    dist_test_mean = np.mean(np.linalg.norm(flat_test_norm[:, None, :] - flat_synth_norm[None, :, :], axis=-1), axis=1)
    
    X_train_features = np.column_stack([dist_train_min, dist_train_mean])
    X_test_features = np.column_stack([dist_test_min, dist_test_mean])
    
    X = np.vstack([X_train_features, X_test_features])
    y = np.concatenate([np.ones(n_train), np.zeros(n_test)])
    
    # Attacker classification
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.35, random_state=42, stratify=y)
    except Exception:
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.35, random_state=42)
        
    attacker = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
    attacker.fit(X_tr, y_tr)
    
    preds = attacker.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    
    # Resistance = 1.0 - 2 * |accuracy - 0.5| (Empirical differential privacy resistance)
    advantage = max(0.0, abs(accuracy - 0.5))
    resistance = max(0.0, 1.0 - 2.0 * advantage)
    return float(round(resistance, 4))


def compute_min_distance(real_data: np.ndarray, synthetic_data: np.ndarray) -> float:
    """
    Computes average normalized minimum distance between real and synthetic patient records (Table 5 in paper: 0.420 ± 0.030).
    A non-zero minimum distance proves the model is not simply memorizing and regurgitating training records.
    
    Distance is normalized by the square root of total dimensions (RMS distance) to ensure scale invariance.
    """
    n_real = len(real_data)
    n_synth = len(synthetic_data)
    flat_real = real_data.reshape(n_real, -1)
    flat_synth = synthetic_data.reshape(n_synth, -1)
    
    # Feature range normalization to [0, 1]
    min_v = np.min(flat_real, axis=0, keepdims=True)
    max_v = np.max(flat_real, axis=0, keepdims=True)
    diff = max_v - min_v
    diff[diff == 0] = 1.0
    
    flat_real_norm = (flat_real - min_v) / diff
    flat_synth_norm = (flat_synth - min_v) / diff
    
    distances = np.min(np.linalg.norm(flat_real_norm[:, None, :] - flat_synth_norm[None, :, :], axis=-1), axis=1)
    
    # Root Mean Square normalized distance: dist / sqrt(num_dimensions)
    rms_distance = np.mean(distances) / np.sqrt(flat_real.shape[1])
    return float(round(rms_distance, 4))


def compute_k_anonymity(synthetic_data: np.ndarray, n_bins: int = 4) -> float:
    """
    Computes empirical k-anonymity score across clinical quasi-identifier trajectory profiles.
    Records are discretized into clinical bins; the score represents the average cluster size
    of identical quasi-identifier groupings in the synthetic population.
    
    Args:
        synthetic_data (np.ndarray): Shape (num_samples, seq_len, num_features)
        n_bins (int): Discretization bins for quasi-identifiers
        
    Returns:
        k_score (float): Average equivalence class size across populated clusters.
    """
    num_samples = len(synthetic_data)
    # Extract clinical trajectory summary as quasi-identifiers (e.g. mean and trend of each vital)
    means = np.mean(synthetic_data, axis=1)  # (N, D)
    trends = synthetic_data[:, -1, :] - synthetic_data[:, 0, :] # (N, D)
    quasi_identifiers = np.hstack([means, trends])
    
    # Discretize into quantile-based bins
    discretized = np.zeros_like(quasi_identifiers)
    for col in range(quasi_identifiers.shape[1]):
        vals = quasi_identifiers[:, col]
        try:
            bins = np.quantile(vals, np.linspace(0, 1, n_bins + 1))
            bins[0] -= 1e-5
            bins[-1] += 1e-5
            discretized[:, col] = np.digitize(vals, bins)
        except Exception:
            discretized[:, col] = 1
            
    _, counts = np.unique(discretized, axis=0, return_counts=True)
    # Return mean cluster size across populated equivalence classes (k-anonymity)
    k_mean = float(np.mean(counts))
    return float(round(k_mean, 2))
