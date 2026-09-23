import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import skew, kurtosis

try:
    from fastdtw import fastdtw
    HAS_FASTDTW = True
except ImportError:
    HAS_FASTDTW = False

def _simple_dtw(s1: np.ndarray, s2: np.ndarray) -> float:
    """Basic Dynamic Time Warping fallback implementation."""
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],
                                          dtw_matrix[i, j - 1],
                                          dtw_matrix[i - 1, j - 1])
    return float(dtw_matrix[n, m])


def compute_autocorrelation(data: np.ndarray, lag: int = 1) -> float:
    """
    Computes autocorrelation at a given lag for all features across time.
    (Matches Table 3 ACF@10 when lag=10).
    
    Args:
        data (np.ndarray): Sequential data of shape (num_samples, seq_len, num_features)
        lag (int): Autocorrelation lag steps.
        
    Returns:
        mean_acf (float): Average autocorrelation coefficient across samples and features.
    """
    num_samples, seq_len, num_features = data.shape
    if seq_len <= lag:
        return 0.0
        
    acfs = []
    for s in range(num_samples):
        for f in range(num_features):
            series = data[s, :, f]
            mean = np.mean(series)
            var = np.var(series) + 1e-8
            
            # Sample autocovariance
            cov = np.mean((series[:-lag] - mean) * (series[lag:] - mean))
            acfs.append(cov / var)
            
    return float(np.mean(acfs))


def compute_dtw_distance(real_data: np.ndarray, synthetic_data: np.ndarray, max_samples: int = 100) -> float:
    """
    Calculates average Dynamic Time Warping (DTW) distance between sequence pairs (Table 3 in paper).
    Uses normalized per-step Euclidean distance without artificial scaling constants.
    
    Args:
        real_data: (N_real, seq_len, D)
        synthetic_data: (N_synth, seq_len, D)
        max_samples: Maximum comparison pairs for computational efficiency
        
    Returns:
        mean_dtw: Average normalized DTW distance per time step
    """
    n_real = len(real_data)
    n_synth = len(synthetic_data)
    num_comparisons = min(max_samples, n_real, n_synth)
    seq_len = real_data.shape[1]
    
    # Scale to [0, 1] normalized trajectories for scale-invariant distance
    min_v = np.min(real_data, axis=(0, 1), keepdims=True)
    max_v = np.max(real_data, axis=(0, 1), keepdims=True)
    diff = max_v - min_v
    diff[diff == 0] = 1.0
    
    r_norm = (real_data - min_v) / diff
    s_norm = (synthetic_data - min_v) / diff
    
    dtw_distances = []
    for i in range(num_comparisons):
        if HAS_FASTDTW:
            dist, _ = fastdtw(r_norm[i], s_norm[i], dist=euclidean)
        else:
            dist = _simple_dtw(r_norm[i], s_norm[i])
        # Standard DTW distance normalized by sequence length (average alignment error per step)
        dtw_distances.append(dist / float(seq_len))
        
    return float(np.mean(dtw_distances))


def compute_skewness_error(real_data: np.ndarray, synthetic_data: np.ndarray) -> float:
    """
    Computes mean absolute error in skewness across all features (Table 3: 0.067 ± 0.009).
    """
    num_features = real_data.shape[-1]
    real_flat = real_data.reshape(-1, num_features)
    synth_flat = synthetic_data.reshape(-1, num_features)
    
    sk_real = skew(real_flat, axis=0, nan_policy='omit')
    sk_synth = skew(synth_flat, axis=0, nan_policy='omit')
    
    return float(np.nanmean(np.abs(sk_real - sk_synth)))


def compute_kurtosis_error(real_data: np.ndarray, synthetic_data: np.ndarray) -> float:
    """
    Computes mean absolute error in kurtosis across all features (Table 3: 0.094 ± 0.012).
    """
    num_features = real_data.shape[-1]
    real_flat = real_data.reshape(-1, num_features)
    synth_flat = synthetic_data.reshape(-1, num_features)
    
    kt_real = kurtosis(real_flat, axis=0, nan_policy='omit')
    kt_synth = kurtosis(synth_flat, axis=0, nan_policy='omit')
    
    return float(np.nanmean(np.abs(kt_real - kt_synth)))
