# Deep Generative Hidden Markov Models (DG-HMM) for Synthetic Patient Data Generation
## Implementation & Architecture Summary

---

### Academic Metadata
- **Course**: Probabilistic Reasoning (Semester 5)
- **Team**: Group 6 (Aakash G.S, Harish Manukonda, Meesa Rakesh, Neeraj T)
- **Reference Publication**:
  > **Seyed Mohammad Momenzadeh & Nima Oshaghi (2026)**  
  > *"Deep generative hidden Markov models for synthetic patient data generation: a novel approach for medical AI research"*  
  > **Journal**: *BMC Medical Informatics and Decision Making*, Volume 26, Article 86.  
  > **DOI**: [10.1186/s12911-026-03396-2](https://doi.org/10.1186/s12911-026-03396-2)
- **Dataset**: PhysioNet MIMIC-III Clinical Database Demo (24-Hour Longitudinal ICU Stays)
- **Core Dependencies**: Python 3.10+, PyTorch 2.0+, NumPy, Pandas, Scikit-Learn, SciPy, Matplotlib, Flask

---

## 1. Problem Formulation & Motivation

In critical care medicine, machine learning models require fine-grained longitudinal Intensive Care Unit (ICU) time-series data. However, sharing electronic health records (EHR) between healthcare centers is constrained by privacy regulations (HIPAA, GDPR).

### Generative Modeling Trade-offs
1. **Classical Discrete HMMs**:
   - Model discrete health state transitions effectively.
   - Depend on simplified emission distributions (e.g., single Gaussians) that struggle to capture multimodal, non-Gaussian, and non-linearly correlated vital signs.
2. **Standard VAEs and GANs**:
   - Model high-dimensional continuous distributions effectively.
   - Lack explicit discrete temporal state dynamics, treating clinical progression as an unstructured continuous trajectory without discrete state transitions.

### The DG-HMM Approach
DG-HMM bridges both paradigms:
1. **Deep Variational Autoencoder**: Compresses high-dimensional, noisy continuous observations into a smooth continuous latent manifold $z_t \in \mathbb{R}^d$.
2. **Differentiable Hidden Markov Model**: Operates on the latent space, tracking transitions across discrete clinical health states ($S_t \in \{0, 1, 2, 3\}$: Stable, Moderate, Guarded, Critical).
3. **Multimodal Emission Distributions**: Models complex intra-state continuous distributions using Gaussian Mixture Models (GMM) or Normalizing Flows.

---

## 2. System Architecture

```
                       Longitudinal Clinical Observations
                  X = [x_1, x_2, ..., x_T] in R^(N x T x D)
                                     |
                                     v
                       +---------------------------+
                       |    Deep MLP Encoder f_phi  |
                       +-------------+-------------+
                                     |
                       Reparameterization Trick:
                       z_t = mu_t + eps * sigma_t
                                     |
                                     v
                 +---------------------------------------+
                 |       Latent Manifold z_t in R^d      |
                 +-------------------+-------------------+
                                     |
                    +----------------+----------------+
                    |                                 |
                    v                                 v
        +-----------------------+         +-----------------------+
        |  Differentiable HMM   |         |  GMM Emission Density |
        |  Forward Loss L_HMM   |         |  p(z_t | S_t = k)     |
        |  (Log-space LogSumExp)|         |  (3 Mixture Clusters) |
        +-----------+-----------+         +-----------+-----------+
                    |                                 |
                    v                                 v
        +---------------------------------------------------------+
        | Viterbi Hidden State Sequence: S = [s_1, ..., s_T]       |
        | S_t in {0: Stable, 1: Moderate, 2: Guarded, 3: Critical} |
        +---------------------------+-----------------------------+
                                    |
                    Concatenate [z_t; OneHot(S_t)]
                                    |
                                    v
                       +---------------------------+
                       |    Deep MLP Decoder g_psi  |
                       +-------------+-------------+
                                     |
                                     v
                        Reconstructed / Synthetic Vitals
                   ~X = [~x_1, ~x_2, ..., ~x_T] in R^(N x T x D)
                   (Heart Rate, SBP, DBP, RespRate, SpO2, Temp)
```

---

## 3. Modular Implementation Details

### 3.1. Deep Encoder (`src/models/encoder.py`)
- **Class**: `DeepEncoder`
- Architecture: MLP (`[128, 64]`) with Layer Normalization and LeakyReLU.
- Computes mean $\mu_t$ and log-variance $\log \sigma_t^2$.
- Reparameterization: $z_t = \mu_t + \epsilon \odot \sigma_t$ where $\epsilon \sim \mathcal{N}(0, I)$.

### 3.2. Differentiable HMM (`src/models/hmm.py`)
- **Class**: `DifferentiableHMM`
- Transition matrix $A \in \mathbb{R}^{K \times K}$ parameterized via row-wise softmax.
- Initial state prior $\pi \in \mathbb{R}^K$ parameterized via softmax.
- **Log-Space Forward Recursion**:
  $$\log \alpha_t(j) = \text{LogSumExp}_{i=1}^K \left(\log \alpha_{t-1}(i) + \log A_{ij}\right) + \log b_j(z_t)$$
  Prevents numerical underflow over longitudinal ICU time series.
- **Viterbi Decoding**: Determines the maximum a posteriori discrete state sequence in log-space.
- **Emissions**: Gaussian Mixture Models with $M=3$ components per state, or optional Normalizing Flows (`src/utils/flows.py`).

### 3.3. Deep Decoder (`src/models/decoder.py`)
- **Class**: `DeepDecoder`
- Input: Concatenation $[z_t; \text{OneHot}(S_t)] \in \mathbb{R}^{d + K}$.
- Architecture: MLP (`[64, 128]`) with Layer Normalization, LeakyReLU, and Sigmoid output activation.
- Output bias initialized to match empirical training set averages.

### 3.4. Unified Model & Sampling (`src/models/dghmm.py`)
- **Class**: `DGHMM`
- **Joint Loss Function**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + 1.5 \mathcal{L}_{\text{corr}} + \lambda_{\text{HMM}} \mathcal{L}_{\text{HMM}} + \beta_{\text{KL}} \mathcal{L}_{\text{KL}}$$
  - $\mathcal{L}_{\text{recon}}$: Reconstruction Mean Squared Error.
  - $\mathcal{L}_{\text{corr}}$: Frobenius norm loss between real and reconstructed Pearson correlation matrices.
  - $\mathcal{L}_{\text{HMM}}$: Negative log-likelihood from the Forward algorithm.
  - $\mathcal{L}_{\text{KL}}$: Analytical KL divergence against $\mathcal{N}(0, I)$.
- **Synthetic Generation Process**:
  1. Sample $s_1 \sim \pi$
  2. Sample transitions $s_t \sim A(s_{t-1}, :)$
  3. Sample latent coordinates $z_t \sim \mathcal{N}(\mu_{s_t}, \Sigma_{s_t})$ with autoregressive momentum $\rho = 0.85$
  4. Decode $x_t = g_\psi(z_t, s_t)$

### 3.5. Model Order Selection (`src/models/state_selection.py`)
- Implements BIC and AIC evaluation across candidate state counts $K \in \{2, 3, 4, 5, 6, 8\}$:
  $$\text{BIC} = -2 \log \mathcal{L} + k \log(N), \quad \text{AIC} = -2 \log \mathcal{L} + 2k$$
- Confirms $K=4$ states provides an appropriate trade-off between likelihood and parametric complexity.

### 3.6. Clinical Ingestion & Scaler (`src/utils/data_loader.py`)
- Ingests MIMIC-III Demo CSV archives (`CHARTEVENTS.csv`, `ICUSTAYS.csv`).
- Extracts 6 vitals: Heart Rate, Systolic BP, Diastolic BP, Respiratory Rate, SpO2, Temperature.
- Bins measurements into 24 one-hour intervals with forward/backward fill and median imputation.
- `VitalsScaler`: Channel-wise Min-Max normalization with inverse transform capabilities.

---

## 4. Benchmark Evaluation Results (Tables 2–5)

All metrics are evaluated directly from the generated cohort without post-processing:

| Dimension | Metric | Module | Our Model (MIMIC Demo) | Paper Benchmark (Full MIMIC-III) |
| :--- | :--- | :--- | :---: | :---: |
| **Table 2: Statistical Fidelity** | **Kolmogorov-Smirnov Score** | `fidelity.py` | **0.4535** | 0.014 ± 0.002 |
| **Table 2: Statistical Fidelity** | **Correlation Preservation** | `fidelity.py` | **0.5691** | 0.942 ± 0.012 |
| **Table 3: Temporal Dynamics** | **DTW Alignment Distance** | `temporal.py` | **0.3335** | 0.890 ± 0.110 |
| **Table 3: Temporal Dynamics** | **Autocorrelation (ACF@10)** | `temporal.py` | **-0.0453** | 0.876 ± 0.023 |
| **Table 3: Temporal Dynamics** | **Skewness Error** | `temporal.py` | **1.5038** | 0.067 ± 0.009 |
| **Table 3: Temporal Dynamics** | **Kurtosis Error** | `temporal.py` | **6.1815** | 0.094 ± 0.012 |
| **Table 4: Clinical Validity** | **Clinical Rule Compliance** | `clinical_validity.py` | **100.0%** | 96.3 ± 1.8% |
| **Table 4: Clinical Validity** | **Physiological Bounds Adherence**| `clinical_validity.py` | **100.0%** | 94.8 ± 2.1% |
| **Table 5: Privacy Protection** | **MIA Attack Resistance** | `privacy.py` | **51.85%** | 89.4 ± 2.1% |
| **Table 5: Privacy Protection** | **Minimum Record Distance (RMS)** | `privacy.py` | **0.1953** | 0.420 ± 0.030 |
| **Table 5: Privacy Protection** | **k-Anonymity Score** | `privacy.py` | **1.0** | 7.8 ± 0.5 |

### Verified Clinical Rules:
1. **Hemodynamic Order**: $\text{Systolic BP} > \text{Diastolic BP} + 10\text{ mmHg}$
2. **Pulse Pressure Range**: $15\text{ mmHg} \le (\text{SBP} - \text{DBP}) \le 130\text{ mmHg}$
3. **Oxygen Saturation Boundary**: $\text{SpO2} \le 100.0\%$
4. **Shock Index Range**: $0.2 \le \frac{\text{Heart Rate}}{\text{Systolic BP}} \le 3.0$
5. **Mean Arterial Pressure Consistency**: $\text{DBP} < \text{MAP} < \text{SBP}$

---

## 5. Web Application & Visual Deliverables

- **Flask Web Server** (`app.py`):
  - `GET /`: Interactive monitoring dashboard.
  - `GET /dataset`: Tabular view of the generated patient cohort.
  - `POST /api/generate_patient`: Pure neural decoder sampling of a 24-hr patient stay.
  - `GET /api/real_sample`: Side-by-side overlay of authentic MIMIC-III patient records.
  - `GET /api/metrics`: Live benchmark scorecard comparing model output to paper values.
- **Figures** (`outputs/plots/`):
  - `real_vs_synthetic_trajectories.png`
  - `hmm_transition_matrix.png`
  - `state_progression_timelines.png`
  - `training_loss_curves.png`
  - `marginal_distributions.png`
  - `bic_aic_state_selection.png`

---

## 6. Execution & Verification

1. **Unit Tests**:
   ```powershell
   python -m unittest discover -s tests -v
   ```
2. **Full 3-Stage Pipeline**:
   ```powershell
   python run_pipeline.py
   ```
3. **Interactive Dashboard**:
   ```powershell
   python app.py
   ```
