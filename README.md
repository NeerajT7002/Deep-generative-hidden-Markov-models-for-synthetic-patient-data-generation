# Deep Generative Hidden Markov Models (DG-HMM) for Synthetic Patient Data Generation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An implementation and empirical evaluation of **Deep Generative Hidden Markov Models (DG-HMM)** for generating sequential, privacy-preserving synthetic Intensive Care Unit (ICU) patient records.

---

## Academic Context & Reference Paper

- **Course**: Probabilistic Reasoning (Semester 5)
- **Team**: Group 6 (Aakash G.S, Harish Manukonda, Meesa Rakesh, Neeraj T)
- **Reference Publication**:
  > **Seyed Mohammad Momenzadeh & Nima Oshaghi (2026)**  
  > *"Deep generative hidden Markov models for synthetic patient data generation: a novel approach for medical AI research"*  
  > **Journal**: *BMC Medical Informatics and Decision Making*, Volume 26, Article 86.  
  > **DOI**: [10.1186/s12911-026-03396-2](https://doi.org/10.1186/s12911-026-03396-2)  
  > **Dataset**: PhysioNet MIMIC-III Clinical Database Demo (24-hour longitudinal ICU stays)

---

## Project Overview

Sharing patient electronic health records (EHR) across institutions is constrained by privacy regulations (such as HIPAA and GDPR). Generative models offer a potential pathway to produce synthetic data that preserves clinical characteristics without directly sharing raw patient records.

However, standard generative models face trade-offs:
1. **Standard VAEs and GANs**: Model continuous observations well, but lack explicit temporal representations of discrete health stages.
2. **Classical Discrete HMMs**: Model discrete transition dynamics well, but struggle to represent high-dimensional, multimodal, and non-linearly correlated continuous clinical measurements.

**DG-HMM** addresses this by combining:
1. **Deep Variational Autoencoder**: Compresses continuous multi-vital observations into a continuous latent space $z_t \in \mathbb{R}^d$.
2. **Differentiable Hidden Markov Model**: Models transitions across discrete clinical health states ($S_t \in \{0, 1, 2, 3\}$: Stable, Moderate, Guarded, Critical) using a vectorized, log-space Forward algorithm.
3. **Multimodal Emission Modeling**: Captures intra-state continuous distributions using Gaussian Mixture Models (GMM) and Normalizing Flows.

---

## Model Architecture

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

## Experimental Results vs. Paper Benchmarks

Our model was evaluated across all four benchmark dimensions described in **Tables 2, 3, 4, and 5 of Momenzadeh & Oshaghi (2026)**:

| Evaluation Dimension | Specific Metric | Our DG-HMM (MIMIC Demo) | Paper Benchmark (Full MIMIC-III) | Reference in Paper |
| :--- | :--- | :---: | :---: | :---: |
| **Statistical Fidelity** | **Kolmogorov-Smirnov (KS) Score** | **0.4535** | **0.014 ± 0.002** | Table 2 (Page 5) |
| **Statistical Fidelity** | **Correlation Preservation** | **0.5691** | **0.942 ± 0.012** | Table 2 (Page 5) |
| **Temporal Dynamics** | **Dynamic Time Warping (DTW)** | **0.3335** | **0.890 ± 0.110** | Table 3 (Page 6) |
| **Temporal Dynamics** | **Autocorrelation (ACF@10)** | **-0.0453** | **0.876 ± 0.023** | Table 3 (Page 6) |
| **Temporal Dynamics** | **Skewness Error** | **1.5038** | **0.067 ± 0.009** | Table 3 (Page 6) |
| **Temporal Dynamics** | **Kurtosis Error** | **6.1815** | **0.094 ± 0.012** | Table 3 (Page 6) |
| **Clinical Validity** | **Clinical Rule Compliance** | **100.0%** | **96.3 ± 1.8%** | Table 4 (Page 7) |
| **Clinical Validity** | **Physiological Bounds Adherence** | **100.0%** | **94.8 ± 2.1%** | Table 4 (Page 7) |
| **Privacy Protection** | **MIA Attack Resistance** | **51.85%** | **89.4 ± 2.1%** | Table 5 (Page 8) |
| **Privacy Protection** | **Minimum Record Distance (RMS)** | **0.1953** | **0.420 ± 0.030** | Table 5 (Page 8) |
| **Privacy Protection** | **k-Anonymity Score** | **1.0** | **7.8 ± 0.5** | Table 5 (Page 8) |

> **Cohort Scope Note**: The published paper evaluated a full cohort of **15,728 ICU patients** over $\ge 48$ hours with multi-modal inputs (vitals, labs, medications, ventilator settings) trained on an NVIDIA RTX 4060 GPU with mixed precision for 8–12 hours per dataset. This project executes on the public **MIMIC-III Demo** (75 stays, 6 continuous vitals, 24 hours), validating the probabilistic formulation, modular implementation, and evaluation pipeline on accessible hardware.

---

## Visual Deliverables

Generated visualization figures are stored in `outputs/plots/`:

| File Name | Description | Purpose |
| :--- | :--- | :--- |
| **`real_vs_synthetic_trajectories.png`** | 24-Hour Trajectory Envelopes | Compares real vs. synthetic vital means and $\pm 1\sigma$ envelopes across 6 vital signs. |
| **`hmm_transition_matrix.png`** | Transition Probability Heatmap | Displays the learned latent transition matrix $A$ across health states. |
| **`state_progression_timelines.png`** | Decoded Patient State Timelines | Shows Viterbi condition timelines over 24-hour stays. |
| **`training_loss_curves.png`** | Optimization Convergence | Multi-loss breakdown showing reconstruction error, HMM likelihood, and KL divergence. |
| **`marginal_distributions.png`** | Feature Density Histograms | Density overlays comparing real vs. synthetic physiological distributions. |
| **`bic_aic_state_selection.png`** | Model Order Selection | BIC and AIC curves across candidate hidden state numbers $K \in \{2, 3, 4, 5, 6, 8\}$. |

---

## Mathematical Formulation

### 1. Joint Optimization Objective
The model is trained end-to-end using a composite objective balancing continuous reconstruction accuracy, temporal Markov chain likelihood, and latent space regularization:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + 1.5 \mathcal{L}_{\text{corr}} + \lambda_{\text{HMM}} \mathcal{L}_{\text{HMM}} + \beta_{\text{KL}} \mathcal{L}_{\text{KL}}$$

Where:
- **Reconstruction Loss**:
  $$\mathcal{L}_{\text{recon}} = \frac{1}{T} \sum_{t=1}^T \|x_t - \tilde{x}_t\|_2^2$$
- **Correlation Consistency**:
  $$\mathcal{L}_{\text{corr}} = \|\text{Corr}(X) - \text{Corr}(\tilde{X})\|_F^2$$
- **HMM Negative Log-Likelihood**:
  $$\mathcal{L}_{\text{HMM}} = -\frac{1}{T} \log p(z_{1:T} \mid \theta_{\text{HMM}})$$
- **Kullback-Leibler Regularization**:
  $$\mathcal{L}_{\text{KL}} = \frac{1}{T} \sum_{t=1}^T D_{\text{KL}}\left(\mathcal{N}(\mu_t, \text{diag}(\sigma_t^2)) \parallel \mathcal{N}(0, I)\right)$$

### 2. Differentiable Forward Algorithm in Log-Space
To backpropagate through discrete Markov state transitions, the Forward recursion is computed in log-space using `torch.logsumexp`:

$$\log \alpha_t(j) = \text{LogSumExp}_{i=1}^K \left(\log \alpha_{t-1}(i) + \log A_{ij}\right) + \log b_j(z_t)$$

Where $b_j(z_t) = \sum_{m=1}^M w_{jm} \mathcal{N}(z_t \mid \mu_{jm}, \Sigma_{jm})$ represents the Gaussian Mixture emission probability of latent state $j$.

### 3. Autoregressive Latent Persistence
To reflect physiological momentum across consecutive hours in steady health states, latent coordinates maintain temporal correlation:

$$z_t = \rho \cdot z_{t-1} + \sqrt{1 - \rho^2} \cdot \zeta_t, \quad \zeta_t \sim \mathcal{N}(\mu_{S_t}, \Sigma_{S_t}), \quad \rho = 0.85$$

---

## Project Structure

```
Deep-generative-hidden-Markov-models-for-synthetic-patient-data-generation/
│
├── mimic-iii-clinical-database-demo-1.4/   # Preprocessed MIMIC-III ICU stays (.npz cache)
│
├── src/
│   ├── models/
│   │   ├── encoder.py                      # Deep MLP Encoder (f_phi)
│   │   ├── decoder.py                      # Deep MLP Decoder (g_psi)
│   │   ├── hmm.py                          # Differentiable Forward HMM & Viterbi Decoder
│   │   ├── dghmm.py                        # Core DG-HMM architecture & generative pipeline
│   │   └── state_selection.py              # BIC & AIC adaptive state selection
│   │
│   ├── utils/
│   │   ├── data_loader.py                  # Clinical vitals extraction, caching & VitalsScaler
│   │   └── flows.py                        # Normalizing flow / RealNVP coupling layers
│   │
│   └── evaluation/
│       ├── fidelity.py                     # KS distance & Pearson correlation preservation
│       ├── temporal.py                     # Normalized DTW distance & autocorrelation
│       ├── privacy.py                      # Distance-based MIA resistance & k-anonymity
│       ├── clinical_validity.py            # Physiological bounds & clinical rule compliance
│       └── visualization.py                # 5-figure publication visualization engine
│
├── outputs/
│   ├── checkpoints/
│   │   └── dghmm_mimic.pt                  # Trained model weights & optimizer checkpoint
│   ├── data/
│   │   ├── synthetic_patient_cohort.csv    # Synthesized 24-hr patient records
│   │   └── synthetic_states.npy            # Discrete Markov state sequences
│   ├── plots/
│   │   ├── real_vs_synthetic_trajectories.png
│   │   ├── hmm_transition_matrix.png
│   │   ├── state_progression_timelines.png
│   │   ├── training_loss_curves.png
│   │   ├── marginal_distributions.png
│   │   └── bic_aic_state_selection.png
│   └── metrics_summary.json                # Complete benchmark metrics report
│
├── templates/
│   ├── index.html                          # Main clinical dashboard UI
│   └── dataset.html                        # Dedicated synthetic dataset viewer UI
├── static/                                 # CSS & JavaScript assets
│
├── tests/
│   └── test_dghmm.py                       # Automated unit testing suite
│
├── app.py                                  # Flask interactive web dashboard
├── train_mimic.py                          # Main training and benchmark pipeline
├── run_pipeline.py                         # 3-stage master execution pipeline
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## Quickstart & Reproducibility Guide

### 1. Environment Setup
Install dependencies:
```powershell
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
Run unit tests verifying encoder, decoder, forward HMM, flows, state selection, and evaluation metrics:
```powershell
python -m unittest discover -s tests -v
```

### 3. Run the Full 3-Stage Pipeline
Execute tests, BIC/AIC state selection, and model training/evaluation:
```powershell
python run_pipeline.py
```

### 4. Launch the Interactive Web Dashboard
Start the local Flask dashboard:
```powershell
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to inspect generated patient telemetry, latent manifold projections, and the paper reproduction scorecard.

---

## Viva & Defense Guide

### Question 1: What is the core probabilistic motivation for DG-HMM?
**Answer**: DG-HMM combines deep neural representations with probabilistic graphical models. Standard deep generative models (VAEs, GANs) lack an explicit discrete state representation for distinct physiological stages. Classical discrete HMMs represent state transitions well, but cannot easily model continuous, multimodal, non-linear clinical vital distributions. DG-HMM operates an HMM over a continuous latent manifold produced by a deep autoencoder with GMM emission heads.

### Question 2: Why is the Forward algorithm implemented in log-space?
**Answer**: In discrete HMMs, multiplying probabilities over long sequences (e.g., 24 hours of measurements) causes numerical underflow. Computing the Forward recursion in log-space using `torch.logsumexp` stabilizes the likelihood computation while remaining fully differentiable, allowing exact marginal likelihood gradients to flow into both HMM parameters and neural network weights.

### Question 3: How is privacy evaluated for the generated data?
**Answer**: Privacy is evaluated using empirical metrics from Table 5 of the paper:
- **Membership Inference Attack (MIA) resistance**: An adversary evaluates distance to nearest real records to guess training set membership.
- **Minimum Record Distance (RMS)**: Measures the distance between each synthetic record and its nearest real training record to detect whether the model simply memorized real patients.
- **k-Anonymity**: Evaluates equivalence class sizes across discretized quasi-identifiers.
These metrics provide empirical privacy indicators rather than absolute formal privacy guarantees.

### Question 4: How do our experimental results relate to the published paper?
**Answer**: The reference paper trained on 15,728 patients over $\ge 48$ hours using high-end GPUs. Our implementation validates the architecture and mathematical principles on the public MIMIC-III Demo cohort (75 patients, 24 hours). While sample size limits statistical fidelity metrics (KS and correlation) compared to the full dataset, the clinical validity rules achieve 100% compliance, and temporal DTW distances remain low.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
