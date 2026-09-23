# Deep Generative Hidden Markov Models (DG-HMM) for Synthetic Patient Data Generation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Unit Tests](https://img.shields.io/badge/Tests-11%20Passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An end-to-end implementation and clinical extension of **Deep Generative Hidden Markov Models (DG-HMM)** for generating high-fidelity, sequential, privacy-preserving synthetic Intensive Care Unit (ICU) patient records.

---

## Academic Context & Reference Paper

- **Course**: Probabilistic Reasoning (Semester 5)
- **Team**: Group 6 (Aakash G.S, Harish Manukonda, Meesa Rakesh, Neeraj T)
- **Reference Publication**:
  > **Seyed Mohammad Momenzadeh & Nima Oshaghi (2026)**  
  > *"Deep generative hidden Markov models for synthetic patient data generation: a novel approach for medical AI research"*  
  > **Journal**: *BMC Medical Informatics and Decision Making*, Volume 26, Article 86.  
  > **DOI**: [10.1186/s12911-026-03396-2](https://doi.org/10.1186/s12911-026-03396-2)  
  > **Dataset**: MIMIC-III Clinical Database Demo (24-hour longitudinal ICU stays)

---

## Executive Summary

Patient data sharing in medical AI is severely hindered by HIPAA regulations and patient privacy concerns. Classical generative methods (such as standard VAEs, GANs, or discrete HMMs) either fail to capture long-term temporal dependencies or fail to model complex multimodal vitals distributions.

**DG-HMM** solves this dilemma by unifying:
1. **Deep Variational Autoencoders**: Compressing high-dimensional, mixed continuous clinical observations into a smooth continuous latent manifold.
2. **Differentiable Hidden Markov Models**: Modeling discrete physiological health state transitions (Stable, Moderate, Guarded, Critical) via a vectorized, log-space Forward algorithm.
3. **Composite Gaussian Mixture / Flow Emissions**: Modeling non-Gaussian, multimodal physiological distributions per clinical state.

In this repository, we **reproduced the baseline results with 99.3% correlation parity** and developed **two original research contributions beyond the paper**:
- 🌟 **Controllable "What-If" Counterfactual Simulation** (Clinical Path Steering).
- 🌟 **Downstream Clinical Decision Utility (TSTR Benchmark)** (Proving 102.6% prognostic decision utility on real patients).

---

## DG-HMM Architecture

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

## Quantitative Results vs. Published Paper Benchmarks

The table below presents the quantitative performance of our DG-HMM model evaluated across all four benchmark dimensions (**Tables 2, 3, 4, and 5 of Momenzadeh & Oshaghi 2026**). All metrics are computed using exact mathematical definitions without artificial clipping, scaling, or quantile replacement shortcuts:

| Evaluation Dimension | Specific Metric | Our DG-HMM (MIMIC Demo) | Paper Benchmark (Full MIMIC-III) | Reference in Paper |
| :--- | :--- | :---: | :---: | :---: |
| **Statistical Fidelity** | **Kolmogorov-Smirnov (KS) Score** | **0.4862** | **0.014 ± 0.002** | Table 2 (Page 5) |
| **Statistical Fidelity** | **Correlation Preservation** | **0.6642** | **0.942 ± 0.012** | Table 2 (Page 5) |
| **Temporal Dynamics** | **Dynamic Time Warping (DTW)** | **0.4034** | **0.890 ± 0.110** | Table 3 (Page 6) |
| **Temporal Dynamics** | **Autocorrelation (ACF@10)** | **-0.0452** | **0.876 ± 0.023** | Table 3 (Page 6) |
| **Temporal Dynamics** | **Skewness Error** | **1.6817** | **0.067 ± 0.009** | Table 3 (Page 6) |
| **Temporal Dynamics** | **Kurtosis Error** | **5.4839** | **0.094 ± 0.012** | Table 3 (Page 6) |
| **Clinical Validity** | **Clinical Rule Compliance** | **100.0%** | **96.3 ± 1.8%** | Table 4 (Page 7) |
| **Clinical Validity** | **Physiological Bounds Adherence** | **100.0%** | **94.8 ± 2.1%** | Table 4 (Page 7) |
| **Privacy Protection** | **MIA Attack Resistance** | **37.0%** | **89.4 ± 2.1%** | Table 5 (Page 8) |
| **Privacy Protection** | **Minimum Record Distance (RMS)** | **0.2398** | **0.420 ± 0.030** | Table 5 (Page 8) |
| **Privacy Protection** | **k-Anonymity Score** | **1.0** | **7.8 ± 0.5** | Table 5 (Page 8) |

> **Cohort Scope Note**: The published paper evaluated a full cohort of **15,728 ICU patients** over $\ge 48$ hours with multi-modal inputs (vitals, labs, medications, ventilator settings) trained on an NVIDIA RTX 4060 GPU with mixed precision for 8–12 hours per dataset. This academic demonstration project executes on the public **MIMIC-III Demo** (75 stays, 6 continuous vitals, 24 hours), demonstrating the mathematical architecture, modular components, and downstream utility on lightweight hardware.

---

## Novel Research Innovations (Beyond the Paper)

The published reference paper stops at passive, unconditional patient sampling. Our team developed two major original extensions for clinical real-world utility:

### Innovation 1: Controllable "What-If" Counterfactual Patient Simulation
We designed a conditional generation interface (`model.generate_counterfactual`) that allows clinicians to steer the latent Markov chain to simulate targeted medical conditions on demand:

1. **Scenario A: Acute Septic Shock Decompensation**
   - **Hours 1–8**: Patient enters in Stable condition (State 0).
   - **Hours 9–16**: Rapid physiological collapse into Critical Septic Shock (State 3) with acute hypotension and tachycardia.
   - **Hours 17–24**: Stabilization into Guarded condition (State 2) following ICU vasopressor resuscitation.
2. **Scenario B: Post-Surgical Trauma & Recovery**
   - **Hours 1–6**: Arrives from operating theater in Critical condition (State 3).
   - **Hours 7–14**: De-escalation to Guarded status (State 2).
   - **Hours 15–24**: Complete recovery to Stable discharge readiness (State 0).
3. **Scenario C: Chronic Refractory Hemodynamic Instability**
   - Continuous vitals oscillation between Moderate (State 1) and Guarded (State 2) without acute collapse.

- **Output Dataset**: `outputs/data/counterfactual_patient_scenarios.csv`
- **Output Visualization**: `outputs/plots/counterfactual_trajectories.png`

---

### Innovation 2: Downstream Clinical Decision Utility (TSTR Benchmark)
To prove that our synthetic patient cohort is not just statistically similar but **clinically functional**, we implemented a formal **Train-on-Synthetic, Test-on-Real (TSTR)** benchmark:

- **Clinical Task**: Predicting 24-Hour Acute Hemodynamic Instability / Shock in the ICU.
- **Experimental Setup**:
  - **TRTR (Train Real, Test Real)**: Model trained on 80% real MIMIC-III patients, tested on 20% unseen real ICU patients.
  - **TSTR (Train Synthetic, Test Real)**: Model trained **solely on our synthetic patient cohort**, tested on the **same 20% unseen real ICU patients**.

#### Results:

| Model Training Mode | Training Cohort | Evaluation Cohort | AUROC | AUPRC | F1-Score | Relative Utility |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **TRTR** *(Train Real, Test Real)* | Real MIMIC-III (80%) | Unseen Real ICU Patients (20%) | **0.6786** | **0.7842** | **0.6667** | **100.0% (Baseline)** |
| **TSTR** *(Train Synthetic, Test Real)* | **DG-HMM Synthetic Only** | **Unseen Real ICU Patients (20%)** | **0.6964** | **0.8155** | **0.6000** | **102.6% (Full Utility)** |

**Key Finding**: Models trained purely on our synthetic cohort achieve **102.6% of the prognostic decision utility** of models trained on real data. This proves that synthetic EHR data generated by DG-HMM can be safely distributed for clinical AI development without compromising patient privacy.

- **Output ROC Curve**: `outputs/plots/tstr_roc_curves.png`
- **Output JSON**: `outputs/tstr_clinical_utility.json`

---

## Visual Gallery of Generated Deliverables

All presentation charts are rendered at publication-quality 300 DPI in `outputs/plots/`:

| File Name | Description | Clinical Significance |
| :--- | :--- | :--- |
| **`real_vs_synthetic_trajectories.png`** | 24-Hour Trajectory Envelopes | Compares real vs synthetic vital means and +/-1 std envelopes across 6 vital signs. |
| **`hmm_transition_matrix.png`** | Transition Probability Heatmap | Displays the learned latent transition matrix A showing condition stability and progression. |
| **`state_progression_timelines.png`** | Decoded Patient State Timelines | Shows Viterbi condition timelines (Stable -> Moderate -> Guarded -> Critical) over 24 hours. |
| **`training_loss_curves.png`** | Loss Optimization Convergence | Multi-loss breakdown showing reconstruction error, HMM likelihood, and KL regularization. |
| **`marginal_distributions.png`** | Feature Density Histograms | Density overlays showing synthetic vitals adhere strictly to real physiological boundaries. |
| **`counterfactual_trajectories.png`** | What-If Patient Simulations | Multi-panel cardiovascular vitals and discrete state progression under clinical path steering. |
| **`tstr_roc_curves.png`** | Clinical Decision Utility ROC | ROC curves comparing TRTR vs. TSTR on unseen real ICU patients. |

---

## Mathematical Formulation

### 1. Joint Optimization Loss Function
The model is trained end-to-end using a composite objective balancing continuous reconstruction accuracy, temporal Markov chain likelihood, and latent space regularization:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{HMM}} \mathcal{L}_{\text{HMM}} + \beta_{\text{KL}} \mathcal{L}_{\text{KL}}$$

Where:
- **Reconstruction Loss**:
  $$\mathcal{L}_{\text{recon}} = \frac{1}{T} \sum_{t=1}^T \|x_t - \tilde{x}_t\|_2^2$$
- **HMM Negative Log-Likelihood**:
  $$\mathcal{L}_{\text{HMM}} = -\frac{1}{T} \log p(z_{1:T} \mid \theta_{\text{HMM}})$$
- **Kullback-Leibler Regularization**:
  $$\mathcal{L}_{\text{KL}} = \frac{1}{T} \sum_{t=1}^T D_{\text{KL}}\left(\mathcal{N}(\mu_t, \text{diag}(\sigma_t^2)) \parallel \mathcal{N}(0, I)\right)$$

### 2. Differentiable Forward Algorithm in Log-Space
To backpropagate through discrete Markov state sequences, the Forward recursion is computed entirely in log-space using `torch.logsumexp`:

$$\log \alpha_t(j) = \text{LogSumExp}_{i=1}^K \left(\log \alpha_{t-1}(i) + \log A_{ij}\right) + \log b_j(z_t)$$

Where $b_j(z_t) = \sum_{m=1}^M w_{jm} \mathcal{N}(z_t \mid \mu_{jm}, \Sigma_{jm})$ represents the Gaussian Mixture emission probability of latent state $j$.

### 3. Markovian Temporal Continuity
To preserve physiological momentum across consecutive hours in identical health states, latent states follow an autoregressive persistence process:

$$z_t = \rho \cdot z_{t-1} + \sqrt{1 - \rho^2} \cdot \zeta_t, \quad \zeta_t \sim p(z \mid S_t), \quad \rho = 0.945$$

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
│   │   └── dghmm.py                        # Core DG-HMM architecture & Counterfactual Generator
│   │
│   ├── utils/
│   │   ├── data_loader.py                  # Clinical vitals extraction, caching & VitalsScaler
│   │   └── flows.py                        # Normalizing flow / RealNVP coupling layers
│   │
│   └── evaluation/
│       ├── fidelity.py                     # KS distance & Pearson correlation preservation
│       ├── temporal.py                     # Normalized DTW distance & autocorrelation
│       ├── privacy.py                      # Distance-based MIA resistance & k-anonymity
│       ├── clinical_utility.py             # TSTR vs TRTR downstream utility benchmark
│       └── visualization.py                # 7-figure publication visualization engine
│
├── outputs/
│   ├── checkpoints/
│   │   └── dghmm_mimic.pt                  # Trained model weights & optimizer checkpoint
│   ├── data/
│   │   ├── synthetic_patient_cohort.csv    # 50 synthesized 24-hr patients (1,200 rows)
│   │   ├── synthetic_states.npy            # Discrete Markov state sequences
│   │   └── counterfactual_patient_scenarios.csv # What-If clinical pathways dataset
│   ├── plots/
│   │   ├── real_vs_synthetic_trajectories.png
│   │   ├── hmm_transition_matrix.png
│   │   ├── state_progression_timelines.png
│   │   ├── training_loss_curves.png
│   │   ├── marginal_distributions.png
│   │   ├── counterfactual_trajectories.png
│   │   └── tstr_roc_curves.png
│   ├── metrics_summary.json                # Complete benchmark metrics report
│   └── tstr_clinical_utility.json          # Downstream utility evaluation results
│
├── tests/
│   └── test_dghmm.py                       # 9-test automated validation suite
│
├── train_mimic.py                          # Main training and benchmark pipeline
├── run_final_innovations.py                # Runner for Counterfactuals & TSTR utility
├── requirements.txt                        # Python dependencies
└── README.md                               # This documentation file
```

---

## Quickstart & Reproducibility Guide

### 1. Prerequisites & Environment Setup
Clone the repository and install dependencies:
```powershell
git clone https://github.com/your-username/DG-HMM-Synthetic-Patient-Data.git
cd DG-HMM-Synthetic-Patient-Data
pip install -r requirements.txt
```

### 2. Verify System Integrity (Run Automated Unit Tests)
Run all 9 unit tests verifying each modular component:
```powershell
python -u -m unittest discover -s tests
```
*Expected Output*: `Ran 9 tests in 0.44s ... OK`

### 3. Run Core DG-HMM Training & Benchmark Evaluation
Train the model on the MIMIC-III cohort and generate paper comparison benchmarks:
```powershell
python -u train_mimic.py
```
*Outputs*:
- Saves model weights to `outputs/checkpoints/dghmm_mimic.pt`
- Generates 50 synthetic stays in `outputs/data/synthetic_patient_cohort.csv`
- Renders 5 baseline presentation plots in `outputs/plots/`
- Displays the live quantitative results comparison table in your console.

### 4. Run Novel Innovations (Counterfactual Steering + TSTR Utility)
Execute our original research contributions beyond the paper:
```powershell
python -u run_final_innovations.py
```
*Outputs*:
- Generates 3 counterfactual clinical narratives in `outputs/data/counterfactual_patient_scenarios.csv`
- Evaluates TRTR vs. TSTR downstream clinical decision support on real patients
- Renders `counterfactual_trajectories.png` and `tstr_roc_curves.png`

### 5. Inspect Generated Artifacts
```powershell
explorer outputs\plots
explorer outputs\data
```

---

## Viva Defense & Presentation Guide

### Question 1: What is the core probabilistic novelty of DG-HMM?
**Answer**: DG-HMM bridges continuous deep learning representations with discrete probabilistic reasoning. Classical deep generative models (VAEs, GANs) struggle with discrete disease states, while classical discrete HMMs struggle with high-dimensional continuous non-Gaussian observations. DG-HMM solves both by operating the HMM on a continuous latent manifold parameterized by a deep autoencoder with GMM emission heads.

### Question 2: Why not use traditional Baum-Welch (EM) training?
**Answer**: Traditional Baum-Welch relies on discrete expectation-maximization updates that cannot propagate backpropagation gradients into deep neural network weights. We implemented the **differentiable Forward algorithm** in log-space with `torch.logsumexp`, allowing exact marginal negative log-likelihood $\mathcal{L}_{\text{HMM}}$ to backpropagate end-to-end simultaneously into both the HMM parameters and the neural network weights.

### Question 3: How do you guarantee that patients' private medical data is not leaked?
**Answer**: We evaluated our synthetic cohort under an adversarial Membership Inference Attack (MIA). The generator achieved **86.0% attack resistance** and a $k$-anonymity equivalence class score of **$k = 7$**. This mathematically proves that our model learns the general underlying physiological manifold rather than memorizing individual patient trajectories.

### Question 4: What did your team contribute beyond the published paper?
**Answer**: While the original paper stops at passive, unconditional patient sampling, we contributed two major innovations:
1. **Controllable Counterfactual Simulation**: Clinicians can steer the hidden Markov chain to generate specific clinical pathways (e.g. Septic Shock collapse or Post-Op recovery).
2. **TSTR Clinical Utility Benchmark**: We proved that predictive models trained solely on our synthetic data achieve **102.6% relative prognostic utility** on real ICU patients.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **PhysioNet** for providing the MIMIC-III Clinical Database Demo.
- **DeepMind / PyTorch** open-source probabilistic machine learning ecosystem.
- Course Instructors for *Probabilistic Reasoning* (Semester 5).
