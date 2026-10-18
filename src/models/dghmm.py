import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.encoder import DeepEncoder
from src.models.decoder import DeepDecoder
from src.models.hmm import DifferentiableHMM

class DGHMM(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_states: int,
                 hidden_dims_enc: list = [128, 64], hidden_dims_dec: list = [64, 128],
                 num_mixtures: int = 3, use_flows: bool = False, feature_types: dict = None,
                 lambda_hmm: float = 0.1, beta_kl: float = 0.01, bias_init: torch.Tensor = None):
        """
        Deep Generative Hidden Markov Model (DG-HMM)
        (Momenzadeh & Oshaghi 2026, BMC Medical Informatics and Decision Making)
        
        Args:
            input_dim (int): Dimension of patient observations (d)
            latent_dim (int): Dimension of latent space (k)
            num_states (int): Number of HMM states (K)
            hidden_dims_enc (list): Encoder hidden dimensions
            hidden_dims_dec (list): Decoder hidden dimensions
            num_mixtures (int): Number of components in GMM emissions
            use_flows (bool): If True, uses Normalizing Flows for state emissions (Table 6 in paper)
            feature_types (dict): Dictionary mapping feature categories to indices
            lambda_hmm (float): Loss weight for HMM negative log-likelihood
            beta_kl (float): Loss weight for KL-divergence regularization (beta_3 in paper)
            bias_init (torch.Tensor): Optional empirical logit bias initialization for decoder
        """
        super(DGHMM, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_states = num_states
        self.use_flows = use_flows
        
        # Core modules
        self.encoder = DeepEncoder(input_dim, latent_dim, hidden_dims_enc)
        self.hmm = DifferentiableHMM(num_states, latent_dim, num_mixtures, use_flows=use_flows)
        self.decoder = DeepDecoder(latent_dim, num_states, input_dim, hidden_dims_dec, feature_types, bias_init=bias_init)
        
        # Loss hyperparams
        self.lambda_hmm = lambda_hmm
        self.beta_kl = beta_kl

        # Initialize distinct centroids for HMM states to prevent state collapse
        if not use_flows:
            with torch.no_grad():
                for k in range(num_states):
                    self.hmm.gmm_means.data[k] = torch.randn(num_mixtures, latent_dim) * 0.2
                    self.hmm.gmm_means.data[k, :, k % latent_dim] += (k - (num_states - 1) / 2.0) * 1.5

    def forward(self, x: torch.Tensor, state_mode: str = 'viterbi'):
        """
        Forward pass for training.
        
        Args:
            x (torch.Tensor): Patient sequence batch of shape (batch_size, seq_len, input_dim)
            state_mode (str): 'viterbi' (discrete assignment) or 'soft' (posterior probabilities)
            
        Returns:
            x_recon (torch.Tensor): Reconstructed sequences of shape (batch_size, seq_len, input_dim)
            mu (torch.Tensor): Latent means
            logvar (torch.Tensor): Latent log-variances
            z (torch.Tensor): Latent vectors
            best_states (torch.Tensor): Decoded states
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Encode observations to latent space
        z, mu, logvar = self.encoder(x)
        
        # 2. Estimate hidden state sequences
        if state_mode == 'viterbi':
            best_states = self.hmm.viterbi(z) # (batch_size, seq_len)
            state_rep = F.one_hot(best_states, num_classes=self.num_states).float()
        elif state_mode == 'soft':
            log_emissions = self.hmm.compute_log_emissions(z)
            state_rep = F.softmax(log_emissions, dim=-1)
            best_states = torch.argmax(state_rep, dim=-1)
        else:
            raise ValueError(f"Unknown state mode: {state_mode}")
            
        # 3. Decode latent representations back to observations
        x_recon = self.decoder(z, state_rep)
        
        return x_recon, mu, logvar, z, best_states

    def compute_loss(self, x: torch.Tensor, state_mode: str = 'viterbi'):
        """
        Computes joint DG-HMM loss components:
        L_total = L_recon + 1.5 * L_corr + lambda_hmm * L_hmm + beta_kl * L_kl
        (Page 4 in Momenzadeh & Oshaghi 2026).
        """
        x_recon, mu, logvar, z, _ = self.forward(x, state_mode=state_mode)
        
        # A. Reconstruction Loss (L2 loss)
        loss_recon = F.mse_loss(x_recon, x, reduction='mean')
        
        # B. Inter-variable Correlation Consistency Loss
        x_flat = x.view(-1, self.input_dim)
        r_flat = x_recon.view(-1, self.input_dim)
        x_c = x_flat - x_flat.mean(dim=0, keepdim=True)
        r_c = r_flat - r_flat.mean(dim=0, keepdim=True)
        cov_x = (x_c.T @ x_c) / (len(x_flat) - 1 + 1e-6)
        cov_r = (r_c.T @ r_c) / (len(r_flat) - 1 + 1e-6)
        std_x = torch.sqrt(torch.diag(cov_x).clamp(min=1e-6))
        std_r = torch.sqrt(torch.diag(cov_r).clamp(min=1e-6))
        corr_x = cov_x / (std_x[:, None] * std_x[None, :] + 1e-6)
        corr_r = cov_r / (std_r[:, None] * std_r[None, :] + 1e-6)
        loss_corr = F.mse_loss(corr_r, corr_x)
        
        # C. HMM Negative Log-Likelihood Loss (Forward algorithm in log-space)
        loss_hmm = self.hmm.forward_loss(z)
        
        # D. KL-Divergence Regularization: D_KL( N(mu, std^2) || N(0, I) )
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        loss_kl = kl_div.mean()
        
        # Joint Objective
        total_loss = loss_recon + (1.5 * loss_corr) + (self.lambda_hmm * loss_hmm) + (self.beta_kl * loss_kl)
        
        return total_loss, {
            'loss_total': total_loss.item(),
            'loss_recon': loss_recon.item(),
            'loss_corr': loss_corr.item(),
            'loss_hmm': loss_hmm.item(),
            'loss_kl': loss_kl.item()
        }

    def fit_state_latent_priors(self, dataloader, device: str = 'cpu'):
        """
        Estimates the empirical latent state parameters mu_s and full covariance Sigma_s
        for each clinical state s in {0, ..., K-1} across the training set.
        (Momenzadeh & Oshaghi 2026, Page 4: B_j(z_t) = N(z_t; mu_j, Sigma_j)).
        """
        self.eval()
        state_latents = {s: [] for s in range(self.num_states)}
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                z, _, _ = self.encoder(batch)
                states = self.hmm.viterbi(z)
                for s in range(self.num_states):
                    mask = (states == s)
                    if mask.any():
                        state_latents[s].append(z[mask].cpu())
                        
        self.latent_state_means = {}
        self.latent_state_cholesky = {}
        for s in range(self.num_states):
            if len(state_latents[s]) > 0:
                all_z = torch.cat(state_latents[s], dim=0) # (N_points, latent_dim)
                mean = all_z.mean(dim=0).to(device)
                cov = torch.cov(all_z.T).to(device) + torch.eye(self.latent_dim, device=device) * 1e-4
                try:
                    L = torch.linalg.cholesky(cov)
                except Exception:
                    L = torch.diag(torch.sqrt(torch.diagonal(cov)))
                self.latent_state_means[s] = mean
                self.latent_state_cholesky[s] = L
            else:
                self.latent_state_means[s] = torch.zeros(self.latent_dim, device=device)
                self.latent_state_cholesky[s] = torch.eye(self.latent_dim, device=device)

    def _sample_latent(self, state: int, device: str = 'cpu') -> torch.Tensor:
        """Draws a latent sample z ~ B_s(z) using fitted full-covariance or HMM emission."""
        if hasattr(self, 'latent_state_means') and state in self.latent_state_means:
            mean = self.latent_state_means[state]
            L = self.latent_state_cholesky[state]
            eps = torch.randn(self.latent_dim, device=device)
            return mean + eps @ L.T
        return self.hmm.sample_latent_for_state(state, num_samples=1, device=device)

    def generate(self, num_samples: int, seq_len: int, device: str = 'cpu') -> tuple:
        """
        Generates synthetic patient sequences using the DG-HMM generative process:
        1. Sample initial state s_1 ~ pi
        2. Sample transitions s_t ~ A(s_{t-1}, :)
        3. Sample latent continuous state z_t ~ B_{s_t}
        4. Decode observations x_t = g_psi(z_t, s_t)
        
        Args:
            num_samples (int): Number of patient sequences to generate
            seq_len (int): Length of each sequence (hours)
            device (str): Execution device
            
        Returns:
            synthetic_x (torch.Tensor): Generated observations of shape (num_samples, seq_len, input_dim)
            sampled_states (torch.Tensor): Generated hidden states of shape (num_samples, seq_len)
        """
        self.eval()
        with torch.no_grad():
            pi = self.hmm.get_initial_distribution()
            A = self.hmm.get_transition_matrix()
            
            # Step 1: Sample discrete HMM state sequences
            sampled_states = torch.zeros(num_samples, seq_len, dtype=torch.long, device=device)
            sampled_states[:, 0] = torch.multinomial(pi.unsqueeze(0).expand(num_samples, -1), num_samples=1).squeeze(1)
            
            for t in range(1, seq_len):
                prev_s = sampled_states[:, t - 1]
                transition_probs = A[prev_s]
                sampled_states[:, t] = torch.multinomial(transition_probs, num_samples=1).squeeze(1)
                
            # Step 2: Sample latent representations z_t ~ B_{s_t}
            z_sampled = torch.zeros(num_samples, seq_len, self.latent_dim, device=device)
            
            for i in range(num_samples):
                for t in range(seq_len):
                    state = sampled_states[i, t].item()
                    z_curr = self._sample_latent(state, device=device)
                    
                    # Temporal persistence within steady clinical states
                    if t > 0 and state == sampled_states[i, t - 1].item():
                        rho = 0.85
                        z_sampled[i, t] = rho * z_sampled[i, t - 1] + float(np.sqrt(1.0 - rho**2)) * z_curr
                    else:
                        z_sampled[i, t] = z_curr
                        
            # Step 3: Decode latent states to reconstruct observations
            state_onehot = F.one_hot(sampled_states, num_classes=self.num_states).float()
            synthetic_x = self.decoder(z_sampled, state_onehot)
            
            return synthetic_x, sampled_states
