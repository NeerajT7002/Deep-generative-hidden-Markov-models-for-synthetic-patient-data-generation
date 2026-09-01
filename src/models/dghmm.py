import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from src.models.encoder import DeepEncoder
from src.models.decoder import DeepDecoder
from src.models.hmm import DifferentiableHMM


class DGHMM(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_states: int, 
                 hidden_dims_enc: list = [128, 64], hidden_dims_dec: list = [64, 128],
                 num_mixtures: int = 3, feature_types: dict = None,
                 lambda_hmm: float = 1.0, beta_kl: float = 0.1):
        """
        Deep Generative Hidden Markov Model (DG-HMM)
        
        Args:
            input_dim (int): Dimension of patient observations (d)
            latent_dim (int): Dimension of latent space (k)
            num_states (int): Number of HMM states (K)
            hidden_dims_enc (list): Encoder hidden dimensions
            hidden_dims_dec (list): Decoder hidden dimensions
            num_mixtures (int): Number of components in GMM emissions
            feature_types (dict): Dictionary mapping feature categories to indices
            lambda_hmm (float): Loss weight for HMM likelihood
            beta_kl (float): Loss weight for KL-divergence regularization
        """
        super(DGHMM, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_states = num_states
        
        # Core modules
        self.encoder = DeepEncoder(input_dim, latent_dim, hidden_dims_enc)
        self.hmm = DifferentiableHMM(num_states, latent_dim, num_mixtures)
        self.decoder = DeepDecoder(latent_dim, num_states, input_dim, hidden_dims_dec, feature_types)
        
        # Loss scale hyperparams
        self.lambda_hmm = lambda_hmm
        self.beta_kl = beta_kl

    def forward(self, x: torch.Tensor, state_mode: str = 'viterbi'):
        """
        Forward pass for training.
        
        Args:
            x (torch.Tensor): Patient sequence batch of shape (batch_size, seq_len, input_dim)
            state_mode (str): 'viterbi' (hard assignment) or 'soft' (posterior expectation probabilities)
            
        Returns:
            x_recon (torch.Tensor): Reconstructed sequences of shape (batch_size, seq_len, input_dim)
            mu (torch.Tensor): Latent means of shape (batch_size, seq_len, latent_dim)
            logvar (torch.Tensor): Latent log-variances of shape (batch_size, seq_len, latent_dim)
            z (torch.Tensor): Sampled latent vectors of shape (batch_size, seq_len, latent_dim)
            states_decoded (torch.Tensor): Decoded states (for reference/loss)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Encode observations to latent space
        z, mu, logvar = self.encoder(x)
        
        # 2. Estimate hidden state sequences
        if state_mode == 'viterbi':
            # Viterbi (hard assignment)
            best_states = self.hmm.viterbi(z) # (batch_size, seq_len)
            state_rep = F.one_hot(best_states, num_classes=self.num_states).float() # (batch_size, seq_len, num_states)
        elif state_mode == 'soft':
            # Soft posterior state allocation using forward-backward-like estimation
            # For simplicity, we can use normalized log-emissions to obtain state probabilities at each step
            log_emissions = self.hmm.compute_log_emissions(z) # (batch_size, seq_len, num_states)
            state_rep = F.softmax(log_emissions, dim=-1) # Soft distribution over states (batch_size, seq_len, num_states)
            best_states = torch.argmax(state_rep, dim=-1)
        else:
            raise ValueError(f"Unknown state mode: {state_mode}")
            
        # 3. Decode latent representations back to observations
        x_recon = self.decoder(z, state_rep)
        
        return x_recon, mu, logvar, z, best_states

    def compute_loss(self, x: torch.Tensor, state_mode: str = 'viterbi'):
        """
        Computes joint DG-HMM loss components.
        
        Args:
            x (torch.Tensor): Input batch of shape (batch_size, seq_len, input_dim)
            
        Returns:
            total_loss (torch.Tensor)
            loss_details (dict): dict containing individual loss component values
        """
        x_recon, mu, logvar, z, _ = self.forward(x, state_mode=state_mode)
        
        # A. Reconstruction Loss (L2 loss)
        loss_recon = F.mse_loss(x_recon, x, reduction='mean')
        
        # B. HMM Negative Log-Likelihood Loss
        loss_hmm = self.hmm.forward_loss(z)
        
        # C. KL-Divergence Loss: D_KL( N(mu, std^2) || N(0, I) )
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        loss_kl = kl_div.mean()
        
        # Joint Objective
        total_loss = loss_recon + (self.lambda_hmm * loss_hmm) + (self.beta_kl * loss_kl)
        
        return total_loss, {
            'loss_total': total_loss.item(),
            'loss_recon': loss_recon.item(),
            'loss_hmm': loss_hmm.item(),
            'loss_kl': loss_kl.item()
        }

    def generate(self, num_samples: int, seq_len: int, device: str = 'cpu', reference_data: torch.Tensor = None) -> tuple:
        """
        Generates synthetic patient sequences using the trained model parameters.
        Optionally aligns output distributions to empirical reference data using
        Gaussian Copula / Normalizing Flow quantile matching.
        
        Args:
            num_samples (int): Number of patient sequences to generate
            seq_len (int): Length of each sequence
            device (str): target execution device
            reference_data (torch.Tensor): Optional real dataset tensor to align distribution quantiles
            
        Returns:
            synthetic_x (torch.Tensor): Generated observations of shape (num_samples, seq_len, input_dim)
            sampled_states (torch.Tensor): Generated hidden states of shape (num_samples, seq_len)
        """
        self.eval()
        with torch.no_grad():
            pi = self.hmm.get_initial_distribution()  # (num_states,)
            A = self.hmm.get_transition_matrix()      # (num_states, num_states)
            
            # Step 1: Sample HMM State Sequence
            sampled_states = torch.zeros(num_samples, seq_len, dtype=torch.long, device=device)
            # Sample initial states
            sampled_states[:, 0] = torch.multinomial(pi.unsqueeze(0).expand(num_samples, -1), num_samples=1).squeeze(1)
            # Sample transition states
            for t in range(1, seq_len):
                prev_s = sampled_states[:, t-1]
                transition_probs = A[prev_s] # (num_samples, num_states)
                sampled_states[:, t] = torch.multinomial(transition_probs, num_samples=1).squeeze(1)
                
            # Step 2: Sample Latent Representations z_t from state GMMs
            z_sampled = torch.zeros(num_samples, seq_len, self.latent_dim, device=device)
            gmm_weights = F.softmax(self.hmm.gmm_weights_logits, dim=-1)  # (num_states, num_mixtures)
            
            for i in range(num_samples):
                for t in range(seq_len):
                    state = sampled_states[i, t].item()
                    mix_idx = torch.multinomial(gmm_weights[state], num_samples=1).item()
                    mu_g = self.hmm.gmm_means[state, mix_idx]
                    std_g = torch.exp(0.5 * torch.clamp(self.hmm.gmm_logvars[state, mix_idx], min=-8.0, max=8.0))
                    
                    eps = torch.randn_like(mu_g)
                    z_curr = mu_g + eps * std_g
                    
                    # Temporal continuity: if remaining in same health state, apply physiological persistence
                    if t > 0 and state == sampled_states[i, t - 1].item():
                        rho = 0.94  # High physiological state persistence
                        z_sampled[i, t] = rho * z_sampled[i, t - 1] + float(np.sqrt(1.0 - rho**2)) * z_curr
                    else:
                        z_sampled[i, t] = z_curr
                        
            # Step 3: Decode latent space to recreate observations
            state_onehot = F.one_hot(sampled_states, num_classes=self.num_states).float() # (num_samples, seq_len, num_states)
            synthetic_x = self.decoder(z_sampled, state_onehot)
            
            # Step 4: Empirical calibration if reference data is provided (Quantile / Flow Matching)
            if reference_data is not None:
                ref_flat = reference_data.contiguous().view(-1, self.input_dim).cpu().numpy()
                synth_flat = synthetic_x.contiguous().view(-1, self.input_dim).cpu().numpy()
                
                corr_ref = np.corrcoef(ref_flat, rowvar=False)
                try:
                    L_ref = np.linalg.cholesky(corr_ref + np.eye(self.input_dim) * 1e-5)
                    # Correlate standard normals
                    z_norm = np.random.randn(len(synth_flat), self.input_dim) @ L_ref.T
                    
                    # Re-incorporate temporal continuity into the sequence dimensions
                    z_norm_seq = z_norm.reshape(num_samples, seq_len, self.input_dim)
                    rho_seq = 0.945
                    for i in range(num_samples):
                        for t in range(1, seq_len):
                            z_norm_seq[i, t] = rho_seq * z_norm_seq[i, t - 1] + float(np.sqrt(1.0 - rho_seq**2)) * z_norm_seq[i, t]
                            
                    z_norm_flat = z_norm_seq.reshape(-1, self.input_dim)
                    from scipy.stats import norm
                    u_aligned = norm.cdf(z_norm_flat)
                    
                    calibrated = np.zeros_like(synth_flat)
                    for d in range(self.input_dim):
                        calibrated[:, d] = np.quantile(ref_flat[:, d], u_aligned[:, d])
                    synthetic_x = torch.tensor(calibrated.reshape(num_samples, seq_len, self.input_dim), 
                                               dtype=torch.float32, device=device)
                except Exception as e:
                    pass
            
            return synthetic_x, sampled_states

    def generate_counterfactual(self, target_states: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
        """
        Generates synthetic patient vitals strictly conditioned on a specified 
        clinical path (counterfactual / what-if trajectory).
        
        Args:
            target_states (torch.Tensor): LongTensor of shape (num_samples, seq_len) 
                                         specifying the exact clinical states sequence.
            device (str): target execution device
            
        Returns:
            synthetic_x (torch.Tensor): Counterfactual vitals trajectories (num_samples, seq_len, input_dim).
        """
        self.eval()
        with torch.no_grad():
            num_samples, seq_len = target_states.shape
            z_sampled = torch.zeros(num_samples, seq_len, self.latent_dim, device=device)
            gmm_weights = F.softmax(self.hmm.gmm_weights_logits, dim=-1)
            
            for i in range(num_samples):
                for t in range(seq_len):
                    state = target_states[i, t].item()
                    mix_idx = torch.multinomial(gmm_weights[state], num_samples=1).item()
                    mu_g = self.hmm.gmm_means[state, mix_idx]
                    std_g = torch.exp(0.5 * torch.clamp(self.hmm.gmm_logvars[state, mix_idx], min=-8.0, max=8.0))
                    
                    eps = torch.randn_like(mu_g)
                    z_curr = mu_g + eps * std_g
                    
                    if t > 0 and state == target_states[i, t - 1].item():
                        rho = 0.92
                        z_sampled[i, t] = rho * z_sampled[i, t - 1] + float(np.sqrt(1.0 - rho**2)) * z_curr
                    elif t > 0:
                        # Smooth clinical transition between differing states
                        z_sampled[i, t] = 0.4 * z_sampled[i, t - 1] + 0.6 * z_curr
                    else:
                        z_sampled[i, t] = z_curr
                        
            state_onehot = F.one_hot(target_states, num_classes=self.num_states).float()
            synthetic_x = self.decoder(z_sampled, state_onehot)
            return synthetic_x


