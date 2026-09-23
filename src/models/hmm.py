import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.flows import NormalizingFlow

class DifferentiableHMM(nn.Module):
    def __init__(self, num_states: int, latent_dim: int, num_mixtures: int = 3, use_flows: bool = False):
        """
        Differentiable Hidden Markov Model (HMM) layer operating in the latent continuous space.
        Supports both Gaussian Mixture Model (GMM) emissions and Normalizing Flow emissions
        (Momenzadeh & Oshaghi 2026, Page 4 and Table 6: "Mixed + Normalizing Flows").
        
        Args:
            num_states (int): Number of hidden states (K)
            latent_dim (int): Dimension of latent continuous manifold (k)
            num_mixtures (int): Number of components in GMM emissions
            use_flows (bool): If True, parameterizes state emissions using Normalizing Flows
        """
        super(DifferentiableHMM, self).__init__()
        self.num_states = num_states
        self.latent_dim = latent_dim
        self.num_mixtures = num_mixtures
        self.use_flows = use_flows
        
        # Initial state logits (unnormalized, shape: (num_states,))
        self.pi_logits = nn.Parameter(torch.zeros(num_states))
        
        # Transition logits (unnormalized, shape: (num_states, num_states))
        self.transition_logits = nn.Parameter(torch.zeros(num_states, num_states))
        
        # Option A: GMM Emission parameters for each hidden state
        self.gmm_weights_logits = nn.Parameter(torch.zeros(num_states, num_mixtures))
        self.gmm_means = nn.Parameter(torch.randn(num_states, num_mixtures, latent_dim) * 0.1)
        self.gmm_logvars = nn.Parameter(torch.zeros(num_states, num_mixtures, latent_dim))
        
        # Option B: Normalizing Flows for each hidden state
        if use_flows:
            self.flows = nn.ModuleList([
                NormalizingFlow(latent_dim, num_layers=3, hidden_dim=64) 
                for _ in range(num_states)
            ])
        else:
            self.flows = None

    def get_transition_matrix(self) -> torch.Tensor:
        """Returns the normalized transition probability matrix A of shape (K, K)"""
        return F.softmax(self.transition_logits, dim=-1)

    def get_initial_distribution(self) -> torch.Tensor:
        """Returns the normalized initial state distribution pi of shape (K,)"""
        return F.softmax(self.pi_logits, dim=-1)

    def compute_log_emissions(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(z_t | s_t) for each state.
        
        Args:
            z (torch.Tensor): Latent representation of shape (batch_size, seq_len, latent_dim)
            
        Returns:
            log_emissions (torch.Tensor): log probabilities of shape (batch_size, seq_len, num_states)
        """
        batch_size, seq_len, _ = z.shape
        
        if self.use_flows and self.flows is not None:
            # Flatten to (batch_size * seq_len, latent_dim) for flow evaluation
            z_flat = z.contiguous().view(-1, self.latent_dim)
            state_log_probs = []
            for k in range(self.num_states):
                # log p(z | s=k) via Normalizing Flow change of variables
                flow_log_p = self.flows[k].log_prob(z_flat) # (batch_size * seq_len,)
                state_log_probs.append(flow_log_p.view(batch_size, seq_len, 1))
            log_emissions = torch.cat(state_log_probs, dim=-1) # (batch_size, seq_len, num_states)
            return log_emissions
            
        # GMM calculation
        # Reshape z to: (batch_size, seq_len, 1, 1, latent_dim)
        z_expanded = z.unsqueeze(2).unsqueeze(3)
        
        # Reshape GMM params: (1, 1, num_states, num_mixtures, latent_dim)
        means = self.gmm_means.unsqueeze(0).unsqueeze(0)
        clamped_logvars = torch.clamp(self.gmm_logvars, min=-8.0, max=8.0)
        logvars = clamped_logvars.unsqueeze(0).unsqueeze(0)
        
        # Calculate log normal density component-wise
        # log N(z; mu, sigma^2) = -0.5 * [ log(2*pi) + log(sigma^2) + (z - mu)^2 / sigma^2 ]
        diff = z_expanded - means
        exponent = -0.5 * (logvars + (diff ** 2) / (torch.exp(logvars) + 1e-6))
        log_2pi = 1.8378770664093453  # ln(2 * pi)
        log_gaussian = exponent - 0.5 * log_2pi
        
        # Sum over latent dimensions (diagonal covariance assumption)
        log_gaussian = log_gaussian.sum(dim=-1) # (batch_size, seq_len, num_states, num_mixtures)
        
        # Incorporate mixture weights
        gmm_weights_log = F.log_softmax(self.gmm_weights_logits, dim=-1).unsqueeze(0).unsqueeze(0)
        weighted_log_gaussian = log_gaussian + gmm_weights_log
        
        # Log-Sum-Exp over mixture components
        log_emissions = torch.logsumexp(weighted_log_gaussian, dim=-1) # (batch_size, seq_len, num_states)
        
        return log_emissions

    def forward_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the HMM negative log-likelihood loss: -log P(Z | theta) using the forward algorithm in log-space.
        
        Args:
            z (torch.Tensor): Latent sequences of shape (batch_size, seq_len, latent_dim)
            
        Returns:
            loss (torch.Tensor): Scalar negative log-likelihood divided by batch_size * seq_len
        """
        batch_size, seq_len, _ = z.shape
        log_emissions = self.compute_log_emissions(z)  # (batch_size, seq_len, num_states)
        
        log_pi = F.log_softmax(self.pi_logits, dim=-1)  # (num_states,)
        log_A = F.log_softmax(self.transition_logits, dim=-1)  # (num_states, num_states)
        
        # Initialize forward variables in log space
        log_alpha = log_pi.unsqueeze(0) + log_emissions[:, 0, :]
        
        # Forward pass over time steps
        for t in range(1, seq_len):
            combined = log_alpha.unsqueeze(2) + log_A.unsqueeze(0)
            log_alpha = torch.logsumexp(combined, dim=1) + log_emissions[:, t, :]
            
        # Log likelihood is log_sum_exp over final alpha states
        log_likelihood = torch.logsumexp(log_alpha, dim=-1) # (batch_size,)
        
        return -log_likelihood.mean() / seq_len

    def compute_sequence_log_likelihood(self, z: torch.Tensor) -> float:
        """
        Computes the total log likelihood log P(Z | theta) across all sequences for BIC/AIC calculation.
        """
        with torch.no_grad():
            batch_size, seq_len, _ = z.shape
            log_emissions = self.compute_log_emissions(z)
            log_pi = F.log_softmax(self.pi_logits, dim=-1)
            log_A = F.log_softmax(self.transition_logits, dim=-1)
            
            log_alpha = log_pi.unsqueeze(0) + log_emissions[:, 0, :]
            for t in range(1, seq_len):
                combined = log_alpha.unsqueeze(2) + log_A.unsqueeze(0)
                log_alpha = torch.logsumexp(combined, dim=1) + log_emissions[:, t, :]
            total_ll = torch.logsumexp(log_alpha, dim=-1).sum().item()
            return total_ll

    def sample_latent_for_state(self, state: int, num_samples: int = 1, device: str = 'cpu') -> torch.Tensor:
        """
        Samples latent vector z ~ B_k(z) from emission distribution of a given state k.
        """
        if self.use_flows and self.flows is not None:
            return self.flows[state].sample(num_samples, device=device)
        else:
            gmm_weights = F.softmax(self.gmm_weights_logits, dim=-1)
            mix_idx = torch.multinomial(gmm_weights[state], num_samples=num_samples, replacement=True)
            mu_g = self.gmm_means[state, mix_idx].squeeze(0)
            logvar_g = torch.clamp(self.gmm_logvars[state, mix_idx].squeeze(0), min=-8.0, max=8.0)
            std_g = torch.exp(0.5 * logvar_g)
            eps = torch.randn_like(mu_g)
            return mu_g + eps * std_g

    def viterbi(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decoding: Find the most probable sequence of hidden states for latent sequences.
        
        Args:
            z (torch.Tensor): Latent sequences of shape (batch_size, seq_len, latent_dim)
            
        Returns:
            states (torch.Tensor): Most probable hidden states of shape (batch_size, seq_len)
        """
        batch_size, seq_len, _ = z.shape
        log_emissions = self.compute_log_emissions(z)
        
        log_pi = F.log_softmax(self.pi_logits, dim=-1)
        log_A = F.log_softmax(self.transition_logits, dim=-1)
        
        v = log_pi.unsqueeze(0) + log_emissions[:, 0, :]
        pointers = []
        
        for t in range(1, seq_len):
            combined = v.unsqueeze(2) + log_A.unsqueeze(0)
            max_vals, argmax_indices = torch.max(combined, dim=1)
            v = max_vals + log_emissions[:, t, :]
            pointers.append(argmax_indices)
            
        best_states = torch.zeros(batch_size, seq_len, dtype=torch.long, device=z.device)
        best_states[:, -1] = torch.argmax(v, dim=-1)
        
        for t in range(seq_len - 2, -1, -1):
            prev_states = best_states[:, t+1]
            pointer = pointers[t]
            best_states[:, t] = pointer.gather(1, prev_states.unsqueeze(1)).squeeze(1)
            
        return best_states
