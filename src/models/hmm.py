import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableHMM(nn.Module):
    def __init__(self, num_states: int, latent_dim: int, num_mixtures: int = 3):
        """
        Differentiable Hidden Markov Model (HMM) layer operating in the latent space.
        
        Args:
            num_states (int): Number of hidden states (K)
            latent_dim (int): Dimension of latent space (k)
            num_mixtures (int): Number of components in Gaussian Mixture Model (GMM) emissions
        """
        super(DifferentiableHMM, self).__init__()
        self.num_states = num_states
        self.latent_dim = latent_dim
        self.num_mixtures = num_mixtures
        
        # Initial state logits (unnormalized, shape: (num_states,))
        self.pi_logits = nn.Parameter(torch.zeros(num_states))
        
        # Transition logits (unnormalized, shape: (num_states, num_states))
        self.transition_logits = nn.Parameter(torch.zeros(num_states, num_states))
        
        # GMM Emission parameters for each hidden state
        # For each state, we have mixture weights, means, and log-covariances (diagonal)
        self.gmm_weights_logits = nn.Parameter(torch.zeros(num_states, num_mixtures))
        self.gmm_means = nn.Parameter(torch.randn(num_states, num_mixtures, latent_dim) * 0.1)
        self.gmm_logvars = nn.Parameter(torch.zeros(num_states, num_mixtures, latent_dim))

    def get_transition_matrix(self) -> torch.Tensor:
        """Returns the normalized transition probability matrix (K, K)"""
        return F.softmax(self.transition_logits, dim=-1)

    def get_initial_distribution(self) -> torch.Tensor:
        """Returns the normalized initial state distribution (K,)"""
        return F.softmax(self.pi_logits, dim=-1)

    def compute_log_emissions(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(z_t | s_t) using GMM for each state.
        
        Args:
            z (torch.Tensor): Latent representation of shape (batch_size, seq_len, latent_dim)
            
        Returns:
            log_emissions (torch.Tensor): log probabilities of shape (batch_size, seq_len, num_states)
        """
        batch_size, seq_len, _ = z.shape
        
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
        # log (w_k * N_k) = log(w_k) + log N_k
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
        # alpha_1(i) = log pi_i + log b_i(z_1)
        # Shape: (batch_size, num_states)
        log_alpha = log_pi.unsqueeze(0) + log_emissions[:, 0, :]
        
        # Forward pass over time steps
        for t in range(1, seq_len):
            # log_alpha_t(j) = log_sum_exp_i ( log_alpha_{t-1}(i) + log_A(i, j) ) + log_emissions_t(j)
            # log_alpha shape: (batch_size, num_states, 1)
            # log_A shape: (1, num_states, num_states)
            combined = log_alpha.unsqueeze(2) + log_A.unsqueeze(0)
            log_alpha = torch.logsumexp(combined, dim=1) + log_emissions[:, t, :]
            
        # Log likelihood is log_sum_exp over the final alpha states for each batch element
        log_likelihood = torch.logsumexp(log_alpha, dim=-1) # (batch_size,)
        
        return -log_likelihood.mean() / seq_len

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
        
        # viterbi log probs: (batch_size, num_states)
        v = log_pi.unsqueeze(0) + log_emissions[:, 0, :]
        
        # Pointer table to reconstruct path
        pointers = []
        
        for t in range(1, seq_len):
            # v_t(j) = max_i ( v_{t-1}(i) + log_A(i, j) ) + log_emissions_t(j)
            combined = v.unsqueeze(2) + log_A.unsqueeze(0)  # (batch_size, num_states, num_states)
            max_vals, argmax_indices = torch.max(combined, dim=1)
            v = max_vals + log_emissions[:, t, :]
            pointers.append(argmax_indices)
            
        # Traceback
        best_states = torch.zeros(batch_size, seq_len, dtype=torch.long, device=z.device)
        best_states[:, -1] = torch.argmax(v, dim=-1)
        
        for t in range(seq_len - 2, -1, -1):
            prev_states = best_states[:, t+1]
            pointer = pointers[t]
            best_states[:, t] = pointer.gather(1, prev_states.unsqueeze(1)).squeeze(1)
            
        return best_states
