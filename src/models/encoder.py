import torch
import torch.nn as nn

class DeepEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list = [128, 64], dropout_rate: float = 0.3):
        """
        Deep Encoder Network f_phi: R^d -> R^k
        
        Args:
            input_dim (int): Dimension of patient observations (d)
            latent_dim (int): Dimension of latent space (k)
            hidden_dims (list): Dimensions of hidden layers
            dropout_rate (float): Dropout probability for regularization
        """
        super(DeepEncoder, self).__init__()
        
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Latent parameters for VAE-style reparameterization
        self.fc_mu = nn.Linear(curr_dim, latent_dim)
        self.fc_logvar = nn.Linear(curr_dim, latent_dim)
        
        # Post-sampling regularization (keeps all parameters active in computation graph)
        self.regularize_latent = True
        self.batch_norm_latent = nn.BatchNorm1d(latent_dim)
        self.dropout_latent = nn.Dropout(dropout_rate)


    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input batch of shape (batch_size, input_dim) or (batch_size * seq_len, input_dim)
            
        Returns:
            z (torch.Tensor): Sampled latent vector (batch_size, seq_len, latent_dim)
            mu (torch.Tensor): Mean of latent distribution
            logvar (torch.Tensor): Log-variance of latent distribution
        """
        # Ensure 2D tensor for BatchNorm1d compatibility in feature extractor
        original_shape = x.shape
        if len(original_shape) == 3:
            # (batch_size, seq_len, input_dim) -> (batch_size * seq_len, input_dim)
            x = x.contiguous().view(-1, original_shape[-1])
            
        features = self.feature_extractor(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        
        # Sample latent variable
        z = self.reparameterize(mu, logvar)
        
        if self.regularize_latent:
            z = self.batch_norm_latent(z)
            z = self.dropout_latent(z)
        
        # Restore sequence shape if input was 3D
        if len(original_shape) == 3:
            z = z.view(original_shape[0], original_shape[1], -1)
            mu = mu.view(original_shape[0], original_shape[1], -1)
            logvar = logvar.view(original_shape[0], original_shape[1], -1)
            
        return z, mu, logvar

