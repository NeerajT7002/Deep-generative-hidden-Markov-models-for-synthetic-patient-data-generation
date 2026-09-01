import torch
import torch.nn as nn

class DeepDecoder(nn.Module):
    def __init__(self, latent_dim: int, num_states: int, output_dim: int, hidden_dims: list = [64, 128], feature_types: dict = None):
        """
        Deep Decoder Network g_psi: R^k x {1, ..., K} -> R^d
        
        Args:
            latent_dim (int): Dimension of latent space (k)
            num_states (int): Number of HMM states (K)
            output_dim (int): Dimension of output observation (d)
            hidden_dims (list): Dimensions of hidden layers
            feature_types (dict): Optional dictionary mapping feature indices to their types 
                                  e.g., {'continuous': [0, 1, 2], 'binary': [3, 4], 'count': [5]}
        """
        super(DeepDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_states = num_states
        self.output_dim = output_dim
        self.feature_types = feature_types
        
        # Input to decoder is concatenation of latent vector z_t and state one-hot encoding e_{s_t}
        input_dim = latent_dim + num_states
        
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim
            
        self.shared_decoder = nn.Sequential(*layers)
        
        # Reconstruction heads
        if self.feature_types is None:
            # Single linear head for general reconstruction
            self.reconstruct_head = nn.Linear(curr_dim, output_dim)
        else:
            # Mixed-type reconstruction heads
            self.heads = nn.ModuleDict()
            for f_type, indices in feature_types.items():
                self.heads[f_type] = nn.Linear(curr_dim, len(indices))

    def forward(self, z: torch.Tensor, state_onehot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (torch.Tensor): Latent representation of shape (batch_size, seq_len, latent_dim) or (batch_size, latent_dim)
            state_onehot (torch.Tensor): One-hot encoded HMM states of shape (batch_size, seq_len, num_states) or (batch_size, num_states)
            
        Returns:
            x_recon (torch.Tensor): Reconstructed observation of shape (batch_size, seq_len, output_dim) or (batch_size, output_dim)
        """
        # Concatenate z_t and e_{s_t} along the last dimension
        combined = torch.cat([z, state_onehot], dim=-1)
        
        # Save shape details
        original_shape = combined.shape
        if len(original_shape) == 3:
            combined = combined.contiguous().view(-1, original_shape[-1])
            
        shared_out = self.shared_decoder(combined)
        
        if self.feature_types is None:
            x_recon = self.reconstruct_head(shared_out)
        else:
            # Reconstruct each feature type block and combine
            x_recon_parts = {}
            for f_type, indices in self.feature_types.items():
                head_out = self.heads[f_type](shared_out)
                
                # Apply activations based on type
                if f_type == 'binary':
                    head_out = torch.sigmoid(head_out)
                elif f_type == 'count':
                    # softplus to ensure non-negative counts
                    head_out = torch.nn.functional.softplus(head_out)

                elif f_type == 'categorical':
                    # Note: We do not apply softmax directly if cross entropy loss expects logits,
                    # but for synthetic generation, we will want probabilities.
                    head_out = torch.softmax(head_out, dim=-1)
                    
                x_recon_parts[f_type] = (head_out, indices)
                
            # Place outputs back in their respective feature indices
            x_recon = torch.zeros((shared_out.shape[0], self.output_dim), device=z.device)
            for f_type, (part_val, indices) in x_recon_parts.items():
                x_recon[:, indices] = part_val
                
        # Restore sequence shape if input was 3D
        if len(original_shape) == 3:
            x_recon = x_recon.view(original_shape[0], original_shape[1], -1)
            
        return x_recon
