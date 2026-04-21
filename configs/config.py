from dataclasses import dataclass


@dataclass
class Config:
    """
    Centralized Configuration for the PCSA-INN Architecture.
    Manages dimensionality, architectural hyperparameters, and training dynamics.
    """
    
    # ==========================================
    # 1. Dimensionality Design (Bijective Constraints)
    # ==========================================
    
    # Physical State (X)
    dim_phys: int = 8          # Target Surface Reflectance (Rrs) dimension

    # Observational Space (Y)
    dim_obs: int = 21          # Input TOA Radiance dimension

    # Latent Space (Z)
    dim_noise: int = 8         # Latent Gaussian Noise (z) dimension

    # Condition Space (C)
    dim_cond_raw: int = 8      # Raw meteorological & geometric condition dimension
    dim_cond_emb: int = 192    # Embedding dimension after FT-Transformer

    # ==========================================
    # 2. Architectural Hyperparameters
    # ==========================================
    
    # cINN Backbone (Affine Coupling Blocks)
    n_blocks: int = 12         # Number of invertible blocks
    subnet_width: int = 1024   # Hidden width for basic subnets (if applicable)
    subnet_depth: int = 4      # Hidden depth for basic subnets (if applicable)
    n_heads: int = 4           # Number of attention heads in specific blocks

    # FT-Transformer Module
    ft_depth: int = 3          # Number of Transformer Encoder layers
    ft_nhead: int = 4          # Number of attention heads
    ft_activation: str = "gelu" # Activation function inside feed-forward network

    # General Regularization
    dropout: float = 0.1       # Dropout rate for attention maps and FF layers

    # ==========================================
    # 3. Training & Optimization Controls
    # ==========================================
    lr: float = 5e-6           # Base learning rate (AdamW)
    soft_clamping: float = 1.5 # Affine scaling coefficient clamp (alpha)
    grad_clamping: float = 2.0 # Maximum gradient norm clipping
    clamping: float = 4.0      # Hard clamping for extreme numerical stability
    weight_decay: float = 1e-4 # L2 penalty (carefully tuned to avoid flow collapse)

    # ==========================================
    # 4. Automatic Dimension Validation (Properties)
    # ==========================================
    @property
    def dim_total(self) -> int:
        """
        Total bijective dimension required by the normalizing flow.
        D_total = D_obs + D_noise (e.g., 21 + 8 = 29)
        """
        return self.dim_obs + self.dim_noise

    @property
    def dim_zeros(self) -> int:
        """
        Number of zero-padding dimensions required for the physical state (X)
        to match the total bijective dimension.
        D_zeros = D_total - D_phys (e.g., 29 - 8 = 21)
        """
        return self.dim_total - self.dim_phys