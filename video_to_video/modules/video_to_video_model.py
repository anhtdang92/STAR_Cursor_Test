import torch
import torch.nn as nn
from .autoencoder import AutoencoderKLTemporalDecoder
from .diffusion import GaussianDiffusion
from .utils import instantiate_from_config
import contextlib
import io
import sys
import logging

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def suppress_model_output():
    """Context manager to suppress model parameter output during loading."""
    # Store original stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # Replace with null output
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    # Temporarily disable logging
    old_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.ERROR)
    
    try:
        yield
    finally:
        # Restore original output
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logging.getLogger().setLevel(old_level)

class VideoToVideo_sr(nn.Module):
    def __init__(self, config, embed_dim=4, ckpt_path=None, ignore_keys=None):
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        
        # Initialize autoencoder with suppressed output
        with suppress_model_output():
            self.autoencoder = AutoencoderKLTemporalDecoder(
                ddconfig=config,
                embed_dim=embed_dim,
                ckpt_path=ckpt_path,
                ignore_keys=ignore_keys
            )
        
        # Initialize diffusion model with suppressed output
        with suppress_model_output():
            self.diffusion = GaussianDiffusion(
                model=self.autoencoder,
                image_size=config['resolution'],
                channels=config['in_channels'],
                timesteps=1000,
                sampling_timesteps=250,
                loss_type='l2',
                objective='pred_noise',
                beta_schedule='linear',
                p2_loss_weight_gamma=0.0,
                p2_loss_weight_k=1.0
            )
        
        # Freeze autoencoder parameters
        for param in self.autoencoder.parameters():
            param.requires_grad = False
    
    def forward(self, x, t=None, noise=None):
        """Forward pass through the model."""
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        
        if noise is None:
            noise = torch.randn_like(x)
        
        return self.diffusion(x, t, noise)
    
    def sample(self, x, t=None):
        """Sample from the model."""
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        
        return self.diffusion.sample(x, t)
    
    def encode(self, x):
        """Encode input to latent space."""
        return self.autoencoder.encode(x)
    
    def decode(self, z):
        """Decode latent space to output."""
        return self.autoencoder.decode(z)
    
    def get_last_layer(self):
        """Get the last layer of the model."""
        return self.autoencoder.decoder.conv_out.weight 