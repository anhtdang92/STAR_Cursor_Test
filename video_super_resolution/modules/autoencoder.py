import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import contextlib
import io
import builtins
import sys

logger = logging.getLogger(__name__)

# Suppress PyTorch's verbose output
def _suppress_print(*args, **kwargs):
    pass

# Store original print function
_original_print = print

@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress all output."""
    # Store original stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_print = builtins.print
    
    # Replace with null output
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    builtins.print = _suppress_print
    
    try:
        yield
    finally:
        # Restore original output
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        builtins.print = old_print

def _load_state_dict_silently(model, state_dict, strict=False):
    """Load state dict without printing parameter names."""
    with suppress_output():
        # Temporarily disable all logging during state dict loading
        old_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        
        try:
            # Create a custom state dict loader that doesn't print parameter names
            def custom_load_state_dict(model, state_dict, strict=False):
                missing_keys = []
                unexpected_keys = []
                
                # Get model state dict
                model_state_dict = model.state_dict()
                
                # Check for missing and unexpected keys
                for key in model_state_dict.keys():
                    if key not in state_dict:
                        missing_keys.append(key)
                
                for key in state_dict.keys():
                    if key not in model_state_dict:
                        unexpected_keys.append(key)
                
                # Load matching parameters silently
                for key, value in state_dict.items():
                    if key in model_state_dict:
                        model_state_dict[key].copy_(value)
                
                return missing_keys, unexpected_keys
            
            # Use custom loader instead of model.load_state_dict
            missing_keys, unexpected_keys = custom_load_state_dict(model, state_dict, strict)
            
            if len(missing_keys) > 0:
                logger.warning(f"Missing keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
            return missing_keys, unexpected_keys
        finally:
            # Restore logging level
            logging.getLogger().setLevel(old_level)

class AutoencoderKLTemporalDecoder(nn.Module):
    def __init__(
        self,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: List[int] = [16],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int = 3,
        resolution: int = 256,
        z_channels: int = 4,
        double_z: bool = True,
        use_linear_attn: bool = False,
        embed_dim: int = 4,
        attn_type: str = 'vanilla'
    ):
        super().__init__()
        self.ch = ch
        self.out_ch = out_ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.resamp_with_conv = resamp_with_conv
        self.in_channels = in_channels
        self.resolution = resolution
        self.z_channels = z_channels
        self.double_z = double_z
        self.use_linear_attn = use_linear_attn
        self.embed_dim = embed_dim
        self.attn_type = attn_type

        # Initialize model components
        self.encoder = Encoder(
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
            double_z=double_z,
            use_linear_attn=use_linear_attn,
            embed_dim=embed_dim,
            attn_type=attn_type
        )

        self.decoder = Decoder(
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
            double_z=double_z,
            use_linear_attn=use_linear_attn,
            embed_dim=embed_dim,
            attn_type=attn_type
        )

    def init_from_ckpt(self, path: str, ignore_keys: List[str] = None):
        """Initialize model weights from checkpoint with suppressed output."""
        logger.info(f"Loading checkpoint from {path}")
        try:
            with suppress_output():
                sd = torch.load(path, map_location='cpu')
                if 'state_dict' in sd:
                    sd = sd['state_dict']
                keys = list(sd.keys())
                for k in keys:
                    for ik in ignore_keys or []:
                        if k.startswith(ik):
                            logger.debug(f"Deleting key {k} from state_dict.")
                            del sd[k]
                _load_state_dict_silently(self, sd, strict=False)
                logger.info(f"Restored from {path}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z) 