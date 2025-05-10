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

class VideoToVideo_sr(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        
        # Log system and CUDA information
        debug_system_resources()
        debug_cuda_memory()
        
        # Initialize autoencoder
        autoencoder_config = {
            'ch': 128,
            'out_ch': 3,
            'ch_mult': (1, 2, 4, 8),
            'num_res_blocks': 2,
            'attn_resolutions': [16],
            'dropout': 0.0,
            'resamp_with_conv': True,
            'in_channels': 3,
            'resolution': 256,
            'z_channels': 4,
            'double_z': True,
            'use_linear_attn': False,
            'attn_type': 'vanilla',
            'loss': {
                'target': 'video_to_video.modules.losses.LPIPSWithDiscriminator',
                'params': {
                    'disc_start': 50001,
                    'kl_weight': 0.000001,
                    'disc_weight': 0.5
                }
            }
        }
        
        logger.info("Initializing autoencoder...")
        self.autoencoder = AutoencoderKLTemporalDecoder(
            ddconfig=autoencoder_config,
            embed_dim=4,
            ckpt_path=None
        )
        
        # Initialize diffusion model
        logger.info("Initializing diffusion model...")
        self.diffusion = GaussianDiffusion(
            denoise_fn=self.autoencoder,
            schedule=noise_schedule(timesteps=1000),
            timesteps=1000,
            loss_type='l2'
        )
        
        # Load model weights with better error handling and debugging
        try:
            logger.info(f"Loading model weights from {cfg.model_path}")
            debug_cuda_memory()  # Log memory before loading
            
            # First try loading as safetensors
            if cfg.model_path.endswith('.safetensors'):
                with suppress_output():
                    state_dict = load_file(cfg.model_path)
                    self.load_state_dict(state_dict, strict=False)
            else:
                # Try loading as PyTorch checkpoint
                try:
                    # First try with weights_only=True
                    try:
                        with suppress_output():
                            load_dict = torch.load(cfg.model_path, map_location='cpu', weights_only=True)
                    except Exception as e:
                        logger.warning(f"Failed to load with weights_only=True, trying without: {e}")
                        with suppress_output():
                            load_dict = torch.load(cfg.model_path, map_location='cpu')
                    
                    if isinstance(load_dict, dict) and 'state_dict' in load_dict:
                        with suppress_output():
                            self.load_state_dict(load_dict['state_dict'], strict=False)
                    else:
                        with suppress_output():
                            self.load_state_dict(load_dict, strict=False)
                except Exception as e:
                    logger.error(f"Error loading model as PyTorch checkpoint: {e}")
                    raise
                    
            debug_cuda_memory()  # Log memory after loading
            debug_model_parameters(self)  # Log model parameters
            
        except Exception as e:
            logger.error(f"Failed to load model from {cfg.model_path}: {e}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
        
        # Initialize model debugger
        self.debugger = ModelDebugger(self) 

    def forward(self, x):
        return self.diffusion(x)

    def test(self, x, steps=None, guide_scale=None, max_chunk_len=None):
        with torch.no_grad():
            return self.diffusion.sample(
                x,
                steps=steps,
                guide_scale=guide_scale,
                max_chunk_len=max_chunk_len
            ) 