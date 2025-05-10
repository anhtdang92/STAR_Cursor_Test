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

def debug_cuda_memory():
    """Log CUDA memory usage in a concise format."""
    if not torch.cuda.is_available():
        return
        
    device = torch.cuda.current_device()
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**2
    
    logger.info(f"CUDA Memory: {memory_allocated:.1f}MB allocated, {memory_reserved:.1f}MB reserved of {total_memory:.1f}MB total")

def debug_system_resources():
    """Log system resource usage in a concise format."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        memory_used = torch.cuda.memory_allocated(device) / 1024**2
        memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**2
        utilization = torch.cuda.utilization(device) if hasattr(torch.cuda, 'utilization') else 0
        
        logger.info(f"GPU: {device_name} - Memory: {memory_used:.1f}MB/{memory_total:.1f}MB - Utilization: {utilization}%")

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
                    _load_state_dict_silently(self, state_dict, strict=False)
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
                        _load_state_dict_silently(self, load_dict['state_dict'], strict=False)
                    else:
                        _load_state_dict_silently(self, load_dict, strict=False)
                except Exception as e:
                    logger.error(f"Error loading model as PyTorch checkpoint: {e}")
                    raise
                    
            debug_cuda_memory()  # Log memory after loading
            logger.info("Model loaded successfully")
            
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