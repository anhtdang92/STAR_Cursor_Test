import torch
import numpy as np
import time
from functools import wraps
import psutil
import GPUtil
from .logger import app_logger, cuda_logger, model_logger

def debug_cuda_memory():
    """Print detailed CUDA memory information"""
    if torch.cuda.is_available():
        cuda_logger.info("CUDA Memory Summary:")
        cuda_logger.info(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        cuda_logger.info(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # Get GPU utilization
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            cuda_logger.info(f"GPU {gpu.id} - {gpu.name}")
            cuda_logger.info(f"Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
            cuda_logger.info(f"GPU Utilization: {gpu.load*100}%")
    else:
        cuda_logger.warning("CUDA is not available")

def debug_system_resources():
    """Print system resource information"""
    app_logger.info("System Resource Summary:")
    app_logger.info(f"CPU Usage: {psutil.cpu_percent()}%")
    app_logger.info(f"RAM Usage: {psutil.virtual_memory().percent}%")
    app_logger.info(f"RAM Available: {psutil.virtual_memory().available / 1024**3:.2f} GB")

def debug_model_parameters(model):
    """Print detailed model parameter information"""
    model_logger.info("Model Parameter Summary:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            model_logger.info(f"{name}: {param_count:,} parameters")
    model_logger.info(f"Total trainable parameters: {total_params:,}")

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        app_logger.info(f"{func.__name__} took {execution_time:.2f} seconds to execute")
        return result
    return wrapper

class ModelDebugger:
    """Class for debugging model operations"""
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks to monitor activations"""
        def hook_fn(module, input, output):
            model_logger.info(f"Module: {module.__class__.__name__}")
            model_logger.info(f"Input shape: {[x.shape for x in input]}")
            model_logger.info(f"Output shape: {output.shape}")
            
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)
                
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    @timing_decorator
    def profile_forward_pass(self, input_tensor):
        """Profile a forward pass through the model"""
        debug_cuda_memory()
        debug_system_resources()
        
        with torch.no_grad():
            output = self.model(input_tensor)
            
        debug_cuda_memory()
        return output

def debug_gradient_flow(model):
    """Debug gradient flow in the model"""
    model_logger.info("Gradient Flow Summary:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            model_logger.info(f"{name}: gradient norm = {grad_norm:.6f}")
        else:
            model_logger.warning(f"{name}: no gradient")

def debug_batch_processing(batch, batch_idx):
    """Debug batch processing information"""
    app_logger.info(f"Processing batch {batch_idx}")
    if isinstance(batch, (torch.Tensor, np.ndarray)):
        app_logger.info(f"Batch shape: {batch.shape}")
        app_logger.info(f"Batch dtype: {batch.dtype}")
        if isinstance(batch, torch.Tensor):
            app_logger.info(f"Device: {batch.device}")
            app_logger.info(f"Requires grad: {batch.requires_grad}")
    else:
        app_logger.info(f"Batch type: {type(batch)}") 