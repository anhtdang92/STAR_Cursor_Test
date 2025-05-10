import torch
import numpy as np
import time
from functools import wraps
import psutil
import GPUtil
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
import gc
from .logger import app_logger, cuda_logger, model_logger

@contextmanager
def cuda_memory_tracking():
    """Context manager for tracking CUDA memory usage"""
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

def debug_cuda_memory() -> Dict[str, float]:
    """Print detailed CUDA memory information with proper error handling"""
    try:
        if not torch.cuda.is_available():
            cuda_logger.warning("CUDA is not available")
            return {}
            
        cuda_logger.info("CUDA Memory Summary:")
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        cuda_logger.info(f"Allocated: {allocated:.2f} MB")
        cuda_logger.info(f"Cached: {cached:.2f} MB")
        
        # Get GPU utilization
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            cuda_logger.info(f"GPU {gpu.id} - {gpu.name}")
            cuda_logger.info(f"Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
            cuda_logger.info(f"GPU Utilization: {gpu.load*100}%")
            
        return {
            'allocated': allocated,
            'cached': cached,
            'gpus': [{
                'id': gpu.id,
                'name': gpu.name,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'utilization': gpu.load
            } for gpu in gpus]
        }
    except Exception as e:
        cuda_logger.error(f"Error getting CUDA memory info: {str(e)}")
        return {}

def debug_system_resources() -> Dict[str, float]:
    """Print system resource information with proper error handling"""
    try:
        app_logger.info("System Resource Summary:")
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        app_logger.info(f"CPU Usage: {cpu_percent}%")
        app_logger.info(f"RAM Usage: {memory.percent}%")
        app_logger.info(f"RAM Available: {memory.available / 1024**3:.2f} GB")
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / 1024**3
        }
    except Exception as e:
        app_logger.error(f"Error getting system resource info: {str(e)}")
        return {}

def debug_model_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Print detailed model parameter information with proper error handling"""
    try:
        model_logger.info("Model Parameter Summary:")
        total_params = 0
        param_counts = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                total_params += param_count
                param_counts[name] = param_count
                model_logger.info(f"{name}: {param_count:,} parameters")
                
        model_logger.info(f"Total trainable parameters: {total_params:,}")
        return {'total_params': total_params, 'param_counts': param_counts}
    except Exception as e:
        model_logger.error(f"Error getting model parameter info: {str(e)}")
        return {'total_params': 0, 'param_counts': {}}

def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time with proper error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            app_logger.info(f"{func.__name__} took {execution_time:.2f} seconds to execute")
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            app_logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds with error: {str(e)}")
            raise
    return wrapper

class ModelDebugger:
    """Class for debugging model operations with proper resource management"""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.hooks = []
        
    def register_hooks(self) -> None:
        """Register forward hooks to monitor activations with proper error handling"""
        try:
            def hook_fn(module, input, output):
                try:
                    model_logger.info(f"Module: {module.__class__.__name__}")
                    model_logger.info(f"Input shape: {[x.shape for x in input]}")
                    model_logger.info(f"Output shape: {output.shape}")
                except Exception as e:
                    model_logger.error(f"Error in hook_fn: {str(e)}")
            
            for name, module in self.model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    hook = module.register_forward_hook(hook_fn)
                    self.hooks.append(hook)
        except Exception as e:
            model_logger.error(f"Error registering hooks: {str(e)}")
                
    def remove_hooks(self) -> None:
        """Remove all registered hooks with proper error handling"""
        try:
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
        except Exception as e:
            model_logger.error(f"Error removing hooks: {str(e)}")
        
    @timing_decorator
    def profile_forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Profile a forward pass through the model with proper resource management"""
        with cuda_memory_tracking():
            debug_cuda_memory()
            debug_system_resources()
            
            try:
                with torch.no_grad():
                    output = self.model(input_tensor)
                return output
            except Exception as e:
                model_logger.error(f"Error in forward pass: {str(e)}")
                raise
            finally:
                debug_cuda_memory()

def debug_gradient_flow(model: torch.nn.Module) -> Dict[str, float]:
    """Debug gradient flow in the model with proper error handling"""
    try:
        model_logger.info("Gradient Flow Summary:")
        grad_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm
                model_logger.info(f"{name}: gradient norm = {grad_norm:.6f}")
            else:
                model_logger.warning(f"{name}: no gradient")
                grad_norms[name] = 0.0
                
        return grad_norms
    except Exception as e:
        model_logger.error(f"Error getting gradient flow info: {str(e)}")
        return {}

def debug_batch_processing(batch: Any, batch_idx: int) -> Dict[str, Any]:
    """Debug batch processing information with proper error handling"""
    try:
        app_logger.info(f"Processing batch {batch_idx}")
        batch_info = {}
        
        if isinstance(batch, (torch.Tensor, np.ndarray)):
            batch_info['shape'] = batch.shape
            batch_info['dtype'] = str(batch.dtype)
            
            if isinstance(batch, torch.Tensor):
                batch_info['device'] = str(batch.device)
                batch_info['requires_grad'] = batch.requires_grad
                
            app_logger.info(f"Batch shape: {batch.shape}")
            app_logger.info(f"Batch dtype: {batch.dtype}")
            if isinstance(batch, torch.Tensor):
                app_logger.info(f"Device: {batch.device}")
                app_logger.info(f"Requires grad: {batch.requires_grad}")
        else:
            batch_info['type'] = type(batch).__name__
            app_logger.info(f"Batch type: {type(batch)}")
            
        return batch_info
    except Exception as e:
        app_logger.error(f"Error getting batch processing info: {str(e)}")
        return {} 