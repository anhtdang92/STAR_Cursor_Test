import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Optional, Union, List

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        image_size: int,
        channels: int,
        timesteps: int = 1000,
        sampling_timesteps: int = 250,
        loss_type: str = 'l2',
        objective: str = 'pred_noise',
        beta_schedule: str = 'linear',
        p2_loss_weight_gamma: float = 0.0,
        p2_loss_weight_k: float = 1.0
    ):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.loss_type = loss_type
        self.objective = objective
        self.beta_schedule = beta_schedule
        self.p2_loss_weight_gamma = p2_loss_weight_gamma
        self.p2_loss_weight_k = p2_loss_weight_k
        
        # Define beta schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, timesteps)
        elif beta_schedule == 'cosine':
            steps = torch.linspace(0, timesteps, timesteps + 1)
            alpha_bar = torch.cos(((steps / timesteps + 0.008) / 1.008) * torch.pi * 0.5) ** 2
            betas = torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Define alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculate posterior parameters
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from q(x_t|x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        return (
            torch.sqrt(self.alphas_cumprod[t])[:, None, None, None] * x_start +
            torch.sqrt(1 - self.alphas_cumprod[t])[:, None, None, None] * noise
        )
    
    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate loss for training."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted = self.model(x_noisy, t)
        
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
        
        if self.loss_type == 'l1':
            loss = torch.abs(predicted - target).mean()
        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(predicted, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample from p(x_{t-1}|x_t)."""
        b, *_, device = *x.shape, x.device
        model_mean = self.posterior_mean_coef1[t] * self.model(x, t) + self.posterior_mean_coef2[t] * x
        model_log_variance = self.posterior_log_variance_clipped[t]
        
        noise = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
        return model_mean + torch.exp(0.5 * model_log_variance) * noise
    
    def p_sample_loop(self, shape: List[int], device: torch.device) -> torch.Tensor:
        """Sample from p(x_T) and denoise."""
        b = shape[0]
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)
        
        return x
    
    def sample(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from the model."""
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        
        return self.p_sample_loop(x.shape, x.device)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model."""
        return self.p_losses(x, t, noise)

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos(((t + s) / (1 + s)) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def identity(t, *args, **kwargs):
    return t

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

def exists(val):
    return val is not None

def reduce(func, pattern, reduction='mean'):
    if reduction == 'mean':
        return func.mean()
    elif reduction == 'sum':
        return func.sum()
    elif reduction == 'none':
        return func
    else:
        raise ValueError(f'invalid reduction {reduction}')

class ModelPrediction:
    def __init__(self, pred_noise, pred_x_start):
        self.pred_noise = pred_noise
        self.pred_x_start = pred_x_start 