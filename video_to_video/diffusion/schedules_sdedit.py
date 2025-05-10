# Copyright (c) Alibaba, Inc. and its affiliates.

import math

import torch
import numpy as np
from typing import Dict, Any


def betas_to_sigmas(betas):
    return torch.sqrt(1 - torch.cumprod(1 - betas, dim=0))


def sigmas_to_betas(sigmas):
    square_alphas = 1 - sigmas**2
    betas = 1 - torch.cat(
        [square_alphas[:1], square_alphas[1:] / square_alphas[:-1]])
    return betas


def logsnrs_to_sigmas(logsnrs):
    return torch.sqrt(torch.sigmoid(-logsnrs))


def sigmas_to_logsnrs(sigmas):
    square_sigmas = sigmas**2
    return torch.log(square_sigmas / (1 - square_sigmas))


def _logsnr_cosine(n, logsnr_min=-15, logsnr_max=15):
    t_min = math.atan(math.exp(-0.5 * logsnr_min))
    t_max = math.atan(math.exp(-0.5 * logsnr_max))
    t = torch.linspace(1, 0, n)
    logsnrs = -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))
    return logsnrs


def _logsnr_cosine_shifted(n, logsnr_min=-15, logsnr_max=15, scale=2):
    logsnrs = _logsnr_cosine(n, logsnr_min, logsnr_max)
    logsnrs += 2 * math.log(1 / scale)
    return logsnrs


def _logsnr_cosine_interp(n,
                          logsnr_min=-15,
                          logsnr_max=15,
                          scale_min=2,
                          scale_max=4):
    t = torch.linspace(1, 0, n)
    logsnrs_min = _logsnr_cosine_shifted(n, logsnr_min, logsnr_max, scale_min)
    logsnrs_max = _logsnr_cosine_shifted(n, logsnr_min, logsnr_max, scale_max)
    logsnrs = t * logsnrs_min + (1 - t) * logsnrs_max
    return logsnrs


def karras_schedule(n, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    ramp = torch.linspace(1, 0, n)
    min_inv_rho = sigma_min**(1 / rho)
    max_inv_rho = sigma_max**(1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho))**rho
    sigmas = torch.sqrt(sigmas**2 / (1 + sigmas**2))
    return sigmas


def logsnr_cosine_interp_schedule(n,
                                  logsnr_min=-15,
                                  logsnr_max=15,
                                  scale_min=2,
                                  scale_max=4):
    return logsnrs_to_sigmas(
        _logsnr_cosine_interp(n, logsnr_min, logsnr_max, scale_min, scale_max))


def noise_schedule(timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02) -> Dict[str, Any]:
    """Create a noise schedule for the diffusion process."""
    # Create beta schedule
    betas = torch.linspace(beta_start, beta_end, timesteps)
    
    # Calculate alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
    
    # Calculate posterior parameters
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    posterior_log_variance_clipped = torch.log(torch.cat([posterior_variance[1:2], posterior_variance[1:]]))
    posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
    posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1. - alphas_cumprod),
        'log_one_minus_alphas_cumprod': torch.log(1. - alphas_cumprod),
        'sqrt_recip_alphas_cumprod': torch.sqrt(1. / alphas_cumprod),
        'sqrt_recipm1_alphas_cumprod': torch.sqrt(1. / alphas_cumprod - 1),
        'posterior_variance': posterior_variance,
        'posterior_log_variance_clipped': posterior_log_variance_clipped,
        'posterior_mean_coef1': posterior_mean_coef1,
        'posterior_mean_coef2': posterior_mean_coef2,
    }