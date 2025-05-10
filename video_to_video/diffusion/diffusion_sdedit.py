import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np

from .schedules_sdedit import karras_schedule
from .solvers_sdedit import sample_dpmpp_2m_sde, sample_heun

from video_to_video.utils.logger import get_logger

logger = get_logger()

__all__ = ['GaussianDiffusion']


def _i(tensor, t, x):
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    return tensor[t.to(tensor.device)].view(shape).to(x.device)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        schedule: Dict[str, torch.Tensor],
        timesteps: int = 1000,
        loss_type: str = 'l2'
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.num_timesteps = timesteps
        
        # Register all schedule parameters as buffers
        for k, v in schedule.items():
            self.register_buffer(k, v)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from q(x_t|x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        return (
            sqrt_alphas_cumprod_t.view(-1, 1, 1, 1) * x_start +
            sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1) * noise
        )
    
    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate loss for training."""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start, t, noise)
        predicted = self.denoise_fn(x_noisy, t)
        
        if self.loss_type == 'l1':
            loss = F.l1_loss(predicted, noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(predicted, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        return loss
    
    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample from p(x_{t-1}|x_t)."""
        denoise_output = self.denoise_fn(x, t)
        
        alpha = self.alphas[t]
        alpha_prev = self.alphas_cumprod_prev[t]
        sigma = self.posterior_variance[t]
        
        # Compute mean
        c1 = torch.sqrt(alpha_prev) * self.betas[t] / (1. - self.alphas_cumprod[t])
        c2 = torch.sqrt(alpha) * (1. - alpha_prev) / (1. - self.alphas_cumprod[t])
        mean = c1 * x + c2 * denoise_output
        
        # Add noise
        noise = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
        var = torch.sqrt(sigma) if t[0] > 0 else 0.
        
        return mean + var * noise
    
    def p_sample_loop(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Sample from p(x_T) and denoise."""
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
            
        return img
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        return self.p_losses(x, t)

    def diffuse(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        xt = _i(self.sqrt_alphas_cumprod, t, x0) * x0 + _i(self.sqrt_one_minus_alphas_cumprod, t, x0) * noise

        return xt

    def get_velocity(self, x0, xt, t):
        sigmas = _i(self.sqrt_one_minus_alphas_cumprod, t, xt)
        alphas = _i(self.sqrt_alphas_cumprod, t, xt)
        velocity = (alphas * xt - x0) / sigmas
        return velocity

    def get_x0(self, v, xt, t):
        sigmas = _i(self.sqrt_one_minus_alphas_cumprod, t, xt)
        alphas = _i(self.sqrt_alphas_cumprod, t, xt)
        x0 = alphas * xt - sigmas * v
        return x0

    def denoise(self,
                xt,
                t,
                s,
                model,
                model_kwargs={},
                guide_scale=None,
                guide_rescale=None,
                clamp=None,
                percentile=None,
                variant_info=None,):
        s = t - 1 if s is None else s

        # hyperparams
        sigmas = _i(self.sqrt_one_minus_alphas_cumprod, t, xt)
        alphas = _i(self.sqrt_alphas_cumprod, t, xt)
        alphas_s = _i(self.sqrt_one_minus_alphas_cumprod, s.clamp(0), xt)
        alphas_s[s < 0] = 1.
        sigmas_s = torch.sqrt(1 - alphas_s**2)

        # precompute variables
        betas = 1 - (alphas / alphas_s)**2
        coef1 = betas * alphas_s / sigmas**2
        coef2 = (alphas * sigmas_s**2) / (alphas_s * sigmas**2)
        var = betas * (sigmas_s / sigmas)**2
        log_var = torch.log(var).clamp_(-20, 20)

        # prediction
        if guide_scale is None:  
            assert isinstance(model_kwargs, dict)
            out = model(xt, t=t, **model_kwargs)
        else:
            # classifier-free guidance
            assert isinstance(model_kwargs, list)
            if len(model_kwargs) > 3:
                y_out = model(xt, t=t, **model_kwargs[0], **model_kwargs[2], **model_kwargs[3], **model_kwargs[4], **model_kwargs[5])
            else:
                y_out = model(xt, t=t, **model_kwargs[0], **model_kwargs[2], variant_info=variant_info)
            if guide_scale == 1.:
                out = y_out
            else:
                if len(model_kwargs) > 3:
                    u_out = model(xt, t=t, **model_kwargs[1], **model_kwargs[2], **model_kwargs[3], **model_kwargs[4], **model_kwargs[5])
                else:
                    u_out = model(xt, t=t, **model_kwargs[1], **model_kwargs[2], variant_info=variant_info)
                out = u_out + guide_scale * (y_out - u_out)

                if guide_rescale is not None:
                    assert guide_rescale >= 0 and guide_rescale <= 1
                    ratio = (
                        y_out.flatten(1).std(dim=1) /  # noqa
                        (out.flatten(1).std(dim=1) + 1e-12)
                    ).view((-1, ) + (1, ) * (y_out.ndim - 1))
                    out *= guide_rescale * ratio + (1 - guide_rescale) * 1.0

        x0 = alphas * xt - sigmas * out

        # restrict the range of x0
        if percentile is not None:
            assert percentile > 0 and percentile <= 1
            s = torch.quantile(x0.flatten(1).abs(), percentile, dim=1)
            s = s.clamp_(1.0).view((-1, ) + (1, ) * (xt.ndim - 1))
            x0 = torch.min(s, torch.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)

        # recompute eps using the restricted x0
        eps = (xt - alphas * x0) / sigmas

        # compute mu (mean of posterior distribution) using the restricted x0
        mu = coef1 * x0 + coef2 * xt
        return mu, var, log_var, x0, eps


    @torch.no_grad()
    def sample(self,
               noise,
               model,
               model_kwargs={},
               condition_fn=None,
               guide_scale=None,
               guide_rescale=None,
               clamp=None,
               percentile=None,
               solver='euler_a',
               solver_mode='fast',
               steps=20,
               t_max=None,
               t_min=None,
               discretization=None,
               discard_penultimate_step=None,
               return_intermediate=None,
               show_progress=False,
               seed=-1,
               chunk_inds=None,
               **kwargs):
        # sanity check
        assert isinstance(steps, (int, torch.LongTensor))
        assert t_max is None or (t_max > 0 and t_max <= self.num_timesteps - 1)
        assert t_min is None or (t_min >= 0 and t_min < self.num_timesteps - 1)
        assert discretization in (None, 'leading', 'linspace', 'trailing')
        assert discard_penultimate_step in (None, True, False)
        assert return_intermediate in (None, 'x0', 'xt')

        # function of diffusion solver
        solver_fn = {
            'heun': sample_heun,
            'dpmpp_2m_sde': sample_dpmpp_2m_sde
        }[solver]

        # options
        schedule = 'karras' if 'karras' in solver else None
        discretization = discretization or 'linspace'
        seed = seed if seed >= 0 else random.randint(0, 2**31)
        if isinstance(steps, torch.LongTensor):
            discard_penultimate_step = False
        if discard_penultimate_step is None:
            discard_penultimate_step = True if solver in (
                'dpm2', 'dpm2_ancestral', 'dpmpp_2m_sde', 'dpm2_karras',
                'dpm2_ancestral_karras', 'dpmpp_2m_sde_karras') else False

        # function for denoising xt to get x0
        intermediates = []

        def model_fn(xt, sigma):
            # denoising
            t = self._sigma_to_t(sigma).repeat(len(xt)).round().long()
            x0 = self.denoise(xt, t, None, model, model_kwargs, guide_scale,
                              guide_rescale, clamp, percentile)[-2]

            # collect intermediate outputs
            if return_intermediate == 'xt':
                intermediates.append(xt)
            elif return_intermediate == 'x0':
                intermediates.append(x0)
            return x0

        mask_cond = model_kwargs[3]['mask_cond']
        def model_chunk_fn(xt, sigma):
            # denoising
            t = self._sigma_to_t(sigma).repeat(len(xt)).round().long()
            O_LEN = chunk_inds[0][-1]-chunk_inds[1][0]
            cut_f_ind = O_LEN//2

            results_list = []
            for i in range(len(chunk_inds)):
                ind_start, ind_end = chunk_inds[i]
                xt_chunk = xt[:,:,ind_start:ind_end].clone()
                cur_f = xt_chunk.size(2)
                model_kwargs[3]['mask_cond'] = mask_cond[:,ind_start:ind_end].clone()
                x0_chunk = self.denoise(xt_chunk, t, None, model, model_kwargs, guide_scale,
                              guide_rescale, clamp, percentile)[-2]
                if i == 0:
                    results_list.append(x0_chunk[:,:,:cur_f+cut_f_ind-O_LEN])
                elif i == len(chunk_inds)-1:
                    results_list.append(x0_chunk[:,:,cut_f_ind:])
                else:
                    results_list.append(x0_chunk[:,:,cut_f_ind:cur_f+cut_f_ind-O_LEN])
            x0 = torch.concat(results_list, dim=2)
            torch.cuda.empty_cache()
            return x0

        # get timesteps
        if isinstance(steps, int):
            steps += 1 if discard_penultimate_step else 0
            t_max = self.num_timesteps - 1 if t_max is None else t_max
            t_min = 0 if t_min is None else t_min

            # discretize timesteps
            if discretization == 'leading':
                steps = torch.arange(t_min, t_max + 1,
                                     (t_max - t_min + 1) / steps).flip(0)
            elif discretization == 'linspace':
                steps = torch.linspace(t_max, t_min, steps)
            elif discretization == 'trailing':
                steps = torch.arange(t_max, t_min - 1,
                                     -((t_max - t_min + 1) / steps))
                if solver_mode == 'fast':
                    t_mid = 500
                    steps1 = torch.arange(t_max, t_mid - 1,
                                            -((t_max - t_mid + 1) / 4))
                    steps2 = torch.arange(t_mid, t_min - 1,
                                            -((t_mid - t_min + 1) / 11))
                    steps = torch.concat([steps1, steps2])
            else:
                raise NotImplementedError(
                    f'{discretization} discretization not implemented')
            steps = steps.clamp_(t_min, t_max)
        steps = torch.as_tensor(
            steps, dtype=torch.float32, device=noise.device)

        # get sigmas
        sigmas = self._t_to_sigma(steps)
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        if schedule == 'karras':
            if sigmas[0] == float('inf'):
                sigmas = karras_schedule(
                    n=len(steps) - 1,
                    sigma_min=sigmas[sigmas > 0].min().item(),
                    sigma_max=sigmas[sigmas < float('inf')].max().item(),
                    rho=7.).to(sigmas)
                sigmas = torch.cat([
                    sigmas.new_tensor([float('inf')]), sigmas,
                    sigmas.new_zeros([1])
                ])
            else:
                sigmas = karras_schedule(
                    n=len(steps),
                    sigma_min=sigmas[sigmas > 0].min().item(),
                    sigma_max=sigmas.max().item(),
                    rho=7.).to(sigmas)
                sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        if discard_penultimate_step:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        
        fn = model_chunk_fn if chunk_inds is not None else model_fn
        x0 = solver_fn(
            noise, fn, sigmas, show_progress=show_progress, **kwargs)
        return (x0, intermediates) if return_intermediate is not None else x0

    @torch.no_grad()
    def sample_sr(self,
               noise,
               model,
               model_kwargs={},
               condition_fn=None,
               guide_scale=None,
               guide_rescale=None,
               clamp=None,
               percentile=None,
               solver='euler_a',
               solver_mode='fast',
               steps=20,
               t_max=None,
               t_min=None,
               discretization=None,
               discard_penultimate_step=None,
               return_intermediate=None,
               show_progress=False,
               seed=-1,
               chunk_inds=None,
               variant_info=None,
               **kwargs):
        # sanity check
        assert isinstance(steps, (int, torch.LongTensor))
        assert t_max is None or (t_max > 0 and t_max <= self.num_timesteps - 1)
        assert t_min is None or (t_min >= 0 and t_min < self.num_timesteps - 1)
        assert discretization in (None, 'leading', 'linspace', 'trailing')
        assert discard_penultimate_step in (None, True, False)
        assert return_intermediate in (None, 'x0', 'xt')

        # function of diffusion solver
        solver_fn = {
            'heun': sample_heun,
            'dpmpp_2m_sde': sample_dpmpp_2m_sde
        }[solver]

        # options
        schedule = 'karras' if 'karras' in solver else None
        discretization = discretization or 'linspace'
        seed = seed if seed >= 0 else random.randint(0, 2**31)
        if isinstance(steps, torch.LongTensor):
            discard_penultimate_step = False
        if discard_penultimate_step is None:
            discard_penultimate_step = True if solver in (
                'dpm2', 'dpm2_ancestral', 'dpmpp_2m_sde', 'dpm2_karras',
                'dpm2_ancestral_karras', 'dpmpp_2m_sde_karras') else False

        # function for denoising xt to get x0
        intermediates = []

        def model_fn(xt, sigma, variant_info=None):
            # denoising
            t = self._sigma_to_t(sigma).repeat(len(xt)).round().long()
            x0 = self.denoise(xt, t, None, model, model_kwargs, guide_scale,
                              guide_rescale, clamp, percentile, variant_info=variant_info)[-2]

            # collect intermediate outputs
            if return_intermediate == 'xt':
                intermediates.append(xt)
            elif return_intermediate == 'x0':
                print('add intermediate outputs x0')
                intermediates.append(x0)
            return x0

        # mask_cond = model_kwargs[3]['mask_cond']
        def model_chunk_fn(xt, sigma, variant_info=None):
            # denoising
            t = self._sigma_to_t(sigma).repeat(len(xt)).round().long()
            O_LEN = chunk_inds[0][-1]-chunk_inds[1][0]
            cut_f_ind = O_LEN//2

            results_list = []
            for i in range(len(chunk_inds)):
                ind_start, ind_end = chunk_inds[i]
                xt_chunk = xt[:,:,ind_start:ind_end].clone()
                model_kwargs[2]['hint_chunk'] = model_kwargs[2]['hint'][:,:,ind_start:ind_end].clone()  # new added
                cur_f = xt_chunk.size(2)
                # model_kwargs[3]['mask_cond'] = mask_cond[:,ind_start:ind_end].clone()
                x0_chunk = self.denoise(xt_chunk, t, None, model, model_kwargs, guide_scale,
                              guide_rescale, clamp, percentile, variant_info=variant_info)[-2]
                if i == 0:
                    results_list.append(x0_chunk[:,:,:cur_f+cut_f_ind-O_LEN])
                elif i == len(chunk_inds)-1:
                    results_list.append(x0_chunk[:,:,cut_f_ind:])
                else:
                    results_list.append(x0_chunk[:,:,cut_f_ind:cur_f+cut_f_ind-O_LEN])
            x0 = torch.concat(results_list, dim=2)
            torch.cuda.empty_cache()
            return x0

        # get timesteps
        if isinstance(steps, int):
            steps += 1 if discard_penultimate_step else 0
            t_max = self.num_timesteps - 1 if t_max is None else t_max
            t_min = 0 if t_min is None else t_min

            # discretize timesteps
            if discretization == 'leading':
                steps = torch.arange(t_min, t_max + 1,
                                     (t_max - t_min + 1) / steps).flip(0)
            elif discretization == 'linspace':
                steps = torch.linspace(t_max, t_min, steps)
            elif discretization == 'trailing':
                steps = torch.arange(t_max, t_min - 1,
                                     -((t_max - t_min + 1) / steps))
                if solver_mode == 'fast':
                    t_mid = 500
                    steps1 = torch.arange(t_max, t_mid - 1,
                                            -((t_max - t_mid + 1) / 4))
                    steps2 = torch.arange(t_mid, t_min - 1,
                                            -((t_mid - t_min + 1) / 11))
                    steps = torch.concat([steps1, steps2])
            else:
                raise NotImplementedError(
                    f'{discretization} discretization not implemented')
            steps = steps.clamp_(t_min, t_max)
        steps = torch.as_tensor(
            steps, dtype=torch.float32, device=noise.device)

        # get sigmas
        sigmas = self._t_to_sigma(steps)
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        if schedule == 'karras':
            if sigmas[0] == float('inf'):
                sigmas = karras_schedule(
                    n=len(steps) - 1,
                    sigma_min=sigmas[sigmas > 0].min().item(),
                    sigma_max=sigmas[sigmas < float('inf')].max().item(),
                    rho=7.).to(sigmas)
                sigmas = torch.cat([
                    sigmas.new_tensor([float('inf')]), sigmas,
                    sigmas.new_zeros([1])
                ])
            else:
                sigmas = karras_schedule(
                    n=len(steps),
                    sigma_min=sigmas[sigmas > 0].min().item(),
                    sigma_max=sigmas.max().item(),
                    rho=7.).to(sigmas)
                sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        if discard_penultimate_step:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        
        
        fn = model_chunk_fn if chunk_inds is not None else model_fn
        x0 = solver_fn(
            noise, fn, sigmas, variant_info=variant_info, show_progress=show_progress, **kwargs)
        return (x0, intermediates) if return_intermediate is not None else x0


    def _sigma_to_t(self, sigma):
        if sigma == float('inf'):
            t = torch.full_like(sigma, len(self.sqrt_one_minus_alphas_cumprod) - 1)
        else:
            log_sigmas = torch.sqrt(self.sqrt_alphas_cumprod**2 /  # noqa
                                    (1 - self.sqrt_alphas_cumprod**2)).log().to(sigma)
            log_sigma = sigma.log()
            dists = log_sigma - log_sigmas[:, None]
            low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(
                max=log_sigmas.shape[0] - 2)
            high_idx = low_idx + 1
            low, high = log_sigmas[low_idx], log_sigmas[high_idx]
            w = (low - log_sigma) / (low - high)
            w = w.clamp(0, 1)
            t = (1 - w) * low_idx + w * high_idx
            t = t.view(sigma.shape)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        return t

    def _t_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigmas = torch.sqrt(self.sqrt_alphas_cumprod**2 /  # noqa
                                (1 - self.sqrt_alphas_cumprod**2)).log().to(t)
        log_sigma = (1 - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx]
        log_sigma[torch.isnan(log_sigma)
                  | torch.isinf(log_sigma)] = float('inf')
        return log_sigma.exp()